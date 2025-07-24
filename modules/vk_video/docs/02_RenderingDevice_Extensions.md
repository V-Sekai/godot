# RenderingDevice Extensions for Vulkan Video

## Brief Description
Extension of Godot's RenderingDevice class to support Vulkan Video operations for hardware-accelerated video decode and encode.

## API Extensions

### Video Session Management

```cpp
// Video session creation info
struct VideoSessionCreateInfo {
    VkVideoCodecOperationFlagBitsKHR codec_operation;
    VkVideoChromaSubsamplingFlagsKHR chroma_subsampling;
    VkVideoPictureResourceInfoKHR picture_format;
    VkVideoPictureResourceInfoKHR reference_format;
    uint32_t max_coded_extent_width;
    uint32_t max_coded_extent_height;
    uint32_t max_dpb_slots;
    uint32_t max_active_reference_pictures;
};

// RenderingDevice extensions
class RenderingDevice {
public:
    // Video session lifecycle
    RID video_session_create(const VideoSessionCreateInfo& p_info);
    void video_session_destroy(RID p_session);
    bool video_session_is_valid(RID p_session) const;

    // Video session parameters (for codec headers)
    RID video_session_parameters_create(RID p_session, const VideoSessionParametersInfo& p_info);
    void video_session_parameters_update(RID p_params, const VideoParametersUpdateInfo& p_info);
    void video_session_parameters_destroy(RID p_params);
};
```

### Video Command Recording

```cpp
// Video coding begin info
struct VideoCodingBeginInfo {
    RID video_session;
    RID video_session_parameters;
    Vector<VideoPictureResourceInfo> reference_slot_info;
};

// Video decode info
struct VideoDecodeInfo {
    RID src_buffer;                    // Bitstream buffer
    uint32_t src_buffer_offset;
    uint32_t src_buffer_range;
    VideoPictureResourceInfo dst_picture_resource;
    Vector<VideoPictureResourceInfo> setup_reference_slot_info;
    Vector<VideoPictureResourceInfo> reference_slot_info;
};

// Command buffer extensions
class RenderingDevice {
public:
    // Video command recording
    void video_cmd_begin_coding(CommandBufferID p_cmd_buffer, const VideoCodingBeginInfo& p_info);
    void video_cmd_decode_frame(CommandBufferID p_cmd_buffer, const VideoDecodeInfo& p_info);
    void video_cmd_encode_frame(CommandBufferID p_cmd_buffer, const VideoEncodeInfo& p_info);
    void video_cmd_end_coding(CommandBufferID p_cmd_buffer);

    // Video memory barriers
    void video_cmd_pipeline_barrier(CommandBufferID p_cmd_buffer,
                                   const Vector<VideoImageMemoryBarrier>& p_image_barriers);
};
```

### Video Resource Creation

```cpp
// Video image creation info
struct VideoImageCreateInfo {
    VkImageType image_type;
    VkFormat format;
    VkExtent3D extent;
    uint32_t mip_levels;
    uint32_t array_layers;
    VkSampleCountFlagBits samples;
    VkImageUsageFlags usage;
    VkVideoProfileListInfoKHR video_profile_list;
};

// Video buffer creation info
struct VideoBufferCreateInfo {
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    VkVideoProfileListInfoKHR video_profile_list;
};

// Resource creation extensions
class RenderingDevice {
public:
    // Video-specific resource creation
    RID video_image_create(const VideoImageCreateInfo& p_info);
    RID video_buffer_create(const VideoBufferCreateInfo& p_info);

    // Video capability queries
    VideoCapabilities video_get_decode_capabilities(VkVideoCodecOperationFlagBitsKHR p_codec);
    VideoCapabilities video_get_encode_capabilities(VkVideoCodecOperationFlagBitsKHR p_codec);
    bool video_format_supported(VkFormat p_format, VkImageUsageFlags p_usage,
                                VkVideoProfileListInfoKHR p_profile_list);
};
```

## Implementation Details

### Video Session Implementation

```cpp
// Internal video session structure
struct VideoSession {
    RDD::VideoSessionID driver_id;
    VkVideoSessionKHR vk_session;
    VkVideoCodecOperationFlagBitsKHR codec_operation;
    VkExtent2D max_coded_extent;
    uint32_t max_dpb_slots;
    uint32_t max_active_reference_pictures;
    VkDeviceMemory session_memory;
    Vector<VkDeviceMemory> bound_memory;
};

// Video session creation
RID RenderingDevice::video_session_create(const VideoSessionCreateInfo& p_info) {
    VideoSession session;

    // Create Vulkan video session
    VkVideoSessionCreateInfoKHR create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR;
    create_info.queueFamilyIndex = get_video_decode_queue_family_index();
    create_info.flags = 0;
    create_info.pVideoProfile = &p_info.video_profile;
    create_info.pictureFormat = p_info.picture_format.imageViewBinding.format;
    create_info.maxCodedExtent = {p_info.max_coded_extent_width, p_info.max_coded_extent_height};
    create_info.referencePictureFormat = p_info.reference_format.imageViewBinding.format;
    create_info.maxDpbSlots = p_info.max_dpb_slots;
    create_info.maxActiveReferencePictures = p_info.max_active_reference_pictures;

    VkResult result = vkCreateVideoSessionKHR(device, &create_info, nullptr, &session.vk_session);
    ERR_FAIL_COND_V(result != VK_SUCCESS, RID());

    // Allocate and bind memory
    _video_session_allocate_memory(session);

    return video_session_owner.make_rid(session);
}
```

### Command Buffer Extensions

```cpp
// Video command recording implementation
void RenderingDevice::video_cmd_begin_coding(CommandBufferID p_cmd_buffer,
                                            const VideoCodingBeginInfo& p_info) {
    VideoSession* session = video_session_owner.get_or_null(p_info.video_session);
    ERR_FAIL_NULL(session);

    VkVideoBeginCodingInfoKHR begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR;
    begin_info.videoSession = session->vk_session;
    begin_info.videoSessionParameters = VK_NULL_HANDLE; // Set if parameters provided
    begin_info.referenceSlotCount = p_info.reference_slot_info.size();

    // Convert reference slot info
    Vector<VkVideoReferenceSlotInfoKHR> reference_slots;
    for (const auto& slot_info : p_info.reference_slot_info) {
        VkVideoReferenceSlotInfoKHR slot = {};
        slot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
        slot.slotIndex = slot_info.slot_index;
        slot.pPictureResource = &slot_info.picture_resource;
        reference_slots.push_back(slot);
    }
    begin_info.pReferenceSlots = reference_slots.ptr();

    vkCmdBeginVideoCodingKHR(p_cmd_buffer, &begin_info);
}

void RenderingDevice::video_cmd_decode_frame(CommandBufferID p_cmd_buffer,
                                           const VideoDecodeInfo& p_info) {
    VkVideoDecodeInfoKHR decode_info = {};
    decode_info.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR;

    // Set up source buffer info
    decode_info.src.sType = VK_STRUCTURE_TYPE_VIDEO_INLINE_QUERY_INFO_KHR;
    decode_info.src.buffer = _get_buffer_from_rid(p_info.src_buffer);
    decode_info.src.offset = p_info.src_buffer_offset;
    decode_info.src.range = p_info.src_buffer_range;

    // Set up destination picture resource
    decode_info.dstPictureResource = p_info.dst_picture_resource;

    // Set up reference pictures
    Vector<VkVideoReferenceSlotInfoKHR> reference_slots;
    for (const auto& ref_info : p_info.reference_slot_info) {
        VkVideoReferenceSlotInfoKHR slot = {};
        slot.sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR;
        slot.slotIndex = ref_info.slot_index;
        slot.pPictureResource = &ref_info.picture_resource;
        reference_slots.push_back(slot);
    }
    decode_info.referenceSlotCount = reference_slots.size();
    decode_info.pReferenceSlots = reference_slots.ptr();

    vkCmdDecodeVideoKHR(p_cmd_buffer, &decode_info);
}
```

### Resource Management

```cpp
// Video image creation with profile support
RID RenderingDevice::video_image_create(const VideoImageCreateInfo& p_info) {
    VkImageCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    create_info.pNext = &p_info.video_profile_list;
    create_info.imageType = p_info.image_type;
    create_info.format = p_info.format;
    create_info.extent = p_info.extent;
    create_info.mipLevels = p_info.mip_levels;
    create_info.arrayLayers = p_info.array_layers;
    create_info.samples = p_info.samples;
    create_info.usage = p_info.usage;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage vk_image;
    VkResult result = vkCreateImage(device, &create_info, nullptr, &vk_image);
    ERR_FAIL_COND_V(result != VK_SUCCESS, RID());

    // Allocate and bind memory
    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device, vk_image, &mem_requirements);

    VkDeviceMemory image_memory = _allocate_video_memory(mem_requirements);
    vkBindImageMemory(device, vk_image, image_memory, 0);

    // Create texture object
    Texture texture;
    texture.driver_id = vk_image;
    texture.type = _convert_image_type(p_info.image_type);
    texture.format = _convert_vk_format(p_info.format);
    texture.width = p_info.extent.width;
    texture.height = p_info.extent.height;
    texture.depth = p_info.extent.depth;
    texture.layers = p_info.array_layers;
    texture.mipmaps = p_info.mip_levels;
    texture.usage_flags = _convert_usage_flags(p_info.usage);

    return texture_owner.make_rid(texture);
}
```

## Integration with Existing Systems

### Command Graph Integration

```cpp
// Video operations integrate with existing command graph
void RenderingDevice::_record_video_decode_commands() {
    // Video decode operations are recorded as part of the command graph
    draw_graph.add_video_decode_pass([&](RDD::CommandBufferID cmd_buffer) {
        video_cmd_begin_coding(cmd_buffer, coding_begin_info);
        video_cmd_decode_frame(cmd_buffer, decode_info);
        video_cmd_end_coding(cmd_buffer);
    });
}
```

### Memory Management Integration

```cpp
// Video memory integrates with existing memory tracking
uint64_t RenderingDevice::get_memory_usage(MemoryType p_type) const {
    switch (p_type) {
        case MEMORY_TEXTURES:
            return texture_memory + video_image_memory;
        case MEMORY_BUFFERS:
            return buffer_memory + video_buffer_memory;
        case MEMORY_VIDEO:
            return video_session_memory + video_image_memory + video_buffer_memory;
        default:
            return texture_memory + buffer_memory + video_session_memory +
                   video_image_memory + video_buffer_memory;
    }
}
```

## Error Handling

### Capability Validation

```cpp
// Validate video capabilities before session creation
Error RenderingDevice::_validate_video_session_create_info(const VideoSessionCreateInfo& p_info) {
    VideoCapabilities caps = video_get_decode_capabilities(p_info.codec_operation);

    ERR_FAIL_COND_V(!caps.codec_supported, ERR_UNAVAILABLE);
    ERR_FAIL_COND_V(p_info.max_coded_extent_width > caps.max_coded_extent.width, ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(p_info.max_coded_extent_height > caps.max_coded_extent.height, ERR_INVALID_PARAMETER);
    ERR_FAIL_COND_V(p_info.max_dpb_slots > caps.max_dpb_slots, ERR_INVALID_PARAMETER);

    return OK;
}
```

### Resource Cleanup

```cpp
// Proper cleanup of video resources
void RenderingDevice::_free_video_session(VideoSession* p_session) {
    if (p_session->vk_session != VK_NULL_HANDLE) {
        vkDestroyVideoSessionKHR(device, p_session->vk_session, nullptr);
    }

    for (VkDeviceMemory memory : p_session->bound_memory) {
        vkFreeMemory(device, memory, nullptr);
    }

    p_session->bound_memory.clear();
}
```

This extension provides a clean, type-safe interface for Vulkan Video operations while integrating seamlessly with Godot's existing rendering architecture.
