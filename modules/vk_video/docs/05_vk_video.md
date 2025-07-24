### **Phase 5A: Vulkan Video Foundation (3-4 days)**

**Priority: CRITICAL - Everything depends on this**

#### Step 1: Extend RenderingDeviceDriverVulkan with Video Extensions

```cpp
// In RenderingDeviceDriverVulkan::_initialize_device_extensions()
void RenderingDeviceDriverVulkan::_register_video_extensions() {
    // Core video extensions
    _register_requested_device_extension("VK_KHR_video_queue", false);
    _register_requested_device_extension("VK_KHR_video_decode_queue", false);

    // AV1 decode support
    _register_requested_device_extension("VK_KHR_video_decode_av1", false);

    // YCbCr conversion support (needed for Phase 5C)
    _register_requested_device_extension("VK_KHR_sampler_ycbcr_conversion", false);

    // Video maintenance extensions
    _register_requested_device_extension("VK_KHR_video_maintenance1", false);
}
```

#### Step 2: Load Video Function Pointers

```cpp
// Add to DeviceFunctions struct
struct DeviceFunctions {
    // Existing functions...

    // Video queue functions
    PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR GetPhysicalDeviceVideoCapabilitiesKHR = nullptr;
    PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR GetPhysicalDeviceVideoFormatPropertiesKHR = nullptr;

    // Video session functions
    PFN_vkCreateVideoSessionKHR CreateVideoSessionKHR = nullptr;
    PFN_vkDestroyVideoSessionKHR DestroyVideoSessionKHR = nullptr;
    PFN_vkCreateVideoSessionParametersKHR CreateVideoSessionParametersKHR = nullptr;
    PFN_vkDestroyVideoSessionParametersKHR DestroyVideoSessionParametersKHR = nullptr;

    // Video decode functions
    PFN_vkCmdDecodeVideoKHR CmdDecodeVideoKHR = nullptr;
    PFN_vkCmdBeginVideoCodingKHR CmdBeginVideoCodingKHR = nullptr;
    PFN_vkCmdEndVideoCodingKHR CmdEndVideoCodingKHR = nullptr;

    // YCbCr conversion functions
    PFN_vkCreateSamplerYcbcrConversionKHR CreateSamplerYcbcrConversionKHR = nullptr;
    PFN_vkDestroySamplerYcbcrConversionKHR DestroySamplerYcbcrConversionKHR = nullptr;
};

// Load in _initialize_device()
void RenderingDeviceDriverVulkan::_load_video_functions() {
    // Video capabilities
    device_functions.GetPhysicalDeviceVideoCapabilitiesKHR =
        (PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR)vkGetDeviceProcAddr(vk_device, "vkGetPhysicalDeviceVideoCapabilitiesKHR");

    // Video session management
    device_functions.CreateVideoSessionKHR =
        (PFN_vkCreateVideoSessionKHR)vkGetDeviceProcAddr(vk_device, "vkCreateVideoSessionKHR");
    device_functions.DestroyVideoSessionKHR =
        (PFN_vkDestroyVideoSessionKHR)vkGetDeviceProcAddr(vk_device, "vkDestroyVideoSessionKHR");

    // Video decode commands
    device_functions.CmdDecodeVideoKHR =
        (PFN_vkCmdDecodeVideoKHR)vkGetDeviceProcAddr(vk_device, "vkCmdDecodeVideoKHR");
    device_functions.CmdBeginVideoCodingKHR =
        (PFN_vkCmdBeginVideoCodingKHR)vkGetDeviceProcAddr(vk_device, "vkCmdBeginVideoCodingKHR");
    device_functions.CmdEndVideoCodingKHR =
        (PFN_vkCmdEndVideoCodingKHR)vkGetDeviceProcAddr(vk_device, "vkCmdEndVideoCodingKHR");

    // YCbCr conversion
    device_functions.CreateSamplerYcbcrConversionKHR =
        (PFN_vkCreateSamplerYcbcrConversionKHR)vkGetDeviceProcAddr(vk_device, "vkCreateSamplerYcbcrConversionKHR");
    device_functions.DestroySamplerYcbcrConversionKHR =
        (PFN_vkDestroySamplerYcbcrConversionKHR)vkGetDeviceProcAddr(vk_device, "vkDestroySamplerYcbcrConversionKHR");
}
```

#### Step 3: Add Video Resource Types

```cpp
// Add to RenderingDeviceDriverVulkan
struct VideoSessionInfo {
    VkVideoSessionKHR vk_video_session = VK_NULL_HANDLE;
    VkVideoSessionParametersKHR vk_parameters = VK_NULL_HANDLE;
    VkVideoProfileInfoKHR profile_info = {};
    uint32_t dpb_slot_count = 0;
    VkFormat decode_output_format = VK_FORMAT_UNDEFINED;
};

struct VideoImageInfo {
    VkImage vk_image = VK_NULL_HANDLE;
    VkImageView vk_image_view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Add to VersatileResource template
using VersatileResource = VersatileResourceTemplate<
    BufferInfo,
    TextureInfo,
    VertexFormatInfo,
    ShaderInfo,
    UniformSetInfo,
    RenderPassInfo,
    CommandBufferInfo,
    VideoSessionInfo,    // NEW
    VideoImageInfo       // NEW
>;
```

### **Phase 5B: Video Session Management (2-3 days)**

**Depends on: Phase 5A complete**

#### Step 1: Implement Video Session Creation

```cpp
// Add to RenderingDeviceDriverVulkan
VideoSessionID RenderingDeviceDriverVulkan::video_session_create(const VideoSessionCreateInfo &p_info) {
    // 1. Query video capabilities
    VkVideoCapabilitiesKHR video_caps = {};
    video_caps.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;

    VkVideoDecodeCapabilitiesKHR decode_caps = {};
    decode_caps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR;
    video_caps.pNext = &decode_caps;

    VkVideoDecodeAV1CapabilitiesKHR av1_caps = {};
    av1_caps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR;
    decode_caps.pNext = &av1_caps;

    VkResult result = device_functions.GetPhysicalDeviceVideoCapabilitiesKHR(
        physical_device, &p_info.profile_info, &video_caps);
    ERR_FAIL_COND_V(result != VK_SUCCESS, VideoSessionID());

    // 2. Create video session
    VkVideoSessionCreateInfoKHR session_create_info = {};
    session_create_info.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR;
    session_create_info.queueFamilyIndex = /* video decode queue family */;
    session_create_info.pVideoProfile = &p_info.profile_info;
    session_create_info.pictureFormat = p_info.output_format;
    session_create_info.maxCodedExtent = { p_info.max_width, p_info.max_height };
    session_create_info.referencePictureFormat = p_info.dpb_format;
    session_create_info.maxDpbSlots = p_info.max_dpb_slots;
    session_create_info.maxActiveReferencePictures = p_info.max_active_refs;

    VkVideoSessionKHR video_session;
    result = device_functions.CreateVideoSessionKHR(vk_device, &session_create_info, nullptr, &video_session);
    ERR_FAIL_COND_V(result != VK_SUCCESS, VideoSessionID());

    // 3. Store in resource system
    VideoSessionInfo *session_info = VersatileResource::allocate<VideoSessionInfo>(resources_allocator);
    session_info->vk_video_session = video_session;
    session_info->profile_info = p_info.profile_info;
    session_info->dpb_slot_count = p_info.max_dpb_slots;
    session_info->decode_output_format = p_info.output_format;

    return VideoSessionID(session_info);
}
```

#### Step 2: Add Video Queue Family Detection

```cpp
// In RenderingDeviceDriverVulkan::initialize()
void RenderingDeviceDriverVulkan::_detect_video_queue_families() {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

    Vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptrw());

    for (uint32_t i = 0; i < queue_family_count; i++) {
        // Check for video decode support
        if (queue_families[i].queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) {
            video_decode_queue_family = i;
            print_verbose("Found video decode queue family: " + itos(i));
        }
    }
}
```

### **Phase 5C: AV1 Decode Implementation (2-3 days)**

**Depends on: Phase 5B complete**

#### Step 1: Connect AV1VulkanDecoder to Real Video Session

```cpp
// Update AV1VulkanDecoder::initialize()
bool AV1VulkanDecoder::initialize(uint32_t width, uint32_t height) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();

    // Create video session for AV1 decode
    Dictionary session_info;
    session_info["codec"] = "AV1";
    session_info["profile"] = "MAIN"; // AV1 Main profile
    session_info["max_width"] = width;
    session_info["max_height"] = height;
    session_info["output_format"] = "NV12"; // YCbCr 4:2:0
    session_info["max_dpb_slots"] = 8; // AV1 reference frames
    session_info["max_active_refs"] = 7; // AV1 allows up to 7 active refs

    video_session = rd->video_session_create(session_info);
    if (!video_session.is_valid()) {
        ERR_PRINT("Failed to create AV1 video session");
        return false;
    }

    // Create decode output images (YCbCr format)
    current_output_texture = _create_video_output_texture(width, height);

    return true;
}
```

#### Step 2: Implement Real AV1 Frame Decode

```cpp
bool AV1VulkanDecoder::decode_frame(const WebMFrame &frame) {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();

    // Parse AV1 bitstream for decode parameters
    AV1DecodeParameters decode_params;
    if (!_parse_av1_frame(frame.buffer, frame.bufferSize, decode_params)) {
        ERR_PRINT("Failed to parse AV1 frame");
        return false;
    }

    // Create bitstream buffer
    RID bitstream_buffer = rd->video_buffer_create(frame.bufferSize, VIDEO_BUFFER_USAGE_BITSTREAM);
    rd->buffer_update(bitstream_buffer, 0, frame.bufferSize, frame.buffer);

    // Begin video decode command recording
    RD::CommandBufferID cmd_buffer = rd->command_buffer_create(video_command_pool);
    rd->command_buffer_begin(cmd_buffer);

    // Begin video coding scope
    Dictionary begin_info;
    begin_info["session"] = video_session;
    begin_info["reference_slots"] = _get_reference_slots();
    rd->command_begin_video_coding(cmd_buffer, begin_info);

    // Record decode command
    Dictionary decode_info;
    decode_info["bitstream_buffer"] = bitstream_buffer;
    decode_info["output_texture"] = current_output_texture;
    decode_info["decode_parameters"] = _av1_params_to_dict(decode_params);
    decode_info["reference_pictures"] = _get_reference_pictures();

    bool success = rd->command_decode_video(cmd_buffer, decode_info);

    // End video coding scope
    rd->command_end_video_coding(cmd_buffer);
    rd->command_buffer_end(cmd_buffer);

    // Submit and wait
    rd->command_queue_submit(video_queue, cmd_buffer);

    return success;
}
```

### **Phase 5D: YCbCr Sampler Integration (1-2 days)**

**Depends on: Phase 5C complete (need decoded YCbCr textures)**

#### Step 1: Create YCbCr Sampler for Decoded Textures

```cpp
// Now we can create YCbCr samplers for the decoded video textures
bool AV1VulkanDecoder::_create_ycbcr_sampler() {
    RenderingDevice *rd = RenderingServer::get_singleton()->get_rendering_device();

    // Create YCbCr sampler for the decoded NV12 texture
    Dictionary sampler_info;
    sampler_info["format"] = "NV12"; // Matches decode output format
    sampler_info["model"] = "BT709"; // Standard for most content
    sampler_info["range"] = "LIMITED"; // Video range
    sampler_info["chroma_filter"] = "LINEAR";
    sampler_info["x_chroma_offset"] = "COSITED_EVEN";
    sampler_info["y_chroma_offset"] = "COSITED_EVEN";

    ycbcr_sampler = rd->ycbcr_sampler_create(sampler_info);
    return ycbcr_sampler.is_valid();
}

RID AV1VulkanDecoder::get_texture_for_rendering() {
    // Return the YCbCr texture that can be sampled directly
    // The YCbCr→RGB conversion happens in the sampler
    return current_output_texture;
}

RID AV1VulkanDecoder::get_ycbcr_sampler() {
    return ycbcr_sampler;
}
```

#### Step 2: Update VideoStreamAV1 to Use YCbCr Sampling

```cpp
void VideoStreamAV1::_update_texture() {
    if (!decoder || !decoder->has_new_frame()) {
        return;
    }

    // Get YCbCr texture and sampler from decoder
    RID ycbcr_texture = decoder->get_texture_for_rendering();
    RID ycbcr_sampler = decoder->get_ycbcr_sampler();

    // Update material to use YCbCr texture with proper sampler
    if (video_material.is_valid()) {
        video_material->set_shader_parameter("video_texture", ycbcr_texture);
        video_material->set_shader_parameter("ycbcr_sampler", ycbcr_sampler);
    }
}
```

### **Phase 5E: Performance & Polish (1-2 days)**

**Depends on: All previous phases complete**

#### Final Integration and Optimization

-   Memory pooling for video textures
-   Frame timing synchronization
-   Error handling and fallbacks
-   Performance profiling and tuning
