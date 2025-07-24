# Phase 5: Vulkan Video Integration

## Overview

Phase 5 implements hardware-accelerated AV1 video decoding using Vulkan Video extensions. This phase integrates the Khronos Vulkan Video API with Godot's RenderingDevice to provide native GPU-accelerated video decode with efficient YCbCr→RGB conversion.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Godot Video Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│ VideoStreamAV1 → AV1VulkanDecoder → RenderingDevice        │
│                                   ↓                        │
│                        Vulkan Video Extensions             │
│                                   ↓                        │
│              ┌─────────────────────────────────────────┐    │
│              │         Hardware Decode Path           │    │
│              │                                         │    │
│              │  AV1 Bitstream → GPU Decoder →         │    │
│              │  YCbCr Texture → YCbCr Sampler →       │    │
│              │  RGB Output (automatic conversion)     │    │
│              └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Key Technologies

- **VK_KHR_video_queue**: Core video processing queue support
- **VK_KHR_video_decode_queue**: Video decode operations
- **VK_KHR_video_decode_av1**: AV1 codec-specific decode support
- **VK_KHR_sampler_ycbcr_conversion**: Hardware YCbCr→RGB conversion
- **VK_KHR_video_maintenance1**: Enhanced video features and bug fixes

## Implementation Phases

### Phase 5A: Vulkan Video Foundation (3-4 days)

**Priority: CRITICAL** - All subsequent phases depend on this foundation.

#### Step 1: Extension Registration

```cpp
// In RenderingDeviceDriverVulkan::_initialize_device_extensions()
void RenderingDeviceDriverVulkan::_register_video_extensions() {
    // Core video extensions
    _register_requested_device_extension("VK_KHR_video_queue", false);
    _register_requested_device_extension("VK_KHR_video_decode_queue", false);
    
    // AV1 decode support
    _register_requested_device_extension("VK_KHR_video_decode_av1", false);
    
    // YCbCr conversion support (needed for Phase 5D)
    _register_requested_device_extension("VK_KHR_sampler_ycbcr_conversion", false);
    
    // Video maintenance extensions
    _register_requested_device_extension("VK_KHR_video_maintenance1", false);
    
    // Log extension availability
    if (is_device_extension_enabled("VK_KHR_video_decode_av1")) {
        print_verbose("Vulkan Video: AV1 hardware decode available");
    } else {
        print_verbose("Vulkan Video: AV1 hardware decode not available, will use software fallback");
    }
}
```

#### Step 2: Function Pointer Loading

```cpp
// Add to DeviceFunctions struct in rendering_device_driver_vulkan.h
struct DeviceFunctions {
    // Existing functions...
    
    // Video capabilities
    PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR GetPhysicalDeviceVideoCapabilitiesKHR = nullptr;
    PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR GetPhysicalDeviceVideoFormatPropertiesKHR = nullptr;
    
    // Video session management
    PFN_vkCreateVideoSessionKHR CreateVideoSessionKHR = nullptr;
    PFN_vkDestroyVideoSessionKHR DestroyVideoSessionKHR = nullptr;
    PFN_vkCreateVideoSessionParametersKHR CreateVideoSessionParametersKHR = nullptr;
    PFN_vkDestroyVideoSessionParametersKHR DestroyVideoSessionParametersKHR = nullptr;
    PFN_vkUpdateVideoSessionParametersKHR UpdateVideoSessionParametersKHR = nullptr;
    
    // Video memory management
    PFN_vkGetVideoSessionMemoryRequirementsKHR GetVideoSessionMemoryRequirementsKHR = nullptr;
    PFN_vkBindVideoSessionMemoryKHR BindVideoSessionMemoryKHR = nullptr;
    
    // Video decode commands
    PFN_vkCmdDecodeVideoKHR CmdDecodeVideoKHR = nullptr;
    PFN_vkCmdBeginVideoCodingKHR CmdBeginVideoCodingKHR = nullptr;
    PFN_vkCmdEndVideoCodingKHR CmdEndVideoCodingKHR = nullptr;
    PFN_vkCmdControlVideoCodingKHR CmdControlVideoCodingKHR = nullptr;
    
    // YCbCr conversion functions
    PFN_vkCreateSamplerYcbcrConversionKHR CreateSamplerYcbcrConversionKHR = nullptr;
    PFN_vkDestroySamplerYcbcrConversionKHR DestroySamplerYcbcrConversionKHR = nullptr;
};

// Load in _initialize_device()
void RenderingDeviceDriverVulkan::_load_video_functions() {
    // Video capabilities
    device_functions.GetPhysicalDeviceVideoCapabilitiesKHR = 
        (PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR)vkGetInstanceProcAddr(vk_instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR");
    device_functions.GetPhysicalDeviceVideoFormatPropertiesKHR = 
        (PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR)vkGetInstanceProcAddr(vk_instance, "vkGetPhysicalDeviceVideoFormatPropertiesKHR");
    
    // Video session management
    device_functions.CreateVideoSessionKHR = 
        (PFN_vkCreateVideoSessionKHR)vkGetDeviceProcAddr(vk_device, "vkCreateVideoSessionKHR");
    device_functions.DestroyVideoSessionKHR = 
        (PFN_vkDestroyVideoSessionKHR)vkGetDeviceProcAddr(vk_device, "vkDestroyVideoSessionKHR");
    device_functions.CreateVideoSessionParametersKHR = 
        (PFN_vkCreateVideoSessionParametersKHR)vkGetDeviceProcAddr(vk_device, "vkCreateVideoSessionParametersKHR");
    device_functions.DestroyVideoSessionParametersKHR = 
        (PFN_vkDestroyVideoSessionParametersKHR)vkGetDeviceProcAddr(vk_device, "vkDestroyVideoSessionParametersKHR");
    device_functions.UpdateVideoSessionParametersKHR = 
        (PFN_vkUpdateVideoSessionParametersKHR)vkGetDeviceProcAddr(vk_device, "vkUpdateVideoSessionParametersKHR");
    
    // Video memory management
    device_functions.GetVideoSessionMemoryRequirementsKHR = 
        (PFN_vkGetVideoSessionMemoryRequirementsKHR)vkGetDeviceProcAddr(vk_device, "vkGetVideoSessionMemoryRequirementsKHR");
    device_functions.BindVideoSessionMemoryKHR = 
        (PFN_vkBindVideoSessionMemoryKHR)vkGetDeviceProcAddr(vk_device, "vkBindVideoSessionMemoryKHR");
    
    // Video decode commands
    device_functions.CmdDecodeVideoKHR = 
        (PFN_vkCmdDecodeVideoKHR)vkGetDeviceProcAddr(vk_device, "vkCmdDecodeVideoKHR");
    device_functions.CmdBeginVideoCodingKHR = 
        (PFN_vkCmdBeginVideoCodingKHR)vkGetDeviceProcAddr(vk_device, "vkCmdBeginVideoCodingKHR");
    device_functions.CmdEndVideoCodingKHR = 
        (PFN_vkCmdEndVideoCodingKHR)vkGetDeviceProcAddr(vk_device, "vkCmdEndVideoCodingKHR");
    device_functions.CmdControlVideoCodingKHR = 
        (PFN_vkCmdControlVideoCodingKHR)vkGetDeviceProcAddr(vk_device, "vkCmdControlVideoCodingKHR");
    
    // YCbCr conversion
    device_functions.CreateSamplerYcbcrConversionKHR = 
        (PFN_vkCreateSamplerYcbcrConversionKHR)vkGetDeviceProcAddr(vk_device, "vkCreateSamplerYcbcrConversionKHR");
    device_functions.DestroySamplerYcbcrConversionKHR = 
        (PFN_vkDestroySamplerYcbcrConversionKHR)vkGetDeviceProcAddr(vk_device, "vkDestroySamplerYcbcrConversionKHR");
    
    // Validate critical functions
    ERR_FAIL_COND_MSG(!device_functions.CreateVideoSessionKHR, "Failed to load vkCreateVideoSessionKHR");
    ERR_FAIL_COND_MSG(!device_functions.CmdDecodeVideoKHR, "Failed to load vkCmdDecodeVideoKHR");
}
```

#### Step 3: Video Queue Family Detection

```cpp
// Add to RenderingDeviceDriverVulkan class
uint32_t video_decode_queue_family = UINT32_MAX;
VkQueue video_decode_queue = VK_NULL_HANDLE;

// In _initialize_device()
void RenderingDeviceDriverVulkan::_detect_video_queue_families() {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    
    Vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptrw());
    
    for (uint32_t i = 0; i < queue_family_count; i++) {
        // Check for video decode support
        if (queue_families[i].queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) {
            video_decode_queue_family = i;
            print_verbose("Vulkan Video: Found video decode queue family: " + itos(i));
            
            // Get the video decode queue
            vkGetDeviceQueue(vk_device, i, 0, &video_decode_queue);
            break;
        }
    }
    
    if (video_decode_queue_family == UINT32_MAX) {
        print_verbose("Vulkan Video: No video decode queue family found, hardware decode unavailable");
    }
}
```

#### Step 4: Video Resource Types

```cpp
// Add to rendering_device_driver_vulkan.h
struct VideoSessionInfo {
    VkVideoSessionKHR vk_video_session = VK_NULL_HANDLE;
    VkVideoSessionParametersKHR vk_parameters = VK_NULL_HANDLE;
    VkVideoProfileInfoKHR profile_info = {};
    VkVideoDecodeAV1ProfileInfoKHR av1_profile_info = {};
    uint32_t dpb_slot_count = 0;
    VkFormat decode_output_format = VK_FORMAT_UNDEFINED;
    VkFormat dpb_format = VK_FORMAT_UNDEFINED;
    VkExtent2D max_coded_extent = {};
    Vector<VkDeviceMemory> session_memory;
};

struct VideoImageInfo {
    VkImage vk_image = VK_NULL_HANDLE;
    VkImageView vk_image_view = VK_NULL_HANDLE;
    VkDeviceMemory vk_memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t array_layers = 1;
    bool is_video_decode_output = false;
    bool is_video_dpb = false;
};

struct YCbCrSamplerInfo {
    VkSamplerYcbcrConversion vk_conversion = VK_NULL_HANDLE;
    VkSampler vk_sampler = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkSamplerYcbcrModelConversion model = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709;
    VkSamplerYcbcrRange range = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
    VkChromaLocation x_chroma_offset = VK_CHROMA_LOCATION_COSITED_EVEN;
    VkChromaLocation y_chroma_offset = VK_CHROMA_LOCATION_COSITED_EVEN;
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
    VideoImageInfo,      // NEW
    YCbCrSamplerInfo     // NEW
>;
```

### Phase 5B: Video Session Management (2-3 days)

**Depends on: Phase 5A complete**

#### Step 1: Video Capabilities Query

```cpp
// Add to RenderingDeviceDriverVulkan
bool RenderingDeviceDriverVulkan::_query_av1_decode_capabilities(
    VkVideoCapabilitiesKHR *video_caps,
    VkVideoDecodeCapabilitiesKHR *decode_caps,
    VkVideoDecodeAV1CapabilitiesKHR *av1_caps) {
    
    // Setup AV1 profile
    VkVideoDecodeAV1ProfileInfoKHR av1_profile = {};
    av1_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
    av1_profile.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
    av1_profile.filmGrainSupport = VK_FALSE; // Start without film grain
    
    VkVideoProfileInfoKHR profile_info = {};
    profile_info.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
    profile_info.pNext = &av1_profile;
    profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
    profile_info.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
    profile_info.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
    profile_info.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
    
    // Query capabilities
    av1_caps->sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR;
    av1_caps->pNext = nullptr;
    
    decode_caps->sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR;
    decode_caps->pNext = av1_caps;
    
    video_caps->sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
    video_caps->pNext = decode_caps;
    
    VkResult result = device_functions.GetPhysicalDeviceVideoCapabilitiesKHR(
        physical_device, &profile_info, video_caps);
    
    if (result != VK_SUCCESS) {
        print_verbose("Vulkan Video: Failed to query AV1 decode capabilities");
        return false;
    }
    
    // Log capabilities
    print_verbose(vformat("Vulkan Video AV1 Capabilities:"));
    print_verbose(vformat("  Max coded extent: %dx%d", video_caps->maxCodedExtent.width, video_caps->maxCodedExtent.height));
    print_verbose(vformat("  Max DPB slots: %d", video_caps->maxDpbSlots));
    print_verbose(vformat("  Max active references: %d", video_caps->maxActiveReferencePictures));
    print_verbose(vformat("  Max level: %d", av1_caps->maxLevel));
    
    return true;
}
```

#### Step 2: Video Session Creation

```cpp
// Add to RenderingDevice API
enum VideoCodec {
    VIDEO_CODEC_AV1,
    VIDEO_CODEC_H264,
    VIDEO_CODEC_H265
};

struct VideoSessionCreateInfo {
    VideoCodec codec = VIDEO_CODEC_AV1;
    uint32_t max_width = 1920;
    uint32_t max_height = 1080;
    VkFormat output_format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; // NV12
    VkFormat dpb_format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    uint32_t max_dpb_slots = 8;
    uint32_t max_active_references = 7;
    bool enable_film_grain = false;
};

// Implementation in RenderingDeviceDriverVulkan
VideoSessionID RenderingDeviceDriverVulkan::video_session_create(const VideoSessionCreateInfo &p_info) {
    ERR_FAIL_COND_V(video_decode_queue_family == UINT32_MAX, VideoSessionID());
    
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
