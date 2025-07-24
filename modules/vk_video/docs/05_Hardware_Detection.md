# Hardware Detection and Capability Management

## Brief Description
Comprehensive hardware detection system for Vulkan Video capabilities, driver compatibility validation, and graceful fallback mechanisms for AV1 video acceleration.

## Core Detection System

### VulkanVideoCapabilities Class

```cpp
// Hardware capability detection and validation
class VulkanVideoCapabilities : public RefCounted {
    GDCLASS(VulkanVideoCapabilities, RefCounted);

public:
    struct AV1DecodeCapabilities {
        bool supported = false;
        VkExtent2D max_coded_extent = {0, 0};
        VkExtent2D min_coded_extent = {0, 0};
        uint32_t max_dpb_slots = 0;
        uint32_t max_active_reference_pictures = 0;
        Vector<uint32_t> supported_profiles;
        Vector<uint32_t> supported_levels;
        Vector<VkFormat> supported_picture_formats;
        Vector<VkFormat> supported_reference_formats;
        bool supports_film_grain = false;
        bool supports_10bit = false;
        bool supports_12bit = false;
    };

    struct AV1EncodeCapabilities {
        bool supported = false;
        VkExtent2D max_coded_extent = {0, 0};
        VkExtent2D min_coded_extent = {0, 0};
        uint32_t max_rate_control_layers = 0;
        uint32_t max_quality_levels = 0;
        Vector<uint32_t> supported_profiles;
        Vector<VkVideoEncodeRateControlModeFlagBitsKHR> rate_control_modes;
        bool supports_b_frames = false;
        bool supports_temporal_layers = false;
    };

    struct DriverInfo {
        String vendor_name;
        String device_name;
        uint32_t driver_version = 0;
        uint32_t api_version = 0;
        String driver_info;
        bool is_beta_driver = false;
        Vector<String> known_issues;
    };

private:
    static VulkanVideoCapabilities *singleton;

    // Cached capabilities
    bool capabilities_cached = false;
    AV1DecodeCapabilities decode_caps;
    AV1EncodeCapabilities encode_caps;
    DriverInfo driver_info;

    // Extension support
    bool vulkan_video_supported = false;
    bool av1_decode_extension_supported = false;
    bool av1_encode_extension_supported = false;

    // Device compatibility
    HashMap<String, bool> device_compatibility_cache;

protected:
    static void _bind_methods();

public:
    VulkanVideoCapabilities();
    virtual ~VulkanVideoCapabilities();

    static VulkanVideoCapabilities *get_singleton();

    // Capability detection
    Error detect_capabilities();
    bool is_vulkan_video_supported() const;
    bool is_av1_decode_supported() const;
    bool is_av1_encode_supported() const;

    // Capability queries
    AV1DecodeCapabilities get_av1_decode_capabilities() const;
    AV1EncodeCapabilities get_av1_encode_capabilities() const;
    DriverInfo get_driver_info() const;

    // Validation
    bool validate_decode_parameters(const VideoSessionCreateInfo &p_info) const;
    bool validate_encode_parameters(const VideoSessionCreateInfo &p_info) const;
    Error get_recommended_settings(Dictionary &r_settings) const;

    // Compatibility
    bool is_device_compatible() const;
    Vector<String> get_compatibility_warnings() const;
    String get_fallback_recommendation() const;

private:
    Error _detect_vulkan_video_support();
    Error _detect_av1_capabilities();
    Error _detect_driver_info();
    void _cache_device_compatibility();
    bool _validate_driver_version() const;
};

// Singleton access
VulkanVideoCapabilities *VulkanVideoCapabilities::singleton = nullptr;

VulkanVideoCapabilities *VulkanVideoCapabilities::get_singleton() {
    if (!singleton) {
        singleton = memnew(VulkanVideoCapabilities);
        singleton->detect_capabilities();
    }
    return singleton;
}
```

## Capability Detection Implementation

### Vulkan Video Extension Detection

```cpp
// Detect Vulkan Video extension support
Error VulkanVideoCapabilities::_detect_vulkan_video_support() {
    RenderingDevice *rd = RenderingDevice::get_singleton();
    ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);

    // Check for required extensions
    Vector<String> required_extensions = {
        "VK_KHR_video_queue",
        "VK_KHR_video_decode_queue"
    };

    Vector<String> optional_extensions = {
        "VK_KHR_video_decode_av1",
        "VK_KHR_video_encode_queue",
        "VK_KHR_video_encode_av1"
    };

    // Query available extensions
    VkPhysicalDevice physical_device = rd->get_device_capabilities().physical_device;
    uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);

    Vector<VkExtensionProperties> available_extensions;
    available_extensions.resize(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, available_extensions.ptrw());

    // Check required extensions
    for (const String &required_ext : required_extensions) {
        bool found = false;
        for (const VkExtensionProperties &ext : available_extensions) {
            if (required_ext == ext.extensionName) {
                found = true;
                break;
            }
        }
        if (!found) {
            vulkan_video_supported = false;
            return ERR_UNAVAILABLE;
        }
    }

    vulkan_video_supported = true;

    // Check optional extensions
    for (const VkExtensionProperties &ext : available_extensions) {
        String ext_name = ext.extensionName;
        if (ext_name == "VK_KHR_video_decode_av1") {
            av1_decode_extension_supported = true;
        } else if (ext_name == "VK_KHR_video_encode_av1") {
            av1_encode_extension_supported = true;
        }
    }

    return OK;
}

// Detect AV1-specific capabilities
Error VulkanVideoCapabilities::_detect_av1_capabilities() {
    if (!av1_decode_extension_supported && !av1_encode_extension_supported) {
        return ERR_UNAVAILABLE;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    VkPhysicalDevice physical_device = rd->get_device_capabilities().physical_device;

    // Query AV1 decode capabilities
    if (av1_decode_extension_supported) {
        VkVideoProfileInfoKHR profile_info = {};
        profile_info.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
        profile_info.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
        profile_info.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
        profile_info.lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
        profile_info.chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;

        VkVideoDecodeAV1ProfileInfoKHR av1_profile = {};
        av1_profile.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR;
        av1_profile.stdProfile = STD_VIDEO_AV1_PROFILE_MAIN;
        profile_info.pNext = &av1_profile;

        VkVideoCapabilitiesKHR capabilities = {};
        capabilities.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;

        VkVideoDecodeCapabilitiesKHR decode_capabilities = {};
        decode_capabilities.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR;
        capabilities.pNext = &decode_capabilities;

        VkVideoDecodeAV1CapabilitiesKHR av1_decode_caps = {};
        av1_decode_caps.sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR;
        decode_capabilities.pNext = &av1_decode_caps;

        VkResult result = vkGetPhysicalDeviceVideoCapabilitiesKHR(physical_device, &profile_info, &capabilities);
        if (result == VK_SUCCESS) {
            decode_caps.supported = true;
            decode_caps.max_coded_extent = capabilities.maxCodedExtent;
            decode_caps.min_coded_extent = capabilities.minCodedExtent;
            decode_caps.max_dpb_slots = capabilities.maxDpbSlots;
            decode_caps.max_active_reference_pictures = capabilities.maxActiveReferencePictures;

            // Query supported profiles
            _query_supported_profiles(physical_device, VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR, decode_caps.supported_profiles);

            // Query supported formats
            _query_supported_formats(physical_device, profile_info, decode_caps.supported_picture_formats, decode_caps.supported_reference_formats);

            // Check advanced features
            decode_caps.supports_film_grain = (av1_decode_caps.maxLevel >= STD_VIDEO_AV1_LEVEL_4_0);
            decode_caps.supports_10bit = _check_10bit_support(physical_device, profile_info);
            decode_caps.supports_12bit = _check_12bit_support(physical_device, profile_info);
        }
    }

    // Query AV1 encode capabilities (similar pattern)
    if (av1_encode_extension_supported) {
        _detect_av1_encode_capabilities(physical_device);
    }

    return OK;
}
```

## Driver Compatibility Detection

### Driver Information Gathering

```cpp
// Detect driver information and compatibility
Error VulkanVideoCapabilities::_detect_driver_info() {
    RenderingDevice *rd = RenderingDevice::get_singleton();
    ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);

    VkPhysicalDevice physical_device = rd->get_device_capabilities().physical_device;

    // Get device properties
    VkPhysicalDeviceProperties device_props = {};
    vkGetPhysicalDeviceProperties(physical_device, &device_props);

    driver_info.device_name = device_props.deviceName;
    driver_info.driver_version = device_props.driverVersion;
    driver_info.api_version = device_props.apiVersion;

    // Get vendor-specific information
    uint32_t vendor_id = device_props.vendorID;
    switch (vendor_id) {
        case 0x10DE: // NVIDIA
            driver_info.vendor_name = "NVIDIA";
            _detect_nvidia_driver_info(device_props);
            break;
        case 0x1002: // AMD
        case 0x1022:
            driver_info.vendor_name = "AMD";
            _detect_amd_driver_info(device_props);
            break;
        case 0x8086: // Intel
            driver_info.vendor_name = "Intel";
            _detect_intel_driver_info(device_props);
            break;
        default:
            driver_info.vendor_name = "Unknown";
            break;
    }

    // Check for beta drivers
    driver_info.is_beta_driver = _is_beta_driver(device_props);

    // Gather known issues
    _gather_known_issues();

    return OK;
}

// NVIDIA-specific driver detection
void VulkanVideoCapabilities::_detect_nvidia_driver_info(const VkPhysicalDeviceProperties &props) {
    uint32_t major = (props.driverVersion >> 22) & 0x3FF;
    uint32_t minor = (props.driverVersion >> 14) & 0xFF;
    uint32_t patch = (props.driverVersion >> 6) & 0xFF;
    uint32_t build = props.driverVersion & 0x3F;

    driver_info.driver_info = vformat("NVIDIA Driver %d.%d.%d.%d", major, minor, patch, build);

    // Check minimum required version for AV1 support
    // NVIDIA AV1 decode requires driver 470.86 or later
    uint32_t min_version = (470 << 22) | (86 << 14);
    if (props.driverVersion < min_version) {
        driver_info.known_issues.push_back("Driver version too old for AV1 support. Minimum: 470.86");
    }

    // Check for RTX 30/40 series requirement
    String device_name = props.deviceName;
    if (!device_name.contains("RTX 30") && !device_name.contains("RTX 40") &&
        !device_name.contains("RTX A") && !device_name.contains("Tesla")) {
        driver_info.known_issues.push_back("AV1 hardware decode requires RTX 30 series or newer");
    }
}

// AMD-specific driver detection
void VulkanVideoCapabilities::_detect_amd_driver_info(const VkPhysicalDeviceProperties &props) {
    // AMD driver version format is different
    uint32_t major = props.driverVersion >> 22;
    uint32_t minor = (props.driverVersion >> 12) & 0x3FF;
    uint32_t patch = props.driverVersion & 0xFFF;

    driver_info.driver_info = vformat("AMD Driver %d.%d.%d", major, minor, patch);

    // Check for RDNA2+ requirement
    String device_name = props.deviceName;
    if (!device_name.contains("RX 6") && !device_name.contains("RX 7") &&
        !device_name.contains("Radeon Pro")) {
        driver_info.known_issues.push_back("AV1 hardware decode requires RDNA2 architecture or newer");
    }

    // Check minimum driver version
    if (props.driverVersion < 0x80000000) { // Placeholder version check
        driver_info.known_issues.push_back("Driver version may not support AV1 hardware decode");
    }
}

// Intel-specific driver detection
void VulkanVideoCapabilities::_detect_intel_driver_info(const VkPhysicalDeviceProperties &props) {
    driver_info.driver_info = vformat("Intel Driver %d", props.driverVersion);

    // Check for Arc GPU requirement
    String device_name = props.deviceName;
    if (!device_name.contains("Arc") && !device_name.contains("Xe")) {
        driver_info.known_issues.push_back("AV1 hardware decode requires Intel Arc or Xe graphics");
    }
}
```

## Validation and Compatibility

### Parameter Validation

```cpp
// Validate decode parameters against hardware capabilities
bool VulkanVideoCapabilities::validate_decode_parameters(const VideoSessionCreateInfo &p_info) const {
    if (!decode_caps.supported) {
        return false;
    }

    // Check resolution limits
    if (p_info.max_coded_extent_width > decode_caps.max_coded_extent.width ||
        p_info.max_coded_extent_height > decode_caps.max_coded_extent.height) {
        return false;
    }

    if (p_info.max_coded_extent_width < decode_caps.min_coded_extent.width ||
        p_info.max_coded_extent_height < decode_caps.min_coded_extent.height) {
        return false;
    }

    // Check DPB requirements
    if (p_info.max_dpb_slots > decode_caps.max_dpb_slots) {
        return false;
    }

    if (p_info.max_active_reference_pictures > decode_caps.max_active_reference_pictures) {
        return false;
    }

    // Check profile support
    bool profile_supported = false;
    for (uint32_t supported_profile : decode_caps.supported_profiles) {
        if (supported_profile == p_info.profile) {
            profile_supported = true;
            break;
        }
    }

    return profile_supported;
}

// Get recommended settings based on hardware capabilities
Error VulkanVideoCapabilities::get_recommended_settings(Dictionary &r_settings) const {
    if (!decode_caps.supported) {
        return ERR_UNAVAILABLE;
    }

    r_settings.clear();

    // Resolution recommendations
    r_settings["max_width"] = decode_caps.max_coded_extent.width;
    r_settings["max_height"] = decode_caps.max_coded_extent.height;
    r_settings["recommended_width"] = MIN(decode_caps.max_coded_extent.width, 3840u); // 4K max
    r_settings["recommended_height"] = MIN(decode_caps.max_coded_extent.height, 2160u);

    // DPB recommendations
    r_settings["max_dpb_slots"] = decode_caps.max_dpb_slots;
    r_settings["recommended_dpb_slots"] = MIN(decode_caps.max_dpb_slots, 8u);
    r_settings["max_reference_frames"] = decode_caps.max_active_reference_pictures;

    // Format recommendations
    Array picture_formats;
    for (VkFormat format : decode_caps.supported_picture_formats) {
        picture_formats.push_back((int)format);
    }
    r_settings["supported_picture_formats"] = picture_formats;

    // Feature support
    r_settings["supports_10bit"] = decode_caps.supports_10bit;
    r_settings["supports_12bit"] = decode_caps.supports_12bit;
    r_settings["supports_film_grain"] = decode_caps.supports_film_grain;

    // Performance recommendations based on vendor
    if (driver_info.vendor_name == "NVIDIA") {
        r_settings["recommended_concurrent_sessions"] = 2;
        r_settings["preferred_memory_type"] = "device_local";
    } else if (driver_info.vendor_name == "AMD") {
        r_settings["recommended_concurrent_sessions"] = 1;
        r_settings["preferred_memory_type"] = "device_local";
    } else if (driver_info.vendor_name == "Intel") {
        r_settings["recommended_concurrent_sessions"] = 1;
        r_settings["preferred_memory_type"] = "host_visible";
    }

    return OK;
}
```

## Fallback and Compatibility Warnings

### Compatibility Assessment

```cpp
// Check overall device compatibility
bool VulkanVideoCapabilities::is_device_compatible() const {
    if (!vulkan_video_supported) {
        return false;
    }

    if (!av1_decode_extension_supported) {
        return false;
    }

    if (!_validate_driver_version()) {
        return false;
    }

    // Check for critical known issues
    for (const String &issue : driver_info.known_issues) {
        if (issue.contains("requires") && issue.contains("newer")) {
            return false; // Hardware too old
        }
    }

    return decode_caps.supported;
}

// Get compatibility warnings
Vector<String> VulkanVideoCapabilities::get_compatibility_warnings() const {
    Vector<String> warnings;

    if (driver_info.is_beta_driver) {
        warnings.push_back("Beta driver detected. Video decoding may be unstable.");
    }

    if (!decode_caps.supports_10bit) {
        warnings.push_back("10-bit AV1 decoding not supported. High-quality content may fall back to software.");
    }

    if (!decode_caps.supports_film_grain) {
        warnings.push_back("Film grain synthesis not supported. Some content may appear different.");
    }

    if (decode_caps.max_coded_extent.width < 3840 || decode_caps.max_coded_extent.height < 2160) {
        warnings.push_back("4K video decoding may not be supported.");
    }

    // Add driver-specific warnings
    warnings.append_array(driver_info.known_issues);

    return warnings;
}

// Get fallback recommendation
String VulkanVideoCapabilities::get_fallback_recommendation() const {
    if (!vulkan_video_supported) {
        return "Vulkan Video not supported. Use software decoder (dav1d/libaom).";
    }

    if (!av1_decode_extension_supported) {
        return "AV1 hardware decode not supported. Use software decoder.";
    }

    if (!_validate_driver_version()) {
        return "Driver too old for reliable AV1 support. Update driver or use software decoder.";
    }

    if (driver_info.vendor_name == "Unknown") {
        return "Unknown GPU vendor. Software decoder recommended for compatibility.";
    }

    return "Hardware acceleration available but may have limitations. Monitor performance.";
}
```

## Usage Examples

### Basic Capability Detection
```cpp
// Check hardware support
VulkanVideoCapabilities *caps = VulkanVideoCapabilities::get_singleton();

if (caps->is_av1_decode_supported()) {
    print("AV1 hardware decode supported");

    auto decode_caps = caps->get_av1_decode_capabilities();
    print("Max resolution: ", decode_caps.max_coded_extent.width, "x", decode_caps.max_coded_extent.height);
    print("Max DPB slots: ", decode_caps.max_dpb_slots);
    print("10-bit support: ", decode_caps.supports_10bit);
} else {
    print("AV1 hardware decode not supported");
    print("Fallback: ", caps->get_fallback_recommendation());
}
```

### Validation and Configuration
```cpp
// Validate and configure video session
VideoSessionCreateInfo session_info;
session_info.max_coded_extent_width = 1920;
session_info.max_coded_extent_height = 1080;
session_info.max_dpb_slots = 8;

VulkanVideoCapabilities *caps = VulkanVideoCapabilities::get_singleton();
if (caps->validate_decode_parameters(session_info)) {
    print("Configuration valid");
} else {
    print("Configuration not supported");

    Dictionary recommended;
    caps->get_recommended_settings(recommended);
    print("Recommended max resolution: ", recommended["recommended_width"], "x", recommended["recommended_height"]);
}

// Check for warnings
Vector<String> warnings = caps->get_compatibility_warnings();
for (const String &warning : warnings) {
    print("Warning: ", warning);
}
```

This hardware detection system provides comprehensive capability assessment and validation for reliable AV1 hardware acceleration across different GPU vendors and driver versions.
