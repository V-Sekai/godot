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

-   **VK_KHR_video_queue**: Core video processing queue support
-   **VK_KHR_video_decode_queue**: Video decode operations
-   **VK_KHR_video_decode_av1**: AV1 codec-specific decode support
-   **VK_KHR_sampler_ycbcr_conversion**: Hardware YCbCr→RGB conversion
-   **VK_KHR_video_maintenance1**: Enhanced video features and bug fixes

## Implementation Status

| Phase | Status           | Description                                                              |
| ----- | ---------------- | ------------------------------------------------------------------------ |
| 5A    | ✅ **COMPLETED** | Vulkan Video Foundation - Extensions, function pointers, queue detection |
| 5B    | ⏳ **PLANNED**   | Video Session Management - Capabilities query, session creation          |
| 5C    | ⏳ **PLANNED**   | Video Memory Management - DPB allocation, memory binding                 |
| 5D    | ⏳ **PLANNED**   | YCbCr Color Conversion - Hardware color space conversion                 |
| 5E    | ⏳ **PLANNED**   | Performance & Polish - Optimization and error handling                   |

## Implementation Phases

### Phase 5A: Vulkan Video Foundation ✅ **COMPLETED**

**Priority: CRITICAL** - All subsequent phases depend on this foundation.

This phase has been **fully implemented** in the Vulkan driver with the following components:

#### ✅ Extension Registration

-   Video extensions registered in `RenderingDeviceDriverVulkan::_initialize_device_extensions()`
-   Includes VK_KHR_video_queue, VK_KHR_video_decode_queue, VK_KHR_video_decode_av1
-   Extension availability logging for debugging

#### ✅ Function Pointer Loading

-   All Vulkan Video function pointers loaded in `DeviceFunctions` struct
-   Includes video capabilities, session management, memory management, and decode commands
-   YCbCr conversion function pointers for color space conversion

#### ✅ Video Queue Family Detection

-   Queue family detection implemented in `_detect_video_queue_families()`
-   Searches for VK_QUEUE_VIDEO_DECODE_BIT_KHR support
-   Graceful fallback when hardware decode unavailable

#### ✅ Video Resource Types

-   Video resource structures defined in Vulkan driver header
-   VideoSessionInfo and VideoBufferInfo structures
-   Integration with VersatileResource template system

**Implementation Files:**

-   `drivers/vulkan/rendering_device_driver_vulkan.h` - Video structures and function pointers
-   `drivers/vulkan/rendering_device_driver_vulkan.cpp` - Extension registration and initialization

---

### Phase 5B: Video Session Management ⏳ **PLANNED**

**Depends on: Phase 5A complete**

This phase will implement video session creation and management for AV1 decode operations.

#### Key Components:

-   **Video Capabilities Query**: Query hardware AV1 decode capabilities and limits
-   **Video Session Creation**: Create Vulkan video sessions with proper AV1 profile configuration
-   **Session Parameter Management**: Handle video session parameters and updates
-   **Validation**: Validate requested parameters against hardware capabilities

**Implementation Files:**

-   `modules/vk_video/vulkan_video_session.h` - Video session management interface
-   `modules/vk_video/vulkan_video_session.cpp` - Session creation and capabilities query
-   Updates to `rendering_device_video_extensions.h` for RenderingDevice API

---

### Phase 5C: Video Memory Management ⏳ **PLANNED**

**Depends on: Phase 5B complete**

This phase will implement memory allocation and binding for video sessions and DPB (Decoded Picture Buffer) management.

#### Step 1: Video Session Memory Requirements

#### Step 2: DPB (Decoded Picture Buffer) Management

### Phase 5D: YCbCr Color Conversion ⏳ **PLANNED**

**Depends on: Phase 5C complete**

This phase implements hardware YCbCr to RGB color space conversion using Vulkan's sampler YCbCr conversion.

#### Step 1: YCbCr Sampler Creation

#### Step 2: Video Decode Output Integration

### Phase 5E: Performance & Polish ⏳ **PLANNED**

**Depends on: All previous phases complete**

This phase focuses on optimization, error handling, and final integration polish.

#### Step 1: Memory Pooling and Optimization

#### Step 2: Error Handling and Fallback

#### Step 3: Performance Monitoring and Debugging

#### Step 4: Integration Testing and Validation

## Next Steps and Roadmap

### Immediate Next Steps (Phase 5B Implementation)

1. **Implement Video Session Management**

    - Add `_query_av1_decode_capabilities()` method
    - Implement `video_session_create()` API
    - Add video session parameter management
    - Test with basic AV1 streams

2. **Add RenderingDevice API Extensions**
    - Define `VideoSessionID` and related types
    - Add video session creation/destruction methods
    - Integrate with existing resource management

### Medium-term Goals (Phases 5C-5D)

3. **Complete Memory Management**

    - Implement video session memory binding
    - Add DPB (Decoded Picture Buffer) management
    - Optimize memory allocation patterns

4. **Implement YCbCr Conversion**
    - Add YCbCr sampler creation
    - Integrate with Godot's material system
    - Test color space conversion accuracy

### Long-term Integration (Phase 5E)

5. **Performance Optimization**

    - Implement texture pooling
    - Add performance monitoring
    - Optimize decode pipeline

6. **Error Handling and Fallback**
    - Comprehensive error handling
    - Software decode fallback
    - Debugging and validation tools

## Testing and Validation

### Hardware Requirements

-   **GPU**: Modern GPU with Vulkan Video support (NVIDIA RTX 40xx series, AMD RDNA3, Intel Arc)
-   **Driver**: Latest graphics drivers with Vulkan Video extensions
-   **OS**: Windows 10/11, Linux with recent kernel

### Test Cases

1. **Extension Detection**: Verify video extensions are properly detected
2. **Queue Family Discovery**: Confirm video decode queues are found
3. **Session Creation**: Test video session creation with various parameters
4. **Memory Binding**: Validate video session memory allocation
5. **Basic Decode**: Test simple AV1 frame decode operation
6. **Color Conversion**: Verify YCbCr to RGB conversion accuracy
7. **Performance**: Measure decode performance vs. software fallback

### Debug Output

Enable verbose logging to monitor video decode operations:

```
Vulkan Video: AV1 hardware decode available
Vulkan Video: Found video decode queue family: 2
Vulkan Video AV1 Capabilities:
  Max coded extent: 8192x8192
  Max DPB slots: 8
  Max active references: 7
  Max level: 23
```

## Conclusion

Phase 5A has been **successfully completed** with all foundation components implemented in the Vulkan driver. The remaining phases (5B-5E) provide a clear roadmap for completing hardware-accelerated AV1 video decoding in Godot.

The implementation follows Vulkan Video best practices and integrates seamlessly with Godot's existing rendering architecture. Once complete, this will provide significant performance improvements for AV1 video playback on supported hardware.
