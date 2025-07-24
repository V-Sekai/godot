# VK Video Module - Architecture Overview

## Brief Description

Hardware-accelerated AV1 video decoding in Godot using Vulkan Video extensions, with current MKV container support and OneEuroFilter-based audio-video synchronization.

## Implementation Status

### ✅ Currently Implemented

-   **VideoStreamMKV**: MKV/WebM container parsing with Opus audio support
-   **OneEuroFilter**: Audio-video synchronization filtering for jitter reduction
-   **Basic Infrastructure**: Module registration and resource management foundation

### 🎯 Target Implementation (Required)

-   **Vulkan Video API**: Hardware-accelerated video decoding using VK*KHR_video*\* extensions
-   **AV1 Codec Support**: Full AV1 hardware decode/encode capabilities
-   **Advanced Resource Management**: DPB management, memory pools, command buffer optimization

### ❌ Not Planned

-   **H.264 Support**: Removed from scope
-   **H.265 Support**: Removed from scope

## System Architecture

### Current Architecture (Phase 1)

```
┌─────────────────────────────────────────────────────────────┐
│                    Godot Engine Core                        │
├─────────────────────────────────────────────────────────────┤
│  VideoStreamPlayer (Control)                               │
│  └── VideoStreamMKV (Resource) ✅ IMPLEMENTED              │
│      └── VideoStreamPlaybackMKV (Playback State)           │
├─────────────────────────────────────────────────────────────┤
│  MKV Container Support                                      │
│  ├── libsimplewebm (MKV/WebM parsing)                      │
│  ├── OpusVorbisDecoder (Audio decoding)                    │
│  └── OneEuroFilter (Audio-video sync) ✅ IMPLEMENTED       │
├─────────────────────────────────────────────────────────────┤
│  Basic Infrastructure                                       │
│  ├── Module registration                                    │
│  ├── Resource format loader                                │
│  └── Class binding system                                  │
└─────────────────────────────────────────────────────────────┘
```

### Target Architecture (Phase 2 - AV1 Focus)

```
┌─────────────────────────────────────────────────────────────┐
│                    Godot Engine Core                        │
├─────────────────────────────────────────────────────────────┤
│  VideoStreamPlayer (Control)                               │
│  ├── VideoStreamMKV (Resource) ✅ CURRENT                  │
│  └── VideoStreamAV1 (Resource) 🎯 TARGET                   │
│      └── VideoStreamPlaybackAV1 (Playback State)           │
├─────────────────────────────────────────────────────────────┤
│  RenderingDevice Extensions (Target)                        │
│  ├── video_session_create()                                │
│  ├── video_cmd_decode_frame()                              │
│  ├── video_cmd_encode_frame()                              │
│  └── video_image_create()                                  │
├─────────────────────────────────────────────────────────────┤
│  Vulkan Video Implementation (Target)                       │
│  ├── VulkanVideoResourceManager                            │
│  ├── VulkanFilterYuvCompute (YCbCr Processing)             │
│  ├── AV1BitstreamParser                                    │
│  └── AudioVideoSynchronizer (OneEuroFilter) ✅ CURRENT     │
├─────────────────────────────────────────────────────────────┤
│  Container Support                                          │
│  ├── MKV/WebM (libsimplewebm) ✅ CURRENT                   │
│  ├── MP4/MOV (Future AV1 containers)                       │
│  └── IVF (AV1 elementary streams)                          │
├─────────────────────────────────────────────────────────────┤
│  Vulkan Driver Layer (Target)                              │
│  ├── VK_KHR_video_queue                                    │
│  ├── VK_KHR_video_decode_queue                             │
│  ├── VK_KHR_video_decode_av1                               │
│  └── VK_KHR_video_encode_av1                               │
└─────────────────────────────────────────────────────────────┘
```

### Vulkan Video AV1 Encoding Support (Target Implementation)

The vk_video module will include AV1 encoding capabilities:

#### Target AV1 Encoding Features

-   **AV1-only support**: Focus on modern AV1 codec for optimal quality/compression
-   **Multi-threaded encoding**: Parallel encoding operations for performance
-   **Rate control**: CBR, VBR, and CQP rate control modes
-   **YCbCr input support**: Direct encoding from YCbCr content
-   **Bit depth support**: 8, 10, and 12-bit AV1 encoding
-   **Hardware acceleration**: GPU-accelerated AV1 encoding with software fallback

#### Target AV1 Encoding Architecture

```cpp
// AV1-focused encoding pipeline
class VulkanVideoAV1Encoder {
public:
    Error initialize(const AV1EncoderSettings& settings);
    Error encode_frame(const YCbCrFrame& input, AV1BitstreamBuffer& output);
    Error finalize();

private:
    VkVideoSessionKHR av1_encode_session;
    VulkanFilterYuvCompute yuv_processor;
    AV1RateController rate_controller;
    AV1SequenceHeader sequence_header;
};
```

### FFmpeg Interop Architecture

#### Container and Format Support

```
┌─────────────────────────────────────────────────────────────┐
│  FFmpeg Integration Layer                                   │
├─────────────────────────────────────────────────────────────┤
│  Container Demuxing                                         │
│  ├── MP4/MOV (libavformat)                                 │
│  ├── MKV/WebM (libavformat)                                │
│  ├── AVI (libavformat)                                     │
│  └── Custom containers                                     │
├─────────────────────────────────────────────────────────────┤
│  YCbCr Processing Pipeline                                  │
│  ├── VulkanFilterYuvCompute                                │
│  │   ├── YCbCr ↔ RGBA conversion                           │
│  │   ├── Format conversion (NV12, I420, etc.)             │
│  │   ├── Bit depth conversion (8/10/12/16-bit)            │
│  │   └── Chroma subsampling (4:4:4, 4:2:2, 4:2:0)        │
│  ├── Hardware acceleration path                            │
│  └── Software fallback path                                │
├─────────────────────────────────────────────────────────────┤
│  Alpha Channel Handling                                     │
│  ├── ❌ Native YCbCrA support (limitation)                 │
│  ├── ✅ RGBA output with alpha=1.0                         │
│  ├── ✅ Software fallback for alpha content                │
│  └── ⚠️  Alpha-enabled formats require FFmpeg path         │
└─────────────────────────────────────────────────────────────┘
```

#### YCbCr Format Support Matrix

| Format   | Planes | Subsampling | Bit Depth | Hardware | Alpha |
| -------- | ------ | ----------- | --------- | -------- | ----- |
| NV12     | 2      | 4:2:0       | 8-bit     | ✅       | ❌    |
| NV21     | 2      | 4:2:0       | 8-bit     | ✅       | ❌    |
| I420     | 3      | 4:2:0       | 8-bit     | ✅       | ❌    |
| P010     | 2      | 4:2:0       | 10-bit    | ✅       | ❌    |
| P012     | 2      | 4:2:0       | 12-bit    | ✅       | ❌    |
| P016     | 2      | 4:2:0       | 16-bit    | ✅       | ❌    |
| YUVA420P | 4      | 4:2:0       | 8-bit     | ❌       | ✅    |
| RGBA     | 1      | 4:4:4       | 8-bit     | ✅       | ✅    |

### Alpha Channel Handling Strategy

#### Current Limitations

```cpp
// Alpha channel limitations in current implementation
class VulkanFilterYuvCompute {
    // ❌ No native YCbCrA (4-component) support
    // ❌ Alpha channels hardcoded to 1.0 in conversions
    // ❌ Limited to 3-plane maximum in current implementation

    // Current alpha handling in YCbCr to RGBA conversion:
    vec4 rgba = vec4(rgb, 1.0); // Alpha always set to fully opaque
};
```

#### Fallback Strategy for Alpha Content

```cpp
// Alpha-enabled content handling
class VideoStreamPlayback {
    bool requires_alpha_fallback(VkFormat format) {
        // Check if format includes alpha channel
        return format_has_alpha(format) && !hardware_supports_alpha(format);
    }

    Ref<VideoStreamPlayback> create_playback() {
        if (requires_alpha_fallback(stream_format)) {
            // Use FFmpeg software path for alpha content
            return memnew(VideoStreamPlaybackFFmpegSoftware);
        }
        return memnew(VideoStreamPlaybackVulkan);
    }
};
```

### Class Hierarchy

#### Current Implementation (Phase 1)

```cpp
// Currently Implemented Resource Classes
VideoStream (base)
└── VideoStreamMKV ✅ IMPLEMENTED
    ├── file_path: String
    ├── mkv_metadata: Dictionary
    ├── audio_decoder: Ref<OpusVorbisDecoder>
    └── duration: double

// Currently Implemented Playback Classes
VideoStreamPlayback (base)
└── VideoStreamPlaybackMKV ✅ IMPLEMENTED
    ├── mkv_parser: Ref<SimpleWebMDemuxer>
    ├── audio_stream: Ref<AudioStreamGenerator>
    ├── av_sync: Ref<OneEuroFilter>
    └── current_frame_texture: RID (placeholder)

// Currently Implemented Support Classes
OneEuroFilter ✅ IMPLEMENTED
├── min_cutoff: double
├── beta: double
├── dcutoff: double
└── filter_state: FilterState
```

#### Target Implementation (Phase 2 - AV1)

```cpp
// Target AV1 Resource Classes
VideoStream (base)
├── VideoStreamMKV ✅ CURRENT
└── VideoStreamAV1 🎯 TARGET
    ├── file_path: String
    ├── sequence_header: AV1SequenceHeader
    ├── hardware_caps: AV1VideoCapabilities
    └── container_format: ContainerFormat

// Target AV1 Playback Classes
VideoStreamPlayback (base)
├── VideoStreamPlaybackMKV ✅ CURRENT
└── VideoStreamPlaybackAV1 🎯 TARGET
    ├── video_session: RID
    ├── resource_manager: Ref<VulkanVideoResourceManager>
    ├── av_sync: Ref<AudioVideoSynchronizer>
    ├── reference_frames: Vector<RID>
    └── av1_decoder: Ref<AV1BitstreamParser>

// Target Support Classes
VulkanVideoResourceManager 🎯 TARGET
├── dpb_pool: VideoMemoryPool
├── bitstream_pool: BitstreamBufferPool
└── command_manager: VideoCommandManager

AudioVideoSynchronizer 🎯 TARGET (extends current OneEuroFilter)
├── av_sync_filter: OneEuroFilter ✅ CURRENT
├── audio_clock_filter: OneEuroFilter ✅ CURRENT
├── video_queue: FrameQueue
└── sync_strategy: SyncStrategy
```

### Integration Points

#### 1. RenderingDevice Integration

```cpp
// New methods added to RenderingDevice class
class RenderingDevice {
public:
    // Video session management
    RID video_session_create(const VideoSessionCreateInfo& info);
    void video_session_destroy(RID session);

    // Video command recording
    void video_cmd_begin_coding(CommandBufferID cmd, const VideoCodingBeginInfo& info);
    void video_cmd_decode_frame(CommandBufferID cmd, const VideoDecodeInfo& info);
    void video_cmd_end_coding(CommandBufferID cmd);

    // Video resource creation
    RID video_image_create(const VideoImageCreateInfo& info);
    RID video_buffer_create(const VideoBufferCreateInfo& info);
};
```

#### 2. VideoStreamPlayer Integration

```cpp
// Existing VideoStreamPlayer automatically supports new format
VideoStreamPlayer player;
Ref<VideoStreamAV1> stream = load("res://video.av1");
player.set_stream(stream);
player.play(); // Hardware decode automatically used if available
```

#### 3. Movie Maker Integration

```cpp
// Hardware encoding for Movie Maker
class MovieMakerAV1Backend {
public:
    Error initialize_encoder(const EncoderSettings& settings);
    Error encode_frame(RID source_texture);
    Error finalize_encoding();
};
```

### Data Flow

#### Current MKV Pipeline (Phase 1)

```
1. File Loading
   VideoStreamMKV::load_file() → Parse MKV headers → Extract metadata

2. Playback Initialization
   VideoStreamPlaybackMKV::play() → Initialize demuxer → Setup audio decoder

3. Frame Processing Loop
   update() → Demux MKV packets → Decode Opus audio → Generate placeholder video

4. Audio-Video Synchronization
   OneEuroFilter → Smooth timing jitter → Maintain sync between audio/video
```

#### Target AV1 Pipeline (Phase 2)

```
1. File Loading
   VideoStreamAV1::load_file() → Parse AV1 headers → Cache sequence info

2. Playback Initialization
   VideoStreamPlaybackAV1::play() → Create Vulkan video session → Allocate DPB

3. Frame Decode Loop
   update() → Parse AV1 bitstream → Submit hardware decode → Present frame

4. Advanced Synchronization
   AudioVideoSynchronizer → OneEuroFilter-based timing → Queue management
```

#### Target AV1 Encode Pipeline (Phase 2)

```
1. Source Preparation
   MovieMaker::capture_frame() → YCbCr conversion → AV1 format validation

2. Encoding Initialization
   VulkanVideoAV1Encoder::initialize() → Create encode session → Allocate buffers

3. Frame Encode Loop
   encode_frame() → YCbCr processing → AV1 hardware encode → Bitstream output

4. Synchronization with OneEuroFilter
   AudioVideoSynchronizer → OneEuroFilter → Timing adjustment
```

#### FFmpeg Interop Pipeline

```
1. Container Demuxing
   FFmpeg::avformat_open_input() → Extract video/audio streams → Parse metadata

2. Format Detection & Conversion
   VulkanFilterYuvCompute → YCbCr format conversion → Hardware compatibility check

3. Alpha Channel Handling
   if (has_alpha_channel) → FFmpeg software path
   else → Vulkan hardware path

4. Fallback Decision Matrix
   Hardware capability → Format support → Performance threshold → Path selection
```

#### Resource Management

```
1. DPB Management
   VideoMemoryPool → Allocate image array → Track reference frames

2. Command Buffer Management
   VideoCommandManager → Pool command buffers → Synchronize execution

3. Memory Synchronization
   Transfer queues → Graphics queues → Present queue
```

### Error Handling Strategy

#### Hardware Detection

```cpp
// Capability detection with graceful fallback
bool VideoStreamAV1::is_hardware_supported() {
    if (!RenderingDevice::get_singleton()->has_feature(FEATURE_VULKAN_VIDEO)) {
        return false;
    }

    VideoCapabilities caps = get_av1_decode_capabilities();
    return caps.supports_profile(sequence_header.profile);
}
```

#### Fallback Mechanisms

```cpp
// Automatic fallback to software decode
Ref<VideoStreamPlayback> VideoStreamAV1::instantiate_playback() {
    if (is_hardware_supported()) {
        return memnew(VideoStreamPlaybackAV1);
    } else {
        // Fallback to software decoder (future implementation)
        return memnew(VideoStreamPlaybackAV1Software);
    }
}
```

### Performance Considerations

#### Memory Management

-   Pre-allocated DPB image arrays
-   Pooled bitstream buffers
-   Command buffer reuse
-   Zero-copy texture sharing

#### Threading Strategy

-   Decode operations on video queue
-   Graphics operations on graphics queue
-   Audio processing on audio thread
-   Synchronization via timeline semaphores

#### Quality Scaling

-   Adaptive resolution based on performance
-   Dynamic bitrate adjustment
-   Frame dropping for real-time playback

### Testing Integration Points

#### Unit Tests

-   Hardware capability detection
-   Resource allocation/deallocation
-   Command buffer recording
-   Error condition handling

#### Integration Tests

-   End-to-end playback scenarios
-   Audio-video synchronization
-   Memory leak detection
-   Performance benchmarking

#### Validation Tests

-   Vulkan Video specification compliance
-   Cross-platform compatibility
-   Driver version compatibility
-   Stress testing with multiple streams

This architecture provides a solid foundation for hardware-accelerated video in Godot while maintaining compatibility with existing systems.
