# VK Video Module - Architecture Overview

## Brief Description
High-level system architecture for Vulkan Video-based hardware acceleration supporting H.264, H.265, and AV1 codecs with both decode and encode capabilities, FFmpeg interop, and YCbCr processing in Godot Engine.

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Godot Engine Core                        │
├─────────────────────────────────────────────────────────────┤
│  VideoStreamPlayer (Control)                               │
│  ├── VideoStreamH264 (Resource)                            │
│  ├── VideoStreamH265 (Resource)                            │
│  ├── VideoStreamAV1 (Resource)                             │
│  ├── VideoStreamPlaybackH264 (Playback State)              │
│  ├── VideoStreamPlaybackH265 (Playback State)              │
│  └── VideoStreamPlaybackAV1 (Playback State)               │
├─────────────────────────────────────────────────────────────┤
│  RenderingDevice Extensions                                 │
│  ├── video_session_create()                                │
│  ├── video_cmd_decode_frame()                              │
│  ├── video_cmd_encode_frame()                              │
│  └── video_image_create()                                  │
├─────────────────────────────────────────────────────────────┤
│  Vulkan Video Implementation                                │
│  ├── VulkanVideoResourceManager                            │
│  ├── VulkanFilterYuvCompute (YCbCr Processing)             │
│  ├── H264BitstreamParser                                   │
│  ├── H265BitstreamParser                                   │
│  ├── AV1BitstreamParser                                    │
│  └── AudioVideoSynchronizer (OneEuroFilter)                │
├─────────────────────────────────────────────────────────────┤
│  FFmpeg Interop Layer                                      │
│  ├── Container Demuxing (.mp4, .mkv, .mov)                 │
│  ├── YCbCr Format Conversion                               │
│  ├── Software Fallback Processing                          │
│  └── Alpha Channel Handling                                │
├─────────────────────────────────────────────────────────────┤
│  Vulkan Driver Layer                                       │
│  ├── VK_KHR_video_decode_h264                              │
│  ├── VK_KHR_video_decode_h265                              │
│  ├── VK_KHR_video_decode_av1                               │
│  ├── VK_KHR_video_encode_h264                              │
│  ├── VK_KHR_video_encode_h265                              │
│  └── VK_KHR_video_encode_av1                               │
└─────────────────────────────────────────────────────────────┘
```

### Vulkan Video Encoding Support

The vk_video module includes comprehensive encoding capabilities through integration with vk_video_samples:

#### Supported Encoding Features
- **Multi-codec support**: H.264, H.265 (HEVC), and AV1 encoding
- **Multi-threaded encoding**: Parallel encoding operations for performance
- **Rate control**: CBR, VBR, and CQP rate control modes
- **YCbCr input support**: Direct encoding from YCbCr content
- **Bit depth support**: 8, 10, 12, and 16-bit encoding
- **Hardware acceleration**: GPU-accelerated encoding with CPU fallback

#### Encoding Architecture
```cpp
// Encoding pipeline components
class VulkanVideoEncoder {
public:
    Error initialize(const EncoderSettings& settings);
    Error encode_frame(const YCbCrFrame& input, BitstreamBuffer& output);
    Error finalize();

private:
    VkVideoSessionKHR encode_session;
    VulkanFilterYuvCompute yuv_processor;
    RateController rate_controller;
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
| Format | Planes | Subsampling | Bit Depth | Hardware | Alpha |
|--------|--------|-------------|-----------|----------|-------|
| NV12   | 2      | 4:2:0       | 8-bit     | ✅       | ❌    |
| NV21   | 2      | 4:2:0       | 8-bit     | ✅       | ❌    |
| I420   | 3      | 4:2:0       | 8-bit     | ✅       | ❌    |
| P010   | 2      | 4:2:0       | 10-bit    | ✅       | ❌    |
| P012   | 2      | 4:2:0       | 12-bit    | ✅       | ❌    |
| P016   | 2      | 4:2:0       | 16-bit    | ✅       | ❌    |
| YUVA420P | 4    | 4:2:0       | 8-bit     | ❌       | ✅    |
| RGBA   | 1      | 4:4:4       | 8-bit     | ✅       | ✅    |

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

```cpp
// Resource Classes
VideoStream (base)
├── VideoStreamH264
│   ├── file_path: String
│   ├── sps: H264SequenceParameterSet
│   └── hardware_caps: VideoCapabilities
├── VideoStreamH265
│   ├── file_path: String
│   ├── sps: H265SequenceParameterSet
│   ├── pps: H265PictureParameterSet
│   └── hardware_caps: VideoCapabilities
└── VideoStreamAV1
    ├── file_path: String
    ├── sequence_header: AV1SequenceHeader
    └── hardware_caps: VideoCapabilities

// Playback Classes
VideoStreamPlayback (base)
├── VideoStreamPlaybackH264
│   ├── video_session: RID
│   ├── resource_manager: Ref<VulkanVideoResourceManager>
│   ├── av_sync: Ref<AudioVideoSynchronizer>
│   └── reference_frames: Vector<RID>
├── VideoStreamPlaybackH265
│   ├── video_session: RID
│   ├── resource_manager: Ref<VulkanVideoResourceManager>
│   ├── av_sync: Ref<AudioVideoSynchronizer>
│   ├── reference_frames: Vector<RID>
│   └── tiles_enabled: bool
└── VideoStreamPlaybackAV1
    ├── video_session: RID
    ├── resource_manager: Ref<VulkanVideoResourceManager>
    └── av_sync: Ref<AudioVideoSynchronizer>

// Support Classes
VulkanVideoResourceManager
├── dpb_pool: VideoMemoryPool
├── bitstream_pool: BitstreamBufferPool
└── command_manager: VideoCommandManager

AudioVideoSynchronizer
├── av_sync_filter: OneEuroFilter
├── audio_clock_filter: OneEuroFilter
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

#### Decode Pipeline
```
1. File Loading
   VideoStreamAV1::load_file() → Parse headers → Cache sequence info

2. Playback Initialization
   VideoStreamPlaybackAV1::play() → Create video session → Allocate DPB

3. Frame Decode Loop
   update() → Parse bitstream → Submit decode → Present frame

4. Synchronization
   AudioVideoSynchronizer → Match timestamps → Queue frames
```

#### Encode Pipeline
```
1. Source Preparation
   MovieMaker::capture_frame() → YCbCr conversion → Format validation

2. Encoding Initialization
   VulkanVideoEncoder::initialize() → Create encode session → Allocate buffers

3. Frame Encode Loop
   encode_frame() → YCbCr processing → Hardware encode → Bitstream output

4. Synchronization with OneEuroFilter
   Conductor::process_audio_video_sync() → OneEuroFilter → Timing adjustment
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
- Pre-allocated DPB image arrays
- Pooled bitstream buffers
- Command buffer reuse
- Zero-copy texture sharing

#### Threading Strategy
- Decode operations on video queue
- Graphics operations on graphics queue
- Audio processing on audio thread
- Synchronization via timeline semaphores

#### Quality Scaling
- Adaptive resolution based on performance
- Dynamic bitrate adjustment
- Frame dropping for real-time playback

### Testing Integration Points

#### Unit Tests
- Hardware capability detection
- Resource allocation/deallocation
- Command buffer recording
- Error condition handling

#### Integration Tests
- End-to-end playback scenarios
- Audio-video synchronization
- Memory leak detection
- Performance benchmarking

#### Validation Tests
- Vulkan Video specification compliance
- Cross-platform compatibility
- Driver version compatibility
- Stress testing with multiple streams

This architecture provides a solid foundation for hardware-accelerated video in Godot while maintaining compatibility with existing systems.
