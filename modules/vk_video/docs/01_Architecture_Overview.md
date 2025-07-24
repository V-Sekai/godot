# VK Video Module - Architecture Overview

## Brief Description
High-level system architecture for Vulkan Video-based hardware acceleration supporting H.264, H.265, and AV1 codecs in Godot Engine.

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
│  └── video_image_create()                                  │
├─────────────────────────────────────────────────────────────┤
│  Vulkan Video Implementation                                │
│  ├── VulkanVideoResourceManager                            │
│  ├── H264BitstreamParser                                   │
│  ├── H265BitstreamParser                                   │
│  ├── AV1BitstreamParser                                    │
│  └── AudioVideoSynchronizer (OneEuroFilter)                │
├─────────────────────────────────────────────────────────────┤
│  Vulkan Driver Layer                                       │
│  ├── VK_KHR_video_decode_h264                              │
│  ├── VK_KHR_video_decode_h265                              │
│  └── VK_KHR_video_decode_av1                               │
└─────────────────────────────────────────────────────────────┘
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
