# VK Video Module Documentation

This directory contains comprehensive documentation for the VK Video module, which provides hardware-accelerated AV1 video decoding in Godot using Vulkan Video extensions.

## Documentation Structure

### Core Architecture
- **[00_VideoStreamAV1.md](00_VideoStreamAV1.md)** - VideoStreamAV1 class overview and basic usage
- **[01_Architecture_Overview.md](01_Architecture_Overview.md)** - Complete system architecture and design
- **[02_RenderingDevice_Extensions.md](02_RenderingDevice_Extensions.md)** - Vulkan Video API integration
- **[03_VideoStream_Classes.md](03_VideoStream_Classes.md)** - Video stream class hierarchy and implementation

### Implementation Guides
- **[04_Resource_Management.md](04_Resource_Management.md)** - Memory and resource management strategies
- **[05_Hardware_Detection.md](05_Hardware_Detection.md)** - Hardware capability detection and fallback mechanisms
- **[07_Movie_Maker_Integration.md](07_Movie_Maker_Integration.md)** - Integration with Godot's Movie Maker
- **[08_Audio_Video_Synchronization_Migration.md](08_Audio_Video_Synchronization_Migration.md)** - Migration guide for synchronization system
- **[09_FFmpeg_Interop_Guide.md](09_FFmpeg_Interop_Guide.md)** - FFmpeg integration and interoperability

### Audio-Video Synchronization
The synchronization system has been migrated to the core Godot engine for universal use across all video formats:

- **[audio_video_sync/one_euro_filter.md](audio_video_sync/one_euro_filter.md)** - OneEuro filter implementation for timing smoothing
- **[audio_video_sync/synchronization_strategies.md](audio_video_sync/synchronization_strategies.md)** - Different synchronization approaches
- **[audio_video_sync/timing_algorithms.md](audio_video_sync/timing_algorithms.md)** - Core timing and clock management algorithms
- **[audio_video_sync/performance_tuning.md](audio_video_sync/performance_tuning.md)** - Performance optimization techniques
- **[audio_video_sync/implementation_guide.md](audio_video_sync/implementation_guide.md)** - Step-by-step implementation guide
- **[audio_video_sync/conductor_implementation.md](audio_video_sync/conductor_implementation.md)** - Advanced conductor pattern implementation

## Quick Start

### Basic Usage

```cpp
// Load and play an AV1 video
Ref<VideoStreamAV1> stream = load("res://video.av1");
VideoStreamPlayer* player = memnew(VideoStreamPlayer);
player->set_stream(stream);
player->play();
```

### Hardware Acceleration

The module automatically detects hardware capabilities and falls back to software decoding when necessary:

```cpp
// Check hardware support
if (VideoStreamAV1::is_hardware_supported()) {
    print("Hardware AV1 decoding available");
} else {
    print("Using software fallback");
}
```

### Audio-Video Synchronization

The new synchronization system is built into the base VideoStreamPlayback class:

```cpp
// Access synchronization controls
Ref<VideoStreamPlayback> playback = stream->instantiate_playback();
Ref<AudioVideoSynchronizer> sync = playback->get_av_synchronizer();

// Configure synchronization
sync->set_sync_mode(AudioVideoSynchronizer::SYNC_MODE_AUDIO_MASTER);
sync->set_use_timing_filter(true);
sync->set_sync_threshold(0.040); // 40ms threshold
```

## Implementation Status

### ✅ Completed
- **Audio-Video Synchronization System** - Migrated to core engine
  - OneEuroFilter for timing smoothing
  - AudioVideoSynchronizer for frame queue management
  - Integration with VideoStreamPlayback base class
  - VideoStreamPlayer automatic synchronization support

- **Core Infrastructure**
  - Module registration and class binding
  - Resource format loaders
  - Basic VideoStreamMKV implementation
  - Documentation framework

### 🎯 In Progress
- **Vulkan Video Integration** - Hardware-accelerated decoding
- **AV1 Codec Support** - Full AV1 decode/encode capabilities
- **Advanced Resource Management** - DPB management and memory pools

### 📋 Planned
- **FFmpeg Interoperability** - Container format support
- **Movie Maker Integration** - Hardware encoding support
- **Performance Optimizations** - Multi-threading and memory efficiency

## Key Features

### Hardware Acceleration
- **Vulkan Video API** - Direct GPU video decoding
- **AV1 Codec Focus** - Modern, efficient video compression
- **Automatic Fallback** - Software decoding when hardware unavailable
- **Cross-Platform** - Windows, Linux, macOS support

### Audio-Video Synchronization
- **Universal System** - Works with all video formats (Theora, AV1, future codecs)
- **OneEuro Filtering** - Advanced timing smoothing to reduce jitter
- **Multiple Sync Modes** - Audio master, video master, or external clock
- **Frame Queue Management** - Intelligent frame dropping and presentation
- **Real-time Statistics** - Sync error monitoring and performance metrics

### Container Support
- **MKV/WebM** - Current implementation with Opus audio
- **MP4/MOV** - Planned AV1 container support
- **IVF** - AV1 elementary stream support

## Architecture Highlights

### Separation of Concerns
- **Hardware Decoding** - Module-specific Vulkan Video implementation
- **Playback Synchronization** - Core engine universal system
- **Container Parsing** - Format-specific implementations
- **Resource Management** - Efficient GPU memory handling

### Performance Design
- **Zero-Copy Textures** - Direct GPU memory sharing
- **Command Buffer Pooling** - Efficient Vulkan command management
- **Timeline Semaphores** - Proper GPU synchronization
- **Adaptive Quality** - Dynamic resolution and bitrate adjustment

### Error Handling
- **Graceful Degradation** - Automatic fallback mechanisms
- **Capability Detection** - Runtime hardware feature detection
- **Resource Recovery** - Robust error handling and cleanup
- **Cross-Platform Compatibility** - Consistent behavior across platforms

## Development Guidelines

### Code Organization
- **Modular Design** - Clear separation between components
- **Resource Management** - RAII patterns and proper cleanup
- **Error Handling** - Comprehensive error checking and recovery
- **Documentation** - Inline documentation and external guides

### Testing Strategy
- **Unit Tests** - Component-level functionality testing
- **Integration Tests** - End-to-end playback scenarios
- **Performance Tests** - Benchmarking and optimization validation
- **Compatibility Tests** - Cross-platform and driver validation

### Future Extensibility
- **Plugin Architecture** - Support for additional codecs
- **API Stability** - Backward-compatible interface design
- **Performance Scaling** - Multi-threaded and multi-GPU support
- **Standards Compliance** - Vulkan Video specification adherence

## Migration Notes

### Audio-Video Synchronization Migration

The synchronization system has been moved from `modules/vk_video/sync/` to the core engine:

- **New Location**: `scene/resources/audio_video_synchronizer.*`
- **OneEuro Filter**: `scene/resources/one_euro_filter.*`
- **Base Class Integration**: `scene/resources/video_stream.*`
- **Player Integration**: `scene/gui/video_stream_player.cpp`

This migration provides:
- **Universal Synchronization** - All video formats benefit from the same sync logic
- **Consistency** - Uniform behavior across different video implementations
- **Maintainability** - Single implementation to maintain and improve
- **Backward Compatibility** - Existing code continues to work without changes

See [08_Audio_Video_Synchronization_Migration.md](08_Audio_Video_Synchronization_Migration.md) for detailed migration information.

## Contributing

When contributing to the VK Video module:

1. **Read the Architecture** - Understand the overall design from [01_Architecture_Overview.md](01_Architecture_Overview.md)
2. **Follow Patterns** - Use established patterns for resource management and error handling
3. **Update Documentation** - Keep documentation current with code changes
4. **Test Thoroughly** - Ensure cross-platform compatibility and performance
5. **Consider Synchronization** - Use the core synchronization system for timing-related features

## Support and Resources

- **Vulkan Video Specification** - Official Khronos documentation
- **AV1 Specification** - Alliance for Open Media documentation
- **Godot Engine Documentation** - Core engine architecture and patterns
- **Performance Profiling** - Use Vulkan validation layers and profiling tools

This documentation provides a comprehensive guide to understanding, implementing, and extending the VK Video module for hardware-accelerated video playback in Godot.
