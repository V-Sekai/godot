# Audio-Video Synchronization Migration Guide

## Overview

The audio-video synchronization system has been migrated from the `modules/vk_video` module to the core Godot video system. This architectural change provides several benefits:

- **Universal Synchronization**: All video formats (Theora, AV1, future codecs) now benefit from the same synchronization logic
- **Consistency**: Uniform synchronization behavior across all video playback
- **Maintainability**: Single implementation to maintain and improve
- **Proper Separation**: Hardware decoding vs. playback synchronization are separate concerns

## New Architecture

### Core Components

The synchronization system now consists of three main components in the core engine:

1. **OneEuroFilter** (`scene/resources/one_euro_filter.h/cpp`)
   - Timing smoothing filter for reducing jitter
   - Moved from `modules/vk_video/sync/`

2. **AudioVideoSynchronizer** (`scene/resources/audio_video_synchronizer.h/cpp`)
   - Main synchronization logic
   - Frame queue management
   - Clock synchronization
   - Statistics tracking

3. **VideoStreamPlayback** (`scene/resources/video_stream.h/cpp`)
   - Base class now includes synchronization support
   - All video implementations inherit synchronization capabilities

### Integration Points

#### VideoStreamPlayer
- Automatically uses synchronized textures when available
- Falls back to direct texture access if synchronization is disabled
- No API changes required for existing code

#### VideoStreamPlayback Base Class
New methods available to all video implementations:

```cpp
// Synchronization control
void set_use_synchronization(bool p_enable);
bool get_use_synchronization() const;
Ref<AudioVideoSynchronizer> get_av_synchronizer() const;

// Clock management
void update_audio_clock(double p_time);
void update_video_clock(double p_time);

// Frame management
void queue_video_frame(const Ref<Texture2D> &p_texture, double p_presentation_time, uint64_t p_frame_number = 0);
Ref<Texture2D> get_synchronized_texture() const;
```

## Migration for Video Implementations

### For vk_video Module

The vk_video module implementations should be updated to:

1. **Remove local synchronization code**
   - Delete `modules/vk_video/sync/` directory
   - Remove synchronization logic from VideoStreamAV1 and VideoStreamMKV

2. **Use base class synchronization**
   - Call `queue_video_frame()` instead of direct texture updates
   - Use `update_audio_clock()` and `update_video_clock()` for timing
   - Let the base class handle frame presentation

### For Other Video Modules

Existing video modules (like Theora) can now benefit from synchronization by:

1. **Enabling synchronization** (enabled by default)
2. **Updating timing calls**:
   ```cpp
   // Update clocks during playback
   update_audio_clock(audio_time);
   update_video_clock(video_time);
   
   // Queue frames instead of direct texture updates
   queue_video_frame(texture, presentation_time, frame_number);
   ```

3. **Using synchronized texture**:
   ```cpp
   // In get_texture() implementation
   if (get_use_synchronization()) {
       return get_synchronized_texture();
   }
   return direct_texture; // Fallback
   ```

## Configuration Options

### AudioVideoSynchronizer Settings

```cpp
// Synchronization mode
enum SyncMode {
    SYNC_MODE_AUDIO_MASTER,  // Audio is master clock (default)
    SYNC_MODE_VIDEO_MASTER,  // Video is master clock
    SYNC_MODE_EXTERNAL       // External clock source
};

// Timing parameters
set_sync_threshold(0.040);           // 40ms sync threshold
set_max_queue_size(3);               // Frame queue size
set_use_timing_filter(true);         // Enable OneEuro filter
```

### OneEuroFilter Parameters

```cpp
// Filter tuning (default values work well for most cases)
filter->update_parameters(0.1, 5.0); // min_cutoff, beta
```

## Benefits of New Architecture

### Performance
- Reduced code duplication
- Optimized frame queue management
- Efficient timing calculations

### Quality
- Consistent synchronization across all video formats
- Advanced timing filters reduce jitter
- Configurable synchronization modes

### Maintainability
- Single implementation to maintain
- Easier to add new synchronization features
- Better separation of concerns

## Backward Compatibility

- **Existing VideoStreamPlayer code**: No changes required
- **Custom video implementations**: Optional migration to use new synchronization
- **Performance**: No performance impact when synchronization is disabled

## Future Enhancements

The new architecture enables future improvements:

- **Advanced synchronization algorithms**
- **Multiple synchronization modes**
- **Better statistics and debugging**
- **Integration with external timing sources**

## Example Implementation

Here's how a video implementation should integrate with the new synchronization system:

```cpp
class VideoStreamPlaybackExample : public VideoStreamPlayback {
    void update(double p_delta) override {
        // Decode video frame
        Ref<Texture2D> frame = decode_next_frame();
        double presentation_time = get_frame_timestamp();
        
        // Update clocks
        update_video_clock(presentation_time);
        if (has_audio()) {
            update_audio_clock(get_audio_timestamp());
        }
        
        // Queue frame for synchronized presentation
        queue_video_frame(frame, presentation_time, frame_number++);
    }
    
    Ref<Texture2D> get_texture() const override {
        // Use synchronized texture if available
        if (get_use_synchronization()) {
            return get_synchronized_texture();
        }
        return current_frame; // Fallback
    }
};
```

This migration provides a solid foundation for high-quality video playback across all supported formats in Godot.
