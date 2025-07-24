# Conductor Implementation Analysis

## Overview

The Conductor class from the rhythm game (`modules/vk_video/thirdparty/rhythm_game/game_state/conductor.gd`) demonstrates a sophisticated real-world implementation of OneEuroFilter for audio-video synchronization. This analysis provides insights for adapting the approach to video playback.

## Architecture Overview

### Dual Clock System

The conductor maintains two independent timing sources:

```gdscript
# Audio thread state - jittery but accurate to what user hears
var _song_time_audio: float = -100

# System time state - stable but can drift from audio
var _song_time_system: float = -100
var _song_time_begin: float = 0

# Filtered combination - best of both worlds
var _filtered_audio_system_delta: float = 0
```

### Key Design Principles

1. **Audio Master Clock**: Audio timing is considered the authoritative source
2. **System Clock Stability**: System time provides stable frame-rate timing
3. **Delta Filtering**: Filter the difference between clocks, not the absolute values
4. **Latency Compensation**: Account for output latency and processing delays

## Detailed Implementation Analysis

### Initialization and Setup

```gdscript
func play() -> void:
    var filter_args := {
        "cutoff": allowed_jitter,    # 0.1 Hz default
        "beta": lag_reduction,       # 5.0 default
    }
    _filter = OneEuroFilter.new(filter_args)

    player.play()
    _is_playing = true

    # Capture precise start time with compensation
    _song_time_begin = (
        Time.get_ticks_usec() / 1000000.0
        + first_beat_offset_ms / 1000.0
        + AudioServer.get_time_to_next_mix()
        + _cached_output_latency
    )
```

**Key Insights:**
- Filter parameters are exposed as user-configurable exports
- Start time calculation includes multiple latency compensations
- Audio processing delays are pre-calculated and cached

### Audio Clock Calculation

```gdscript
func _process(_delta: float) -> void:
    # Handle web platform timing bug
    var last_mix := AudioServer.get_time_since_last_mix()
    if last_mix > 1000:
        last_mix = 0

    # Calculate audio-based time (jittery but accurate)
    _song_time_audio = (
        player.get_playback_position()
        - first_beat_offset_ms / 1000.0    # Content offset
        + last_mix                         # Inter-frame smoothing
        - _cached_output_latency           # Hardware compensation
    )

    # Calculate system-based time (stable but can drift)
    _song_time_system = (Time.get_ticks_usec() / 1000000.0) - _song_time_begin
    _song_time_system *= player.pitch_scale
```

**Key Insights:**
- Robust handling of platform-specific timing bugs
- Multiple compensation factors applied to audio timing
- System time scaled by playback rate for variable speed support

### OneEuroFilter Application

```gdscript
func _physics_process(delta: float) -> void:
    if not _is_playing:
        return

    # Filter the DELTA between clocks, not the absolute values
    var audio_system_delta := _song_time_audio - _song_time_system
    _filtered_audio_system_delta = _filter.filter(audio_system_delta, delta)

    # Final time combines system stability with audio accuracy
    # var song_time := _song_time_system + _filtered_audio_system_delta
```

**Key Insights:**
- Filtering applied to the difference, not absolute times
- Runs at physics rate (60 Hz) for consistent filter behavior
- Final time calculation combines both clock sources

### Public API Design

```gdscript
## Returns the current beat of the song.
func get_current_beat() -> float:
    var song_time := _song_time_system + _filtered_audio_system_delta
    return song_time / get_beat_duration()

## Returns the current beat of the song without smoothing.
func get_current_beat_raw() -> float:
    return _song_time_audio / get_beat_duration()
```

**Key Insights:**
- Provides both filtered and raw timing for different use cases
- Beat-based API abstracts timing complexity from game logic
- Raw timing available for debugging and comparison

## Adaptation for Video Synchronization

### Video-Specific Considerations

```cpp
class VideoStreamPlaybackAV1 {
private:
    // Timing state (adapted from conductor pattern)
    double video_time_pts = 0.0;        // From video timestamps (like audio_time)
    double video_time_system = 0.0;     // System clock based (like system_time)
    double video_time_begin = 0.0;      // Playback start reference

    // OneEuroFilter for delta smoothing
    OneEuroFilter av_sync_filter;
    double filtered_av_delta = 0.0;

    // Cached latency values
    double cached_display_latency = 0.0;
    double cached_decode_latency = 0.0;

public:
    void initialize_playback() {
        // Initialize filter with video-appropriate parameters
        av_sync_filter = OneEuroFilter(0.1, 5.0);  // Conservative settings

        // Cache display pipeline latency
        cached_display_latency = get_display_latency();
        cached_decode_latency = estimate_decode_latency();

        // Set reference time with compensation
        video_time_begin = get_system_time() + cached_display_latency;
    }
};
```

### Frame Timing Calculation

```cpp
void update_video_timing(double delta_time) {
    // Calculate PTS-based time (accurate but jittery)
    double video_time_pts = current_frame.pts - cached_decode_latency;

    // Calculate system-based time (stable)
    video_time_system = get_system_time() - video_time_begin;

    // Apply OneEuroFilter to the delta
    double pts_system_delta = video_time_pts - video_time_system;
    filtered_av_delta = av_sync_filter.filter(pts_system_delta, delta_time);
}

double get_corrected_video_time() {
    return video_time_system + filtered_av_delta;
}
```

### Audio-Video Synchronization

```cpp
bool should_present_frame(const FrameInfo& frame) {
    double corrected_video_time = get_corrected_video_time();
    double audio_time = get_audio_clock();

    // Calculate presentation timing
    double frame_presentation_time = frame.pts - cached_decode_latency;
    double av_delta = frame_presentation_time - audio_time;

    // Use threshold for presentation decision
    return abs(av_delta) <= sync_threshold;
}
```

## Parameter Tuning for Video

### Recommended Settings

```cpp
// Conservative (high quality, some latency)
OneEuroFilter video_sync_filter(0.1, 5.0);

// Balanced (good quality, moderate latency)
OneEuroFilter video_sync_filter(0.3, 8.0);

// Responsive (lower quality, minimal latency)
OneEuroFilter video_sync_filter(0.8, 15.0);
```

### Adaptive Parameter Adjustment

```cpp
void adjust_sync_parameters(double sync_quality, double frame_rate) {
    if (sync_quality < 0.8) {
        // Poor sync - increase smoothing
        double new_cutoff = av_sync_filter.get_min_cutoff() * 0.8;
        double new_beta = av_sync_filter.get_beta() * 1.2;
        av_sync_filter.update_parameters(new_cutoff, new_beta);
    } else if (sync_quality > 0.95) {
        // Excellent sync - reduce latency
        double new_cutoff = av_sync_filter.get_min_cutoff() * 1.1;
        double new_beta = av_sync_filter.get_beta() * 0.9;
        av_sync_filter.update_parameters(new_cutoff, new_beta);
    }
}
```

## Error Handling and Edge Cases

### Platform-Specific Issues

From the conductor implementation:

```gdscript
# Handle web platform timing bug
var last_mix := AudioServer.get_time_since_last_mix()
if last_mix > 1000:
    last_mix = 0
```

**Video Adaptation:**
```cpp
double get_frame_decode_time() {
    double decode_time = get_last_decode_duration();

    // Handle platform-specific timing anomalies
    if (decode_time > 1.0) {  // Unreasonably long decode time
        decode_time = estimated_decode_time;
    }

    return decode_time;
}
```

### Pause and Resume Handling

```gdscript
@export var is_paused: bool = false:
    set(value):
        if player:
            player.stream_paused = value
```

**Video Adaptation:**
```cpp
void set_paused(bool paused) {
    if (paused != is_paused) {
        if (paused) {
            pause_time = get_system_time();
        } else {
            // Adjust reference time to account for pause duration
            double pause_duration = get_system_time() - pause_time;
            video_time_begin += pause_duration;

            // Reset filter to avoid artifacts
            av_sync_filter.reset();
        }
        is_paused = paused;
    }
}
```

### Debugging and Monitoring

From conductor (commented debug code):

```gdscript
# Uncomment this to show the difference between raw and filtered time.
#var song_time := _song_time_system + _filtered_audio_system_delta
#print("Error: %+.1f ms" % [abs(song_time - _song_time_audio) * 1000.0])
```

**Video Adaptation:**
```cpp
void debug_sync_quality() {
    double corrected_time = get_corrected_video_time();
    double raw_pts_time = current_frame.pts;
    double error_ms = abs(corrected_time - raw_pts_time) * 1000.0;

    print_line(vformat("A/V Sync Error: %+.1f ms", error_ms));
    print_line(vformat("Filter Delta: %+.1f ms", filtered_av_delta * 1000.0));
}
```

## Performance Considerations

### Update Frequency

The conductor runs filtering at physics rate (60 Hz):

```gdscript
func _physics_process(delta: float) -> void:
    # Filter update here for consistent timing
```

**Video Adaptation:**
- Run sync updates at consistent rate (60 Hz or display refresh rate)
- Separate from variable frame decode rate
- Ensures filter receives consistent delta times

### Memory and CPU Impact

```gdscript
# Minimal state - just two float values per filter
var _filter: OneEuroFilter
var _filtered_audio_system_delta: float = 0
```

**Benefits:**
- Very low memory footprint
- O(1) computational complexity
- Suitable for real-time video processing

## Integration Checklist

### Required Components

1. **Dual Clock System**: Implement both PTS-based and system-based timing
2. **OneEuroFilter**: Integrate filter for delta smoothing
3. **Latency Compensation**: Account for decode and display delays
4. **Parameter Tuning**: Expose filter parameters for optimization
5. **Error Handling**: Robust handling of timing anomalies
6. **Debug Interface**: Provide sync quality monitoring

### Testing Validation

1. **Timing Accuracy**: Verify sync quality across different content
2. **Parameter Sensitivity**: Test filter behavior with various settings
3. **Platform Compatibility**: Validate across different operating systems
4. **Performance Impact**: Measure CPU and memory overhead
5. **Edge Case Handling**: Test pause/resume, seeking, variable playback rates

The conductor implementation provides a proven foundation for implementing robust audio-video synchronization using OneEuroFilter in the VK Video module.
