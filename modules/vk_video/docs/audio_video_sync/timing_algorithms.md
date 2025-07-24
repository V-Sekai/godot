# Timing Algorithms

## Overview

This document details the core timing calculation methods used in audio-video synchronization, based on the OneEuroFilter approach demonstrated in the rhythm game conductor and adapted for video playback.

## Core Timing Concepts

### Timestamp Types

#### Presentation Timestamp (PTS)

-   **Definition**: When a frame should be displayed relative to stream start
-   **Source**: Video container/codec metadata
-   **Characteristics**: Accurate but can be jittery due to decode timing variations
-   **Usage**: Primary timing reference for frame presentation decisions

#### Decode Timestamp (DTS)

-   **Definition**: When a frame should be decoded
-   **Source**: Video container/codec metadata
-   **Characteristics**: Usually earlier than PTS for B-frames
-   **Usage**: Decode scheduling and buffer management

#### System Clock Time

-   **Definition**: High-resolution system timer
-   **Source**: Operating system (Time.get_ticks_usec() in Godot)
-   **Characteristics**: Stable and monotonic, but can drift from media clocks
-   **Usage**: Provides stable timing base for filtering

## Dual Clock Architecture

### Clock Sources

```cpp
class VideoTimingManager {
private:
    // Primary timing sources
    double pts_clock = 0.0;        // From video timestamps
    double system_clock = 0.0;     // From system timer
    double audio_clock = 0.0;      // From audio subsystem

    // Reference points
    double playback_start_time = 0.0;
    double stream_start_pts = 0.0;

    // Latency compensation
    double decode_latency = 0.0;
    double display_latency = 0.0;
    double audio_latency = 0.0;
};
```

### Clock Update Methods

#### PTS Clock Update

```cpp
void update_pts_clock(const FrameInfo& frame) {
    // Raw PTS time with latency compensation
    pts_clock = frame.pts - decode_latency - display_latency;

    // Handle stream start offset
    if (stream_start_pts > 0.0) {
        pts_clock -= stream_start_pts;
    }
}
```

#### System Clock Update

```cpp
void update_system_clock() {
    double current_time = get_high_resolution_time();
    system_clock = current_time - playback_start_time;

    // Apply playback rate scaling
    system_clock *= playback_rate;
}
```

#### Audio Clock Update

```cpp
void update_audio_clock() {
    // Get audio playback position with compensation
    audio_clock = audio_player.get_playback_position() - audio_latency;

    // Add inter-frame smoothing (like conductor implementation)
    double time_since_mix = AudioServer.get_time_since_last_mix();
    if (time_since_mix < 1.0) {  // Sanity check
        audio_clock += time_since_mix;
    }
}
```

## OneEuroFilter Integration

### Delta-Based Filtering

The key insight from the conductor implementation is to filter the **difference** between clocks rather than absolute values:

```cpp
class SyncFilter {
private:
    OneEuroFilter av_delta_filter;
    OneEuroFilter pts_system_filter;

    double filtered_av_delta = 0.0;
    double filtered_pts_delta = 0.0;

public:
    void update_filters(double delta_time) {
        // Audio-video synchronization
        double av_delta = audio_clock - pts_clock;
        filtered_av_delta = av_delta_filter.filter(av_delta, delta_time);

        // PTS-system clock synchronization
        double pts_system_delta = pts_clock - system_clock;
        filtered_pts_delta = pts_system_filter.filter(pts_system_delta, delta_time);
    }

    double get_corrected_video_time() {
        return system_clock + filtered_pts_delta;
    }

    double get_av_sync_offset() {
        return filtered_av_delta;
    }
};
```

### Multi-Stage Filtering

For complex scenarios, multiple filter stages can be used:

```cpp
class AdvancedSyncFilter {
private:
    // Stage 1: Smooth PTS jitter
    OneEuroFilter pts_smoothing_filter;

    // Stage 2: Audio-video sync
    OneEuroFilter av_sync_filter;

    // Stage 3: System clock drift compensation
    OneEuroFilter drift_compensation_filter;

public:
    double calculate_presentation_time(const FrameInfo& frame, double delta_time) {
        // Stage 1: Smooth raw PTS
        double smoothed_pts = pts_smoothing_filter.filter(frame.pts, delta_time);

        // Stage 2: Sync with audio
        double av_delta = smoothed_pts - audio_clock;
        double filtered_av_delta = av_sync_filter.filter(av_delta, delta_time);
        double av_synced_time = audio_clock + filtered_av_delta;

        // Stage 3: Compensate for system clock drift
        double system_delta = av_synced_time - system_clock;
        double filtered_system_delta = drift_compensation_filter.filter(system_delta, delta_time);

        return system_clock + filtered_system_delta;
    }
};
```

## Frame Presentation Algorithms

### Basic Presentation Decision

```cpp
bool should_present_frame(const FrameInfo& frame) {
    double presentation_time = calculate_presentation_time(frame);
    double current_time = get_corrected_video_time();

    double time_diff = presentation_time - current_time;

    // Present if within threshold
    return abs(time_diff) <= presentation_threshold;
}
```

### Adaptive Threshold Calculation

```cpp
double calculate_adaptive_threshold(double frame_rate, double sync_quality) {
    double base_threshold = 1.0 / (frame_rate * 2.0);  // Half frame duration

    // Adjust based on sync quality
    if (sync_quality < 0.8) {
        return base_threshold * 1.5;  // More lenient for poor sync
    } else if (sync_quality > 0.95) {
        return base_threshold * 0.7;  // Stricter for good sync
    }

    return base_threshold;
}
```

### Frame Drop/Duplicate Logic

```cpp
enum FrameAction {
    PRESENT_FRAME,
    DROP_FRAME,
    DUPLICATE_FRAME,
    WAIT_FOR_TIME
};

FrameAction determine_frame_action(const FrameInfo& frame) {
    double presentation_time = calculate_presentation_time(frame);
    double current_time = get_corrected_video_time();
    double time_diff = presentation_time - current_time;

    if (time_diff < -drop_threshold) {
        return DROP_FRAME;  // Frame is too late
    } else if (time_diff > duplicate_threshold) {
        return DUPLICATE_FRAME;  // Need to show current frame longer
    } else if (abs(time_diff) <= presentation_threshold) {
        return PRESENT_FRAME;  // Perfect timing
    } else {
        return WAIT_FOR_TIME;  // Wait for proper timing
    }
}
```

## Latency Compensation

### Decode Latency Estimation

```cpp
class DecodeLatencyEstimator {
private:
    double running_average = 0.0;
    int sample_count = 0;
    static const int MAX_SAMPLES = 60;

public:
    void update_estimate(double decode_time) {
        if (sample_count < MAX_SAMPLES) {
            running_average = (running_average * sample_count + decode_time) / (sample_count + 1);
            sample_count++;
        } else {
            // Exponential moving average
            running_average = running_average * 0.95 + decode_time * 0.05;
        }
    }

    double get_estimate() const {
        return running_average;
    }
};
```

### Display Latency Measurement

```cpp
double measure_display_latency() {
    // Platform-specific implementation
    #ifdef WINDOWS
        return get_windows_display_latency();
    #elif LINUX
        return get_linux_display_latency();
    #elif MACOS
        return get_macos_display_latency();
    #else
        return 16.67;  // Assume 1 frame at 60Hz
    #endif
}
```

### Audio Latency Compensation

```cpp
double get_audio_latency() {
    // Use cached value from conductor pattern
    static double cached_latency = AudioServer.get_output_latency();

    // Periodically update (audio latency can change)
    static uint64_t last_update = 0;
    uint64_t current_time = Time.get_ticks_msec();

    if (current_time - last_update > 5000) {  // Update every 5 seconds
        cached_latency = AudioServer.get_output_latency();
        last_update = current_time;
    }

    return cached_latency;
}
```

## Synchronization Quality Metrics

### Sync Error Calculation

```cpp
struct SyncMetrics {
    double avg_error = 0.0;
    double max_error = 0.0;
    double error_variance = 0.0;
    int frames_dropped = 0;
    int frames_duplicated = 0;
    double sync_quality = 1.0;
};

void update_sync_metrics(SyncMetrics& metrics, double av_error) {
    // Update running average
    metrics.avg_error = metrics.avg_error * 0.95 + abs(av_error) * 0.05;

    // Track maximum error
    metrics.max_error = max(metrics.max_error * 0.99, abs(av_error));

    // Calculate sync quality (0.0 = poor, 1.0 = perfect)
    double normalized_error = metrics.avg_error / presentation_threshold;
    metrics.sync_quality = max(0.0, 1.0 - normalized_error);
}
```

### Quality-Based Parameter Adjustment

```cpp
void adjust_filter_parameters(SyncMetrics& metrics) {
    if (metrics.sync_quality < 0.7) {
        // Poor sync - increase smoothing
        double new_cutoff = av_sync_filter.get_min_cutoff() * 0.8;
        double new_beta = av_sync_filter.get_beta() * 1.3;
        av_sync_filter.update_parameters(new_cutoff, new_beta);

        // Also increase presentation threshold
        presentation_threshold *= 1.1;
    } else if (metrics.sync_quality > 0.95) {
        // Excellent sync - reduce latency
        double new_cutoff = av_sync_filter.get_min_cutoff() * 1.1;
        double new_beta = av_sync_filter.get_beta() * 0.9;
        av_sync_filter.update_parameters(new_cutoff, new_beta);

        // Tighten presentation threshold
        presentation_threshold *= 0.95;
    }
}
```

## Variable Playback Rate Support

### Rate-Scaled Timing

```cpp
void update_timing_for_playback_rate(double rate) {
    // Scale system clock progression
    system_clock_scale = rate;

    // Adjust filter parameters for rate
    if (rate != 1.0) {
        // Faster playback needs more responsive filtering
        double rate_factor = sqrt(abs(rate));
        double adjusted_cutoff = base_cutoff * rate_factor;
        double adjusted_beta = base_beta * rate_factor;

        av_sync_filter.update_parameters(adjusted_cutoff, adjusted_beta);
    }
}
```

### Seek Handling

```cpp
void handle_seek(double seek_time) {
    // Reset all timing state
    playback_start_time = get_high_resolution_time() - seek_time;
    stream_start_pts = seek_time;

    // Reset filters to avoid artifacts
    av_sync_filter.reset();
    pts_system_filter.reset();

    // Clear metrics
    sync_metrics = SyncMetrics{};
}
```

## Performance Optimization

### Update Frequency Management

```cpp
class TimingUpdateManager {
private:
    static const double UPDATE_INTERVAL = 1.0 / 60.0;  // 60 Hz
    double last_update_time = 0.0;

public:
    bool should_update_timing() {
        double current_time = get_high_resolution_time();
        if (current_time - last_update_time >= UPDATE_INTERVAL) {
            last_update_time = current_time;
            return true;
        }
        return false;
    }
};
```

### Computational Efficiency

```cpp
// Pre-calculate frequently used values
struct TimingConstants {
    double frame_duration;
    double half_frame_duration;
    double presentation_threshold;
    double drop_threshold;
    double duplicate_threshold;

    void update_for_framerate(double fps) {
        frame_duration = 1.0 / fps;
        half_frame_duration = frame_duration * 0.5;
        presentation_threshold = frame_duration * 0.4;
        drop_threshold = frame_duration * 0.8;
        duplicate_threshold = frame_duration * 1.2;
    }
};
```

These timing algorithms provide the foundation for robust audio-video synchronization using the OneEuroFilter approach, ensuring smooth playback across various conditions and platforms.
