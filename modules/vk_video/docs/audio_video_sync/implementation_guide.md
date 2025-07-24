# Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing OneEuroFilter-based audio-video synchronization in the VK Video module, based on the proven approach from the rhythm game conductor.

## Prerequisites

### Required Components

1. **OneEuroFilter Class**: Core filtering implementation
2. **Audio Clock Access**: Interface to audio playback timing
3. **Video Timestamp Access**: PTS/DTS from decoded frames
4. **System Timer**: High-resolution timing source
5. **Latency Measurement**: Audio and display pipeline delays

### Dependencies

```cpp
// Required headers
#include "core/math/math_funcs.h"
#include "core/os/time.h"
#include "servers/audio_server.h"
#include "servers/rendering_server.h"
```

## Step 1: OneEuroFilter Integration

### Basic Filter Implementation

```cpp
// File: modules/vk_video/sync/one_euro_filter.h
#ifndef ONE_EURO_FILTER_H
#define ONE_EURO_FILTER_H

#include "core/object/ref_counted.h"

class OneEuroFilter : public RefCounted {
    GDCLASS(OneEuroFilter, RefCounted);

private:
    struct LowPassFilter {
        double last_value = 0.0;

        double filter(double value, double alpha) {
            double result = alpha * value + (1.0 - alpha) * last_value;
            last_value = result;
            return result;
        }

        void reset() { last_value = 0.0; }
    };

    double min_cutoff;
    double beta;
    double d_cutoff;
    LowPassFilter x_filter;
    LowPassFilter dx_filter;

    double calculate_alpha(double rate, double cutoff) const {
        double tau = 1.0 / (2.0 * Math::PI * cutoff);
        double te = 1.0 / rate;
        return 1.0 / (1.0 + tau / te);
    }

protected:
    static void _bind_methods();

public:
    OneEuroFilter(double p_min_cutoff = 0.1, double p_beta = 5.0);

    double filter(double value, double delta_time);
    void reset();
    void update_parameters(double p_min_cutoff, double p_beta);

    double get_min_cutoff() const { return min_cutoff; }
    double get_beta() const { return beta; }
};

#endif // ONE_EURO_FILTER_H
```

### Filter Implementation

```cpp
// File: modules/vk_video/sync/one_euro_filter.cpp
#include "one_euro_filter.h"

OneEuroFilter::OneEuroFilter(double p_min_cutoff, double p_beta)
    : min_cutoff(p_min_cutoff), beta(p_beta), d_cutoff(p_min_cutoff) {
}

double OneEuroFilter::filter(double value, double delta_time) {
    if (delta_time <= 0.0) {
        return value;  // Skip filtering for invalid delta
    }

    double rate = 1.0 / delta_time;
    double dx = (value - x_filter.last_value) * rate;

    double edx = dx_filter.filter(dx, calculate_alpha(rate, d_cutoff));
    double cutoff = min_cutoff + beta * Math::abs(edx);

    return x_filter.filter(value, calculate_alpha(rate, cutoff));
}

void OneEuroFilter::reset() {
    x_filter.reset();
    dx_filter.reset();
}

void OneEuroFilter::update_parameters(double p_min_cutoff, double p_beta) {
    min_cutoff = p_min_cutoff;
    beta = p_beta;
    d_cutoff = p_min_cutoff;
}

void OneEuroFilter::_bind_methods() {
    ClassDB::bind_method(D_METHOD("filter", "value", "delta_time"), &OneEuroFilter::filter);
    ClassDB::bind_method(D_METHOD("reset"), &OneEuroFilter::reset);
    ClassDB::bind_method(D_METHOD("update_parameters", "min_cutoff", "beta"), &OneEuroFilter::update_parameters);
    ClassDB::bind_method(D_METHOD("get_min_cutoff"), &OneEuroFilter::get_min_cutoff);
    ClassDB::bind_method(D_METHOD("get_beta"), &OneEuroFilter::get_beta);
}
```

## Step 2: Audio-Video Synchronizer

### Core Synchronizer Class

```cpp
// File: modules/vk_video/sync/av_synchronizer.h
#ifndef AV_SYNCHRONIZER_H
#define AV_SYNCHRONIZER_H

#include "core/object/ref_counted.h"
#include "one_euro_filter.h"

class AVSynchronizer : public RefCounted {
    GDCLASS(AVSynchronizer, RefCounted);

public:
    struct FrameInfo {
        double pts = 0.0;
        double dts = 0.0;
        uint64_t frame_number = 0;
        bool keyframe = false;
    };

    struct SyncMetrics {
        double avg_error = 0.0;
        double max_error = 0.0;
        double sync_quality = 1.0;
        int frames_dropped = 0;
        int frames_duplicated = 0;
    };

private:
    // Timing state (based on conductor pattern)
    double audio_clock = 0.0;
    double video_pts_clock = 0.0;
    double system_clock = 0.0;
    double playback_start_time = 0.0;

    // OneEuroFilter for delta smoothing
    Ref<OneEuroFilter> av_sync_filter;
    double filtered_av_delta = 0.0;

    // Latency compensation
    double cached_audio_latency = 0.0;
    double cached_display_latency = 0.0;
    double cached_decode_latency = 0.0;

    // Synchronization parameters
    double sync_threshold = 0.04;  // 40ms
    double drop_threshold = 0.1;   // 100ms
    double duplicate_threshold = 0.02;  // 20ms

    // Performance metrics
    SyncMetrics current_metrics;
    bool is_playing = false;

protected:
    static void _bind_methods();

public:
    AVSynchronizer();

    // Initialization
    void initialize(double p_sync_threshold = 0.04);
    void start_playback();
    void stop_playback();
    void reset();

    // Clock updates (call from main thread)
    void update_audio_clock();
    void update_video_clock(const FrameInfo& frame);
    void update_system_clock();

    // Synchronization (call from physics thread for consistent timing)
    void update_synchronization(double delta_time);

    // Frame presentation decisions
    bool should_present_frame(const FrameInfo& frame);
    bool should_drop_frame(const FrameInfo& frame);
    bool should_duplicate_frame();

    // Configuration
    void set_sync_threshold(double threshold);
    void set_filter_parameters(double min_cutoff, double beta);

    // Metrics and debugging
    Dictionary get_sync_metrics() const;
    double get_corrected_video_time() const;
    double get_av_sync_offset() const;

private:
    void update_latency_cache();
    void update_sync_metrics(double av_error, bool frame_dropped, bool frame_duplicated);
    double get_high_resolution_time() const;
};

#endif // AV_SYNCHRONIZER_H
```

### Synchronizer Implementation

```cpp
// File: modules/vk_video/sync/av_synchronizer.cpp
#include "av_synchronizer.h"
#include "servers/audio_server.h"
#include "core/os/time.h"

AVSynchronizer::AVSynchronizer() {
    av_sync_filter.instantiate();
    av_sync_filter->update_parameters(0.1, 5.0);  // Conservative defaults
}

void AVSynchronizer::initialize(double p_sync_threshold) {
    sync_threshold = p_sync_threshold;
    update_latency_cache();
    reset();
}

void AVSynchronizer::start_playback() {
    playback_start_time = get_high_resolution_time();
    is_playing = true;

    // Reset filter state for clean startup
    av_sync_filter->reset();
    filtered_av_delta = 0.0;

    // Reset metrics
    current_metrics = SyncMetrics{};
}

void AVSynchronizer::stop_playback() {
    is_playing = false;
}

void AVSynchronizer::reset() {
    audio_clock = 0.0;
    video_pts_clock = 0.0;
    system_clock = 0.0;
    filtered_av_delta = 0.0;

    if (av_sync_filter.is_valid()) {
        av_sync_filter->reset();
    }

    current_metrics = SyncMetrics{};
}

void AVSynchronizer::update_audio_clock() {
    if (!is_playing) return;

    // Get audio playback position with latency compensation
    // This mimics the conductor implementation
    audio_clock = AudioServer::get_singleton()->get_playback_position() - cached_audio_latency;

    // Add inter-frame smoothing
    double time_since_mix = AudioServer::get_singleton()->get_time_since_last_mix();
    if (time_since_mix < 1.0) {  // Sanity check for web platform bug
        audio_clock += time_since_mix;
    }
}

void AVSynchronizer::update_video_clock(const FrameInfo& frame) {
    if (!is_playing) return;

    // Update PTS-based clock with latency compensation
    video_pts_clock = frame.pts - cached_decode_latency - cached_display_latency;
}

void AVSynchronizer::update_system_clock() {
    if (!is_playing) return;

    system_clock = get_high_resolution_time() - playback_start_time;
}

void AVSynchronizer::update_synchronization(double delta_time) {
    if (!is_playing) return;

    // Apply OneEuroFilter to the audio-video delta (key insight from conductor)
    double av_delta = video_pts_clock - audio_clock;
    filtered_av_delta = av_sync_filter->filter(av_delta, delta_time);

    // Update performance metrics
    update_sync_metrics(av_delta, false, false);
}

bool AVSynchronizer::should_present_frame(const FrameInfo& frame) {
    if (!is_playing) return false;

    double corrected_video_time = get_corrected_video_time();
    double frame_time = frame.pts - cached_decode_latency - cached_display_latency;
    double time_diff = frame_time - corrected_video_time;

    return Math::abs(time_diff) <= sync_threshold;
}

bool AVSynchronizer::should_drop_frame(const FrameInfo& frame) {
    if (!is_playing) return false;

    double corrected_video_time = get_corrected_video_time();
    double frame_time = frame.pts - cached_decode_latency - cached_display_latency;
    double time_diff = frame_time - corrected_video_time;

    // Drop if frame is too late
    bool should_drop = time_diff < -drop_threshold;

    if (should_drop) {
        current_metrics.frames_dropped++;
    }

    return should_drop;
}

bool AVSynchronizer::should_duplicate_frame() {
    if (!is_playing) return false;

    // Check if we need to show current frame longer
    double time_since_last = system_clock - video_pts_clock;
    bool should_duplicate = time_since_last > duplicate_threshold;

    if (should_duplicate) {
        current_metrics.frames_duplicated++;
    }

    return should_duplicate;
}

double AVSynchronizer::get_corrected_video_time() const {
    // Combine audio clock with filtered delta for best of both worlds
    return audio_clock + filtered_av_delta;
}

double AVSynchronizer::get_av_sync_offset() const {
    return filtered_av_delta;
}

void AVSynchronizer::update_latency_cache() {
    cached_audio_latency = AudioServer::get_singleton()->get_output_latency();

    // Platform-specific display latency estimation
    cached_display_latency = 16.67 / 1000.0;  // Assume 1 frame at 60Hz

    // Decode latency will be estimated during playback
    cached_decode_latency = 10.0 / 1000.0;  // 10ms initial estimate
}

void AVSynchronizer::update_sync_metrics(double av_error, bool frame_dropped, bool frame_duplicated) {
    // Update running average
    current_metrics.avg_error = current_metrics.avg_error * 0.95 + Math::abs(av_error) * 0.05;

    // Track maximum error
    current_metrics.max_error = MAX(current_metrics.max_error * 0.99, Math::abs(av_error));

    // Calculate sync quality (0.0 = poor, 1.0 = perfect)
    double normalized_error = current_metrics.avg_error / sync_threshold;
    current_metrics.sync_quality = MAX(0.0, 1.0 - normalized_error);

    // Update frame counters
    if (frame_dropped) current_metrics.frames_dropped++;
    if (frame_duplicated) current_metrics.frames_duplicated++;
}

double AVSynchronizer::get_high_resolution_time() const {
    return Time::get_singleton()->get_ticks_usec() / 1000000.0;
}

Dictionary AVSynchronizer::get_sync_metrics() const {
    Dictionary metrics;
    metrics["avg_error_ms"] = current_metrics.avg_error * 1000.0;
    metrics["max_error_ms"] = current_metrics.max_error * 1000.0;
    metrics["sync_quality"] = current_metrics.sync_quality;
    metrics["frames_dropped"] = current_metrics.frames_dropped;
    metrics["frames_duplicated"] = current_metrics.frames_duplicated;
    metrics["av_sync_offset_ms"] = filtered_av_delta * 1000.0;
    return metrics;
}

void AVSynchronizer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "sync_threshold"), &AVSynchronizer::initialize, DEFVAL(0.04));
    ClassDB::bind_method(D_METHOD("start_playback"), &AVSynchronizer::start_playback);
    ClassDB::bind_method(D_METHOD("stop_playback"), &AVSynchronizer::stop_playback);
    ClassDB::bind_method(D_METHOD("reset"), &AVSynchronizer::reset);

    ClassDB::bind_method(D_METHOD("update_synchronization", "delta_time"), &AVSynchronizer::update_synchronization);
    ClassDB::bind_method(D_METHOD("get_sync_metrics"), &AVSynchronizer::get_sync_metrics);
    ClassDB::bind_method(D_METHOD("get_corrected_video_time"), &AVSynchronizer::get_corrected_video_time);
    ClassDB::bind_method(D_METHOD("get_av_sync_offset"), &AVSynchronizer::get_av_sync_offset);

    ClassDB::bind_method(D_METHOD("set_sync_threshold", "threshold"), &AVSynchronizer::set_sync_threshold);
    ClassDB::bind_method(D_METHOD("set_filter_parameters", "min_cutoff", "beta"), &AVSynchronizer::set_filter_parameters);
}
```

## Step 3: VideoStreamPlaybackAV1 Integration

### Integration Points

```cpp
// File: modules/vk_video/video_stream_playback_av1.h (additions)
class VideoStreamPlaybackAV1 : public VideoStreamPlayback {
private:
    // Add synchronization components
    Ref<AVSynchronizer> av_synchronizer;

    // Timing update management
    double last_sync_update_time = 0.0;
    static constexpr double SYNC_UPDATE_INTERVAL = 1.0 / 60.0;  // 60 Hz

public:
    // Override existing methods to integrate synchronization
    virtual void play() override;
    virtual void stop() override;
    virtual void update(double p_delta) override;
    virtual Ref<Texture2D> get_texture() const override;

    // New synchronization methods
    void initialize_synchronization();
    bool should_present_current_frame();
    Dictionary get_sync_debug_info();
};
```

### Implementation Integration

```cpp
// File: modules/vk_video/video_stream_playback_av1.cpp (additions)

void VideoStreamPlaybackAV1::play() {
    // Existing play logic...

    // Initialize synchronization
    initialize_synchronization();
    av_synchronizer->start_playback();
}

void VideoStreamPlaybackAV1::stop() {
    if (av_synchronizer.is_valid()) {
        av_synchronizer->stop_playback();
    }

    // Existing stop logic...
}

void VideoStreamPlaybackAV1::initialize_synchronization() {
    if (!av_synchronizer.is_valid()) {
        av_synchronizer.instantiate();
        av_synchronizer->initialize();

        // Configure for video playback (balanced settings)
        av_synchronizer->set_filter_parameters(0.1, 5.0);
        av_synchronizer->set_sync_threshold(0.04);  // 40ms threshold
    }
}

void VideoStreamPlaybackAV1::update(double p_delta) {
    // Existing update logic...

    if (av_synchronizer.is_valid() && is_playing()) {
        // Update clocks at main thread rate
        av_synchronizer->update_audio_clock();
        av_synchronizer->update_system_clock();

        // Update video clock when new frame is decoded
        if (has_new_frame()) {
            AVSynchronizer::FrameInfo frame_info;
            frame_info.pts = get_current_frame_pts();
            frame_info.dts = get_current_frame_dts();
            frame_info.frame_number = get_current_frame_number();
            frame_info.keyframe = is_current_frame_keyframe();

            av_synchronizer->update_video_clock(frame_info);
        }

        // Update synchronization at consistent rate
        double current_time = Time::get_singleton()->get_ticks_usec() / 1000000.0;
        if (current_time - last_sync_update_time >= SYNC_UPDATE_INTERVAL) {
            av_synchronizer->update_synchronization(SYNC_UPDATE_INTERVAL);
            last_sync_update_time = current_time;
        }
    }
}

Ref<Texture2D> VideoStreamPlaybackAV1::get_texture() const {
    // Check if current frame should be presented
    if (av_synchronizer.is_valid() && !should_present_current_frame()) {
        // Return previous frame or null if timing isn't right
        return get_previous_texture();
    }

    // Existing texture retrieval logic...
    return current_texture;
}

bool VideoStreamPlaybackAV1::should_present_current_frame() {
    if (!av_synchronizer.is_valid()) {
        return true;  // Fallback to always present
    }

    AVSynchronizer::FrameInfo frame_info;
    frame_info.pts = get_current_frame_pts();
    frame_info.dts = get_current_frame_dts();
    frame_info.frame_number = get_current_frame_number();
    frame_info.keyframe = is_current_frame_keyframe();

    // Check if frame should be dropped
    if (av_synchronizer->should_drop_frame(frame_info)) {
        advance_to_next_frame();  // Skip this frame
        return false;
    }

    // Check if frame should be presented
    return av_synchronizer->should_present_frame(frame_info);
}

Dictionary VideoStreamPlaybackAV1::get_sync_debug_info() {
    if (!av_synchronizer.is_valid()) {
        return Dictionary();
    }

    Dictionary debug_info = av_synchronizer->get_sync_metrics();
    debug_info["corrected_video_time"] = av_synchronizer->get_corrected_video_time();
    debug_info["av_sync_offset_ms"] = av_synchronizer->get_av_sync_offset() * 1000.0;

    return debug_info;
}
```

## Step 4: Module Registration

### SCsub Configuration

```python
# File: modules/vk_video/SCsub
#!/usr/bin/env python

Import("env")
Import("env_modules")

env_vk_video = env_modules.Clone()

# Add source files
env_vk_video.add_source_files(env.modules_sources, "*.cpp")
env_vk_video.add_source_files(env.modules_sources, "sync/*.cpp")

# Add include paths
env_vk_video.Prepend(CPPPATH=["#modules/vk_video"])
env_vk_video.Prepend(CPPPATH=["#modules/vk_video/sync"])
```

### Module Registration

```cpp
// File: modules/vk_video/register_types.cpp
#include "register_types.h"
#include "core/object/class_db.h"
#include "sync/one_euro_filter.h"
#include "sync/av_synchronizer.h"
#include "video_stream_av1.h"
#include "video_stream_playback_av1.h"

void initialize_vk_video_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    // Register synchronization classes
    GDREGISTER_CLASS(OneEuroFilter);
    GDREGISTER_CLASS(AVSynchronizer);

    // Register video classes
    GDREGISTER_CLASS(VideoStreamAV1);
    GDREGISTER_CLASS(VideoStreamPlaybackAV1);
}

void uninitialize_vk_video_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}
```

## Step 5: Testing and Validation

### Basic Functionality Test

```gdscript
# File: test_av_sync.gd
extends Node

func test_basic_synchronization():
    var av_sync = AVSynchronizer.new()
    av_sync.initialize(0.04)  # 40ms threshold

    # Test filter parameters
    av_sync.set_filter_parameters(0.1, 5.0)

    # Simulate playback
    av_sync.start_playback()

    # Test synchronization update
    for i in range(60):  # 1 second at 60fps
        av_sync.update_synchronization(1.0/60.0)

        var metrics = av_sync.get_sync_metrics()
        print("Frame %d: Sync Quality = %.2f" % [i, metrics.sync_quality])

    av_sync.stop_playback()

func test_video_playback():
    var video_stream = VideoStreamAV1.new()
    video_stream.file = "res://test_video.av1"

    var video_player = VideoStreamPlayer.new()
    video_player.stream = video_stream
    add_child(video_player)

    video_player.play()

    # Monitor sync quality
    var timer = Timer.new()
    timer.wait_time = 1.0
    timer.timeout.connect(_print_sync_stats.bind(video_player))
    add_child(timer)
    timer.start()

func _print_sync_stats(player: VideoStreamPlayer):
    var playback = player.get_stream_playback()
    if playback.has_method("get_sync_debug_info"):
        var debug_info = playback.get_sync_debug_info()
        print("Sync Stats: ", debug_info)
```

### Performance Benchmark

```gdscript
# File: benchmark_av_sync.gd
extends Node

func benchmark_filter_performance():
    var filter = OneEuroFilter.new()
    filter.update_parameters(0.1, 5.0)

    var start_time = Time.get_ticks_usec()
    var iterations = 10000

    for i in range(iterations):
        var test_value = sin(i * 0.1) + randf_range(-0.1, 0.1)  # Noisy sine wave
        filter.filter(test_value, 1.0/60.0)

    var end_time = Time.get_ticks_usec()
    var total_time = (end_time - start_time) / 1000.0  # Convert to milliseconds

    print("Filter Performance:")
    print("  Iterations: %d" % iterations)
    print("  Total Time: %.2f ms" % total_time)
    print("  Time per Call: %.3f μs" % (total_time * 1000.0 / iterations))
    print("  Calls per Second: %.0f" % (iterations / (total_time / 1000.0)))
```

## Step 6: Configuration and Tuning

### Project Settings Integration

```cpp
// File: modules/vk_video/vk_video_settings.cpp
void register_vk_video_settings() {
    // Audio-Video Sync Settings
    GLOBAL_DEF("video/av_sync/filter_cutoff", 0.1);
    GLOBAL_DEF("video/av_sync/filter_beta", 5.0);
    GLOBAL_DEF("video/av_sync/sync_threshold_ms", 40.0);
    GLOBAL_DEF("video/av_sync/drop_threshold_ms", 100.0);
    GLOBAL_DEF("video/av_sync/adaptive_tuning", true);

    // Performance Settings
    GLOBAL_DEF("video/av_sync/update_frequency", 60.0);
    GLOBAL_DEF("video/av_sync/enable_metrics", false);

    // Platform-specific defaults
    #ifdef MOBILE_ENABLED
    GLOBAL_DEF("video/av_sync/power_aware", true);
    #endif
}
```

### Runtime Configuration

```gdscript
# Example configuration in project
func configure_av_sync_for_music_video():
    # Tighter sync for music content
    ProjectSettings.set_setting("video/av_sync/filter_cutoff", 0.05)
    ProjectSettings.set_setting("video/av_sync/filter_beta", 3.0)
    ProjectSettings.set_setting("video/av_sync/sync_threshold_ms", 20.0)

func configure_av_sync_for_gaming():
    # Lower latency for interactive content
    ProjectSettings.set_setting("video/av_sync/filter_cutoff", 0.8)
    ProjectSettings.set_setting("video/av_sync/filter_beta", 20.0)
    ProjectSettings.set_setting("video/av_sync/sync_threshold_ms", 15.0)
```

## Troubleshooting

### Common Issues

1. **High Sync Error**: Increase filter smoothing (lower cutoff)
2. **High Latency**: Decrease filter smoothing (higher cutoff, higher beta)
3. **Unstable Sync**: Check audio latency measurement accuracy
4. **Frame Drops**: Adjust drop threshold or improve decode performance
5. **Platform-Specific Issues**: Verify timing API accuracy

### Debug Output

```cpp
// Enable debug output for troubleshooting
void AVSynchronizer::enable_debug_output(bool enabled) {
    debug_output_enabled = enabled;

    if (enabled) {
        print_line("AV Sync Debug Output Enabled");
        print_line("Filter Parameters: cutoff=%.3f, beta=%.1f",
                  av_sync_filter->get_min_cutoff(), av_sync_filter->get_beta());
        print_line("Sync Threshold: %.1f ms", sync_threshold * 1000.0);
    }
}
```

This implementation guide provides a complete foundation for integrating OneEuroFilter-based audio-video synchronization into the VK Video module, based on the proven approach from the rhythm game conductor.
