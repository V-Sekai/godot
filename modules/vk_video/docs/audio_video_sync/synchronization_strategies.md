# Synchronization Strategies

## Overview

This document outlines different approaches to audio-video synchronization, their trade-offs, and implementation strategies using the OneEuroFilter framework. Each strategy is suited for different use cases and performance requirements.

## Master Clock Strategies

### Audio Master Clock (Recommended)

The audio master approach treats audio timing as authoritative, with video synchronized to match.

#### Advantages
- **Perceptually Optimal**: Audio timing discontinuities are more noticeable than video
- **Stable Reference**: Audio hardware provides consistent timing
- **Proven Approach**: Used successfully in the rhythm game conductor
- **Natural Buffering**: Audio systems typically have built-in buffering

#### Implementation
```cpp
class AudioMasterSync {
private:
    OneEuroFilter av_sync_filter;
    double audio_clock = 0.0;
    double video_pts = 0.0;
    double filtered_av_delta = 0.0;

public:
    void update_synchronization(double delta_time) {
        // Audio is the master clock
        audio_clock = get_audio_playback_position();

        // Filter the audio-video delta
        double av_delta = video_pts - audio_clock;
        filtered_av_delta = av_sync_filter.filter(av_delta, delta_time);

        // Corrected video time follows audio
        corrected_video_time = audio_clock + filtered_av_delta;
    }

    bool should_present_frame(const FrameInfo& frame) {
        double frame_time = frame.pts;
        double time_diff = frame_time - corrected_video_time;
        return abs(time_diff) <= presentation_threshold;
    }
};
```

#### Use Cases
- Standard video playback
- Music videos and rhythm games
- Applications where audio quality is critical
- Real-time streaming

### Video Master Clock

Video timing drives synchronization, with audio adjusted to match.

#### Advantages
- **Visual Smoothness**: Ensures consistent frame presentation
- **Predictable Timing**: Video frame rates are typically constant
- **Lower Video Latency**: Direct video timing without filtering delays

#### Disadvantages
- **Audio Artifacts**: May cause audio dropouts or pitch changes
- **Complex Audio Adjustment**: Requires sophisticated audio resampling
- **Perceptual Issues**: Audio timing errors are more noticeable

#### Implementation
```cpp
class VideoMasterSync {
private:
    OneEuroFilter video_smoothing_filter;
    double video_clock = 0.0;
    double audio_pts = 0.0;
    AudioResampler resampler;

public:
    void update_synchronization(double delta_time) {
        // Video is the master clock
        video_clock = get_smoothed_video_time(delta_time);

        // Adjust audio to match video
        double av_delta = audio_pts - video_clock;
        if (abs(av_delta) > audio_adjustment_threshold) {
            resampler.adjust_rate(calculate_correction_rate(av_delta));
        }
    }

private:
    double get_smoothed_video_time(double delta_time) {
        double raw_video_time = current_frame.pts;
        return video_smoothing_filter.filter(raw_video_time, delta_time);
    }
};
```

#### Use Cases
- Video editing applications
- Frame-accurate playback requirements
- Applications where visual smoothness is paramount

### External Master Clock

An external timing source (system clock, network time, etc.) drives synchronization.

#### Advantages
- **Consistent Reference**: Independent of media stream variations
- **Multi-Stream Sync**: Can synchronize multiple streams to common reference
- **Network Synchronization**: Suitable for distributed playback

#### Implementation
```cpp
class ExternalMasterSync {
private:
    OneEuroFilter audio_sync_filter;
    OneEuroFilter video_sync_filter;
    double external_clock = 0.0;
    double playback_start_time = 0.0;

public:
    void update_synchronization(double delta_time) {
        // External clock is the master
        external_clock = get_system_time() - playback_start_time;

        // Sync both audio and video to external clock
        double audio_delta = audio_pts - external_clock;
        double video_delta = video_pts - external_clock;

        filtered_audio_delta = audio_sync_filter.filter(audio_delta, delta_time);
        filtered_video_delta = video_sync_filter.filter(video_delta, delta_time);
    }
};
```

#### Use Cases
- Multi-room audio/video systems
- Network streaming with multiple clients
- Professional broadcast applications

## Adaptive Synchronization

### Quality-Based Strategy Selection

Automatically choose the best synchronization strategy based on current conditions.

```cpp
class AdaptiveSyncManager {
private:
    AudioMasterSync audio_master;
    VideoMasterSync video_master;
    ExternalMasterSync external_master;

    SyncStrategy current_strategy = AUDIO_MASTER;
    SyncMetrics metrics;

public:
    void update_strategy() {
        if (metrics.audio_quality < 0.7 && metrics.video_quality > 0.9) {
            switch_to_strategy(VIDEO_MASTER);
        } else if (metrics.network_jitter > high_jitter_threshold) {
            switch_to_strategy(EXTERNAL_MASTER);
        } else {
            switch_to_strategy(AUDIO_MASTER);  // Default
        }
    }

private:
    void switch_to_strategy(SyncStrategy new_strategy) {
        if (new_strategy != current_strategy) {
            // Reset filters when switching strategies
            reset_all_filters();
            current_strategy = new_strategy;
        }
    }
};
```

### Dynamic Parameter Adjustment

Adjust OneEuroFilter parameters based on synchronization quality.

```cpp
class DynamicParameterAdjuster {
private:
    struct ParameterSet {
        double min_cutoff;
        double beta;
        double presentation_threshold;
    };

    ParameterSet conservative = {0.1, 5.0, 40.0};   // High quality, some latency
    ParameterSet balanced = {0.3, 8.0, 25.0};       // Balanced
    ParameterSet responsive = {0.8, 15.0, 15.0};    // Low latency, less smoothing

public:
    void adjust_parameters(OneEuroFilter& filter, const SyncMetrics& metrics) {
        ParameterSet target;

        if (metrics.sync_quality > 0.95) {
            target = responsive;  // Excellent sync - reduce latency
        } else if (metrics.sync_quality < 0.8) {
            target = conservative;  // Poor sync - increase smoothing
        } else {
            target = balanced;  // Good sync - balanced approach
        }

        // Smooth parameter transitions
        smooth_parameter_transition(filter, target);
    }

private:
    void smooth_parameter_transition(OneEuroFilter& filter, const ParameterSet& target) {
        double current_cutoff = filter.get_min_cutoff();
        double current_beta = filter.get_beta();

        // Gradual adjustment to avoid artifacts
        double new_cutoff = current_cutoff * 0.9 + target.min_cutoff * 0.1;
        double new_beta = current_beta * 0.9 + target.beta * 0.1;

        filter.update_parameters(new_cutoff, new_beta);
    }
};
```

## Frame Management Strategies

### Conservative Frame Management

Prioritizes quality over real-time performance.

```cpp
class ConservativeFrameManager {
public:
    FrameAction determine_action(const FrameInfo& frame, double current_time) {
        double time_diff = frame.pts - current_time;

        // Wide tolerance for frame presentation
        if (abs(time_diff) <= conservative_threshold) {
            return PRESENT_FRAME;
        }

        // Rarely drop frames
        if (time_diff < -drop_threshold * 2.0) {
            return DROP_FRAME;
        }

        // Wait for proper timing
        return WAIT_FOR_TIME;
    }

private:
    static constexpr double conservative_threshold = 50.0;  // 50ms tolerance
    static constexpr double drop_threshold = 100.0;        // 100ms before dropping
};
```

### Aggressive Frame Management

Prioritizes real-time performance over quality.

```cpp
class AggressiveFrameManager {
public:
    FrameAction determine_action(const FrameInfo& frame, double current_time) {
        double time_diff = frame.pts - current_time;

        // Tight tolerance for frame presentation
        if (abs(time_diff) <= aggressive_threshold) {
            return PRESENT_FRAME;
        }

        // Quick to drop late frames
        if (time_diff < -drop_threshold) {
            return DROP_FRAME;
        }

        // Duplicate frames when ahead
        if (time_diff > duplicate_threshold) {
            return DUPLICATE_FRAME;
        }

        return WAIT_FOR_TIME;
    }

private:
    static constexpr double aggressive_threshold = 15.0;   // 15ms tolerance
    static constexpr double drop_threshold = 30.0;        // 30ms before dropping
    static constexpr double duplicate_threshold = 20.0;   // 20ms before duplicating
};
```

### Adaptive Frame Management

Adjusts frame management based on system performance and sync quality.

```cpp
class AdaptiveFrameManager {
private:
    ConservativeFrameManager conservative;
    AggressiveFrameManager aggressive;

    double current_threshold = 25.0;  // Start balanced
    double target_threshold = 25.0;

public:
    FrameAction determine_action(const FrameInfo& frame, double current_time,
                               const SyncMetrics& metrics) {
        update_thresholds(metrics);

        double time_diff = frame.pts - current_time;

        if (abs(time_diff) <= current_threshold) {
            return PRESENT_FRAME;
        }

        // Adaptive drop threshold based on sync quality
        double drop_threshold = current_threshold * 2.0;
        if (metrics.sync_quality < 0.7) {
            drop_threshold *= 1.5;  // More lenient when sync is poor
        }

        if (time_diff < -drop_threshold) {
            return DROP_FRAME;
        }

        return WAIT_FOR_TIME;
    }

private:
    void update_thresholds(const SyncMetrics& metrics) {
        if (metrics.sync_quality > 0.95) {
            target_threshold = 15.0;  // Tighten for excellent sync
        } else if (metrics.sync_quality < 0.8) {
            target_threshold = 40.0;  // Loosen for poor sync
        } else {
            target_threshold = 25.0;  // Balanced
        }

        // Smooth threshold transitions
        current_threshold = current_threshold * 0.95 + target_threshold * 0.05;
    }
};
```

## Network Streaming Strategies

### Buffer-Based Synchronization

Maintains a buffer of frames to handle network jitter.

```cpp
class BufferedStreamSync {
private:
    std::queue<FrameInfo> frame_buffer;
    OneEuroFilter network_jitter_filter;
    double target_buffer_duration = 200.0;  // 200ms buffer
    double current_buffer_duration = 0.0;

public:
    void update_buffering(double delta_time) {
        current_buffer_duration = calculate_buffer_duration();

        // Filter buffer duration to smooth network variations
        double filtered_duration = network_jitter_filter.filter(
            current_buffer_duration, delta_time);

        // Adjust playback rate based on buffer level
        if (filtered_duration < target_buffer_duration * 0.8) {
            slow_down_playback();
        } else if (filtered_duration > target_buffer_duration * 1.2) {
            speed_up_playback();
        }
    }

private:
    void slow_down_playback() {
        // Slightly slow down to build buffer
        set_playback_rate(0.98);
    }

    void speed_up_playback() {
        // Slightly speed up to reduce buffer
        set_playback_rate(1.02);
    }
};
```

### Predictive Synchronization

Uses network timing patterns to predict and compensate for delays.

```cpp
class PredictiveStreamSync {
private:
    OneEuroFilter network_delay_filter;
    std::vector<double> delay_history;
    double predicted_delay = 0.0;

public:
    void update_prediction(double measured_delay, double delta_time) {
        // Filter network delay measurements
        double filtered_delay = network_delay_filter.filter(measured_delay, delta_time);

        // Update delay history
        delay_history.push_back(filtered_delay);
        if (delay_history.size() > 60) {  // Keep 1 second of history at 60fps
            delay_history.erase(delay_history.begin());
        }

        // Predict future delay based on trend
        predicted_delay = calculate_delay_trend();
    }

    double get_compensated_presentation_time(const FrameInfo& frame) {
        return frame.pts + predicted_delay;
    }

private:
    double calculate_delay_trend() {
        if (delay_history.size() < 10) {
            return delay_history.back();
        }

        // Simple linear regression for trend prediction
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        int n = delay_history.size();

        for (int i = 0; i < n; i++) {
            sum_x += i;
            sum_y += delay_history[i];
            sum_xy += i * delay_history[i];
            sum_x2 += i * i;
        }

        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        double intercept = (sum_y - slope * sum_x) / n;

        // Predict next value
        return slope * n + intercept;
    }
};
```

## Platform-Specific Strategies

### Mobile Device Optimization

Optimized for battery life and thermal constraints.

```cpp
class MobileOptimizedSync {
private:
    OneEuroFilter power_aware_filter;
    bool low_power_mode = false;
    double cpu_temperature = 0.0;

public:
    void update_power_awareness() {
        cpu_temperature = get_cpu_temperature();

        if (cpu_temperature > thermal_threshold || get_battery_level() < 0.2) {
            enable_low_power_mode();
        } else {
            disable_low_power_mode();
        }
    }

private:
    void enable_low_power_mode() {
        if (!low_power_mode) {
            // Reduce filter update frequency
            power_aware_filter.update_parameters(0.05, 3.0);  // More aggressive smoothing

            // Increase frame drop tolerance
            presentation_threshold *= 1.5;

            low_power_mode = true;
        }
    }

    void disable_low_power_mode() {
        if (low_power_mode) {
            // Restore normal filter parameters
            power_aware_filter.update_parameters(0.1, 5.0);

            // Restore normal thresholds
            presentation_threshold /= 1.5;

            low_power_mode = false;
        }
    }

    static constexpr double thermal_threshold = 70.0;  // Celsius
};
```

### Desktop High-Performance

Optimized for high refresh rate displays and low latency.

```cpp
class HighPerformanceSync {
private:
    OneEuroFilter high_freq_filter;
    double display_refresh_rate = 144.0;  // High refresh rate display

public:
    void initialize_for_display() {
        display_refresh_rate = get_display_refresh_rate();

        // Adjust filter parameters for high refresh rate
        double frame_time = 1.0 / display_refresh_rate;
        double responsive_cutoff = 2.0;  // More responsive for high refresh
        double responsive_beta = 20.0;   // Reduce lag aggressively

        high_freq_filter.update_parameters(responsive_cutoff, responsive_beta);

        // Tighter presentation threshold for high refresh
        presentation_threshold = frame_time * 0.3;
    }

    void update_for_variable_refresh_rate(double current_refresh_rate) {
        if (abs(current_refresh_rate - display_refresh_rate) > 1.0) {
            display_refresh_rate = current_refresh_rate;
            initialize_for_display();
        }
    }
};
```

## Strategy Selection Guidelines

### Use Case Matrix

| Use Case | Recommended Strategy | Filter Parameters | Notes |
|----------|---------------------|-------------------|-------|
| Standard Video | Audio Master | cutoff=0.1, beta=5.0 | Balanced quality/latency |
| Music Videos | Audio Master | cutoff=0.05, beta=3.0 | Prioritize audio sync |
| Gaming/Interactive | Video Master | cutoff=0.5, beta=15.0 | Low latency critical |
| Streaming | Adaptive | Dynamic | Handle network variations |
| Mobile | Power-Aware | Thermal-dependent | Battery optimization |
| High Refresh | High-Performance | cutoff=2.0, beta=20.0 | Responsive for 120Hz+ |

### Performance Considerations

```cpp
// Strategy selection based on system capabilities
SyncStrategy select_optimal_strategy() {
    SystemCapabilities caps = get_system_capabilities();

    if (caps.is_mobile_device && caps.battery_level < 0.3) {
        return POWER_OPTIMIZED;
    }

    if (caps.display_refresh_rate > 100.0) {
        return HIGH_PERFORMANCE;
    }

    if (caps.network_connection && caps.network_jitter > 50.0) {
        return BUFFERED_STREAMING;
    }

    return AUDIO_MASTER;  // Default for most cases
}
```

These synchronization strategies provide a comprehensive framework for implementing robust audio-video sync across different platforms and use cases using the OneEuroFilter approach.
