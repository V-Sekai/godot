# Performance Tuning

## Overview

This document provides guidelines for optimizing OneEuroFilter parameters and monitoring synchronization performance in the VK Video module. Proper tuning ensures optimal audio-video sync quality while maintaining system performance.

## Parameter Tuning Guidelines

### Understanding Filter Parameters

#### min_cutoff (Minimum Cutoff Frequency)

-   **Physical Meaning**: Baseline smoothing strength
-   **Units**: Hertz (Hz)
-   **Effect**: Lower values = more smoothing, higher latency
-   **Tuning Range**: 0.01 - 10.0 Hz

```cpp
// Parameter effects visualization
void demonstrate_cutoff_effects() {
    OneEuroFilter low_cutoff(0.05, 5.0);    // Heavy smoothing, ~320ms latency
    OneEuroFilter med_cutoff(0.5, 5.0);     // Moderate smoothing, ~32ms latency
    OneEuroFilter high_cutoff(2.0, 5.0);    // Light smoothing, ~8ms latency
}
```

#### beta (Speed Coefficient)

-   **Physical Meaning**: Responsiveness to rapid changes
-   **Units**: Dimensionless
-   **Effect**: Higher values = less lag during rapid signal changes
-   **Tuning Range**: 0.0 - 50.0

```cpp
// Beta parameter comparison
void demonstrate_beta_effects() {
    OneEuroFilter low_beta(0.1, 1.0);      // Slow response to changes
    OneEuroFilter med_beta(0.1, 10.0);     // Balanced response
    OneEuroFilter high_beta(0.1, 30.0);    // Fast response to changes
}
```

### Systematic Parameter Tuning

#### Step 1: Baseline Measurement

```cpp
struct BaselineMetrics {
    double avg_sync_error = 0.0;
    double max_sync_error = 0.0;
    double sync_stability = 0.0;
    double filter_latency = 0.0;
    int measurement_count = 0;
};

BaselineMetrics measure_baseline(OneEuroFilter& filter,
                                const std::vector<double>& test_signal) {
    BaselineMetrics metrics;
    double delta_time = 1.0 / 60.0;  // 60 FPS

    for (size_t i = 1; i < test_signal.size(); i++) {
        double filtered = filter.filter(test_signal[i], delta_time);
        double error = abs(filtered - test_signal[i]);

        metrics.avg_sync_error += error;
        metrics.max_sync_error = max(metrics.max_sync_error, error);
        metrics.measurement_count++;
    }

    metrics.avg_sync_error /= metrics.measurement_count;
    metrics.filter_latency = estimate_group_delay(filter);

    return metrics;
}
```

#### Step 2: Parameter Sweep

```cpp
struct TuningResult {
    double min_cutoff;
    double beta;
    double quality_score;
    BaselineMetrics metrics;
};

std::vector<TuningResult> parameter_sweep(const std::vector<double>& test_signal) {
    std::vector<TuningResult> results;

    // Cutoff frequency sweep
    for (double cutoff = 0.05; cutoff <= 2.0; cutoff *= 1.5) {
        // Beta parameter sweep
        for (double beta = 1.0; beta <= 30.0; beta *= 1.5) {
            OneEuroFilter filter(cutoff, beta);
            BaselineMetrics metrics = measure_baseline(filter, test_signal);

            // Calculate composite quality score
            double quality_score = calculate_quality_score(metrics);

            results.push_back({cutoff, beta, quality_score, metrics});
        }
    }

    return results;
}

double calculate_quality_score(const BaselineMetrics& metrics) {
    // Weighted combination of metrics
    double error_score = 1.0 / (1.0 + metrics.avg_sync_error * 1000.0);  // Lower error = higher score
    double latency_score = 1.0 / (1.0 + metrics.filter_latency * 10.0);  // Lower latency = higher score
    double stability_score = metrics.sync_stability;

    return 0.5 * error_score + 0.3 * latency_score + 0.2 * stability_score;
}
```

#### Step 3: Validation Testing

```cpp
bool validate_parameters(double min_cutoff, double beta,
                        const std::vector<TestScenario>& scenarios) {
    OneEuroFilter filter(min_cutoff, beta);

    for (const auto& scenario : scenarios) {
        BaselineMetrics metrics = test_scenario(filter, scenario);

        // Check if parameters meet requirements for this scenario
        if (metrics.avg_sync_error > scenario.max_allowed_error ||
            metrics.filter_latency > scenario.max_allowed_latency) {
            return false;
        }
    }

    return true;
}
```

### Application-Specific Tuning

#### Standard Video Playback

```cpp
struct VideoPlaybackTuning {
    static constexpr double min_cutoff = 0.1;   // 100ms smoothing window
    static constexpr double beta = 5.0;         // Moderate responsiveness
    static constexpr double max_latency = 50.0; // 50ms acceptable latency

    static OneEuroFilter create_filter() {
        return OneEuroFilter(min_cutoff, beta);
    }

    static bool validate_performance(const BaselineMetrics& metrics) {
        return metrics.avg_sync_error < 10.0 &&  // <10ms average error
               metrics.filter_latency < max_latency;
    }
};
```

#### Music Video / Rhythm Games

```cpp
struct MusicVideoTuning {
    static constexpr double min_cutoff = 0.05;  // More smoothing for stability
    static constexpr double beta = 3.0;         // Conservative responsiveness
    static constexpr double max_latency = 30.0; // Tighter latency requirement

    static OneEuroFilter create_filter() {
        return OneEuroFilter(min_cutoff, beta);
    }

    static bool validate_performance(const BaselineMetrics& metrics) {
        return metrics.avg_sync_error < 5.0 &&   // <5ms average error
               metrics.filter_latency < max_latency;
    }
};
```

#### Interactive/Gaming Applications

```cpp
struct InteractiveTuning {
    static constexpr double min_cutoff = 0.8;   // Minimal smoothing
    static constexpr double beta = 20.0;        // High responsiveness
    static constexpr double max_latency = 15.0; // Very tight latency

    static OneEuroFilter create_filter() {
        return OneEuroFilter(min_cutoff, beta);
    }

    static bool validate_performance(const BaselineMetrics& metrics) {
        return metrics.filter_latency < max_latency;  // Latency is primary concern
    }
};
```

## Performance Monitoring

### Real-Time Metrics Collection

```cpp
class SyncPerformanceMonitor {
private:
    struct PerformanceMetrics {
        // Timing metrics
        double avg_sync_error = 0.0;
        double max_sync_error = 0.0;
        double sync_error_variance = 0.0;

        // Quality metrics
        double sync_quality = 1.0;
        int frames_dropped = 0;
        int frames_duplicated = 0;

        // Performance metrics
        double filter_cpu_time = 0.0;
        double update_frequency = 60.0;

        // History for trend analysis
        std::deque<double> error_history;
        std::deque<double> latency_history;
    };

    PerformanceMetrics current_metrics;
    std::chrono::high_resolution_clock::time_point last_update;

public:
    void update_metrics(double sync_error, double filter_latency, bool frame_dropped) {
        auto now = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration<double>(now - last_update).count();

        // Update timing metrics
        update_sync_error_metrics(sync_error);
        update_latency_metrics(filter_latency);

        // Update quality metrics
        if (frame_dropped) {
            current_metrics.frames_dropped++;
        }

        // Update performance metrics
        current_metrics.update_frequency = 1.0 / delta;

        last_update = now;
    }

    Dictionary get_performance_report() const {
        Dictionary report;

        // Basic metrics
        report["avg_sync_error_ms"] = current_metrics.avg_sync_error * 1000.0;
        report["max_sync_error_ms"] = current_metrics.max_sync_error * 1000.0;
        report["sync_quality"] = current_metrics.sync_quality;

        // Performance metrics
        report["frames_dropped"] = current_metrics.frames_dropped;
        report["frames_duplicated"] = current_metrics.frames_duplicated;
        report["update_frequency"] = current_metrics.update_frequency;

        // Trend analysis
        report["error_trend"] = calculate_error_trend();
        report["stability_score"] = calculate_stability_score();

        return report;
    }

private:
    void update_sync_error_metrics(double error) {
        // Exponential moving average
        current_metrics.avg_sync_error =
            current_metrics.avg_sync_error * 0.95 + abs(error) * 0.05;

        // Track maximum
        current_metrics.max_sync_error =
            max(current_metrics.max_sync_error * 0.99, abs(error));

        // Update history
        current_metrics.error_history.push_back(error);
        if (current_metrics.error_history.size() > 300) {  // 5 seconds at 60fps
            current_metrics.error_history.pop_front();
        }

        // Calculate variance
        update_error_variance();
    }

    void update_error_variance() {
        if (current_metrics.error_history.size() < 10) return;

        double mean = 0.0;
        for (double error : current_metrics.error_history) {
            mean += abs(error);
        }
        mean /= current_metrics.error_history.size();

        double variance = 0.0;
        for (double error : current_metrics.error_history) {
            double diff = abs(error) - mean;
            variance += diff * diff;
        }
        variance /= current_metrics.error_history.size();

        current_metrics.sync_error_variance = variance;
    }

    double calculate_stability_score() const {
        if (current_metrics.sync_error_variance == 0.0) return 1.0;

        // Lower variance = higher stability
        return 1.0 / (1.0 + current_metrics.sync_error_variance * 1000.0);
    }
};
```

### Adaptive Parameter Adjustment

```cpp
class AdaptiveParameterTuner {
private:
    OneEuroFilter* target_filter;
    SyncPerformanceMonitor* monitor;

    // Tuning state
    double current_cutoff;
    double current_beta;
    double target_cutoff;
    double target_beta;

    // Adaptation parameters
    double adaptation_rate = 0.1;
    double quality_threshold_low = 0.7;
    double quality_threshold_high = 0.95;

public:
    AdaptiveParameterTuner(OneEuroFilter* filter, SyncPerformanceMonitor* perf_monitor)
        : target_filter(filter), monitor(perf_monitor) {
        current_cutoff = filter->get_min_cutoff();
        current_beta = filter->get_beta();
        target_cutoff = current_cutoff;
        target_beta = current_beta;
    }

    void update_adaptation() {
        Dictionary metrics = monitor->get_performance_report();
        double sync_quality = metrics["sync_quality"];
        double avg_error = metrics["avg_sync_error_ms"];

        // Determine target parameters based on performance
        if (sync_quality < quality_threshold_low) {
            // Poor sync - increase smoothing
            target_cutoff = current_cutoff * 0.8;
            target_beta = current_beta * 1.2;
        } else if (sync_quality > quality_threshold_high && avg_error < 5.0) {
            // Excellent sync - reduce latency
            target_cutoff = current_cutoff * 1.1;
            target_beta = current_beta * 0.9;
        }

        // Clamp to reasonable ranges
        target_cutoff = clamp(target_cutoff, 0.01, 5.0);
        target_beta = clamp(target_beta, 1.0, 50.0);

        // Smooth parameter transitions
        apply_parameter_changes();
    }

private:
    void apply_parameter_changes() {
        // Gradual parameter adjustment to avoid artifacts
        current_cutoff = lerp(current_cutoff, target_cutoff, adaptation_rate);
        current_beta = lerp(current_beta, target_beta, adaptation_rate);

        // Update filter parameters
        target_filter->update_parameters(current_cutoff, current_beta);
    }

    double lerp(double a, double b, double t) {
        return a + t * (b - a);
    }

    double clamp(double value, double min_val, double max_val) {
        return max(min_val, min(max_val, value));
    }
};
```

## Debugging and Diagnostics

### Sync Quality Visualization

```cpp
class SyncQualityVisualizer {
private:
    std::vector<double> sync_error_history;
    std::vector<double> filter_output_history;
    std::vector<double> raw_input_history;

public:
    void record_sample(double raw_input, double filtered_output, double sync_error) {
        raw_input_history.push_back(raw_input);
        filter_output_history.push_back(filtered_output);
        sync_error_history.push_back(sync_error);

        // Keep last 5 seconds of data at 60fps
        const size_t max_samples = 300;
        if (raw_input_history.size() > max_samples) {
            raw_input_history.erase(raw_input_history.begin());
            filter_output_history.erase(filter_output_history.begin());
            sync_error_history.erase(sync_error_history.begin());
        }
    }

    void generate_debug_output() {
        print_line("=== Sync Quality Report ===");
        print_line(vformat("Samples: %d", sync_error_history.size()));

        if (!sync_error_history.empty()) {
            double avg_error = calculate_average(sync_error_history);
            double max_error = *std::max_element(sync_error_history.begin(), sync_error_history.end());
            double min_error = *std::min_element(sync_error_history.begin(), sync_error_history.end());

            print_line(vformat("Average Error: %.2f ms", avg_error * 1000.0));
            print_line(vformat("Max Error: %.2f ms", max_error * 1000.0));
            print_line(vformat("Min Error: %.2f ms", min_error * 1000.0));

            // Calculate filter effectiveness
            double raw_variance = calculate_variance(raw_input_history);
            double filtered_variance = calculate_variance(filter_output_history);
            double noise_reduction = (raw_variance - filtered_variance) / raw_variance * 100.0;

            print_line(vformat("Noise Reduction: %.1f%%", noise_reduction));
        }
    }

private:
    double calculate_average(const std::vector<double>& data) {
        double sum = 0.0;
        for (double value : data) sum += value;
        return sum / data.size();
    }

    double calculate_variance(const std::vector<double>& data) {
        double mean = calculate_average(data);
        double variance = 0.0;
        for (double value : data) {
            double diff = value - mean;
            variance += diff * diff;
        }
        return variance / data.size();
    }
};
```

### Performance Profiling

```cpp
class FilterPerformanceProfiler {
private:
    struct ProfileData {
        uint64_t total_calls = 0;
        double total_time_us = 0.0;
        double min_time_us = std::numeric_limits<double>::max();
        double max_time_us = 0.0;
    };

    ProfileData profile_data;

public:
    class ScopedTimer {
    private:
        FilterPerformanceProfiler* profiler;
        std::chrono::high_resolution_clock::time_point start_time;

    public:
        ScopedTimer(FilterPerformanceProfiler* p) : profiler(p) {
            start_time = std::chrono::high_resolution_clock::now();
        }

        ~ScopedTimer() {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::micro>(end_time - start_time);
            profiler->record_timing(duration.count());
        }
    };

    ScopedTimer start_timing() {
        return ScopedTimer(this);
    }

    void record_timing(double time_us) {
        profile_data.total_calls++;
        profile_data.total_time_us += time_us;
        profile_data.min_time_us = min(profile_data.min_time_us, time_us);
        profile_data.max_time_us = max(profile_data.max_time_us, time_us);
    }

    Dictionary get_profile_report() {
        Dictionary report;

        if (profile_data.total_calls > 0) {
            double avg_time = profile_data.total_time_us / profile_data.total_calls;

            report["total_calls"] = (int64_t)profile_data.total_calls;
            report["avg_time_us"] = avg_time;
            report["min_time_us"] = profile_data.min_time_us;
            report["max_time_us"] = profile_data.max_time_us;
            report["total_time_ms"] = profile_data.total_time_us / 1000.0;

            // Calculate performance metrics
            double calls_per_second = profile_data.total_calls / (profile_data.total_time_us / 1000000.0);
            report["calls_per_second"] = calls_per_second;
        }

        return report;
    }

    void reset_profile() {
        profile_data = ProfileData{};
    }
};

// Usage example
FilterPerformanceProfiler profiler;

double filtered_value;
{
    auto timer = profiler.start_timing();
    filtered_value = filter.filter(input_value, delta_time);
}
```

## Platform-Specific Optimizations

### Mobile Device Optimizations

```cpp
class MobilePerformanceOptimizer {
private:
    bool low_power_mode = false;
    double cpu_usage_threshold = 80.0;
    double battery_threshold = 20.0;

public:
    void update_power_state() {
        double cpu_usage = get_cpu_usage_percentage();
        double battery_level = get_battery_percentage();

        bool should_enable_low_power =
            cpu_usage > cpu_usage_threshold ||
            battery_level < battery_threshold;

        if (should_enable_low_power != low_power_mode) {
            low_power_mode = should_enable_low_power;
            apply_power_optimizations();
        }
    }

private:
    void apply_power_optimizations() {
        if (low_power_mode) {
            // Reduce filter update frequency
            set_filter_update_rate(30.0);  // 30 Hz instead of 60 Hz

            // Use more aggressive smoothing to reduce CPU load
            adjust_filter_parameters(0.05, 3.0);

            // Increase frame drop tolerance
            increase_presentation_threshold(1.5);
        } else {
            // Restore normal operation
            set_filter_update_rate(60.0);
            restore_normal_filter_parameters();
            restore_normal_presentation_threshold();
        }
    }
};
```

### High-Performance Desktop Optimizations

```cpp
class DesktopPerformanceOptimizer {
private:
    double display_refresh_rate = 60.0;
    bool variable_refresh_rate = false;

public:
    void optimize_for_display() {
        display_refresh_rate = get_display_refresh_rate();
        variable_refresh_rate = supports_variable_refresh_rate();

        if (display_refresh_rate > 100.0) {
            // High refresh rate optimizations
            optimize_for_high_refresh_rate();
        }

        if (variable_refresh_rate) {
            // VRR optimizations
            optimize_for_variable_refresh_rate();
        }
    }

private:
    void optimize_for_high_refresh_rate() {
        // More responsive filtering for high refresh rates
        double responsive_cutoff = 2.0;
        double responsive_beta = 20.0;
        adjust_filter_parameters(responsive_cutoff, responsive_beta);

        // Tighter presentation thresholds
        double frame_time = 1.0 / display_refresh_rate;
        set_presentation_threshold(frame_time * 0.3);
    }

    void optimize_for_variable_refresh_rate() {
        // Adaptive filtering based on current refresh rate
        enable_adaptive_refresh_rate_filtering();

        // Looser frame timing for VRR displays
        enable_vrr_frame_timing();
    }
};
```

These performance tuning guidelines provide a comprehensive framework for optimizing OneEuroFilter-based audio-video synchronization across different platforms and use cases.
