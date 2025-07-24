# OneEuroFilter Implementation

## Overview

The OneEuroFilter is a simple yet effective algorithm for smoothing noisy signals while maintaining low latency. Originally developed for human-computer interaction, it has proven highly effective for audio-video synchronization in the VK Video module.

## Mathematical Foundation

### Core Algorithm

The OneEuroFilter combines two low-pass filters to achieve optimal smoothing:

1. **Position Filter**: Smooths the main signal
2. **Velocity Filter**: Smooths the signal's rate of change

```
α(fc) = 1 / (1 + τ/Te)
where:
- τ = 1 / (2π × fc)  (time constant)
- Te = 1 / rate      (sampling period)
- fc = cutoff frequency
```

### Adaptive Cutoff Calculation

```
fc = fcmin + β × |dx|
where:
- fcmin = minimum cutoff frequency
- β = speed coefficient
- dx = filtered velocity
```

## GDScript Implementation

Based on the rhythm game implementation in `modules/vk_video/thirdparty/rhythm_game/globals/one_euro_filter.gd`:

```gdscript
class_name OneEuroFilter

var min_cutoff: float
var beta: float
var d_cutoff: float
var x_filter: LowPassFilter
var dx_filter: LowPassFilter

func _init(args: Variant) -> void:
    min_cutoff = args.cutoff
    beta = args.beta
    d_cutoff = args.cutoff
    x_filter = LowPassFilter.new()
    dx_filter = LowPassFilter.new()

func alpha(rate: float, cutoff: float) -> float:
    var tau: float = 1.0 / (2 * PI * cutoff)
    var te: float = 1.0 / rate
    return 1.0 / (1.0 + tau/te)

func filter(value: float, delta: float) -> float:
    var rate: float = 1.0 / delta
    var dx: float = (value - x_filter.last_value) * rate

    var edx: float = dx_filter.filter(dx, alpha(rate, d_cutoff))
    var cutoff: float = min_cutoff + beta * abs(edx)
    return x_filter.filter(value, alpha(rate, cutoff))

class LowPassFilter:
    var last_value: float

    func _init() -> void:
        last_value = 0

    func filter(value: float, alpha: float) -> float:
        var result := alpha * value + (1 - alpha) * last_value
        last_value = result
        return result
```

## C++ Implementation Template

For integration into the VK Video module:

```cpp
class OneEuroFilter {
private:
    struct LowPassFilter {
        double last_value = 0.0;

        double filter(double value, double alpha) {
            double result = alpha * value + (1.0 - alpha) * last_value;
            last_value = result;
            return result;
        }
    };

    double min_cutoff;
    double beta;
    double d_cutoff;
    LowPassFilter x_filter;
    LowPassFilter dx_filter;

    double calculate_alpha(double rate, double cutoff) {
        double tau = 1.0 / (2.0 * Math_PI * cutoff);
        double te = 1.0 / rate;
        return 1.0 / (1.0 + tau / te);
    }

public:
    OneEuroFilter(double p_min_cutoff, double p_beta)
        : min_cutoff(p_min_cutoff), beta(p_beta), d_cutoff(p_min_cutoff) {}

    double filter(double value, double delta_time) {
        double rate = 1.0 / delta_time;
        double dx = (value - x_filter.last_value) * rate;

        double edx = dx_filter.filter(dx, calculate_alpha(rate, d_cutoff));
        double cutoff = min_cutoff + beta * Math::abs(edx);

        return x_filter.filter(value, calculate_alpha(rate, cutoff));
    }

    void reset() {
        x_filter.last_value = 0.0;
        dx_filter.last_value = 0.0;
    }
};
```

## Parameter Configuration

### Key Parameters

#### min_cutoff (Minimum Cutoff Frequency)
- **Purpose**: Controls baseline jitter reduction
- **Range**: 0.1 - 10.0 Hz
- **Effect**: Lower values = more smoothing, higher latency
- **Typical Values**:
  - Audio sync: 0.1 - 1.0 Hz
  - Video timing: 0.5 - 2.0 Hz
  - Real-time applications: 1.0 - 5.0 Hz

#### beta (Speed Coefficient)
- **Purpose**: Controls lag reduction when signal changes rapidly
- **Range**: 0.0 - 50.0
- **Effect**: Higher values = less lag during rapid changes
- **Typical Values**:
  - Conservative: 1.0 - 5.0
  - Balanced: 5.0 - 15.0
  - Aggressive: 15.0 - 30.0

#### d_cutoff (Derivative Cutoff)
- **Purpose**: Smooths velocity calculations
- **Range**: Usually same as min_cutoff
- **Effect**: Affects responsiveness to signal changes

### Tuning Guidelines

#### For Audio-Video Synchronization
```gdscript
# Conservative (high quality, some latency)
var filter = OneEuroFilter.new({
    "cutoff": 0.1,
    "beta": 5.0
})

# Balanced (good quality, moderate latency)
var filter = OneEuroFilter.new({
    "cutoff": 0.5,
    "beta": 10.0
})

# Responsive (lower quality, minimal latency)
var filter = OneEuroFilter.new({
    "cutoff": 1.0,
    "beta": 20.0
})
```

#### Rhythm Game Settings (from conductor.gd)
```gdscript
@export var allowed_jitter: float = 0.1    # min_cutoff
@export var lag_reduction: float = 5       # beta
```

## Usage Patterns

### Basic Filtering
```gdscript
var filter = OneEuroFilter.new({"cutoff": 0.1, "beta": 5.0})

func _process(delta):
    var raw_value = get_noisy_signal()
    var filtered_value = filter.filter(raw_value, delta)
    use_filtered_value(filtered_value)
```

### Audio-Video Sync Application
```gdscript
var av_sync_filter = OneEuroFilter.new({"cutoff": 0.1, "beta": 5.0})
var audio_clock = 0.0
var video_clock = 0.0

func _physics_process(delta):
    var av_delta = video_clock - audio_clock
    var filtered_delta = av_sync_filter.filter(av_delta, delta)
    var corrected_video_time = audio_clock + filtered_delta
```

### Adaptive Parameter Adjustment
```gdscript
func adjust_filter_parameters(sync_quality: float):
    if sync_quality < 0.8:
        # Increase smoothing for poor sync
        filter.min_cutoff *= 0.8
        filter.beta *= 1.2
    elif sync_quality > 0.95:
        # Reduce latency for good sync
        filter.min_cutoff *= 1.1
        filter.beta *= 0.9
```

## Performance Characteristics

### Computational Complexity
- **Time Complexity**: O(1) per filter operation
- **Memory Usage**: Minimal (2 float values per filter)
- **CPU Impact**: Very low, suitable for real-time applications

### Latency Analysis
- **Group Delay**: Approximately 1/(2π × effective_cutoff)
- **Typical Latency**: 10-50ms depending on parameters
- **Trade-off**: Lower cutoff = more smoothing but higher latency

### Stability
- **Numerical Stability**: Excellent for typical parameter ranges
- **Edge Cases**: Handle division by zero in delta_time
- **Reset Behavior**: Clean startup after reset() call

## Integration Considerations

### Initialization
```cpp
// Initialize with appropriate parameters for use case
OneEuroFilter timing_filter(0.1, 5.0);  // Conservative settings
OneEuroFilter position_filter(1.0, 15.0); // Responsive settings
```

### Error Handling
```cpp
double filter_with_validation(double value, double delta) {
    if (delta <= 0.0) {
        return value;  // Skip filtering for invalid delta
    }
    return filter.filter(value, delta);
}
```

### Thread Safety
The OneEuroFilter is not inherently thread-safe. For multi-threaded applications:
- Use separate filter instances per thread
- Or protect with appropriate synchronization primitives

## Testing and Validation

### Unit Tests
```gdscript
func test_filter_stability():
    var filter = OneEuroFilter.new({"cutoff": 1.0, "beta": 5.0})
    var constant_input = 10.0
    var delta = 1.0/60.0  # 60 FPS

    # After sufficient iterations, output should converge to input
    for i in range(300):  # 5 seconds at 60 FPS
        var output = filter.filter(constant_input, delta)

    assert(abs(output - constant_input) < 0.01)
```

### Performance Benchmarks
```gdscript
func benchmark_filter_performance():
    var filter = OneEuroFilter.new({"cutoff": 0.1, "beta": 5.0})
    var start_time = Time.get_ticks_usec()

    for i in range(10000):
        filter.filter(randf() * 100.0, 1.0/60.0)

    var end_time = Time.get_ticks_usec()
    print("10k operations took: ", (end_time - start_time), " microseconds")
```

## References

- [Original 1€ Filter Paper](https://gery.casiez.net/1euro/) - Casiez, G., Roussel, N. and Vogel, D.
- [Interactive Demo](https://gery.casiez.net/1euro/InteractiveDemo/) - Parameter visualization
- [Godot XR Kit](https://github.com/patrykkalinowski/godot-xr-kit) - GDScript implementation source
