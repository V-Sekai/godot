# Audio-Video Synchronization Documentation

## Overview

This directory contains detailed documentation for audio-video synchronization in the VK Video module, with a focus on the OneEuroFilter implementation derived from the rhythm game example.

**Status Update (July 2025)**: Following the recent FFmpeg integration merge (commit 89f1c70), the VK Video module now operates alongside enhanced video importing capabilities, providing a complete pipeline from import to hardware-accelerated playback.

## Quick Start

The VK Video module uses a proven OneEuroFilter-based approach for audio-video synchronization, originally implemented in the rhythm game conductor system. This approach provides:

- **Jitter Reduction**: Smooths out timing inconsistencies between audio and video streams
- **Low Latency**: Maintains responsive synchronization with minimal delay
- **Adaptive Filtering**: Automatically adjusts to different playback conditions
- **Proven Reliability**: Based on real-world rhythm game timing requirements
- **FFmpeg Compatibility**: Works seamlessly with content imported via the new FFmpeg pipeline

## Document Index

### Core Implementation
- **[OneEuroFilter](one_euro_filter.md)** - Mathematical foundation and implementation details
- **[Conductor Implementation](conductor_implementation.md)** - Real-world usage example from rhythm game
- **[Timing Algorithms](timing_algorithms.md)** - Core timing calculation methods

### Synchronization Strategies
- **[Synchronization Strategies](synchronization_strategies.md)** - Different approaches and trade-offs
- **[Performance Tuning](performance_tuning.md)** - Parameter optimization and monitoring

### Implementation Guide
- **[Implementation Guide](implementation_guide.md)** - Step-by-step integration instructions


## Key Concepts

### OneEuroFilter Basics
The OneEuroFilter is a simple yet effective algorithm for smoothing noisy signals while maintaining low latency. It uses two main parameters:
- **Cutoff Frequency (β)**: Controls jitter reduction (lower = less jitter)
- **Beta (β)**: Controls lag reduction (higher = less lag)

### Audio-Video Sync Challenges
- **Clock Drift**: System clocks and audio clocks can drift apart over time
- **Jitter**: Frame timing can vary due to system load and hardware variations
- **Latency**: Processing delays must be compensated for accurate synchronization
- **Platform Differences**: Different platforms have varying audio/video pipeline characteristics

### Conductor Pattern
The rhythm game conductor demonstrates a sophisticated approach:
1. **Dual Clock System**: Maintains both audio-based and system-based time
2. **Delta Filtering**: Applies OneEuroFilter to the difference between clocks
3. **Adaptive Compensation**: Adjusts for output latency and processing delays
4. **Robust Error Handling**: Handles edge cases like audio underruns

## Integration with VK Video

The OneEuroFilter approach integrates with the VK Video module at several levels:

### VideoStreamPlaybackAV1
```cpp
class VideoStreamPlaybackAV1 {
    OneEuroFilter timing_filter;
    double audio_clock = 0.0;
    double video_clock = 0.0;
    double filtered_av_delta = 0.0;
};
```

### Frame Presentation
```cpp
bool should_present_frame(double current_time) {
    double raw_delay = video_pts - audio_clock;
    double filtered_delay = timing_filter.filter(raw_delay, delta_time);
    return filtered_delay <= presentation_threshold;
}
```

### FFmpeg Integration Benefits
The recent FFmpeg integration provides additional opportunities for synchronization optimization:
- **Metadata Extraction**: Frame rate and timing information for parameter tuning
- **Quality Settings**: Import-time quality controls that affect sync requirements
- **Format Support**: Unified sync approach across MP4, WebM, AV1, and other formats
- **Cross-module Compatibility**: Consistent timing behavior between Theora and VK Video modules

## Getting Started

1. **Read [OneEuroFilter](one_euro_filter.md)** to understand the mathematical foundation
2. **Study [Conductor Implementation](conductor_implementation.md)** for a working example
3. **Review [Implementation Guide](implementation_guide.md)** for integration steps
4. **Consult [Performance Tuning](performance_tuning.md)** for optimization

## References

- [1€ Filter Paper](https://gery.casiez.net/1euro/) - Original research and mathematical foundation
- [Godot XR Kit Implementation](https://github.com/patrykkalinowski/godot-xr-kit) - Source of the GDScript implementation
- Rhythm Game Conductor - Real-world timing system demonstrating the approach
- [FFmpeg Integration](../../theora/editor/resource_importer_video.cpp) - Recent video importing enhancements
