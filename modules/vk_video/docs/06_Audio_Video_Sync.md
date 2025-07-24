# Audio-Video Synchronization

## Brief Description
Timestamp-based audio-video synchronization system for hardware-decoded AV1 video with frame queue management, clock synchronization, and adaptive playback control.

## Core Synchronization System

### AudioVideoSynchronizer Class

```cpp
// Main audio-video synchronization controller
class AudioVideoSynchronizer : public RefCounted {
    GDCLASS(AudioVideoSynchronizer, RefCounted);

public:
    enum ClockSource {
        CLOCK_AUDIO,        // Audio master clock (default)
        CLOCK_VIDEO,        // Video master clock
        CLOCK_EXTERNAL      // External system clock
    };

    struct FrameInfo {
        RID texture;
        double pts = 0.0;           // Presentation timestamp
        double dts = 0.0;           // Decode timestamp
        uint64_t frame_number = 0;
        bool keyframe = false;
        double duration = 0.0;
    };

    struct AudioInfo {
        Vector<float> samples;
        double pts = 0.0;
        int sample_rate = 48000;
        int channels = 2;
        double duration = 0.0;
    };

private:
    // Clock management
    ClockSource clock_source = CLOCK_AUDIO;
    double master_clock = 0.0;
    double audio_clock = 0.0;
    double video_clock = 0.0;
    double system_clock_base = 0.0;
    
    // Frame queues
    Queue<FrameInfo> decoded_frames;
    Queue<AudioInfo> audio_packets;
    uint32_t max_video_queue_size = 5;
    uint32_t max_audio_queue_size = 10;
    
    // Current presentation
    FrameInfo current_frame;
    bool has_current_frame = false;
    double last_frame_time = 0.0;
    
    // Synchronization parameters
    double av_sync_threshold = 0.04;    // 40ms threshold
    double max_frame_drop_threshold = 0.1; // 100ms max drop
    double duplicate_frame_threshold = 0.02; // 20ms duplicate
    
    // Statistics
    uint64_t frames_displayed = 0;
    uint64_t frames_dropped = 0;
    uint64_t frames_duplicated = 0;
    double avg_frame_delay = 0.0;
    
    // Configuration
    double target_frame_rate = 30.0;
    bool adaptive_sync = true;
    bool drop_frames_enabled = true;

protected:
    static void _bind_methods();

public:
    AudioVideoSynchronizer();
    virtual ~AudioVideoSynchronizer();
    
    // Initialization
    void initialize(double p_frame_rate, ClockSource p_clock_source = CLOCK_AUDIO);
    void reset();
    
    // Frame management
    void queue_decoded_frame(RID p_texture, double p_pts, double p_dts = -1.0, bool p_keyframe = false);
    void queue_audio_packet(const Vector<float> &p_samples, double p_pts, int p_sample_rate, int p_channels);
    
    // Clock synchronization
    void update_master_clock(double p_delta);
    void set_audio_clock(double p_time);
    void set_video_clock(double p_time);
    double get_master_clock() const;
    
    // Frame presentation
    Ref<Texture2D> get_current_frame();
    bool should_present_frame(double p_current_time);
    void advance_frame();
    
    // Audio synchronization
    bool get_audio_samples(Vector<float> &r_samples, int p_requested_frames);
    double get_audio_delay() const;
    
    // Configuration
    void set_clock_source(ClockSource p_source);
    ClockSource get_clock_source() const;
    void set_sync_threshold(double p_threshold);
    void set_adaptive_sync(bool p_enabled);
    void set_frame_dropping(bool p_enabled);
    
    // Statistics
    Dictionary get_sync_statistics() const;
    void reset_statistics();

private:
    void _update_clocks(double p_delta);
    double _calculate_frame_delay();
    bool _should_drop_frame(const FrameInfo &p_frame);
    bool _should_duplicate_frame();
    void _adjust_playback_rate(double p_delay);
};
```

## Clock Management Implementation

### Master Clock Synchronization

```cpp
// Update master clock based on selected source
void AudioVideoSynchronizer::update_master_clock(double p_delta) {
    switch (clock_source) {
        case CLOCK_AUDIO:
            master_clock = audio_clock;
            break;
        case CLOCK_VIDEO:
            master_clock = video_clock;
            break;
        case CLOCK_EXTERNAL:
            master_clock += p_delta;
            break;
    }
    
    _update_clocks(p_delta);
}

// Internal clock update logic
void AudioVideoSynchronizer::_update_clocks(double p_delta) {
    // Update video clock if not master
    if (clock_source != CLOCK_VIDEO) {
        video_clock += p_delta;
    }
    
    // Update audio clock if not master
    if (clock_source != CLOCK_AUDIO) {
        audio_clock += p_delta;
    }
    
    // Calculate synchronization delay
    double av_delay = video_clock - audio_clock;
    
    // Apply adaptive synchronization
    if (adaptive_sync && Math::abs(av_delay) > av_sync_threshold) {
        _adjust_playback_rate(av_delay);
    }
}

// Adjust playback rate for synchronization
void AudioVideoSynchronizer::_adjust_playback_rate(double p_delay) {
    if (Math::abs(p_delay) > max_frame_drop_threshold) {
        // Large delay - consider frame dropping/duplication
        if (p_delay > 0 && drop_frames_enabled) {
            // Video ahead - drop frames
            while (!decoded_frames.is_empty() && p_delay > duplicate_frame_threshold) {
                FrameInfo frame = decoded_frames.front();
                decoded_frames.pop_front();
                frames_dropped++;
                p_delay -= (1.0 / target_frame_rate);
            }
        } else if (p_delay < 0) {
            // Video behind - duplicate current frame
            if (has_current_frame && Math::abs(p_delay) > duplicate_frame_threshold) {
                frames_duplicated++;
            }
        }
    }
}
```

## Frame Queue Management

### Video Frame Queue

```cpp
// Queue decoded video frame
void AudioVideoSynchronizer::queue_decoded_frame(RID p_texture, double p_pts, double p_dts, bool p_keyframe) {
    FrameInfo frame;
    frame.texture = p_texture;
    frame.pts = p_pts;
    frame.dts = (p_dts >= 0) ? p_dts : p_pts;
    frame.frame_number = frames_displayed + decoded_frames.size();
    frame.keyframe = p_keyframe;
    frame.duration = 1.0 / target_frame_rate;
    
    // Maintain queue size
    while (decoded_frames.size() >= max_video_queue_size) {
        FrameInfo dropped_frame = decoded_frames.front();
        decoded_frames.pop_front();
        frames_dropped++;
        
        // Release texture resource
        RenderingDevice::get_singleton()->free(dropped_frame.texture);
    }
    
    decoded_frames.push_back(frame);
}

// Get current frame for presentation
Ref<Texture2D> AudioVideoSynchronizer::get_current_frame() {
    if (!has_current_frame) {
        return Ref<Texture2D>();
    }
    
    // Convert RID to Texture2D
    RenderingDevice *rd = RenderingDevice::get_singleton();
    return rd->texture_get_rd_texture(current_frame.texture);
}

// Check if frame should be presented
bool AudioVideoSynchronizer::should_present_frame(double p_current_time) {
    if (decoded_frames.is_empty()) {
        return false;
    }
    
    FrameInfo next_frame = decoded_frames.front();
    double frame_time = next_frame.pts;
    double time_diff = frame_time - master_clock;
    
    // Check if it's time to present this frame
    if (time_diff <= av_sync_threshold) {
        return true;
    }
    
    // Check if frame should be dropped
    if (_should_drop_frame(next_frame)) {
        decoded_frames.pop_front();
        frames_dropped++;
        RenderingDevice::get_singleton()->free(next_frame.texture);
        return should_present_frame(p_current_time); // Check next frame
    }
    
    return false;
}

// Advance to next frame
void AudioVideoSynchronizer::advance_frame() {
    if (decoded_frames.is_empty()) {
        return;
    }
    
    // Release previous frame
    if (has_current_frame) {
        RenderingDevice::get_singleton()->free(current_frame.texture);
    }
    
    // Get next frame
    current_frame = decoded_frames.front();
    decoded_frames.pop_front();
    has_current_frame = true;
    frames_displayed++;
    
    // Update video clock
    video_clock = current_frame.pts;
    last_frame_time = current_frame.pts;
    
    // Update statistics
    double frame_delay = _calculate_frame_delay();
    avg_frame_delay = (avg_frame_delay * 0.9) + (frame_delay * 0.1);
}
```

### Audio Queue Management

```cpp
// Queue audio packet
void AudioVideoSynchronizer::queue_audio_packet(const Vector<float> &p_samples, double p_pts, 
                                               int p_sample_rate, int p_channels) {
    AudioInfo audio;
    audio.samples = p_samples;
    audio.pts = p_pts;
    audio.sample_rate = p_sample_rate;
    audio.channels = p_channels;
    audio.duration = (double)p_samples.size() / (p_sample_rate * p_channels);
    
    // Maintain queue size
    while (audio_packets.size() >= max_audio_queue_size) {
        audio_packets.pop_front();
    }
    
    audio_packets.push_back(audio);
}

// Get audio samples for playback
bool AudioVideoSynchronizer::get_audio_samples(Vector<float> &r_samples, int p_requested_frames) {
    if (audio_packets.is_empty()) {
        r_samples.clear();
        return false;
    }
    
    AudioInfo &audio = audio_packets.front();
    int samples_per_frame = audio.channels;
    int available_frames = audio.samples.size() / samples_per_frame;
    int frames_to_copy = MIN(p_requested_frames, available_frames);
    
    // Copy samples
    r_samples.resize(frames_to_copy * samples_per_frame);
    for (int i = 0; i < frames_to_copy * samples_per_frame; i++) {
        r_samples.write[i] = audio.samples[i];
    }
    
    // Update audio clock
    double consumed_duration = (double)frames_to_copy / audio.sample_rate;
    audio_clock += consumed_duration;
    
    // Remove consumed samples
    if (frames_to_copy == available_frames) {
        audio_packets.pop_front();
    } else {
        // Partial consumption - remove consumed samples
        Vector<float> remaining_samples;
        int remaining_count = audio.samples.size() - (frames_to_copy * samples_per_frame);
        remaining_samples.resize(remaining_count);
        for (int i = 0; i < remaining_count; i++) {
            remaining_samples.write[i] = audio.samples[frames_to_copy * samples_per_frame + i];
        }
        audio.samples = remaining_samples;
        audio.pts += consumed_duration;
        audio.duration -= consumed_duration;
    }
    
    return true;
}
```

## Synchronization Algorithms

### Frame Timing Calculation

```cpp
// Calculate frame delay for statistics
double AudioVideoSynchronizer::_calculate_frame_delay() {
    if (!has_current_frame) {
        return 0.0;
    }
    
    double expected_time = last_frame_time + (1.0 / target_frame_rate);
    double actual_time = current_frame.pts;
    return actual_time - expected_time;
}

// Determine if frame should be dropped
bool AudioVideoSynchronizer::_should_drop_frame(const FrameInfo &p_frame) {
    if (!drop_frames_enabled) {
        return false;
    }
    
    double frame_delay = p_frame.pts - master_clock;
    
    // Drop if frame is too late
    if (frame_delay < -max_frame_drop_threshold) {
        return true;
    }
    
    // Don't drop keyframes unless severely delayed
    if (p_frame.keyframe && frame_delay > -max_frame_drop_threshold * 2) {
        return false;
    }
    
    return false;
}

// Determine if current frame should be duplicated
bool AudioVideoSynchronizer::_should_duplicate_frame() {
    if (!has_current_frame) {
        return false;
    }
    
    double time_since_last = master_clock - last_frame_time;
    return time_since_last > duplicate_frame_threshold;
}
```

## Adaptive Synchronization

### Dynamic Sync Adjustment

```cpp
// Adaptive synchronization based on playback conditions
class AdaptiveSyncController : public RefCounted {
    GDCLASS(AdaptiveSyncController, RefCounted);

private:
    struct SyncMetrics {
        double avg_delay = 0.0;
        double delay_variance = 0.0;
        uint32_t dropped_frames = 0;
        uint32_t duplicated_frames = 0;
        double sync_quality = 1.0;
    };
    
    SyncMetrics current_metrics;
    Vector<double> delay_history;
    uint32_t history_size = 60; // 1 second at 60fps
    
    // Adaptive parameters
    double base_sync_threshold = 0.04;
    double min_sync_threshold = 0.01;
    double max_sync_threshold = 0.1;

public:
    AdaptiveSyncController();
    
    void update_metrics(double p_delay, bool p_frame_dropped, bool p_frame_duplicated);
    double get_adaptive_threshold();
    bool should_enable_frame_dropping();
    Dictionary get_quality_metrics();

private:
    void _calculate_variance();
    double _calculate_sync_quality();
};

// Update synchronization metrics
void AdaptiveSyncController::update_metrics(double p_delay, bool p_frame_dropped, bool p_frame_duplicated) {
    // Add to delay history
    delay_history.push_back(p_delay);
    if (delay_history.size() > history_size) {
        delay_history.remove_at(0);
    }
    
    // Update counters
    if (p_frame_dropped) current_metrics.dropped_frames++;
    if (p_frame_duplicated) current_metrics.duplicated_frames++;
    
    // Calculate average delay
    double sum = 0.0;
    for (double delay : delay_history) {
        sum += delay;
    }
    current_metrics.avg_delay = sum / delay_history.size();
    
    // Calculate variance
    _calculate_variance();
    
    // Update sync quality
    current_metrics.sync_quality = _calculate_sync_quality();
}

// Get adaptive synchronization threshold
double AdaptiveSyncController::get_adaptive_threshold() {
    // Adjust threshold based on delay variance
    double adaptive_factor = 1.0 + (current_metrics.delay_variance * 2.0);
    double threshold = base_sync_threshold * adaptive_factor;
    
    return CLAMP(threshold, min_sync_threshold, max_sync_threshold);
}
```

## Performance Monitoring

### Synchronization Statistics

```cpp
// Get comprehensive synchronization statistics
Dictionary AudioVideoSynchronizer::get_sync_statistics() const {
    Dictionary stats;
    
    // Basic counters
    stats["frames_displayed"] = frames_displayed;
    stats["frames_dropped"] = frames_dropped;
    stats["frames_duplicated"] = frames_duplicated;
    
    // Performance metrics
    stats["avg_frame_delay"] = avg_frame_delay;
    stats["current_av_delay"] = video_clock - audio_clock;
    stats["video_queue_size"] = decoded_frames.size();
    stats["audio_queue_size"] = audio_packets.size();
    
    // Clock information
    stats["master_clock"] = master_clock;
    stats["audio_clock"] = audio_clock;
    stats["video_clock"] = video_clock;
    stats["clock_source"] = (int)clock_source;
    
    // Quality metrics
    double drop_rate = (frames_displayed > 0) ? (double)frames_dropped / frames_displayed : 0.0;
    double duplicate_rate = (frames_displayed > 0) ? (double)frames_duplicated / frames_displayed : 0.0;
    stats["frame_drop_rate"] = drop_rate;
    stats["frame_duplicate_rate"] = duplicate_rate;
    
    // Sync quality (0.0 = poor, 1.0 = perfect)
    double sync_quality = 1.0 - (Math::abs(avg_frame_delay) / av_sync_threshold);
    stats["sync_quality"] = CLAMP(sync_quality, 0.0, 1.0);
    
    return stats;
}
```

## Usage Examples

### Basic Synchronization Setup
```cpp
// Initialize audio-video synchronizer
Ref<AudioVideoSynchronizer> av_sync = memnew(AudioVideoSynchronizer);
av_sync->initialize(30.0, AudioVideoSynchronizer::CLOCK_AUDIO);
av_sync->set_sync_threshold(0.04); // 40ms threshold
av_sync->set_adaptive_sync(true);

// Queue decoded frames
av_sync->queue_decoded_frame(texture_rid, pts, dts, is_keyframe);

// Update during playback
av_sync->update_master_clock(delta_time);

// Get current frame for display
if (av_sync->should_present_frame(current_time)) {
    Ref<Texture2D> frame = av_sync->get_current_frame();
    if (frame.is_valid()) {
        display_frame(frame);
        av_sync->advance_frame();
    }
}
```

### Advanced Configuration
```cpp
// Configure for low-latency streaming
av_sync->set_clock_source(AudioVideoSynchronizer::CLOCK_EXTERNAL);
av_sync->set_sync_threshold(0.02); // 20ms for low latency
av_sync->set_frame_dropping(true);

// Monitor synchronization quality
Dictionary stats = av_sync->get_sync_statistics();
print("Sync quality: ", stats["sync_quality"]);
print("Frame drop rate: ", stats["frame_drop_rate"] * 100.0, "%");
print("A/V delay: ", stats["current_av_delay"] * 1000.0, "ms");

// Adaptive threshold adjustment
if (stats["sync_quality"] < 0.8) {
    double current_threshold = av_sync->get_sync_threshold();
    av_sync->set_sync_threshold(current_threshold * 1.2);
}
```

This synchronization system provides robust, adaptive audio-video sync with comprehensive monitoring and automatic quality adjustment for optimal playback experience.
