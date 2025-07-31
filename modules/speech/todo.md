# Speech Module Design Document - V-Sekai VOIP Redesign

## Executive Summary

Complete redesign of godot-speech module to achieve stable VOIP for V-Sekai multiplayer. Replaces unreliable jitter buffer system with One Euro Filter timing, implements comprehensive testing without multi-client requirements, and provides clean APIs for developers.

## Current Problems - Root Cause Analysis

-   **System Already Broken**: Current godot-speech is completely unusable and avoided
-   **Jitter Buffer Instability**: Pitch manipulation causes audio artifacts and timing drift
-   **Complex API Surface**: Multiple classes (Speech/SpeechProcessor/SpeechDecoder) create confusion
-   **Testing Bottleneck**: Requires launching multiple V-Sekai instances, blocking development
-   **No Compatibility Risk**: Since system is broken, we can redesign from scratch

## Architecture Redesign

### 1. Timing System Overhaul

**Replace**: Jitter buffer with pitch adjustment (STREAM_SPEEDUP_PITCH/SLOWDOWN)
**With**: One Euro Filter for smooth audio timing

```cpp
class OneEuroFilter {
    float cutoff;     // Jitter reduction (0.1 default)
    float beta;       // Lag reduction (5.0 default)
    float filter(float value, float delta_time);
};
```

**Implementation**:

-   Port conductor.gd OneEuroFilter to C++
-   Apply to audio timestamp deltas, not pitch
-   Eliminate audio artifacts from speed changes
-   Maintain perfect audio quality while smoothing timing

### 2. Unified API Design

**Current**: Speech + SpeechProcessor + SpeechDecoder (3 classes)
**New**: Single VoiceChat class with clean interface

```cpp
class VoiceChat : public Node {
public:
    // Connection Management
    bool connect_to_peer(int peer_id);
    void disconnect_from_peer(int peer_id);
    void disconnect_all();

    // Audio Control
    void set_microphone_enabled(bool enabled);
    void set_voice_volume(float volume);
    void set_voice_filter_enabled(bool enabled);

    // Status
    Dictionary get_connection_stats();
    Array get_connected_peers();
};
```

### 3. Testing Infrastructure - Zero Multi-Client Dependency

#### Mock Audio Framework

```cpp
enum class TestAudioType {
    SINE_WAVE,          // Basic tone
    WHITE_NOISE,        // Random audio
    SPEECH_LIKE,        // Vowel patterns
    SILENCE            // Zero audio
};

enum class NetworkTest {
    PERFECT,           // No latency/loss
    TYPICAL,           // 50ms, 1% loss
    POOR              // 200ms, 5% loss
};

class MockAudioDevice {
    PackedVector2Array generate(TestAudioType type, float duration);
    void set_network(NetworkTest condition);
};
```

#### Direct Frame Testing

```cpp
class VoiceChat : public Node {
    // Testing interface - inject synthetic data into audio pipeline
    void inject_audio_frames(PackedVector2Array frames);
    PackedVector2Array get_processed_frames();

    // Use existing audio capture for real testing
    bool start_recording(); // Already captures raw audio frames
    PackedVector2Array get_raw_microphone_frames();
};
```

#### Embedded Doctest Integration

-   Add doctest to SCsub build system
-   Tests run during module compilation
-   No separate test executables required
-   Automatic CI integration

```cpp
// In speech.cpp
#ifdef TESTS_ENABLED
#include "doctest.h"

TEST_CASE("OneEuroFilter reduces jitter") {
    OneEuroFilter filter(0.1f, 5.0f);
    // Test jittery input produces smooth output
    REQUIRE(filter.filter(noisy_data) < threshold);
}

TEST_CASE("Codec roundtrip preserves quality") {
    MockAudioDevice mock;
    auto input = mock.generate(TestAudioType::SPEECH_LIKE, 1.0f);
    auto compressed = codec.encode(input);
    auto output = codec.decode(compressed);
    REQUIRE(audio_similarity(input, output) > 0.95f);
}
#endif
```

## Implementation Roadmap

### Phase 1: Core Stability

**Deliverables**:

-   [ ] OneEuroFilter C++ implementation
-   [ ] Replace jitter buffer logic in Speech class
-   [ ] Basic VoiceChat API wrapper
-   [ ] Eliminate pitch-based timing adjustments

**Success Criteria**:

-   Audio quality matches input without artifacts
-   Timing stability under network jitter
-   No backward compatibility needed (system is broken)

### Phase 2: Testing Infrastructure

**Deliverables**:

-   [ ] MockAudioDevice with synthetic data generation
-   [ ] Direct frame injection testing interface
-   [ ] Doctest integration in SCsub
-   [ ] Comprehensive test suite (>90% coverage)

**Test Categories**:

-   Codec accuracy tests
-   Network simulation tests
-   Memory leak detection
-   Performance benchmarks
-   Filter parameter validation

### Phase 3: API Unification

**Deliverables**:

-   [ ] Complete VoiceChat class implementation
-   [ ] Remove old Speech/SpeechProcessor APIs entirely
-   [ ] Clean integration examples
-   [ ] Performance optimization

**Success Criteria**:

-   Single-class API for all VOIP operations
-   80% reduction in integration code (aggressive cleanup)
-   Improved performance characteristics

### Phase 4: Validation & Documentation

**Deliverables**:

-   [ ] Benchmark against two-voip-godot-4
-   [ ] Multi-peer stress testing (synthetic)
-   [ ] Complete API documentation
-   [ ] V-Sekai integration testing

**Benchmarks**:

-   Latency: <50ms end-to-end
-   CPU usage: <5% per voice stream
-   Memory: <10MB per active connection
-   Quality: >95% similarity after compression

## Technical Specifications

### OneEuroFilter Parameters

```cpp
struct FilterConfig {
    float cutoff = 0.1f;        // Lower = less jitter, more lag
    float beta = 5.0f;          // Higher = less lag, more jitter
    float min_cutoff = 1.0f;    // Minimum cutoff frequency
    float derivate_cutoff = 1.0f; // Derivative filter cutoff
};
```

### Audio Pipeline

```
Microphone → Opus Encode → Network → Opus Decode → OneEuroFilter → Speaker
                ↑                                        ↑
            MockAudio (testing)              Direct Frame Testing
```

### Network Protocol

-   Modify existing packet format to include timing metadata for OneEuroFilter
-   Add timestamp fields for filter synchronization
-   No legacy client support needed (clean break)

### Memory Management

-   Circular buffer for filter history
-   Automatic cleanup on peer disconnect

## Testing Strategy Details

### Unit Tests (Doctest)

```cpp
TEST_SUITE("OneEuroFilter") {
    TEST_CASE("Initialization") { /* ... */ }
    TEST_CASE("Jitter reduction") { /* ... */ }
    TEST_CASE("Lag compensation") { /* ... */ }
}

TEST_SUITE("VoiceChat") {
    TEST_CASE("Peer connection") { /* ... */ }
    TEST_CASE("Audio routing") { /* ... */ }
    TEST_CASE("Error handling") { /* ... */ }
}
```

### Integration Tests (Synthetic)

```cpp
TEST_CASE("Multi-peer voice chat simulation") {
    VoiceChat host, client1, client2;
    MockAudioDevice mock;

    // Simulate 3-way conversation
    host.connect_to_peer(1);
    host.connect_to_peer(2);

    // Inject test audio and verify routing
    auto test_audio = mock.generate(TestAudioType::SPEECH_LIKE, 1.0f);
    host.inject_audio(test_audio);

    REQUIRE(client1.get_received_audio().size() > 0);
    REQUIRE(client2.get_received_audio().size() > 0);
}
```

### Performance Tests

```cpp
TEST_CASE("CPU usage under load") {
    VoiceChat chat;
    for(int i = 0; i < 10; ++i) {
        chat.connect_to_peer(i);
    }

    auto start_time = get_cpu_time();
    simulate_voice_activity(1000); // 1 second
    auto cpu_usage = get_cpu_time() - start_time;

    REQUIRE(cpu_usage < 0.05); // <5% CPU
}
```

## Risk Mitigation


### Performance Risks

-   **Risk**: OneEuroFilter adds CPU overhead
-   **Mitigation**: Optimize filter implementation, benchmark against current system

### Quality Risks

-   **Risk**: Filter introduces audio latency
-   **Mitigation**: Tunable parameters, A/B testing with users

## Success Metrics

### Technical Metrics

-   **Stability**: Zero crashes in 24-hour stress test
-   **Quality**: >95% audio similarity after compression/filtering
-   **Latency**: <50ms end-to-end voice transmission
-   **CPU**: <5% per active voice stream

### Developer Experience

-   **API Simplicity**: 50% reduction in integration code lines
-   **Testing Speed**: Tests run in <30 seconds
-   **Documentation**: 100% API coverage with examples

### User Experience

-   **Voice Quality**: Subjective quality rating >4/5
-   **Connection Reliability**: >99% successful connections
-   **Audio Sync**: <20ms audio/visual desync in gameplay

## Conclusion

This design eliminates all open questions by providing:

1. **Specific technical solution**: OneEuroFilter replaces jitter buffer
2. **Clear testing strategy**: Mock audio + doctest, no multi-client needed
3. **Defined API surface**: Single VoiceChat class
4. **Measurable success criteria**: Performance benchmarks and quality metrics
5. **Risk mitigation**: Compatibility and performance safeguards
6. **Implementation timeline**: Phased approach

The result will be stable, maintainable VOIP that enables V-Sekai multiplayer without developer friction.
