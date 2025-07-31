/**************************************************************************/
/*  mock_audio_device.h                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

// Forward declarations
template <typename T>
class Vector;
struct Vector2;
typedef Vector<Vector2> PackedVector2Array;

/**
 * MockAudioDevice - Generates synthetic audio data for testing
 * 
 * This class provides various types of test audio signals that can be used
 * to test the speech module without requiring real microphone input or
 * multiple client instances.
 */
class MockAudioDevice {
public:
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

private:
    TestAudioType current_audio_type = TestAudioType::SINE_WAVE;
    NetworkTest network_condition = NetworkTest::PERFECT;
    
    // Audio generation parameters
    float sample_rate = 48000.0f;
    float frequency = 440.0f;  // A4 note for sine wave
    float phase = 0.0f;
    
    // Network simulation parameters
    float latency_ms = 0.0f;
    float packet_loss_percent = 0.0f;
    
    // Internal state for speech-like generation
    int vowel_index = 0;
    float vowel_time = 0.0f;
    
    // Generate specific audio types
    PackedVector2Array generate_sine_wave(float duration);
    PackedVector2Array generate_white_noise(float duration);
    PackedVector2Array generate_speech_like(float duration);
    PackedVector2Array generate_silence(float duration);

public:
    MockAudioDevice();
    
    // Main generation function
    PackedVector2Array generate(TestAudioType type, float duration);
    
    // Configuration
    void set_audio_type(TestAudioType type);
    void set_network_condition(NetworkTest condition);
    void set_sample_rate(float rate);
    void set_frequency(float freq);
    
    // Getters
    TestAudioType get_audio_type() const;
    NetworkTest get_network_condition() const;
    float get_sample_rate() const;
    float get_frequency() const;
    float get_latency_ms() const;
    float get_packet_loss_percent() const;
    
    // Network simulation
    bool should_drop_packet() const;
    float get_simulated_latency() const;
    
    // Utility functions
    static float audio_similarity(const PackedVector2Array &a, const PackedVector2Array &b);
    static float calculate_rms(const PackedVector2Array &audio);
};
