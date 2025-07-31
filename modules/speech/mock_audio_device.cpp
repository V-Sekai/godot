/**************************************************************************/
/*  mock_audio_device.cpp                                                 */
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

#include "mock_audio_device.h"

#include <cmath>
#include <cstdlib>
#include <algorithm>

// Simple Vector2 implementation for testing
struct Vector2 {
    float x, y;
    Vector2() : x(0), y(0) {}
    Vector2(float p_x, float p_y) : x(p_x), y(p_y) {}
};

// Simple Vector implementation for testing
template<typename T>
class Vector {
private:
    T* data;
    int size_val;
    int capacity_val;

public:
    Vector() : data(nullptr), size_val(0), capacity_val(0) {}
    
    ~Vector() {
        if (data) {
            delete[] data;
        }
    }
    
    void resize(int new_size) {
        if (new_size > capacity_val) {
            T* new_data = new T[new_size];
            for (int i = 0; i < size_val; i++) {
                new_data[i] = data[i];
            }
            if (data) {
                delete[] data;
            }
            data = new_data;
            capacity_val = new_size;
        }
        size_val = new_size;
    }
    
    int size() const { return size_val; }
    
    T& operator[](int index) { return data[index]; }
    const T& operator[](int index) const { return data[index]; }
    
    T* write = data;
};

typedef Vector<Vector2> PackedVector2Array;

MockAudioDevice::MockAudioDevice() {
    // Initialize network conditions based on default
    set_network_condition(NetworkTest::PERFECT);
}

PackedVector2Array MockAudioDevice::generate(TestAudioType type, float duration) {
    switch (type) {
        case TestAudioType::SINE_WAVE:
            return generate_sine_wave(duration);
        case TestAudioType::WHITE_NOISE:
            return generate_white_noise(duration);
        case TestAudioType::SPEECH_LIKE:
            return generate_speech_like(duration);
        case TestAudioType::SILENCE:
            return generate_silence(duration);
        default:
            return generate_silence(duration);
    }
}

PackedVector2Array MockAudioDevice::generate_sine_wave(float duration) {
    int sample_count = static_cast<int>(sample_rate * duration);
    PackedVector2Array audio_data;
    audio_data.resize(sample_count);
    
    for (int i = 0; i < sample_count; i++) {
        float sample_value = std::sin(phase) * 0.5f; // 50% amplitude
        audio_data.write[i] = Vector2(sample_value, sample_value); // Stereo
        
        // Update phase
        phase += 2.0f * 3.14159265359f * frequency / sample_rate;
        if (phase > 2.0f * 3.14159265359f) {
            phase -= 2.0f * 3.14159265359f;
        }
    }
    
    return audio_data;
}

PackedVector2Array MockAudioDevice::generate_white_noise(float duration) {
    int sample_count = static_cast<int>(sample_rate * duration);
    PackedVector2Array audio_data;
    audio_data.resize(sample_count);
    
    for (int i = 0; i < sample_count; i++) {
        float sample_value = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.3f; // 30% amplitude
        audio_data.write[i] = Vector2(sample_value, sample_value); // Stereo
    }
    
    return audio_data;
}

PackedVector2Array MockAudioDevice::generate_speech_like(float duration) {
    int sample_count = static_cast<int>(sample_rate * duration);
    PackedVector2Array audio_data;
    audio_data.resize(sample_count);
    
    // Vowel formant frequencies (simplified)
    float vowel_freqs[][3] = {
        {730, 1090, 2440}, // 'a' as in "father"
        {270, 2290, 3010}, // 'i' as in "beat"
        {300, 870, 2240},  // 'u' as in "boot"
        {530, 1840, 2480}, // 'e' as in "bet"
        {570, 840, 2410}   // 'o' as in "bought"
    };
    
    for (int i = 0; i < sample_count; i++) {
        float time = static_cast<float>(i) / sample_rate;
        
        // Change vowel every 0.2 seconds
        if (time - vowel_time > 0.2f) {
            vowel_index = (vowel_index + 1) % 5;
            vowel_time = time;
        }
        
        // Generate formant-based speech-like sound
        float sample_value = 0.0f;
        for (int formant = 0; formant < 3; formant++) {
            float freq = vowel_freqs[vowel_index][formant];
            float amplitude = 0.1f / (formant + 1); // Decreasing amplitude for higher formants
            sample_value += amplitude * std::sin(2.0f * 3.14159265359f * freq * time);
        }
        
        // Apply envelope to make it more speech-like
        float envelope = 0.5f + 0.3f * std::sin(2.0f * 3.14159265359f * 5.0f * time); // 5Hz modulation
        sample_value *= envelope * 0.3f; // Overall amplitude
        
        audio_data.write[i] = Vector2(sample_value, sample_value); // Stereo
    }
    
    return audio_data;
}

PackedVector2Array MockAudioDevice::generate_silence(float duration) {
    int sample_count = static_cast<int>(sample_rate * duration);
    PackedVector2Array audio_data;
    audio_data.resize(sample_count);
    
    for (int i = 0; i < sample_count; i++) {
        audio_data.write[i] = Vector2(0.0f, 0.0f);
    }
    
    return audio_data;
}

void MockAudioDevice::set_audio_type(TestAudioType type) {
    current_audio_type = type;
}

void MockAudioDevice::set_network_condition(NetworkTest condition) {
    network_condition = condition;
    
    switch (condition) {
        case NetworkTest::PERFECT:
            latency_ms = 0.0f;
            packet_loss_percent = 0.0f;
            break;
        case NetworkTest::TYPICAL:
            latency_ms = 50.0f;
            packet_loss_percent = 1.0f;
            break;
        case NetworkTest::POOR:
            latency_ms = 200.0f;
            packet_loss_percent = 5.0f;
            break;
    }
}

void MockAudioDevice::set_sample_rate(float rate) {
    sample_rate = rate;
}

void MockAudioDevice::set_frequency(float freq) {
    frequency = freq;
}

MockAudioDevice::TestAudioType MockAudioDevice::get_audio_type() const {
    return current_audio_type;
}

MockAudioDevice::NetworkTest MockAudioDevice::get_network_condition() const {
    return network_condition;
}

float MockAudioDevice::get_sample_rate() const {
    return sample_rate;
}

float MockAudioDevice::get_frequency() const {
    return frequency;
}

float MockAudioDevice::get_latency_ms() const {
    return latency_ms;
}

float MockAudioDevice::get_packet_loss_percent() const {
    return packet_loss_percent;
}

bool MockAudioDevice::should_drop_packet() const {
    float random_value = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    return random_value < packet_loss_percent;
}

float MockAudioDevice::get_simulated_latency() const {
    // Add some random variation to latency (±20%)
    float variation = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.4f;
    return latency_ms * (1.0f + variation);
}

float MockAudioDevice::audio_similarity(const PackedVector2Array &a, const PackedVector2Array &b) {
    if (a.size() != b.size()) {
        return 0.0f; // Different sizes = no similarity
    }
    
    if (a.size() == 0) {
        return 1.0f; // Both empty = perfect similarity
    }
    
    float sum_diff_squared = 0.0f;
    float sum_a_squared = 0.0f;
    float sum_b_squared = 0.0f;
    
    for (int i = 0; i < a.size(); i++) {
        Vector2 sample_a = a[i];
        Vector2 sample_b = b[i];
        
        // Use left channel for comparison
        float diff = sample_a.x - sample_b.x;
        sum_diff_squared += diff * diff;
        sum_a_squared += sample_a.x * sample_a.x;
        sum_b_squared += sample_b.x * sample_b.x;
    }
    
    // Calculate normalized cross-correlation
    float denominator = std::sqrt(sum_a_squared * sum_b_squared);
    if (denominator < 1e-10f) {
        return 1.0f; // Both signals are essentially zero
    }
    
    // Return similarity as 1 - normalized_difference
    float normalized_diff = std::sqrt(sum_diff_squared) / denominator;
    return std::max(0.0f, 1.0f - normalized_diff);
}

float MockAudioDevice::calculate_rms(const PackedVector2Array &audio) {
    if (audio.size() == 0) {
        return 0.0f;
    }
    
    float sum_squared = 0.0f;
    for (int i = 0; i < audio.size(); i++) {
        Vector2 sample = audio[i];
        // Use left channel for RMS calculation
        sum_squared += sample.x * sample.x;
    }
    
    return std::sqrt(sum_squared / audio.size());
}
