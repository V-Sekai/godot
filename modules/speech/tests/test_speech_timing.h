/**************************************************************************/
/*  test_speech_timing.h                                                  */
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

#include "../one_euro_filter.h"
#include "../mock_audio_device.h"
#include "tests/test_macros.h"

namespace TestSpeechTiming {

TEST_CASE("[SpeechTiming] OneEuroFilter and MockAudioDevice integration") {
	MockAudioDevice device;
	OneEuroFilter filter;
	
	// Test that both components work together
	device.start_playback();
	
	// Simulate timing data with jitter
	float base_timing = 100.0f; // 100ms base timing
	float filtered_timing = filter.filter(base_timing, 1.0f / 60.0f);
	
	CHECK_MESSAGE(filtered_timing > 0.0f, "SpeechTiming: Filtered timing is positive");
	CHECK_MESSAGE(device.is_playing(), "SpeechTiming: Audio device is active during timing sync");
	
	device.stop_playback();
}

TEST_CASE("[SpeechTiming] Network jitter compensation") {
	MockAudioDevice device;
	OneEuroFilter filter;
	
	// Set up network conditions
	device.set_network_delay(50);  // 50ms base delay
	device.set_network_jitter(20); // 20ms jitter
	
	// Test timing synchronization under network conditions
	float timing_values[] = { 100.0f, 120.0f, 80.0f, 110.0f, 90.0f };
	float filtered_values[5];
	
	for (int i = 0; i < 5; ++i) {
		filtered_values[i] = filter.filter(timing_values[i], 1.0f / 60.0f);
	}
	
	// Check that filtering reduces variation
	float max_variation = 0.0f;
	for (int i = 1; i < 5; ++i) {
		float variation = filtered_values[i] - filtered_values[i-1];
		if (variation < 0) variation = -variation;
		if (variation > max_variation) max_variation = variation;
	}
	
	CHECK_MESSAGE(max_variation < 30.0f, "SpeechTiming: Filter reduces timing variation under network jitter");
}

TEST_CASE("[SpeechTiming] Performance under load") {
	MockAudioDevice device;
	OneEuroFilter filter;
	
	device.start_playback();
	
	// Simulate high-frequency timing updates
	const int iterations = 1000;
	float timing_sum = 0.0f;
	
	for (int i = 0; i < iterations; ++i) {
		float input_timing = 100.0f + (i % 10) * 2.0f; // Small variations
		float filtered_timing = filter.filter(input_timing, 1.0f / 60.0f);
		timing_sum += filtered_timing;
	}
	
	float average_timing = timing_sum / iterations;
	CHECK_MESSAGE(average_timing > 90.0f && average_timing < 110.0f, "SpeechTiming: Average timing remains stable under load");
	
	device.stop_playback();
}

} // namespace TestSpeechTiming
