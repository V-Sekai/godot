/**************************************************************************/
/*  test_one_euro_filter.h                                               */
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
#include "core/math/math_funcs.h"
#include "core/os/time.h"
#include "tests/test_macros.h"

namespace TestOneEuroFilter {

TEST_CASE("[OneEuroFilter] Basic initialization and configuration") {
	OneEuroFilter filter;
	
	// Test default configuration through getters
	CHECK_MESSAGE(filter.get_cutoff() == 0.1f, "OneEuroFilter: Default cutoff frequency is 0.1");
	CHECK_MESSAGE(filter.get_beta() == 5.0f, "OneEuroFilter: Default beta value is 5.0");
	CHECK_MESSAGE(filter.get_min_cutoff() == 1.0f, "OneEuroFilter: Default min_cutoff is 1.0");
	CHECK_MESSAGE(filter.get_derivate_cutoff() == 1.0f, "OneEuroFilter: Default derivate_cutoff is 1.0");
	
	// Test configuration setting
	OneEuroFilter::FilterConfig custom_config;
	custom_config.cutoff = 0.2f;
	custom_config.beta = 10.0f;
	custom_config.min_cutoff = 2.0f;
	custom_config.derivate_cutoff = 2.0f;
	
	filter.set_config(custom_config);
	OneEuroFilter::FilterConfig retrieved_config = filter.get_config();
	
	CHECK_MESSAGE(retrieved_config.cutoff == 0.2f, "OneEuroFilter: Custom cutoff frequency is set correctly");
	CHECK_MESSAGE(retrieved_config.beta == 10.0f, "OneEuroFilter: Custom beta value is set correctly");
	CHECK_MESSAGE(retrieved_config.min_cutoff == 2.0f, "OneEuroFilter: Custom min_cutoff is set correctly");
	CHECK_MESSAGE(retrieved_config.derivate_cutoff == 2.0f, "OneEuroFilter: Custom derivate_cutoff is set correctly");
}

TEST_CASE("[OneEuroFilter] Basic filtering functionality") {
	OneEuroFilter filter;
	const float delta_time = 1.0f / 60.0f; // 60 FPS
	
	// Test that filter produces reasonable output
	float input = 1.0f;
	float output = filter.filter(input, delta_time);
	CHECK_MESSAGE(output > 0.0f && output <= 1.0f, "OneEuroFilter: Filter produces reasonable output for positive input");
	
	// Test with zero input
	output = filter.filter(0.0f, delta_time);
	CHECK_MESSAGE(output >= 0.0f, "OneEuroFilter: Filter handles zero input correctly");
	
	// Test with negative input
	output = filter.filter(-1.0f, delta_time);
	CHECK_MESSAGE(output <= 0.0f, "OneEuroFilter: Filter handles negative input correctly");
}

TEST_CASE("[OneEuroFilter] Jitter reduction") {
	OneEuroFilter filter;
	const float delta_time = 1.0f / 60.0f;
	
	// Create jittery input signal
	float base_value = 100.0f;
	float jitter_amplitude = 10.0f;
	
	// Apply filter to jittery signal
	float previous_output = filter.filter(base_value, delta_time);
	float max_variation = 0.0f;
	
	for (int i = 0; i < 10; ++i) {
		// Add random jitter
		float jitter = (i % 2 == 0 ? jitter_amplitude : -jitter_amplitude);
		float jittery_input = base_value + jitter;
		
		float output = filter.filter(jittery_input, delta_time);
		float variation = Math::abs(output - previous_output);
		max_variation = Math::max(max_variation, variation);
		previous_output = output;
	}
	
	// Filtered output should have less variation than input jitter
	CHECK_MESSAGE(max_variation < jitter_amplitude, "OneEuroFilter: Filter reduces jitter in signal");
}

TEST_CASE("[OneEuroFilter] Reset functionality") {
	OneEuroFilter filter;
	const float delta_time = 1.0f / 60.0f;
	
	// Apply some filtering to build up internal state
	for (int i = 0; i < 10; ++i) {
		filter.filter(100.0f + i * 10.0f, delta_time);
	}
	
	// Reset the filter
	filter.reset();
	
	// After reset, filter should behave as if newly initialized
	float first_output = filter.filter(50.0f, delta_time);
	
	// Create a new filter for comparison
	OneEuroFilter fresh_filter;
	float fresh_output = fresh_filter.filter(50.0f, delta_time);
	
	CHECK_MESSAGE(Math::abs(first_output - fresh_output) < 0.001f, "OneEuroFilter: Reset restores filter to initial state");
}

TEST_CASE("[OneEuroFilter] Edge cases and robustness") {
	OneEuroFilter filter;
	
	// Test with very small delta time
	float output = filter.filter(1.0f, 0.0001f);
	CHECK_MESSAGE(!Math::is_nan(output) && Math::is_finite(output), "OneEuroFilter: Handles very small delta time without producing NaN or infinite values");
	
	// Test with zero delta time
	output = filter.filter(1.0f, 0.0f);
	CHECK_MESSAGE(!Math::is_nan(output) && Math::is_finite(output), "OneEuroFilter: Handles zero delta time gracefully");
	
	// Test with very large input values
	output = filter.filter(1000000.0f, 1.0f / 60.0f);
	CHECK_MESSAGE(!Math::is_nan(output) && Math::is_finite(output), "OneEuroFilter: Handles large input values without overflow");
	
	// Test with very small input values
	output = filter.filter(0.000001f, 1.0f / 60.0f);
	CHECK_MESSAGE(!Math::is_nan(output) && Math::is_finite(output), "OneEuroFilter: Handles very small input values correctly");
}

} // namespace TestOneEuroFilter
