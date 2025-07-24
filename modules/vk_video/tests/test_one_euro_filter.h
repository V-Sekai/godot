/**************************************************************************/
/*  test_one_euro_filter.h                                                */
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

#include "scene/resources/one_euro_filter.h"

#include "tests/test_macros.h"

namespace TestOneEuroFilter {

TEST_CASE("[OneEuroFilter] Constructor and basic functionality") {
	OneEuroFilter filter(1.0, 5.0);
	CHECK_MESSAGE(
			filter.get_min_cutoff() == doctest::Approx(1.0),
			"OneEuroFilter constructor should set min_cutoff correctly.");
	CHECK_MESSAGE(
			filter.get_beta() == doctest::Approx(5.0),
			"OneEuroFilter constructor should set beta correctly.");
}

TEST_CASE("[OneEuroFilter] Constant input convergence") {
	OneEuroFilter filter(1.0, 5.0);
	double constant = 10.0;
	double result = constant;

	// Simulate 5 seconds at 60fps
	for (int i = 0; i < 300; i++) {
		result = filter.filter(constant, 1.0 / 60.0);
	}

	CHECK_MESSAGE(
			Math::abs(result - constant) < 0.01,
			"OneEuroFilter should converge to constant input value.");
}

TEST_CASE("[OneEuroFilter] Noise reduction") {
	OneEuroFilter filter(0.1, 5.0); // Low cutoff for smoothing
	double base_value = 5.0;
	double noise_amplitude = 0.5;

	// Generate noisy signal and measure variance
	double sum_input_variance = 0.0;
	double sum_output_variance = 0.0;
	double prev_input = base_value;
	double prev_output = base_value;

	for (int i = 0; i < 100; i++) {
		double noisy_input = base_value + (Math::randf() - 0.5) * noise_amplitude * 2;
		double filtered_output = filter.filter(noisy_input, 1.0 / 60.0);

		if (i > 0) {
			sum_input_variance += Math::pow(noisy_input - prev_input, 2);
			sum_output_variance += Math::pow(filtered_output - prev_output, 2);
		}

		prev_input = noisy_input;
		prev_output = filtered_output;
	}

	CHECK_MESSAGE(
			sum_output_variance < sum_input_variance,
			"OneEuroFilter should reduce signal variance.");
}

TEST_CASE("[OneEuroFilter] Parameter effects") {
	OneEuroFilter low_cutoff(0.1, 5.0);
	OneEuroFilter high_cutoff(2.0, 5.0);

	double noisy_value = 10.0 + Math::randf();
	double low_result = low_cutoff.filter(noisy_value, 1.0 / 60.0);
	double high_result = high_cutoff.filter(noisy_value, 1.0 / 60.0);

	// Higher cutoff should be closer to input (less smoothing)
	CHECK_MESSAGE(
			Math::abs(high_result - noisy_value) < Math::abs(low_result - noisy_value),
			"Higher cutoff should provide less smoothing.");
}

TEST_CASE("[OneEuroFilter] Reset functionality") {
	OneEuroFilter filter(1.0, 5.0);

	// Filter some values
	filter.filter(10.0, 1.0 / 60.0);
	filter.filter(15.0, 1.0 / 60.0);

	filter.reset();

	// After reset, should behave like new filter
	double result1 = filter.filter(5.0, 1.0 / 60.0);

	OneEuroFilter fresh_filter(1.0, 5.0);
	double result2 = fresh_filter.filter(5.0, 1.0 / 60.0);

	CHECK_MESSAGE(
			Math::abs(result1 - result2) < 0.001,
			"Reset filter should behave like fresh filter.");
}

TEST_CASE("[OneEuroFilter] Edge cases") {
	OneEuroFilter filter(0.1, 5.0);

	// Test zero delta time
	double result = filter.filter(10.0, 0.0);
	CHECK_MESSAGE(
			result == 10.0,
			"Filter should return input value for zero delta time.");

	// Test negative delta time
	result = filter.filter(10.0, -1.0);
	CHECK_MESSAGE(
			result == 10.0,
			"Filter should return input value for negative delta time.");
}

} // namespace TestOneEuroFilter
