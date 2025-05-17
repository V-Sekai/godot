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

#include "../one_euro_filter.h"
#include "core/math/math_funcs.h"
#include "core/object/ref_counted.h" 
#include "tests/test_macros.h"

namespace TestOneEuroFilter {

TEST_CASE("[OneEuroFilter] Initialization and Configuration") {
	Ref<OneEuroFilter> filter_default = memnew(OneEuroFilter);
	REQUIRE(filter_default.is_valid());
	CHECK(Math::is_equal_approx(filter_default->get_min_cutoff_frequency(), 1.0));
	CHECK(Math::is_equal_approx(filter_default->get_beta_value(), 0.0));

	Ref<OneEuroFilter> filter_params = memnew(OneEuroFilter(2.0f, 0.5f));
	REQUIRE(filter_params.is_valid());
	CHECK(Math::is_equal_approx(filter_params->get_min_cutoff_frequency(), 2.0));
	CHECK(Math::is_equal_approx(filter_params->get_beta_value(), 0.5));

	filter_default->configure(3.0, 0.7);
	CHECK(Math::is_equal_approx(filter_default->get_min_cutoff_frequency(), 3.0));
	CHECK(Math::is_equal_approx(filter_default->get_beta_value(), 0.7));

	filter_default->set_min_cutoff_frequency(4.0);
	CHECK(Math::is_equal_approx(filter_default->get_min_cutoff_frequency(), 4.0));

	filter_default->set_beta_value(0.9);
	CHECK(Math::is_equal_approx(filter_default->get_beta_value(), 0.9));
}

TEST_CASE("[OneEuroFilter] Apply and Reset") {
	Ref<OneEuroFilter> filter = memnew(OneEuroFilter(1.0f, 0.007f));
	REQUIRE(filter.is_valid());

	double initial_value = 10.0;
	double delta_time = 1.0 / 60.0;

	double filtered_value1 = filter->apply(initial_value, delta_time);
	CHECK_MESSAGE(!Math::is_nan(filtered_value1), "Filtered value 1 is NaN");
	CHECK_MESSAGE(!Math::is_inf(filtered_value1), "Filtered value 1 is Inf");

	double next_value = 12.0;
	double filtered_value2 = filter->apply(next_value, delta_time);
	CHECK_MESSAGE(!Math::is_nan(filtered_value2), "Filtered value 2 is NaN");
	CHECK_MESSAGE(!Math::is_inf(filtered_value2), "Filtered value 2 is Inf");
	CHECK_MESSAGE(!Math::is_equal_approx(filtered_value2, filtered_value1), "Filter output should have changed when input changed from 10.0 to 12.0, but the output remained effectively the same.");

	filter->reset();
	double val_after_reset = filter->apply(initial_value, delta_time);
	CHECK_MESSAGE(!Math::is_nan(val_after_reset), "Value after reset is NaN");
	CHECK_MESSAGE(!Math::is_inf(val_after_reset), "Value after reset is Inf");
	CHECK_MESSAGE(Math::is_equal_approx(val_after_reset, filtered_value1), "Value after reset should be same as first apply if input is same.");

	double filtered_zero_dt = filter->apply(15.0, 0.0);
	CHECK_MESSAGE(Math::is_equal_approx(filtered_zero_dt, val_after_reset), "Filter with zero delta_time should return last filtered value.");
}

} // namespace TestOneEuroFilter
