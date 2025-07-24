/**************************************************************************/
/*  one_euro_filter.cpp                                                  */
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

#include "one_euro_filter.h"

OneEuroFilter::OneEuroFilter(double p_min_cutoff, double p_beta)
		: min_cutoff(p_min_cutoff), beta(p_beta), d_cutoff(p_min_cutoff) {
}

double OneEuroFilter::filter(double value, double delta_time) {
	if (delta_time <= 0.0) {
		return value; // Skip filtering for invalid delta
	}

	double rate = 1.0 / delta_time;
	double dx = (value - x_filter.last_value) * rate;

	double edx = dx_filter.filter(dx, calculate_alpha(rate, d_cutoff));
	double cutoff = min_cutoff + beta * Math::abs(edx);

	return x_filter.filter(value, calculate_alpha(rate, cutoff));
}

void OneEuroFilter::reset() {
	x_filter.reset();
	dx_filter.reset();
}

void OneEuroFilter::update_parameters(double p_min_cutoff, double p_beta) {
	min_cutoff = p_min_cutoff;
	beta = p_beta;
	d_cutoff = p_min_cutoff;
}

void OneEuroFilter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("filter", "value", "delta_time"), &OneEuroFilter::filter);
	ClassDB::bind_method(D_METHOD("reset"), &OneEuroFilter::reset);
	ClassDB::bind_method(D_METHOD("update_parameters", "min_cutoff", "beta"), &OneEuroFilter::update_parameters);
	ClassDB::bind_method(D_METHOD("get_min_cutoff"), &OneEuroFilter::get_min_cutoff);
	ClassDB::bind_method(D_METHOD("get_beta"), &OneEuroFilter::get_beta);
}
