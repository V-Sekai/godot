/**************************************************************************/
/*  one_euro_filter.cpp                                                   */
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

void OneEuroFilter::_bind_methods() {
	ClassDB::bind_method(D_METHOD("configure", "new_min_cutoff", "new_beta"), &OneEuroFilter::configure);
	ClassDB::bind_method(D_METHOD("reset"), &OneEuroFilter::reset);
	ClassDB::bind_method(D_METHOD("apply", "value", "delta_time"), &OneEuroFilter::apply);

	ClassDB::bind_method(D_METHOD("set_min_cutoff_frequency", "value"), &OneEuroFilter::set_min_cutoff_frequency);
	ClassDB::bind_method(D_METHOD("get_min_cutoff_frequency"), &OneEuroFilter::get_min_cutoff_frequency);
	ClassDB::bind_method(D_METHOD("set_beta_value", "value"), &OneEuroFilter::set_beta_value);
	ClassDB::bind_method(D_METHOD("get_beta_value"), &OneEuroFilter::get_beta_value);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_cutoff_freq"), "set_min_cutoff_frequency", "get_min_cutoff_frequency");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "beta_val"), "set_beta_value", "get_beta_value");
}

void OneEuroFilter::_clear_filters() {
	if (x_lpf) {
		memdelete(x_lpf);
		x_lpf = nullptr;
	}
	if (dx_lpf) {
		memdelete(dx_lpf);
		dx_lpf = nullptr;
	}
}

void OneEuroFilter::_initialize_filters() {
	_clear_filters();

	x_lpf = memnew(LowPassFilter);
	dx_lpf = memnew(LowPassFilter);
	initialized = true;
}

double OneEuroFilter::_compute_alpha(double p_rate, double p_cutoff_f) const {
	if (p_rate < CMP_EPSILON) {
		return 0.0f;
	}
	if (p_cutoff_f < CMP_EPSILON) {
		return 0.0f;
	}

	double tau = 1.0f / (2.0f * Math::PI * p_cutoff_f);
	double te = 1.0f / p_rate;
	return 1.0f / (1.0f + tau / te);
}

double OneEuroFilter::apply(double p_value, double p_delta_time) {
	ERR_FAIL_COND_V_MSG(!initialized || !x_lpf || !dx_lpf, p_value, "OneEuroFilter not properly initialized.");

	if (p_delta_time < CMP_EPSILON) {
		return x_lpf->last_value;
	}

	double rate = 1.0f / p_delta_time;

	double dx = (p_value - x_lpf->last_value) * rate;

	double edx = dx_lpf->filter(dx, _compute_alpha(rate, d_cutoff_freq));

	double current_x_cutoff = min_cutoff_freq + beta_val * Math::abs(edx);
	if (current_x_cutoff < CMP_EPSILON) {
		current_x_cutoff = CMP_EPSILON;
	}

	return x_lpf->filter(p_value, _compute_alpha(rate, current_x_cutoff));
}

void OneEuroFilter::reset() {
	if (x_lpf) {
		x_lpf->reset();
	}
	if (dx_lpf) {
		dx_lpf->reset();
	}
}

void OneEuroFilter::configure(double p_new_min_cutoff, double p_new_beta) {
	min_cutoff_freq = p_new_min_cutoff;
	beta_val = p_new_beta;
	d_cutoff_freq = p_new_min_cutoff;

	if (min_cutoff_freq < CMP_EPSILON) {
		min_cutoff_freq = CMP_EPSILON;
	}
	if (d_cutoff_freq < CMP_EPSILON) {
		d_cutoff_freq = CMP_EPSILON;
	}

	if (x_lpf) {
		x_lpf->reset();
	}
	if (dx_lpf) {
		dx_lpf->reset();
	}

	if (!initialized) {
		_initialize_filters();
	}
}

void OneEuroFilter::set_min_cutoff_frequency(double p_val) {
	min_cutoff_freq = p_val;
	// d_cutoff_freq is often kept in sync with min_cutoff_freq in OneEuroFilter implementations
	// If you want them to be independently configurable, you'll need a separate setter for d_cutoff_freq.
	d_cutoff_freq = p_val;
	if (min_cutoff_freq < CMP_EPSILON) {
		min_cutoff_freq = CMP_EPSILON;
	}
	if (d_cutoff_freq < CMP_EPSILON) {
		d_cutoff_freq = CMP_EPSILON;
	}
	// It's good practice to reset filters if parameters change significantly
	if (initialized) {
		x_lpf->reset();
		dx_lpf->reset();
	}
}

double OneEuroFilter::get_min_cutoff_frequency() const {
	return min_cutoff_freq;
}

void OneEuroFilter::set_beta_value(double p_val) {
	beta_val = p_val;
	// It's good practice to reset filters if parameters change significantly
	if (initialized) {
		// x_lpf->reset(); // Beta primarily affects dx_lpf's influence on x_lpf's cutoff
		dx_lpf->reset(); // Resetting dx_lpf might be sufficient or desired
	}
}

double OneEuroFilter::get_beta_value() const {
	return beta_val;
}
