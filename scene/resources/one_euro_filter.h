/**************************************************************************/
/*  one_euro_filter.h                                                     */
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

#include "core/math/math_funcs.h"
#include "core/object/ref_counted.h"

class OneEuroFilter : public RefCounted {
	GDCLASS(OneEuroFilter, RefCounted);

private:
	struct LowPassFilter {
		double last_value = 0.0;

		double filter(double value, double alpha) {
			double result = alpha * value + (1.0 - alpha) * last_value;
			last_value = result;
			return result;
		}

		void reset() { last_value = 0.0; }
	};

	double min_cutoff;
	double beta;
	double d_cutoff;
	LowPassFilter x_filter;
	LowPassFilter dx_filter;

	double calculate_alpha(double rate, double cutoff) const {
		double tau = 1.0 / (2.0 * Math::PI * cutoff);
		double te = 1.0 / rate;
		return 1.0 / (1.0 + tau / te);
	}

protected:
	static void _bind_methods();

public:
	OneEuroFilter(double p_min_cutoff = 0.1, double p_beta = 5.0);

	double filter(double value, double delta_time);
	void reset();
	void update_parameters(double p_min_cutoff, double p_beta);

	double get_min_cutoff() const { return min_cutoff; }
	double get_beta() const { return beta; }
};
