/**************************************************************************/
/*  curve_utility.h                                                       */
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

#include "../../util/containers/std_vector.h"
#include "../../util/godot/core/rect2i.h"
#include "../../util/godot/macros.h"
#include "../../util/math/interval.h"

ZN_GODOT_FORWARD_DECLARE(class Curve)

namespace zylann {

struct CurveMonotonicSection {
	float x_min;
	float x_max;
	// Note: Y values are not necessarily in increasing order.
	// Their name only means to correspond to X coordinates.
	float y_min;
	float y_max;
};

struct CurveRangeData {
	StdVector<CurveMonotonicSection> sections;
};

static const float CURVE_RANGE_MARGIN = CMP_EPSILON;

// Gathers monotonic sections of a curve, at baked resolution.
// Within one section, the curve has only one of the following properties:
// - Be stationary or decrease
// - Be stationary or increase
// Which means, within one section, given a range of input values defined by a min and max,
// we can quickly calculate an accurate range of output values by sampling the curve only at the two points.
void get_curve_monotonic_sections(Curve &curve, StdVector<CurveMonotonicSection> &sections);
// Gets the range of Y values for a range of X values on a curve, using precalculated monotonic segments
math::Interval get_curve_range(Curve &curve, const StdVector<CurveMonotonicSection> &sections, math::Interval x);

// Legacy
math::Interval get_curve_range(Curve &curve, bool &is_monotonic_increasing);

} // namespace zylann
