/**************************************************************************/
/*  keyframe_reduce.cpp                                                   */
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

#include "keyframe_reduce.h"

using Vector2Bezier = BezierKeyframeReduce::Vector2Bezier;
using Bezier = BezierKeyframeReduce::Bezier;

#include <vector>

using namespace std;

// Based on https://github.com/robertjoosten/maya-keyframe-reduction

// MIT License

// Copyright (c) 2019 Robert Joosten

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

LocalVector<double> BezierKeyframeReduce::float_range(double p_start, double p_end, double p_step) {
	LocalVector<double> values;
	while (p_start < p_end) {
		values.push_back(p_start);
		p_start += p_step;
	}

	return values;
};

void BezierKeyframeReduce::fit_cubic(const LocalVector<Bezier> &p_curves, LocalVector<Bezier> &r_keyframes, int32_t p_first, int32_t p_last, Vector2Bezier p_tan_1, Vector2Bezier p_tan_2, real_t p_error) {
	//use heuristic if region only has two points in it
	if (p_last - p_first == 1) {
		// get points
		Vector2Bezier pt1 = p_curves[p_first].time_value;
		Vector2Bezier pt2 = p_curves[p_last].time_value;

		// get distance between points
		real_t dist = pt1.distance_between(pt2) / 3;

		// add curve
		add_curve(r_keyframes, pt1, pt1 + p_tan_1 * dist, pt2 + p_tan_2 * dist, pt2);
		return;
	}

	// parameterize points, and attempt to fit curve
	HashMap<int, Vector2Bezier> uPrime = chord_length_parameterize(p_curves, p_first, p_last);
	real_t errorThreshold = MAX(p_error, p_error * 4);

	// get iterations, when weighted tangents is turned off the bezier
	// generator cannot be improved using multiple iterations.
	int32_t iterations = 4;
	real_t maxError = 0;
	int32_t maxIndex = 0;
	for (int32_t i = 0; i < iterations; i++) {
		// generate curve
		LocalVector<Vector2Bezier> curve = generate_bezier(p_curves, p_first, p_last, uPrime, p_tan_1, p_tan_2);

		// find max deviation of points to fitted curve
		Vector2Bezier max_error_vec = find_max_error(p_curves, p_first, p_last, curve, uPrime);
		maxError = max_error_vec.x;
		maxIndex = max_error_vec.y;

		// validate max error and add curve
		if (maxError < p_error) {
			add_curve(r_keyframes, curve[0], curve[1], curve[2], curve[3]);
			return;
		}

		// if error not too large, try reparameterization and iteration
		if (maxError >= errorThreshold) {
			break;
		}
		reparameterize(p_curves, p_first, p_last, uPrime, curve);
		errorThreshold = maxError;
	}

	// fitting failed -- split at max error point and fit recursively
	Bezier tanCenter = (p_curves[maxIndex - 1] - p_curves[maxIndex + 1]).normalized();

	fit_cubic(p_curves, r_keyframes, p_first, maxIndex, p_tan_1, tanCenter.time_value, p_error);
	fit_cubic(p_curves, r_keyframes, maxIndex, p_last, tanCenter.time_value * -1, p_tan_2, p_error);
}

// 	@param Vector2Bezier pt1:
// 	@param Vector2Bezier tan1:
// 	@param Vector2Bezier tan2:
// 	@param Vector2Bezier pt2:
void BezierKeyframeReduce::add_curve(LocalVector<Bezier> &r_curves, Vector2Bezier p_pt_1, Vector2Bezier p_tan_1, Vector2Bezier p_tan_2, Vector2Bezier p_pt_2) {
	// update previous keyframe with out handle
	Bezier &prev = r_curves[r_curves.size() - 1];
	prev.out_handle = p_tan_1 - p_pt_1;

	// create new keyframe
	Bezier keyframe(p_pt_2, Vector2Bezier(p_tan_2 - p_pt_2), Vector2Bezier());
	r_curves.push_back(keyframe);
}

// Based on the weighted tangent setting either use a least-squares
// method to find Bezier controls points for a region or use Wu/Barsky
// heuristic.
// @param int first:
// @param int last:
// @param dict uPrime:
// @param Vector2Bezier tan1:
// @param Vector2Bezier tan2:
LocalVector<Vector2Bezier> BezierKeyframeReduce::generate_bezier(const LocalVector<Bezier> &p_curves, int32_t p_first, int32_t p_last, HashMap<int, Vector2Bezier> p_u_prime, Vector2Bezier p_tan_1, Vector2Bezier p_tan_2) {
	Vector2Bezier pt1 = p_curves[p_first].time_value;
	Vector2Bezier pt2 = p_curves[p_last].time_value;

	Vector2Bezier alpha1;
	Vector2Bezier alpha2;
	Vector2Bezier handle1;
	Vector2Bezier handle2;
	// use least-squares method to find Bezier control points for region.
	LocalVector<Vector2Bezier> C;
	C.resize(4);
	LocalVector<Vector2Bezier> X;
	X.resize(2);
	int32_t range_i = p_last - p_first + 1;
	for (int32_t i = 0; i < range_i; i++) {
		Vector2Bezier u = p_u_prime[i];
		Vector2Bezier t = Vector2Bezier(1, 1) - u;
		Vector2Bezier b = Vector2Bezier(3, 3) * u * t;
		Vector2Bezier b0 = t * t * t;
		Vector2Bezier b1 = b * t;
		Vector2Bezier b2 = b * u;
		Vector2Bezier b3 = u * u * u;
		Vector2Bezier a1 = p_tan_1 * b1;
		Vector2Bezier a2 = p_tan_2 * b2;
		Vector2Bezier tmp = (p_curves[p_first + i].time_value - pt1 * (b0 + b1) - pt2 * (b2 + b3));
		C[0] += a1 * a1;
		C[1] += a1 * a2;
		C[2] = C[1];
		C[3] += a2 * a2;
		X[0] += a1 * tmp;
		X[1] += a2 * tmp;
	}

	//compute the determinants of C and X
	Vector2Bezier detC0C1 = C[0] * C[3] - C[2] * C[1];
	if (detC0C1.abs().x > FLT_EPSILON && detC0C1.abs().y > FLT_EPSILON) {
		//kramer's rule
		Vector2Bezier detC0X = C[0] * X[1] - C[2] * X[0];
		Vector2Bezier detXC1 = X[0] * C[3] - X[1] * C[1];

		//derive alpha values
		alpha1 = detXC1 / detC0C1;
		alpha2 = detC0X / detC0C1;
	} else {
		//matrix is under-determined, try assuming alpha1 == alpha2
		Vector2Bezier c0 = C[0] + C[1];
		Vector2Bezier c1 = C[2] + C[3];
		if (c0.abs().x > FLT_EPSILON && c0.abs().y > FLT_EPSILON) {
			alpha1 = alpha2 = X[0] / c0;
		} else if (c1.abs().x > FLT_EPSILON && c1.abs().y > FLT_EPSILON) {
			alpha1 = alpha2 = X[1] / c1;
		}
	}

	// if alpha negative, use the Wu/Barsky heuristic (see text)
	// (if alpha is 0, you get coincident control points that lead to
	// divide by zero in any subsequent NewtonRaphsonRootFind() call.
	real_t segment_length = pt2.distance_between(pt1);
	const real_t epsilon = FLT_EPSILON * segment_length;
	if ((alpha1.x < epsilon && alpha1.y < epsilon) && (alpha2.x < epsilon && alpha2.y < epsilon)) {
		// fall back on standard (probably inaccurate) formula,
		// and subdivide further if needed.
		alpha1 = alpha2 = Vector2Bezier(segment_length / 3, segment_length / 3);
	}
	// check if the found control points are in the right order when
	// projected onto the line through pt1 and pt2.
	Vector2Bezier line = pt2 - pt1;

	// control points 1 and 2 are positioned an alpha distance out
	// on the tangent vectors, left and right, respectively
	handle1 = p_tan_1 * alpha1;
	handle2 = p_tan_2 * alpha2;
	Vector2Bezier a = ((handle1 * line) - (handle2 * line));
	Vector2Bezier b = Vector2Bezier(segment_length, segment_length) * Vector2Bezier(segment_length, segment_length);
	if (a.x > b.x && a.y > b.y) {
		// fall back to the Wu/Barsky heuristic above.
		alpha1 = alpha2 = Vector2Bezier(segment_length / 3, segment_length / 3);
		handle1 = handle2 = Vector2Bezier();
	}

	// first and last control points of the Bezier curve are
	// positioned exactly at the first and last data points
	// Control points 1 and 2 are positioned an alpha distance out
	// on the tangent vectors, left and right, respectively
	LocalVector<Vector2Bezier> curve;
	curve.resize(4);
	curve[0] = pt1;
	Vector2Bezier pt2_tmp = p_tan_1 * alpha1;
	if (handle1 == Vector2Bezier()) {
		pt2_tmp = handle1;
	}
	curve[1] = pt1 + pt2_tmp;
	Vector2Bezier pt3_tmp = p_tan_2 * alpha2;
	if (handle2 == Vector2Bezier()) {
		pt2_tmp = handle2;
	}
	curve[2] = pt2 + pt3_tmp;
	curve[3] = pt2;
	return curve;
}

// Given set of points and their parameterization, try to find a better
// parameterization.
// @param int first:
// @param int last:
// @param dict u:
// @param list curve:
void BezierKeyframeReduce::reparameterize(LocalVector<Bezier> p_existing_curves, int32_t p_first, int32_t p_last, HashMap<int32_t, Vector2Bezier> &r_u, LocalVector<Vector2Bezier> p_curves) {
	int32_t i_range = p_last + 1;
	for (int32_t i = p_first; i < i_range; i++) {
		r_u[i - p_first] = find_root(p_curves, p_existing_curves[i].time_value, r_u[i - p_first]);
	}
}

// Use Newton-Raphson iteration to find better root.
// @param list curve:
// @param Vector2Bezier point
// @param Vector2Bezier u
// @return New root point
// @rtype Vector2Bezier
Vector2Bezier BezierKeyframeReduce::find_root(LocalVector<Vector2Bezier> p_curves, Vector2Bezier p_curve, Vector2Bezier p_u) {
	// generate control vertices for Q'
	LocalVector<Vector2Bezier> curve1;
	for (int32_t curve_i = 0; curve_i < 3; curve_i++) {
		curve1.push_back(p_curves[curve_i + 1] - p_curves[curve_i] * 3);
	}
	LocalVector<Vector2Bezier> curve2;
	for (int32_t curve_i = 0; curve_i < 2; curve_i++) {
		// generate control vertices for Q''
		curve2.push_back((curve1[curve_i + 1] - curve1[curve_i]) * 2);
	}

	// compute Q(u), Q'(u) and Q''(u)
	Vector2Bezier pt = evaluate(3, p_curves, p_u);
	Vector2Bezier pt1 = evaluate(2, curve1, p_u);
	Vector2Bezier pt2 = evaluate(1, curve2, p_u);
	Vector2Bezier diff = pt - p_curve;
	Vector2Bezier df = (pt1 * pt1) + (diff * pt2);

	// compute f(u) / f'(u)
	if (df.abs().x < DBL_EPSILON && df.abs().y < DBL_EPSILON) {
		return p_u;
	}
	// u = u - f(u) / f'(u)
	return p_u - (diff * pt1) / df;
}

// Evaluate a bezier curve at a particular parameter value.
// @param int degree:
// @param list curve:
// @param float t:
// @return  Vector2Bezier point on curve
Vector2Bezier BezierKeyframeReduce::evaluate(int32_t p_degree, LocalVector<Vector2Bezier> p_curves, Vector2Bezier p_t) {
	// copy array
	LocalVector<Vector2Bezier> curves = p_curves;

	// triangle computation
	int32_t i_range = p_degree + 1;
	for (int32_t i = 1; i < i_range; i++) {
		int32_t j_range = p_degree - i + 1;
		for (int32_t j = 0; j < j_range; j++) {
			curves[j] = (curves[j] * (Vector2Bezier(1.0f, 1.0f) - p_t)) + (curves[j + 1] * p_t);
		}
	}

	return curves[0];
}

// Assign parameter values to digitized points using relative distances
// between points.
// @param int first:
// @param int last:
// @return dictionary of chord length parameterization
HashMap<int, Vector2Bezier> BezierKeyframeReduce::chord_length_parameterize(LocalVector<Bezier> p_curves, int32_t p_first, int32_t p_last) {
	HashMap<int, Vector2Bezier> u;
	u.insert(0, Vector2Bezier());
	int32_t range_i = p_last + 1;
	for (int32_t i = p_first + 1; i < range_i; i++) {
		real_t value = p_curves[i].time_value.distance_between(
				p_curves[i - 1].time_value);
		u.insert(i - p_first, u[i - p_first - 1] + Vector2Bezier(value, value));
	}
	int32_t m = p_last - p_first;
	int32_t range_j = m + 1;
	for (int32_t j = 1; j < range_j; j++) {
		u[j] = u[j] / u[m];
	}
	return u;
}

// Find the maximum squared distance of digitized points to fitted
// curve.
// @param int first:
// @param int last:
// @param list curve:
// @param dict u:
// @return tuple of Max distance and max index
Vector2Bezier BezierKeyframeReduce::find_max_error(const LocalVector<Bezier> &p_existing_curves, int32_t p_first, int32_t p_last, LocalVector<Vector2Bezier> p_curves, HashMap<int, Vector2Bezier> p_u) {
	real_t maxDist = 0.0f;
	real_t maxIndex = Math::floor((p_last - p_first + 1.0f) / 2.0f);

	for (int32_t i = p_first + 1; i < p_last; i++) {
		Vector2Bezier P = evaluate(3, p_curves, p_u[i - p_first]);
		real_t dist = (P - p_existing_curves[i].time_value).length();

		if (dist >= maxDist) {
			maxDist = dist;
			maxIndex = i;
		}
	}

	return Vector2Bezier(maxDist, maxIndex);
}

real_t BezierKeyframeReduce::min_real_list(LocalVector<real_t> p_reals) {
	if (p_reals.is_empty()) {
		return 0;
	}
	real_t min = p_reals[0];
	for (int32_t real_i = 0; real_i < p_reals.size(); real_i++) {
		min = MIN(min, p_reals[real_i]);
	}
	return min;
}

real_t BezierKeyframeReduce::max_real_list(LocalVector<real_t> p_reals) {
	if (p_reals.is_empty()) {
		return 0.0f;
	}
	real_t max = p_reals[0];
	for (int32_t real_i = 0; real_i < p_reals.size(); real_i++) {
		max = MAX(max, p_reals[real_i]);
	}
	return max;
}

real_t BezierKeyframeReduce::sum_real_list(LocalVector<real_t> p_reals) {
	if (p_reals.is_empty()) {
		return 0.0f;
	}
	real_t sum = 0.0f;
	for (int32_t real_i = 0; real_i < p_reals.size(); real_i++) {
		sum += p_reals[real_i];
	}
	return sum;
}

// The automatic tangent split will take the average of all values and
// the average of just the minimum and maximum value and remaps that on
// a logarithmic scale, this will give a predicted split angle value.
// All angles will be processed to see if they fall in or outside that
// threshold.
// @param list of angles:
// @return list of split indices
LocalVector<int32_t> BezierKeyframeReduce::_find_tangent_split_auto(LocalVector<real_t> p_angles) {
	// get angles from points
	LocalVector<int32_t> splits;

	// get average variables
	real_t min_list = min_real_list(p_angles);
	real_t minAngle = min_list;
	if (min_list == 0) {
		minAngle = 0.00001;
	}
	real_t maxAngle = max_real_list(p_angles);
	real_t average = (minAngle + maxAngle) * 0.5;
	real_t mean = sum_real_list(p_angles) / p_angles.size() * 0.5;

	// get value at which to split
	real_t threshold = (Math::log(average) - Math::log(mean)) /
			(Math::log(maxAngle) - Math::log(minAngle)) * average;

	if (mean * 10 > average) {
		return LocalVector<int32_t>();
	}

	// split based on angles
	for (int32_t angle_i = 0; angle_i < p_angles.size(); angle_i++) {
		if (p_angles[angle_i] > threshold) {
			splits.push_back(angle_i + 1);
		}
	}

	return splits;
}

// Loop existing frames and see if any keyframes contain tangents that
// are not unified. If this is the case the index of the closest sampled
// point will be returned.
// @param list frames
// @param int start
// @param int end
// @param int/float step
// @return list of split indices
LocalVector<int32_t> BezierKeyframeReduce::_find_tangent_split_existing(const LocalVector<Bezier> p_frames, int32_t p_start, int32_t p_end, real_t p_step) {
	LocalVector<int32_t> splits;
	for (int32_t i = 0; i < p_frames.size(); i++) {
		// get closest index
		int32_t index = int((i - p_start) / p_step);

		// validate split
		Vector2 frame_abs = (p_frames[i].out_handle - p_frames[i].in_handle).abs();
		if (frame_abs.x > CMP_EPSILON && frame_abs.y > CMP_EPSILON) {
			splits.push_back(index);
		}
	}

	return splits;
}

// The threshold tangent split will process all angles and check if that
// angle falls in or outside of user provided threshold.
// @param list angles:
// @param int/float threshold:
// @return list of split indices
LocalVector<int32_t> BezierKeyframeReduce::_find_tangent_split_threshold(LocalVector<real_t> p_angles, real_t p_threshold) {
	LocalVector<int32_t> splits;
	// split based on angles
	for (int32_t i = 0; i < p_angles.size(); i++) {
		if (p_angles[i] > p_threshold) {
			splits.push_back(i + 1);
		}
	}
	return splits;
}

// Split provided points list based on the split indices provided. The
// lists will have a duplicate end and start points relating to each
// other.
// @param list p_curves:
// @param list p_split:
// @return list of split bezier points
LocalVector<Bezier> BezierKeyframeReduce::_split_points(const LocalVector<Bezier> &p_curves, LocalVector<int32_t> &p_split) {
	// validate split
	if (p_split.is_empty()) {
		return p_curves;
	}

	// complete split with adding start and end frames
	if (p_split[0] != 0) {
		p_split.insert(0, 0);
	}

	if (p_split[p_split.size() - 1] != p_curves.size()) {
		p_split.push_back(p_curves.size());
	}

	// make sure split is sorted and doesn't contain any duplicates
	HashSet<int32_t> split_set;
	for (int32_t split_i = 0; split_i < p_split.size(); split_i++) {
		split_set.insert(p_split[split_i]);
	}
	LocalVector<int32_t> split_list;
	for (int32_t split : split_set) {
		split_list.push_back(split);
	}
	p_split = split_list;
	p_split.sort();

	// split range for looping
	LocalVector<int32_t> splitA;
	for (int32_t i = 0; i < p_split.size() - 1; i++) {
		splitA.push_back(p_split[i]);
	}
	LocalVector<int32_t> splitB;
	for (int32_t i = 1; i < p_split.size(); i++) {
		if (!p_split.size()) {
			break;
		}
		splitB.push_back(p_split[i]);
	}

	// get lists
	LocalVector<Bezier> final_points;
	LocalVector<Vector2Bezier> zipped_splits;
	for (int32_t split_i = 0; split_i < splitA.size() && split_i < splitB.size(); split_i++) {
		zipped_splits.push_back(Vector2Bezier(splitA[split_i], splitB[split_i]));
	}
	for (int32_t zip_i = 0; zip_i < zipped_splits.size(); zip_i++) {
		Vector2Bezier zip = zipped_splits[zip_i];
		for (int32_t split_i = zip.x; split_i < zip.y; split_i++) {
			if (split_i >= p_curves.size()) {
				break;
			}
			final_points.push_back(p_curves[split_i]);
		}
	}
	return final_points;
}

// Ported from Paper.js -
// The Swiss Army Knife of Vector Graphics Scripting. http://paperjs.org/
// Fit bezier curves to the points based on the provided maximum error
// value and the bezier weighted tangents.
// @return list of bezier segments
LocalVector<Bezier> BezierKeyframeReduce::fit(FitState p_state) {
	real_t error = p_state.max_error;
	int32_t length = p_state.points.size();
	if (length == 0) {
		return LocalVector<Bezier>();
	}
	LocalVector<Bezier> segments;
	// add first point as a keyframe
	segments.push_back(p_state.points[0]);

	// return segments if there is only 1 point
	if (length == 1) {
		return segments;
	}

	// get tangents
	Vector2Bezier tan1 = (p_state.points[1].time_value - p_state.points[0].time_value).normalized();
	Vector2Bezier tan2 = (p_state.points[length - 2].time_value - p_state.points[length - 1].time_value).normalized();

	// fit cubic
	fit_cubic(p_state.points, segments, 0, length - 1, tan1, tan2, error);

	return segments;
}

LocalVector<real_t> BezierKeyframeReduce::get_values(LocalVector<Bezier> p_curves, LocalVector<double> p_frames) {
	LocalVector<real_t> values;
	for (int32_t frame_i = 0; frame_i < p_frames.size(); frame_i++) {
		int32_t frame = p_frames[frame_i];
		Vector2Bezier vec2 = p_curves[frame].time_value;
		real_t value = vec2.y;
		values.push_back(value);
	}
	return values;
}

// Sample the current animation curve based on the start and end frame,
// and the provided step size. Vector2Ds and angles will be returned.
// @param int start:
// @param int end:
// @param int/float step:
// @return list of sample points and angles
BezierKeyframeReduce::KeyframeTime BezierKeyframeReduce::sample(const LocalVector<Bezier> p_curves, int32_t p_start, int32_t p_end, real_t p_step) {
	BezierKeyframeReduce::KeyframeTime frame_values;
	// get frames and values
	LocalVector<double> frames = float_range(p_start, p_end, p_step);
	LocalVector<real_t> values = get_values(p_curves, frames);

	// get points and angles
	LocalVector<real_t> angles;
	LocalVector<Vector2Bezier> points;
	for (int32_t frame_i = 0; frame_i < frames.size(); frame_i++) {
		double time = frames[frame_i];
		real_t value = values[frame_i];
		Vector2Bezier point = Vector2Bezier(time, value);
		points.push_back(point);
	}
	for (int32_t i = 1; i < points.size() - 2; i++) {
		Vector2Bezier v1 = points[i - 1] - points[i];
		Vector2Bezier v2 = points[i + 1] - points[i];
		real_t angle_to = v1.angle_to(v2);
		real_t rad = Math_PI - angle_to;
		real_t deg = Math::rad_to_deg(rad);
		angles.push_back(deg);
	}

	frame_values.points = points;
	frame_values.angles = angles;
	return frame_values;
}

// Reduce the number of keyframes on the animation curve. Useful when
// you are working with baked curves.
// @param LocalVector<Bezier> r_points
// @param LocalVector<Bezier> r_keyframes
// @param KeyframeReductionSetting p_settings
// @return float Reduction rate
real_t BezierKeyframeReduce::reduce(const LocalVector<Bezier> &p_points, LocalVector<Bezier> &r_keyframes, KeyframeReductionSetting p_settings) {
	if (!p_points.size()) {
		return 0.0f;
	}
	LocalVector<Bezier> points = p_points;
	// get start and end frames
	int32_t start = 0;
	int32_t end = points.size() - 1;

	// get sample frames and values
	BezierKeyframeReduce::KeyframeTime sampled_frame_values = sample(points, start, end, p_settings.step_size);

	// get split indices
	LocalVector<int32_t> split;

	LocalVector<int32_t> tangent_split_auto = _find_tangent_split_auto(sampled_frame_values.angles);
	for (int i = 0; i < tangent_split_auto.size(); ++i) {
		split.push_back(tangent_split_auto[i]);
	}

	LocalVector<int32_t> tangent_split_existing = _find_tangent_split_existing(p_points, start, end, p_settings.step_size);
	for (int i = 0; i < tangent_split_existing.size(); ++i) {
		split.push_back(tangent_split_existing[i]);
	}

	LocalVector<int32_t> tangent_split_threshold = _find_tangent_split_threshold(sampled_frame_values.angles, p_settings.tangent_split_angle_threshold_in_degrees);
	for (int i = 0; i < tangent_split_threshold.size(); ++i) {
		split.push_back(tangent_split_threshold[i]);
	}

	{
		// get split points
		LocalVector<Bezier> split_points = _split_points(points, split);

		// fit points and get keyframes
		FitState state;
		state.max_error = p_settings.max_error;
		state.points = split_points;
		r_keyframes = fit(state);
	}
	if (points.size() <= r_keyframes.size()) {
		r_keyframes = points;
		return 0;
	}

	real_t rate = r_keyframes.size() / float(points.size());
	return rate;
}
