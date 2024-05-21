/**************************************************************************/
/*  keyframe_reduce.h                                                     */
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

#ifndef KEYFRAME_REDUCE_H
#define KEYFRAME_REDUCE_H

#include "core/math/math_funcs.h"
#include "core/object/ref_counted.h"
#include <vector>

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

class BezierKeyframeReduce : public RefCounted {
	GDCLASS(BezierKeyframeReduce, RefCounted);

public:
	class Vector2Bezier : public Vector2 {
	public:
		// Distance between two points
		// @return real_t distance
		real_t distance_between(Vector2Bezier p_other) const {
			return (*this - p_other).length();
		}
		// Signed angle between points
		// @return real_t
		real_t signed_angle(Vector2 p_this, Vector2 p_other) {
			real_t angle = p_this.angle_to(p_other);
			real_t cross = p_this.cross(p_other);
			if (cross < 0) {
				return -angle;
			}
			return angle;
		}
		Vector2Bezier() {
		}
		Vector2Bezier(real_t p_x, real_t p_y) {
			x = p_x;
			y = p_y;
		}
		Vector2Bezier(Vector2 p_vector2) {
			x = p_vector2.x;
			y = p_vector2.y;
		}
	};
	struct Bezier {
		Vector2Bezier in_handle; //relative (x always <0)
		Vector2Bezier out_handle; //relative (x always >0)
		Vector2Bezier time_value;
		bool weighted_tangents = true;
		Bezier normalized() const {
			Bezier bezier = *this;
			bezier.in_handle = bezier.in_handle.normalized();
			bezier.out_handle = bezier.out_handle.normalized();
			bezier.time_value = bezier.time_value.normalized();
			return bezier;
		}
		Bezier operator-(Bezier p_other) const {
			Bezier bezier = *this;
			bezier.in_handle -= p_other.in_handle;
			bezier.out_handle -= p_other.out_handle;
			bezier.time_value -= p_other.time_value;
			return bezier;
		}
		Bezier() {
		}
		Bezier(Vector2Bezier p_time_value, Vector2Bezier p_in_handle, Vector2Bezier p_out_handle) {
			time_value = p_time_value;
			in_handle = p_in_handle;
			out_handle = p_out_handle;
		}
	};

	struct FitState {
		LocalVector<Bezier> points;
		real_t max_error;
	};

public:
	struct KeyframeReductionSetting {
		// Maximum allowed error when reducing the animation curves.
		real_t max_error = 0.1f;

		// Step size at which to sample the animation curves."
		real_t step_size = 0.5f;
		// The threshold to split tangents.
		real_t tangent_split_angle_threshold_in_degrees = 5.0f;
	};

private:
	// @param int/float p_start
	// @param int/float p_end
	// @param int/float p_step
	// @return LocalVector<String> range
	LocalVector<double> float_range(double p_start, double p_end, double p_step);

	// 	Add cubic bezier points between the provided first and last index
	// 	and it's tangents. Based in the weighted tangent settings the
	// 	iterations will be adjusted to gain speed. If a curve can be matched
	// 	the curve will be added to the keyframes, if not the curve will be
	// 	split at the point of max error and the function will be called
	// 	again.
	// 	@param uint first
	// 	@param uint last
	// 	@param Vector2Bezier tan1
	// 	@param Vector2Bezier tan2
	void fit_cubic(const LocalVector<Bezier> &p_curves, LocalVector<Bezier> &r_keyframes, uint32_t p_first, uint32_t p_last, Vector2Bezier p_tan_1, Vector2Bezier p_tan_2, real_t p_error);

	// 	@param Vector2Bezier pt1:
	// 	@param Vector2Bezier tan1:
	// 	@param Vector2Bezier tan2:
	// 	@param Vector2Bezier pt2:
	void add_curve(LocalVector<Bezier> &r_curves, Vector2Bezier p_pt_1, Vector2Bezier p_tan_1, Vector2Bezier p_tan_2, Vector2Bezier p_pt_2);

	// Based on the weighted tangent setting either use a least-squares
	// method to find Bezier controls points for a region or use Wu/Barsky
	// heuristic.
	// @param int first:
	// @param int last:
	// @param dict uPrime:
	// @param Vector2Bezier tan1:
	// @param Vector2Bezier tan2:
	LocalVector<Vector2Bezier> generate_bezier(const LocalVector<Bezier> &p_curves, uint32_t p_first, uint32_t p_last, HashMap<int, Vector2Bezier> p_u_prime, Vector2Bezier p_tan_1, Vector2Bezier p_tan_2);

	// Given set of points and their parameterization, try to find a better
	// parameterization.
	// @param int first:
	// @param int last:
	// @param dict u:
	// @param list curve:
	void reparameterize(LocalVector<Bezier> p_existing_curves, uint32_t p_first, uint32_t p_last, HashMap<int32_t, Vector2Bezier> &r_u, LocalVector<Vector2Bezier> p_curves);

	// Use Newton-Raphson iteration to find better root.
	// @param list curve:
	// @param Vector2Bezier point
	// @param Vector2Bezier u
	// @return New root point
	// @rtype Vector2Bezier
	Vector2Bezier find_root(LocalVector<Vector2Bezier> p_curves, Vector2Bezier p_curve, Vector2Bezier p_u);

	// Evaluate a bezier curve at a particular parameter value.
	// @param int degree:
	// @param list curve:
	// @param float t:
	// @return  Vector2Bezier point on curve
	Vector2Bezier evaluate(int32_t p_degree, LocalVector<Vector2Bezier> p_curves, Vector2Bezier p_t);

	// Assign parameter values to digitized points using relative distances
	// between points.
	// @param uint first:
	// @param uint last:
	// @return dictionary of chord length parameterization
	HashMap<int, Vector2Bezier> chord_length_parameterize(LocalVector<Bezier> p_curves, uint32_t p_first, uint32_t p_last);

	// Find the maximum squared distance of digitized points to fitted
	// curve.
	// @param uint first:
	// @param uint last:
	// @param list curve:
	// @param dict u:
	// @return tuple of Max distance and max index
	Vector2Bezier find_max_error(const LocalVector<Bezier> &p_existing_curves, uint32_t p_first, uint32_t p_last, LocalVector<Vector2Bezier> p_curves, HashMap<int, Vector2Bezier> p_u);

	real_t min_real_list(LocalVector<real_t> p_reals);

	real_t max_real_list(LocalVector<real_t> p_reals);

	real_t sum_real_list(LocalVector<real_t> p_reals);

	// The automatic tangent split will take the average of all values and
	// the average of just the minimum and maximum value and remaps that on
	// a logarithmic scale, this will give a predicted split angle value.
	// All angles will be processed to see if they fall in or outside that
	// threshold.
	// @param list of angles:
	// @return list of split indices
	LocalVector<int32_t> _find_tangent_split_auto(LocalVector<real_t> p_angles);

	// Loop existing frames and see if any keyframes contain tangents that
	// are not unified. If this is the case the index of the closest sampled
	// point will be returned.
	// @param list frames
	// @param int start
	// @param int end
	// @param int/float step
	// @return list of split indices
	LocalVector<int32_t> _find_tangent_split_existing(const LocalVector<Bezier> p_frames, uint32_t p_start, uint32_t p_end, real_t p_step);

	// The threshold tangent split will process all angles and check if that
	// angle falls in or outside of user provided threshold.
	// @param list angles:
	// @param int/float threshold:
	// @return list of split indices
	LocalVector<int32_t> _find_tangent_split_threshold(LocalVector<real_t> p_angles, real_t p_threshold);

	// Split provided points list based on the split indices provided. The
	// lists will have a duplicate end and start points relating to each
	// other.
	// @param list p_curves:
	// @param list p_split:
	// @return list of split bezier points
	LocalVector<Bezier> _split_points(const LocalVector<Bezier> &p_curves, LocalVector<uint32_t> &p_split);

	// Ported from Paper.js -
	// The Swiss Army Knife of Vector Graphics Scripting. http://paperjs.org/
	// Fit bezier curves to the points based on the provided maximum error
	// value and the bezier weighted tangents.
	// @return list of bezier segments
	LocalVector<Bezier> fit(FitState p_state);

	struct KeyframeTime {
		LocalVector<Vector2Bezier> points;
		LocalVector<real_t> angles;
	};

	LocalVector<real_t> get_values(LocalVector<Bezier> p_curves, LocalVector<double> p_frames);

	// Sample the current animation curve based on the start and end frame,
	// and the provided step size. Vector2Ds and angles will be returned.
	// @param int start:
	// @param int end:
	// @param int/float step:
	// @return list of sample points and angles
	KeyframeTime sample(const LocalVector<Bezier> p_curves, uint32_t p_start, uint32_t p_end, real_t p_step);

public:
	// Reduce the number of keyframes on the animation curve. Useful when
	// you are working with baked curves.
	// @param LocalVector<Bezier> r_points
	// @param LocalVector<Bezier> r_keyframes
	// @param KeyframeReductionSetting p_settings
	// @return float Reduction rate
	real_t reduce(const LocalVector<Bezier> &p_points, LocalVector<Bezier> &r_keyframes, KeyframeReductionSetting p_settings);
};
#endif // KEYFRAME_REDUCE_H
