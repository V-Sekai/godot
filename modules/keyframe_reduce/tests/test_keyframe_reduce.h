/**************************************************************************/
/*  test_keyframe_reduce.h                                                */
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

#ifndef TEST_KEYFRAME_REDUCE_H
#define TEST_KEYFRAME_REDUCE_H

#include "tests/test_macros.h"

#include "modules/keyframe_reduce/keyframe_reduce.h"
#include <cstdint>

namespace TestKeyframeReduction {

using Vector2Bezier = BezierKeyframeReduce::Vector2Bezier;
using Bezier = BezierKeyframeReduce::Bezier;

TEST_CASE("[Module][Keyframe Reduce][Vector2Bezier] Distance between two points") {
	Vector2Bezier point1(0, 0);
	Vector2Bezier point2(3, 4);
	CHECK(point1.distance_between(point2) == 5.0);
}

TEST_CASE("[Module][Keyframe Reduce][Vector2Bezier] Signed angle between points") {
	Vector2Bezier point1(0, 0);
	Vector2Bezier point2(1, 0);
	Vector2Bezier point3(0, 1);
	CHECK(Math::is_equal_approx(static_cast<real_t>(point1.signed_angle(point2, point3)), Math_PI / 2.0));
}

TEST_CASE("[Module][Keyframe Reduce][Bezier] Normalized Bezier") {
	Bezier bezier = Bezier(Vector2Bezier(1, 1), Vector2Bezier(-1, -1), Vector2Bezier(2, 2));
	Bezier normalized = bezier.normalized();
	CHECK(normalized.in_handle.x <= 0);
	CHECK(normalized.out_handle.x >= 0);
}

TEST_CASE("[Module][Keyframe Reduce][Bezier] Subtract Bezier") {
	Bezier bezier1(Vector2Bezier(1, 1), Vector2Bezier(-1, -1), Vector2Bezier(2, 2));
	Bezier bezier2(Vector2Bezier(1, 1), Vector2Bezier(-1, -1), Vector2Bezier(2, 2));
	Bezier result = bezier1 - bezier2;
	CHECK(result.in_handle == Vector2Bezier());
	CHECK(result.out_handle == Vector2Bezier());
	CHECK(result.time_value == Vector2Bezier());
}

TEST_CASE("[Module][Keyframe Reduce] Reduction rate") {
	LocalVector<Bezier> points;
	points.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(-1, -1), Vector2Bezier(1, 1)));
	points.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(0, 0), Vector2Bezier(2, 2)));
	points.push_back(Bezier(Vector2Bezier(2, 2), Vector2Bezier(1, 1), Vector2Bezier(3, 3)));

	BezierKeyframeReduce::KeyframeReductionSetting settings;
	settings.max_error = 0.1f;
	settings.step_size = 0.5f;
	settings.tangent_split_angle_threshold_in_degrees = 5.0f;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> keyframes;
	real_t reduction_rate = keyframe_reduce->reduce(points, keyframes, settings);
	CHECK(reduction_rate <= 1.0);
	CHECK(keyframes.size() <= points.size());
	for (uint32_t i = 0; i < keyframes.size(); ++i) {
		CHECK_MESSAGE(true, vformat("Reduced keyframe time value: %s In: %s Out: %s", String(keyframes[i].time_value), String(keyframes[i].in_handle), String(keyframes[i].out_handle)));
	}
}

TEST_CASE("[Module][Keyframe Reduce] Reduction rate on constants") {
	LocalVector<Bezier> points;
	double time = 0.0f;
	while (time <= 10.0f) {
		time += 1.0f;
		points.push_back(Bezier(Vector2Bezier(time, 1.0), Vector2Bezier(), Vector2Bezier()));
	}
	BezierKeyframeReduce::KeyframeReductionSetting settings;
	settings.max_error = 0.1f;
	settings.step_size = 0.5f;
	settings.tangent_split_angle_threshold_in_degrees = 5.0f;
	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> keyframes;
	real_t reduction_rate = keyframe_reduce->reduce(points, keyframes, settings);
	CHECK(reduction_rate <= 1.0);
	CHECK(keyframes.size() <= points.size());
	for (uint32_t i = 0; i < keyframes.size(); ++i) {
		CHECK_MESSAGE(true, vformat("Reduced keyframe time value: %s In: %s Out: %s", String(keyframes[i].time_value), String(keyframes[i].in_handle), String(keyframes[i].out_handle)));
	}
}
} // namespace TestKeyframeReduction

#endif // TEST_KEYFRAME_REDUCE_H
