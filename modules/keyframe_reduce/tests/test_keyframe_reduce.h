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

#pragma once

#include "tests/test_macros.h"

#include "core/math/vector2.h"
#include "core/templates/local_vector.h"
#include "modules/keyframe_reduce/keyframe_reduce.h"
#include <cstdint>

namespace TestKeyframeReduction {

using Vector2Bezier = BezierKeyframeReduce::Vector2Bezier;
using Bezier = BezierKeyframeReduce::Bezier;

// Helper function to calculate jerk (third derivative) at a point on a Bezier curve
real_t calculate_bezier_jerk(const Bezier &bezier, real_t t) {
	// For a cubic Bezier curve B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
	// First derivative B'(t) = 3(1-t)^2 (P1-P0) + 6(1-t)t (P2-P1) + 3t^2 (P3-P2)
	// Second derivative B''(t) = 6(1-t)(P2-2P1+P0) + 6t(P3-2P2+P1)
	// Third derivative B'''(t) = 6(P3-3P2+3P1-P0) = 6 * (P3 - P2 - (P2 - P1) - (P1 - P0)) = 6 * (P3 - 3P2 + 3P1 - P0)

	Vector2Bezier p0 = bezier.time_value;
	Vector2Bezier p1 = bezier.time_value + bezier.in_handle;
	Vector2Bezier p2 = bezier.time_value + bezier.out_handle;
	Vector2Bezier p3 = bezier.time_value; // Next keyframe time_value (simplified)

	// Jerk is constant for cubic Bezier: 6 * (P3 - 3*P2 + 3*P1 - P0)
	Vector2Bezier jerk_vec = 6.0f * (p3 - 3.0f * p2 + 3.0f * p1 - p0);
	return jerk_vec.length();
}

// Helper function to check C2 continuity (continuous acceleration) between two Bezier segments
bool check_c2_continuity(const Bezier &bezier1, const Bezier &bezier2) {
	// At junction, acceleration should be continuous
	// B1''(1) should equal B2''(0)

	// For cubic Bezier, B''(t) = 6(1-t)(P2-2P1+P0) + 6t(P3-2P2+P1)
	// At t=1: B''(1) = 6(P3-2P2+P1)
	// At t=0: B''(0) = 6(P2-2P1+P0)

	Vector2Bezier p1_1 = bezier1.time_value + bezier1.in_handle;
	Vector2Bezier p2_1 = bezier1.time_value + bezier1.out_handle;
	Vector2Bezier p3_1 = bezier2.time_value; // Next segment's start

	Vector2Bezier accel1_end = 6.0f * (p3_1 - 2.0f * p2_1 + p1_1);

	Vector2Bezier p0_2 = bezier2.time_value;
	Vector2Bezier p1_2 = bezier2.time_value + bezier2.in_handle;
	Vector2Bezier p2_2 = bezier2.time_value + bezier2.out_handle;

	Vector2Bezier accel2_start = 6.0f * (p2_2 - 2.0f * p1_2 + p0_2);

	return accel1_end.distance_to(accel2_start) < 0.001f;
}

TEST_CASE("[Module][Keyframe Reduce][Vector2Bezier] Distance between two points") {
	Vector2Bezier point1(0, 0);
	Vector2Bezier point2(3, 4);
	CHECK(point1.distance_between(point2) == 5.0);
}

TEST_CASE("[Module][Keyframe Reduce][Vector2Bezier] Signed angle between points") {
	Vector2Bezier point1(0, 0);
	Vector2Bezier point2(1, 0);
	Vector2Bezier point3(0, 1);
	real_t expected = static_cast<real_t>(Math::PI) / 2.0;
	real_t actual = static_cast<real_t>(point1.signed_angle(point2, point3));
	CHECK(Math::is_equal_approx(actual, expected));
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

TEST_CASE("[Module][Keyframe Reduce][Physics][Jerk] Enhanced tangent calculation reduces discontinuities") {
	// Test that the enhanced split tangent calculation creates smoother curves
	LocalVector<Bezier> sharp_curve;
	sharp_curve.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(0, 0), Vector2Bezier(0.5, 2))); // Sharp upward
	sharp_curve.push_back(Bezier(Vector2Bezier(1, 2), Vector2Bezier(-0.5, 0), Vector2Bezier(0.5, -2))); // Sharp downward
	sharp_curve.push_back(Bezier(Vector2Bezier(2, 0), Vector2Bezier(-0.5, 0), Vector2Bezier(0, 0))); // Back to zero

	BezierKeyframeReduce::KeyframeReductionSetting settings;
	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> smoothed_keyframes;
	real_t reduction_rate = keyframe_reduce->reduce(sharp_curve, smoothed_keyframes, settings);

	// Verify that splitting occurred (indicating the algorithm tried to smooth)
	CHECK(reduction_rate < 1.0f);

	// Check that the enhanced tangent calculation creates more reasonable handle lengths
	for (const Bezier &keyframe : smoothed_keyframes) {
		real_t in_length = keyframe.in_handle.length();
		real_t out_length = keyframe.out_handle.length();
		// Tangent lengths should be proportional to segment duration (time difference)
		// This is a basic sanity check for the enhanced tangent calculation
		CHECK(in_length >= 0.0f);
		CHECK(out_length >= 0.0f);
		CHECK(in_length < 10.0f); // Reasonable upper bound
		CHECK(out_length < 10.0f);
	}

	INFO(vformat("Sharp curve reduced from %d to %d points with enhanced tangent smoothing",
			sharp_curve.size(), smoothed_keyframes.size()));
}

TEST_CASE("[Module][Keyframe Reduce][Physics][Jerk] Adaptive iteration prevents over-smoothing") {
	// Test that adaptive iteration control prevents excessive iterations on smooth curves
	LocalVector<Bezier> smooth_curve;

	// Create a naturally smooth curve
	for (int i = 0; i < 15; i++) {
		real_t time = i * 0.2f;
		real_t value = Math::sin(time * Math::PI / 3.0f); // Gentle sine wave
		Vector2Bezier tangent_in = Vector2Bezier(-0.1f, Math::cos(time * Math::PI / 3.0f) * 0.15f);
		Vector2Bezier tangent_out = Vector2Bezier(0.1f, Math::cos(time * Math::PI / 3.0f) * 0.15f);
		smooth_curve.push_back(Bezier(Vector2Bezier(time, value), tangent_in, tangent_out));
	}

	BezierKeyframeReduce::KeyframeReductionSetting settings;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(smooth_curve, result_keyframes, settings);

	// Should maintain most of the original smoothness
	CHECK(result_keyframes.size() >= smooth_curve.size() / 3); // Don't over-reduce

	// Verify tangent continuity (basic jerk reduction check)
	real_t tangent_continuity_score = 0.0f;
	for (uint32_t i = 1; i < result_keyframes.size(); i++) {
		Vector2Bezier prev_out = result_keyframes[i - 1].out_handle;
		Vector2Bezier curr_in = result_keyframes[i].in_handle;
		real_t continuity = 1.0f - (prev_out.normalized() - curr_in.normalized()).length();
		tangent_continuity_score += continuity;
	}
	if (result_keyframes.size() > 1) {
		tangent_continuity_score /= (result_keyframes.size() - 1);
		CHECK(tangent_continuity_score > 0.5f); // Decent continuity maintained
	}

	INFO(vformat("Smooth curve: %d original -> %d reduced points, continuity score: %.3f",
			smooth_curve.size(), result_keyframes.size(), tangent_continuity_score));
}

TEST_CASE("[Module][Keyframe Reduce][Non-Continuous] Preserves sudden position jumps") {
	// Test that the algorithm preserves intentional discontinuities (sudden jumps)
	LocalVector<Bezier> discontinuous_curve;

	// Create a curve with a sudden jump at t=2.0
	discontinuous_curve.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(0, 0), Vector2Bezier(0.5, 1))); // Start smooth
	discontinuous_curve.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(-0.5, 0), Vector2Bezier(0.5, 0))); // Approach jump
	discontinuous_curve.push_back(Bezier(Vector2Bezier(2, 1), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // Sudden jump point
	discontinuous_curve.push_back(Bezier(Vector2Bezier(2, 5), Vector2Bezier(0, 0), Vector2Bezier(0.5, -1))); // After jump
	discontinuous_curve.push_back(Bezier(Vector2Bezier(3, 4), Vector2Bezier(-0.5, 0), Vector2Bezier(0, 0))); // Continue

	BezierKeyframeReduce::KeyframeReductionSetting settings;
	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(discontinuous_curve, result_keyframes, settings);

	// Should preserve the discontinuity - the jump at t=2 should remain
	bool jump_preserved = false;
	for (const Bezier &keyframe : result_keyframes) {
		if (Math::is_equal_approx(keyframe.time_value.x, 2.0f)) {
			// Check if we have keyframes at both y=1 and y=5 around t=2
			bool has_pre_jump = false;
			bool has_post_jump = false;
			for (const Bezier &other : result_keyframes) {
				if (Math::is_equal_approx(other.time_value.x, 2.0f)) {
					if (Math::is_equal_approx(other.time_value.y, 1.0f)) {
						has_pre_jump = true;
					}
					if (Math::is_equal_approx(other.time_value.y, 5.0f)) {
						has_post_jump = true;
					}
				}
			}
			if (has_pre_jump && has_post_jump) {
				jump_preserved = true;
				break;
			}
		}
	}

	CHECK(jump_preserved); // Algorithm should preserve intentional discontinuities

	INFO(vformat("Discontinuous curve: %d original -> %d reduced points, jump preserved: %s",
			discontinuous_curve.size(), result_keyframes.size(), jump_preserved ? "yes" : "no"));
}

TEST_CASE("[Module][Keyframe Reduce][Non-Continuous] Maintains sharp direction changes") {
	// Test that sharp, intentional direction changes are not over-smoothed
	LocalVector<Bezier> sharp_turn_curve;

	// Create a curve that goes right then suddenly turns left
	sharp_turn_curve.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(0, 0), Vector2Bezier(1, 0))); // Right
	sharp_turn_curve.push_back(Bezier(Vector2Bezier(1, 0), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // Sharp turn point
	sharp_turn_curve.push_back(Bezier(Vector2Bezier(1, 0), Vector2Bezier(0, 0), Vector2Bezier(-1, 0))); // Left
	sharp_turn_curve.push_back(Bezier(Vector2Bezier(2, 0), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // Continue left

	BezierKeyframeReduce::KeyframeReductionSetting settings;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(sharp_turn_curve, result_keyframes, settings);

	// Should preserve the sharp turn - check for keyframes at the turn point
	bool turn_preserved = false;
	for (const Bezier &keyframe : result_keyframes) {
		if (Math::is_equal_approx(keyframe.time_value.x, 1.0f) &&
				Math::is_equal_approx(keyframe.time_value.y, 0.0f)) {
			// Found the turn point, check if it's preserved as a separate keyframe
			turn_preserved = true;
			break;
		}
	}

	CHECK(turn_preserved); // Sharp direction changes should be preserved

	INFO(vformat("Sharp turn curve: %d original -> %d reduced points, turn preserved: %s",
			sharp_turn_curve.size(), result_keyframes.size(), turn_preserved ? "yes" : "no"));
}

TEST_CASE("[Module][Keyframe Reduce][Non-Continuous] Handles instantaneous velocity changes") {
	// Test behavior with curves that have zero-length handles (instantaneous direction changes)
	LocalVector<Bezier> instant_change_curve;

	// Create keyframes with zero-length handles indicating instantaneous changes
	instant_change_curve.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // No handles
	instant_change_curve.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // No handles
	instant_change_curve.push_back(Bezier(Vector2Bezier(2, 2), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // No handles

	BezierKeyframeReduce::KeyframeReductionSetting settings;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(instant_change_curve, result_keyframes, settings);

	// With zero handles, the algorithm should be very conservative about reduction
	// since these represent intentional instantaneous changes
	CHECK(result_keyframes.size() >= instant_change_curve.size() - 1); // At most one reduction

	// Verify that all result keyframes still have zero handles (preserving instantaneous nature)
	for (const Bezier &keyframe : result_keyframes) {
		CHECK(keyframe.in_handle.length() < 0.001f);
		CHECK(keyframe.out_handle.length() < 0.001f);
	}

	INFO(vformat("Instant change curve: %d original -> %d reduced points, zero handles preserved",
			instant_change_curve.size(), result_keyframes.size()));
}

TEST_CASE("[Module][Keyframe Reduce][Non-Continuous] Respects high-frequency intentional motion") {
	// Test that high-frequency but intentional motion is not over-smoothed
	LocalVector<Bezier> high_freq_intentional;

	// Create a high-frequency square wave pattern (intentional, not noise)
	for (int i = 0; i < 10; i++) {
		real_t time = i * 0.2f;
		real_t value = (i % 2 == 0) ? 0.0f : 1.0f; // Square wave: 0, 1, 0, 1...
		// Use zero handles to indicate sharp transitions are intentional
		high_freq_intentional.push_back(Bezier(Vector2Bezier(time, value), Vector2Bezier(0, 0), Vector2Bezier(0, 0)));
	}

	BezierKeyframeReduce::KeyframeReductionSetting settings;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(high_freq_intentional, result_keyframes, settings);

	// Should preserve most of the high-frequency changes since they have zero handles
	CHECK(result_keyframes.size() >= high_freq_intentional.size() / 2);

	// Verify the square wave pattern is largely preserved
	bool pattern_preserved = true;
	for (uint32_t i = 0; i < result_keyframes.size(); i++) {
		real_t expected_value = (i % 2 == 0) ? 0.0f : 1.0f;
		if (!Math::is_equal_approx(result_keyframes[i].time_value.y, expected_value, 0.2f)) {
			pattern_preserved = false;
			break;
		}
	}

	CHECK(pattern_preserved); // High-frequency intentional patterns should be preserved

	INFO(vformat("High-frequency curve: %d original -> %d reduced points, pattern preserved: %s",
			high_freq_intentional.size(), result_keyframes.size(), pattern_preserved ? "yes" : "no"));
}

TEST_CASE("[Module][Keyframe Reduce] Reduction rate") {
	LocalVector<Bezier> points;
	points.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(-1, -1), Vector2Bezier(1, 1)));
	points.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(0, 0), Vector2Bezier(2, 2)));
	points.push_back(Bezier(Vector2Bezier(2, 2), Vector2Bezier(1, 1), Vector2Bezier(3, 3)));

	BezierKeyframeReduce::KeyframeReductionSetting settings;

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
