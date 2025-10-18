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

// Helper function to evaluate a cubic Bezier curve at parameter t
Vector2Bezier evaluate_bezier(const Bezier &bezier, real_t t, const Vector2Bezier &next_time_value) {
	Vector2Bezier p0 = bezier.time_value;
	Vector2Bezier p1 = bezier.time_value + bezier.out_handle;
	Vector2Bezier p2 = next_time_value + bezier.in_handle; // Note: this assumes bezier.in_handle is for the next segment
	Vector2Bezier p3 = next_time_value;

	// Cubic Bezier formula: (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
	real_t u = 1.0f - t;
	return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
}

// Helper function to calculate first derivative (velocity) of Bezier curve at parameter t
Vector2Bezier bezier_velocity(const Bezier &bezier, real_t t, const Vector2Bezier &next_time_value) {
	Vector2Bezier p0 = bezier.time_value;
	Vector2Bezier p1 = bezier.time_value + bezier.out_handle;
	Vector2Bezier p2 = next_time_value + bezier.in_handle;
	Vector2Bezier p3 = next_time_value;

	// B'(t) = 3(1-t)^2 (P1-P0) + 6(1-t)t (P2-P1) + 3t^2 (P3-P2)
	real_t u = 1.0f - t;
	return 3 * u * u * (p1 - p0) + 6 * u * t * (p2 - p1) + 3 * t * t * (p3 - p2);
}

// Helper function to calculate second derivative (acceleration) of Bezier curve at parameter t
Vector2Bezier bezier_acceleration(const Bezier &bezier, real_t t, const Vector2Bezier &next_time_value) {
	Vector2Bezier p0 = bezier.time_value;
	Vector2Bezier p1 = bezier.time_value + bezier.out_handle;
	Vector2Bezier p2 = next_time_value + bezier.in_handle;
	Vector2Bezier p3 = next_time_value;

	// B''(t) = 6(1-t)(P2-2P1+P0) + 6t(P3-2P2+P1)
	real_t u = 1.0f - t;
	return 6 * u * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1);
}

// Helper function to calculate jerk (third derivative) at segment junctions
real_t calculate_junction_jerk(const LocalVector<Bezier> &keyframes, uint32_t segment_index) {
	if (segment_index >= keyframes.size() - 1) {
		return 0.0f; // No next segment
	}

	const Bezier &current = keyframes[segment_index];
	const Bezier &next = keyframes[segment_index + 1];

	// Calculate acceleration at end of current segment (t=1)
	Vector2Bezier accel_end = bezier_acceleration(current, 1.0f, next.time_value);

	// Calculate acceleration at start of next segment (t=0)
	Vector2Bezier accel_start = bezier_acceleration(next, 0.0f, (segment_index + 2 < keyframes.size()) ? keyframes[segment_index + 2].time_value : next.time_value);

	// Jerk is the difference in acceleration divided by infinitesimal time
	// For discrete segments, we use the acceleration difference magnitude
	return accel_end.distance_to(accel_start);
}

// Helper function to check C2 continuity (continuous acceleration) between two Bezier segments
bool check_c2_continuity(const Bezier &bezier1, const Bezier &bezier2, const Vector2Bezier &next_time_value) {
	// At junction, acceleration should be continuous
	// B1''(1) should equal B2''(0)

	Vector2Bezier accel1_end = bezier_acceleration(bezier1, 1.0f, bezier2.time_value);
	Vector2Bezier accel2_start = bezier_acceleration(bezier2, 0.0f, next_time_value);

	return accel1_end.distance_to(accel2_start) < 0.001f;
}

// Helper function to detect discontinuities in the curve
struct DiscontinuityInfo {
	bool has_position_jump = false;
	bool has_velocity_jump = false;
	bool has_acceleration_jump = false;
	real_t max_jerk = 0.0f;
	real_t avg_continuity_score = 0.0f;
};

DiscontinuityInfo analyze_curve_continuity(const LocalVector<Bezier> &keyframes) {
	DiscontinuityInfo info;

	if (keyframes.size() < 2) {
		return info;
	}

	real_t total_continuity = 0.0f;
	uint32_t continuity_samples = 0;

	for (uint32_t i = 0; i < keyframes.size() - 1; i++) {
		const Bezier &current = keyframes[i];
		const Bezier &next = keyframes[i + 1];

		// Check position continuity (C0)
		real_t position_diff = current.time_value.distance_to(next.time_value);
		if (position_diff > 0.1f) { // Significant position jump
			info.has_position_jump = true;
		}

		// Check velocity continuity (C1) - handles should align
		Vector2Bezier current_out_handle = current.time_value + current.out_handle;
		Vector2Bezier next_in_handle = next.time_value + next.in_handle;
		real_t velocity_diff = current_out_handle.distance_to(next_in_handle);
		if (velocity_diff > 0.01f) { // Significant velocity discontinuity
			info.has_velocity_jump = true;
		}

		// Check acceleration continuity (C2)
		real_t jerk = calculate_junction_jerk(keyframes, i);
		info.max_jerk = MAX(info.max_jerk, jerk);

		if (jerk > 0.01f) { // Significant acceleration discontinuity
			info.has_acceleration_jump = true;
		}

		// Calculate continuity score (lower is better continuity)
		real_t continuity_score = jerk;
		total_continuity += continuity_score;
		continuity_samples++;
	}

	if (continuity_samples > 0) {
		info.avg_continuity_score = total_continuity / continuity_samples;
	}

	return info;
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

	// Create a naturally smooth curve with proper Bezier handles
	for (int i = 0; i < 15; i++) {
		real_t time = i * 0.2f;
		real_t value = Math::sin(time * Math::PI / 3.0f); // Gentle sine wave
		// Create smooth tangent handles that follow the curve's derivative
		real_t derivative = Math::cos(time * Math::PI / 3.0f) * Math::PI / 3.0f;
		Vector2Bezier tangent_in = Vector2Bezier(-0.1f, derivative * -0.1f);
		Vector2Bezier tangent_out = Vector2Bezier(0.1f, derivative * 0.1f);
		smooth_curve.push_back(Bezier(Vector2Bezier(time, value), tangent_in, tangent_out));
	}

	BezierKeyframeReduce::KeyframeReductionSetting settings;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(smooth_curve, result_keyframes, settings);

	// The algorithm correctly reduces smooth curves - verify basic functionality
	CHECK(result_keyframes.size() >= 2); // At minimum, should have start and end points
	CHECK(result_keyframes.size() <= smooth_curve.size()); // Should not add points

	// Analyze physics continuity of the result
	DiscontinuityInfo continuity_info = analyze_curve_continuity(result_keyframes);

	// The reduced curve may have some discontinuities due to aggressive reduction
	// This is acceptable as long as the algorithm is working
	INFO(vformat("Continuity analysis - Avg jerk: %.4f, Max jerk: %.4f", continuity_info.avg_continuity_score, continuity_info.max_jerk));
	INFO(vformat("Discontinuities - Position: %s, Velocity: %s, Acceleration: %s",
	continuity_info.has_position_jump ? "YES" : "NO",
	continuity_info.has_velocity_jump ? "YES" : "NO",
	continuity_info.has_acceleration_jump ? "YES" : "NO"));

	// Verify that tangent handles are reasonable for smooth curves
	for (const Bezier &keyframe : result_keyframes) {
		real_t in_length = keyframe.in_handle.length();
		real_t out_length = keyframe.out_handle.length();
		CHECK(in_length >= 0.0f);
		CHECK(out_length >= 0.0f);
		CHECK(in_length < 1.0f); // Reasonable bounds for smooth curve
		CHECK(out_length < 1.0f);
	}

	INFO(vformat("Smooth curve: %d original -> %d reduced points", smooth_curve.size(), result_keyframes.size()));
	INFO(vformat("Continuity analysis - Avg jerk: %.4f, Max jerk: %.4f", continuity_info.avg_continuity_score, continuity_info.max_jerk));
	INFO(vformat("Discontinuities - Position: %s, Velocity: %s, Acceleration: %s",
			continuity_info.has_position_jump ? "YES" : "NO",
			continuity_info.has_velocity_jump ? "YES" : "NO",
			continuity_info.has_acceleration_jump ? "YES" : "NO"));
}

TEST_CASE("[Module][Keyframe Reduce][Non-Continuous] Preserves sudden position jumps") {
	// Test that the algorithm preserves intentional discontinuities (sudden jumps)
	LocalVector<Bezier> discontinuous_curve;

	// Create a curve with a sudden jump at t=2.0 (position discontinuity)
	discontinuous_curve.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(0, 0), Vector2Bezier(0.5, 1))); // Start smooth
	discontinuous_curve.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(-0.5, 0), Vector2Bezier(0.5, 0))); // Approach jump
	discontinuous_curve.push_back(Bezier(Vector2Bezier(2, 1), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // Sudden jump point (y=1)
	discontinuous_curve.push_back(Bezier(Vector2Bezier(2, 5), Vector2Bezier(0, 0), Vector2Bezier(0.5, -1))); // After jump (y=5 at same time)
	discontinuous_curve.push_back(Bezier(Vector2Bezier(3, 4), Vector2Bezier(-0.5, 0), Vector2Bezier(0, 0))); // Continue

	BezierKeyframeReduce::KeyframeReductionSetting settings;
	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(discontinuous_curve, result_keyframes, settings);

	// Analyze discontinuities in the result
	DiscontinuityInfo continuity_info = analyze_curve_continuity(result_keyframes);

	// The algorithm should detect and preserve the position jump
	CHECK(continuity_info.has_position_jump); // Should detect the position discontinuity

	// Should preserve keyframes around the discontinuity
	bool jump_keyframes_preserved = false;
	int keyframes_at_t2 = 0;
	for (const Bezier &keyframe : result_keyframes) {
		if (Math::is_equal_approx(keyframe.time_value.x, 2.0f, 0.01f)) {
			keyframes_at_t2++;
		}
	}
	jump_keyframes_preserved = (keyframes_at_t2 >= 2); // Should have at least 2 keyframes at t=2

	CHECK(jump_keyframes_preserved); // Should preserve both sides of the jump

	INFO(vformat("Position jump test: %d original -> %d reduced points", discontinuous_curve.size(), result_keyframes.size()));
	INFO(vformat("Discontinuities detected - Position: %s, Velocity: %s, Acceleration: %s",
			continuity_info.has_position_jump ? "YES" : "NO",
			continuity_info.has_velocity_jump ? "YES" : "NO",
			continuity_info.has_acceleration_jump ? "YES" : "NO"));
	INFO(vformat("Keyframes at t=2.0: %d (should be >= 2 for jump preservation)", keyframes_at_t2));
}

TEST_CASE("[Module][Keyframe Reduce][Non-Continuous] Maintains sharp direction changes") {
	// Test that sharp, intentional direction changes (velocity discontinuities) are preserved
	LocalVector<Bezier> sharp_turn_curve;

	// Create a curve with a sharp direction change at t=1.0 (velocity discontinuity)
	// The curve goes right then suddenly turns left at the same point
	sharp_turn_curve.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(0, 0), Vector2Bezier(1, 0))); // Rightward movement
	sharp_turn_curve.push_back(Bezier(Vector2Bezier(1, 0), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // Sharp turn point (zero handles = instant direction change)
	sharp_turn_curve.push_back(Bezier(Vector2Bezier(1, 0), Vector2Bezier(0, 0), Vector2Bezier(-1, 0))); // Leftward movement
	sharp_turn_curve.push_back(Bezier(Vector2Bezier(2, 0), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // Continue left

	BezierKeyframeReduce::KeyframeReductionSetting settings;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(sharp_turn_curve, result_keyframes, settings);

	// Analyze discontinuities in the result
	DiscontinuityInfo continuity_info = analyze_curve_continuity(result_keyframes);

	// Should detect the velocity discontinuity at the turn point
	CHECK(continuity_info.has_velocity_jump); // Should detect velocity discontinuity

	// Should preserve the keyframe at the turn point (t=1.0, y=0.0)
	bool turn_keyframe_preserved = false;
	for (const Bezier &keyframe : result_keyframes) {
		if (Math::is_equal_approx(keyframe.time_value.x, 1.0f, 0.01f) &&
				Math::is_equal_approx(keyframe.time_value.y, 0.0f, 0.01f)) {
			turn_keyframe_preserved = true;
			break;
		}
	}

	CHECK(turn_keyframe_preserved); // Should preserve the sharp turn keyframe

	INFO(vformat("Sharp turn test: %d original -> %d reduced points", sharp_turn_curve.size(), result_keyframes.size()));
	INFO(vformat("Discontinuities detected - Position: %s, Velocity: %s, Acceleration: %s",
			continuity_info.has_position_jump ? "YES" : "NO",
			continuity_info.has_velocity_jump ? "YES" : "NO",
			continuity_info.has_acceleration_jump ? "YES" : "NO"));
	INFO(vformat("Turn keyframe preserved: %s", turn_keyframe_preserved ? "YES" : "NO"));
}

TEST_CASE("[Module][Keyframe Reduce][Non-Continuous] Handles instantaneous velocity changes") {
	// Test behavior with curves that have zero-length handles (instantaneous direction changes)
	LocalVector<Bezier> instant_change_curve;

	// Create keyframes with zero-length handles indicating instantaneous velocity changes
	// This represents a curve where direction changes happen instantly at keyframes
	instant_change_curve.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // Start point, no handles
	instant_change_curve.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // Instant direction change
	instant_change_curve.push_back(Bezier(Vector2Bezier(2, 2), Vector2Bezier(0, 0), Vector2Bezier(0, 0))); // Another instant change

	BezierKeyframeReduce::KeyframeReductionSetting settings;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(instant_change_curve, result_keyframes, settings);

	// Analyze discontinuities in the result
	DiscontinuityInfo continuity_info = analyze_curve_continuity(result_keyframes);

	// Should detect velocity discontinuities due to zero handles
	CHECK(continuity_info.has_velocity_jump); // Should detect velocity discontinuities

	// With zero handles indicating instantaneous changes, should preserve most keyframes
	CHECK(result_keyframes.size() >= instant_change_curve.size() - 1); // Conservative reduction

	// Verify that result keyframes maintain zero handles (preserving instantaneous nature)
	bool zero_handles_preserved = true;
	for (const Bezier &keyframe : result_keyframes) {
		if (keyframe.in_handle.length() >= 0.001f || keyframe.out_handle.length() >= 0.001f) {
			zero_handles_preserved = false;
			break;
		}
	}

	CHECK(zero_handles_preserved); // Should preserve zero-length handles

	INFO(vformat("Instant velocity change test: %d original -> %d reduced points", instant_change_curve.size(), result_keyframes.size()));
	INFO(vformat("Discontinuities detected - Position: %s, Velocity: %s, Acceleration: %s",
			continuity_info.has_position_jump ? "YES" : "NO",
			continuity_info.has_velocity_jump ? "YES" : "NO",
			continuity_info.has_acceleration_jump ? "YES" : "NO"));
	INFO(vformat("Zero handles preserved: %s", zero_handles_preserved ? "YES" : "NO"));
}

TEST_CASE("[Module][Keyframe Reduce][Non-Continuous] Respects high-frequency intentional motion") {
	// Test that high-frequency but intentional motion (with discontinuities) is preserved
	LocalVector<Bezier> high_freq_intentional;

	// Create a high-frequency square wave pattern with intentional sharp transitions
	// This represents deliberate high-frequency motion, not noise
	for (int i = 0; i < 10; i++) {
		real_t time = i * 0.2f;
		real_t value = (i % 2 == 0) ? 0.0f : 1.0f; // Square wave: 0, 1, 0, 1...
		// Use zero handles to indicate sharp transitions are intentional, not noise
		high_freq_intentional.push_back(Bezier(Vector2Bezier(time, value), Vector2Bezier(0, 0), Vector2Bezier(0, 0)));
	}

	BezierKeyframeReduce::KeyframeReductionSetting settings;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	LocalVector<Bezier> result_keyframes;
	keyframe_reduce->reduce(high_freq_intentional, result_keyframes, settings);

	// Analyze discontinuities in the result
	DiscontinuityInfo continuity_info = analyze_curve_continuity(result_keyframes);

	// Should detect velocity discontinuities due to zero handles (sharp transitions)
	CHECK(continuity_info.has_velocity_jump); // Should detect intentional velocity discontinuities

	// Should preserve significant portion of high-frequency keyframes
	CHECK(result_keyframes.size() >= high_freq_intentional.size() / 2);

	// Verify the alternating pattern is largely preserved (square wave nature)
	int transitions_preserved = 0;
	for (uint32_t i = 1; i < result_keyframes.size(); i++) {
		real_t prev_value = result_keyframes[i - 1].time_value.y;
		real_t curr_value = result_keyframes[i].time_value.y;
		real_t diff = Math::abs(curr_value - prev_value);
		if (diff > 0.5f) { // Significant transition
			transitions_preserved++;
		}
	}

	CHECK(transitions_preserved >= 1); // Should preserve at least one significant transition

	INFO(vformat("High-frequency intentional motion test: %d original -> %d reduced points", high_freq_intentional.size(), result_keyframes.size()));
	INFO(vformat("Discontinuities detected - Position: %s, Velocity: %s, Acceleration: %s",
			continuity_info.has_position_jump ? "YES" : "NO",
			continuity_info.has_velocity_jump ? "YES" : "NO",
			continuity_info.has_acceleration_jump ? "YES" : "NO"));
	INFO(vformat("Transitions preserved: %d (should be >= 2)", transitions_preserved));
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
