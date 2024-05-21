#ifndef TEST_VECTOR2BEZIER_H
#define TEST_VECTOR2BEZIER_H

#include "tests/test_macros.h"

#include "modules/keyframe_reduce/keyframe_reduce.h"

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
	CHECK(point1.signed_angle(point2, point3) == Math_PI / 2.0);
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
	Vector<Bezier> points;
	points.push_back(Bezier(Vector2Bezier(0, 0), Vector2Bezier(-1, -1), Vector2Bezier(1, 1)));
	points.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(0, 0), Vector2Bezier(2, 2)));
	points.push_back(Bezier(Vector2Bezier(2, 2), Vector2Bezier(1, 1), Vector2Bezier(3, 3)));

	BezierKeyframeReduce::KeyframeReductionSetting settings;
	settings.max_error = 0.1f;
	settings.step_size = 0.5f;
	settings.tangent_split_angle_threshold_in_degrees = 5.0f;

	Ref<BezierKeyframeReduce> keyframe_reduce;
	keyframe_reduce.instantiate();
	Vector<Bezier> keyframes;
	real_t reduction_rate = keyframe_reduce->reduce(points, keyframes, settings);
	CHECK(reduction_rate <= 1.0);
	CHECK(keyframes.size() <= points.size());
}

TEST_CASE("[Module][Keyframe Reduce] Reduction rate on constants") {
    // Create a vector of Bezier curves with constant values
    Vector<Bezier> points;
    points.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(1, 1), Vector2Bezier(1, 1)));
    points.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(1, 1), Vector2Bezier(1, 1)));
    points.push_back(Bezier(Vector2Bezier(1, 1), Vector2Bezier(1, 1), Vector2Bezier(1, 1)));

    // Define the keyframe reduction settings
    BezierKeyframeReduce::KeyframeReductionSetting settings;
    settings.max_error = 0.1f;
    settings.step_size = 0.5f;
    settings.tangent_split_angle_threshold_in_degrees = 5.0f;

    // Instantiate a BezierKeyframeReduce object
    Ref<BezierKeyframeReduce> keyframe_reduce;
    keyframe_reduce.instantiate();

    // Create a vector to hold the reduced keyframes
    Vector<Bezier> keyframes;

    // Perform the reduction and get the reduction rate
    real_t reduction_rate = keyframe_reduce->reduce(points, keyframes, settings);

    // Check that the reduction rate is less than or equal to 1.0
    CHECK(reduction_rate <= 1.0);

    // Check that the size of the reduced keyframes is less than or equal to the original points size
    CHECK(keyframes.size() <= points.size());
}
} // namespace TestKeyframeReduction

#endif // TEST_BEZIER_H
