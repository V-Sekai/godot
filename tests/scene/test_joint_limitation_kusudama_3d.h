/**************************************************************************/
/*  test_joint_limitation_kusudama_3d.h                                   */
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
#include "scene/resources/3d/joint_limitation_kusudama_3d.h"
#include "tests/test_macros.h"

namespace TestJointLimitationKusudama3D {

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point inside single cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f); // 30 degrees

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_open_cones(cones);

	// Point at the center of the cone should be returned as-is
	Vector3 test_point = control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));

	// Point slightly off center but still within cone
	Vector3 point_in_cone = Quaternion(Vector3(1, 0, 0), radius * 0.5f).xform(control_point);
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), point_in_cone);
	CHECK(result.is_equal_approx(point_in_cone));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point outside single cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f); // 30 degrees

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_open_cones(cones);

	// Point far outside the cone should be clamped to boundary
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);

	// Result should be on the cone boundary, not the original point
	CHECK_FALSE(result.is_equal_approx(test_point));

	// Result should be close to the control point (within radius)
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f); // Allow small tolerance
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point with zero radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = 0.0f; // Zero radius - only exact point allowed

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_open_cones(cones);

	// Point at exact control point should be returned
	Vector3 test_point = control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(control_point));

	// Any other point should be clamped to control point
	Vector3 outside_point = Vector3(1, 0, 0).normalized();
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), outside_point);
	CHECK(result.is_equal_approx(control_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test multiple cones - point between cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create two cones
	Vector3 control_point1 = Vector3(1, 0, 0).normalized();
	Vector3 control_point2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f); // 45 degrees

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point1.x, control_point1.y, control_point1.z, radius));
	cones.push_back(Vector4(control_point2.x, control_point2.y, control_point2.z, radius));
	limitation->set_open_cones(cones);

	// Point between the two cones should be handled by path logic
	Vector3 point_between = (control_point1 + control_point2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_between);

	// Result should be valid (not NaN)
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f); // Should be normalized
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point on path between two adjacent cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create two cones with overlapping paths
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(60.0f); // Large enough to create paths

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_open_cones(cones);

	// Point exactly on the great circle path between cones
	Vector3 path_point = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path_point);

	// Should be on or near the path
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be closer to the path than the original if original was outside
	// Test with a point clearly between but not exactly on path
	Vector3 between_point = Vector3(0.7, 0.7, 0.1).normalized();
	Vector3 result2 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between_point);
	CHECK(result2.is_finite());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point outside both cones but in path region") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0f); // Smaller radius

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_open_cones(cones);

	// Point outside both cones but in the region between them
	Vector3 outside_point = Vector3(0.5, 0.5, 0.7).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), outside_point);

	// Should be clamped to the path between cones
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be closer to the path than original
	real_t dist_to_cp1 = result.angle_to(cp1);
	real_t dist_to_cp2 = result.angle_to(cp2);
	// Should be reachable from at least one cone
	CHECK((dist_to_cp1 <= radius + 0.1f || dist_to_cp2 <= radius + 0.1f));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path between cones with different radii") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius1 = Math::deg_to_rad(45.0f);
	real_t radius2 = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius1));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius2));
	limitation->set_open_cones(cones);

	// Point between cones should still work with different radii
	Vector3 between = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path wrapping from last to first cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones in a loop
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(50.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	limitation->set_open_cones(cones);

	// Point between last and first cone (wrapping path)
	Vector3 between_last_first = (cp3 + cp1).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), between_last_first);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point in tangent circle region between cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_open_cones(cones);

	// Point that should be in the tangent circle region
	// This is in the region where the tangent circle connects the two cones
	Vector3 tangent_region_point = Vector3(0.6, 0.6, 0.5).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), tangent_region_point);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be on or near the tangent path
	// The angle to either control point should be reasonable
	real_t angle1 = result.angle_to(cp1);
	real_t angle2 = result.angle_to(cp2);
	CHECK((angle1 < Math::PI / 2 || angle2 < Math::PI / 2)); // Should be in hemisphere of at least one cone
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test multiple paths - point closest to which path") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones forming a triangle
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(40.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	limitation->set_open_cones(cones);

	// Point that could be in path between cp1-cp2 or cp2-cp3
	// Should find the closest valid path
	Vector3 test_point = Vector3(0.4, 0.5, 0.3).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), test_point);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be valid and normalized
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path with very close cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones very close together
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0.99, 0.1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_open_cones(cones);

	// Point between very close cones
	Vector3 between = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path with nearly opposite cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones nearly opposite each other
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(-0.9, 0.1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_open_cones(cones);

	// Point between nearly opposite cones
	Vector3 between = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test multiple cones - point in first cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point1 = Vector3(1, 0, 0).normalized();
	Vector3 control_point2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point1.x, control_point1.y, control_point1.z, radius));
	cones.push_back(Vector4(control_point2.x, control_point2.y, control_point2.z, radius));
	limitation->set_open_cones(cones);

	// Point inside first cone should be returned as-is
	Vector3 point_in_cone1 = Quaternion(Vector3(0, 1, 0), radius * 0.3f).xform(control_point1);
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_in_cone1);
	CHECK(result.is_equal_approx(point_in_cone1));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test empty cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector<Vector4> empty_cones;
	limitation->set_open_cones(empty_cones);

	// With no cones and orientationally constrained, should return input
	limitation->set_orientationally_constrained(true);
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));

	// With orientationally unconstrained, should return input
	limitation->set_orientationally_constrained(false);
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test orientationally unconstrained") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_open_cones(cones);
	limitation->set_orientationally_constrained(false);

	// Should return input regardless of cone constraints
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test axial limits getters and setters") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	real_t min_angle = Math::deg_to_rad(-45.0f);
	real_t range_angle = Math::deg_to_rad(90.0f);

	limitation->set_axial_limits(min_angle, range_angle);
	CHECK(Math::is_equal_approx(limitation->get_min_axial_angle(), min_angle));
	CHECK(Math::is_equal_approx(limitation->get_range_angle(), range_angle));

	limitation->set_axially_constrained(true);
	CHECK(limitation->is_axially_constrained() == true);

	limitation->set_axially_constrained(false);
	CHECK(limitation->is_axially_constrained() == false);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test open cones getters and setters") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector<Vector4> cones;
	cones.push_back(Vector4(1, 0, 0, Math::deg_to_rad(30.0f)));
	cones.push_back(Vector4(0, 1, 0, Math::deg_to_rad(45.0f)));
	cones.push_back(Vector4(0, 0, 1, Math::deg_to_rad(60.0f)));

	limitation->set_open_cones(cones);
	Vector<Vector4> retrieved = limitation->get_open_cones();

	CHECK(retrieved.size() == 3);
	CHECK(Math::is_equal_approx(retrieved[0].w, Math::deg_to_rad(30.0f)));
	CHECK(Math::is_equal_approx(retrieved[1].w, Math::deg_to_rad(45.0f)));
	CHECK(Math::is_equal_approx(retrieved[2].w, Math::deg_to_rad(60.0f)));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test large radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(170.0f); // Very large cone

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_open_cones(cones);

	// Most points should be inside such a large cone
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);

	// Should be valid
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test three cones in sequence") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones forming a path
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	limitation->set_open_cones(cones);

	// Test point in first cone
	Vector3 point1 = Quaternion(Vector3(0, 1, 0), radius * 0.3f).xform(cp1);
	Vector3 result1 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point1);
	CHECK(result1.is_equal_approx(point1));

	// Test point in second cone
	Vector3 point2 = Quaternion(Vector3(1, 0, 0), radius * 0.3f).xform(cp2);
	Vector3 result2 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point2);
	CHECK(result2.is_equal_approx(point2));

	// Test point between cones (should use path logic)
	Vector3 point_between = (cp1 + cp2).normalized();
	Vector3 result3 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_between);
	CHECK(result3.is_finite());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test edge case - parallel vectors") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(1, 0, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_open_cones(cones);

	// Test with input parallel to control point (should handle gracefully)
	Vector3 parallel_point = control_point * 2.0f; // Same direction, different length
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), parallel_point);

	// Should normalize and return (point is inside cone)
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test edge case - opposite direction") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_open_cones(cones);

	// Point in opposite direction (180 degrees away)
	Vector3 opposite = -control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), opposite);

	// Should clamp to boundary
	CHECK(result.is_finite());
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test make_space method") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 forward = Vector3(0, 1, 0);
	Vector3 right = Vector3(1, 0, 0);
	Quaternion offset = Quaternion();

	Quaternion space = limitation->make_space(forward, right, offset);

	// Should return a valid quaternion
	CHECK(space.is_normalized());
	CHECK(space.is_finite());
}

} // namespace TestJointLimitationKusudama3D
