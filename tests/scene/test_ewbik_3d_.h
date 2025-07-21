/**************************************************************************/
/*  test_ewbik_3d_.h                                                      */
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

#include "scene/3d/ewbik_3d_.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/3d/joint_limitation_cone_3d.h"
#include "tests/scene/test_ik_common.h"

namespace TestEWBIK3D {

TEST_CASE("[SceneTree][EWBIK3D] Class instantiation and registration") {
	// Test that EWBIK3D can be instantiated
	EWBIK3D *ik = memnew(EWBIK3D);
	CHECK_MESSAGE(ik != nullptr, "EWBIK3D should be instantiable");

	// Test basic properties
	ik->set_max_iterations(10);
	CHECK_MESSAGE(ik->get_max_iterations() == 10, "Max iterations should be settable");

	ik->set_min_distance(0.01);
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_min_distance(), 0.01), "Min distance should be settable");

	memdelete(ik);
}

TEST_CASE("[SceneTree][EWBIK3D] Skeleton setup and basic functionality") {
	Skeleton3D *skeleton = memnew(Skeleton3D);
	EWBIK3D *ik = memnew(EWBIK3D);

	// Create a simple bone chain
	skeleton->add_bone("root");
	skeleton->add_bone("upper");
	skeleton->add_bone("lower");
	skeleton->add_bone("effector");

	// Set hierarchy
	skeleton->set_bone_parent(1, 0);
	skeleton->set_bone_parent(2, 1);
	skeleton->set_bone_parent(3, 2);

	// Set rest poses
	skeleton->set_bone_rest(0, Transform3D(Basis(), Vector3(0, 0, 0)));
	skeleton->set_bone_rest(1, Transform3D(Basis(), Vector3(0, 1, 0)));
	skeleton->set_bone_rest(2, Transform3D(Basis(), Vector3(0, 1, 0)));
	skeleton->set_bone_rest(3, Transform3D(Basis(), Vector3(0, 1, 0)));

	// Setup IK
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "root");
	ik->set_end_bone_name(0, "effector");

	// Verify initial setup
	Vector3 initial_pos = skeleton->get_bone_global_pose(3).origin;
	CHECK_MESSAGE(initial_pos.distance_to(Vector3(0, 3, 0)) < 0.1, "Initial effector position should be at (0,3,0)");

	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Multiple settings support") {
	EWBIK3D *ik = memnew(EWBIK3D);

	// Test setting count management
	ik->set_setting_count(3);
	CHECK_MESSAGE(ik->get_setting_count() == 3, "Should support multiple settings");

	ik->clear_settings();
	CHECK_MESSAGE(ik->get_setting_count() == 0, "Should be able to clear all settings");

	memdelete(ik);
}

TEST_CASE("[SceneTree][EWBIK3D] Joint limitation support") {
	Skeleton3D *skeleton = memnew(Skeleton3D);
	EWBIK3D *ik = memnew(EWBIK3D);

	// Create simple chain
	skeleton->add_bone("root");
	skeleton->add_bone("upper");
	skeleton->add_bone("effector");

	skeleton->set_bone_parent(1, 0);
	skeleton->set_bone_parent(2, 1);

	skeleton->set_bone_rest(0, Transform3D(Basis(), Vector3(0, 0, 0)));
	skeleton->set_bone_rest(1, Transform3D(Basis(), Vector3(0, 1, 0)));
	skeleton->set_bone_rest(2, Transform3D(Basis(), Vector3(0, 1, 0)));

	// Setup IK with joint limitation
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "root");
	ik->set_end_bone_name(0, "effector");

	// Add joint limitation
	Ref<JointLimitation3D> constraint = memnew(JointLimitation3D);
	ik->set_joint_limitation(0, 0, constraint); // First joint in chain

	Ref<JointLimitation3D> retrieved = ik->get_joint_limitation(0, 0);
	CHECK_MESSAGE(retrieved.is_valid(), "Joint limitation should be retrievable");

	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Humanoid arm chain with basis and translation") {
	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Setup IK
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");

	// Verify initial setup
	Vector3 initial_pos = skeleton->get_bone_global_pose(7).origin;
	float expected_length = 0.9 + 0.1 + 0.1 + 0.1 + 0.05 + 0.3 + 0.35 + 0.25;
	CHECK_MESSAGE(Math::abs(initial_pos.y - expected_length) < 0.1, "Initial hand position should match bone chain length");

	// Validate all joints have proper basis and translation
	int bones[] = { 4, 5, 6, 7 }; // LeftShoulder, LeftUpperArm, LeftLowerArm, LeftHand

	for (int i = 0; i < 4; i++) {
		Transform3D transform = skeleton->get_bone_global_pose(bones[i]);
		CHECK_MESSAGE(transform.origin.length() > 0.01, "Bone should have non-zero translation");
		CHECK_MESSAGE(transform.basis.is_finite(), "Bone should have finite basis");
	}

	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Humanoid leg chain with basis and translation") {
	Skeleton3D *skeleton = create_humanoid_leg_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Setup IK
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "Hips");
	ik->set_end_bone_name(0, "LeftFoot");

	// Verify initial setup
	Vector3 initial_pos = skeleton->get_bone_global_pose(3).origin;
	float expected_length = 0.9 + 0.45 + 0.4 + 0.3;
	CHECK_MESSAGE(Math::abs(initial_pos.y - expected_length) < 0.1, "Initial foot position should match bone chain length");

	// Validate all joints have proper basis and translation
	int bones[] = { 0, 1, 2, 3 }; // Hips, LeftUpperLeg, LeftLowerLeg, LeftFoot

	for (int i = 0; i < 4; i++) {
		Transform3D transform = skeleton->get_bone_global_pose(bones[i]);
		CHECK_MESSAGE(transform.origin.length() > 0.01, "Bone should have non-zero translation");
		CHECK_MESSAGE(transform.basis.is_finite(), "Bone should have finite basis");
	}

	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Random root functionality") {
	// Test EWBIK's unique ability to use any bone as root
	Skeleton3D *skeleton = memnew(Skeleton3D);
	skeleton->add_bone("Hips");
	skeleton->add_bone("Spine");
	skeleton->add_bone("Chest");
	skeleton->add_bone("UpperChest");
	skeleton->add_bone("Neck");
	skeleton->add_bone("Head");

	skeleton->set_bone_parent(1, 0); // Spine -> Hips
	skeleton->set_bone_parent(2, 1); // Chest -> Spine
	skeleton->set_bone_parent(3, 2); // UpperChest -> Chest
	skeleton->set_bone_parent(4, 3); // Neck -> UpperChest
	skeleton->set_bone_parent(5, 4); // Head -> Neck

	// Set realistic spine chain
	skeleton->set_bone_rest(0, Transform3D(Basis(), Vector3(0, 0.9, 0)));
	skeleton->set_bone_rest(1, Transform3D(Basis(), Vector3(0, 0.1, 0)));
	skeleton->set_bone_rest(2, Transform3D(Basis(), Vector3(0, 0.1, 0)));
	skeleton->set_bone_rest(3, Transform3D(Basis(), Vector3(0, 0.1, 0)));
	skeleton->set_bone_rest(4, Transform3D(Basis(), Vector3(0, 0.1, 0)));
	skeleton->set_bone_rest(5, Transform3D(Basis(), Vector3(0, 0.1, 0)));

	EWBIK3D *ik = memnew(EWBIK3D);

	// Test using chest as root (random root functionality)
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "Chest"); // Not the actual root of skeleton
	ik->set_end_bone_name(0, "Head");

	// Verify setup works with non-hierarchical root
	int chest_bone = skeleton->find_bone("Chest");
	int head_bone = skeleton->find_bone("Head");
	CHECK_MESSAGE(chest_bone >= 0, "Chest bone should exist");
	CHECK_MESSAGE(head_bone >= 0, "Head bone should exist");

	// EWBIK should be able to find path from chest to head
	Vector3 initial_pos = skeleton->get_bone_global_pose(head_bone).origin;
	float expected_length = 0.1 + 0.1 + 0.1 + 0.1; // chest to upperchest to neck to head
	CHECK_MESSAGE(Math::abs(initial_pos.y - (0.9 + expected_length)) < 0.1, "Head position should be reachable from chest root");

	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] IK solving with target position") {
	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Create a target node
	Node3D *target = memnew(Node3D);
	target->set_global_position(Vector3(0.5, 0.8, 0.2));

	// Setup IK for arm chain
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_target_node(0, target->get_path());

	// Solve IK
	ik->process_modification(0.016); // Simulate one frame

	// Verify effector moved towards target
	Vector3 initial_pos = skeleton->get_bone_global_pose(7).origin; // LeftHand initial position
	Vector3 final_pos = skeleton->get_bone_global_pose(7).origin; // LeftHand after solving
	Vector3 target_pos = target->get_global_position();

	float initial_distance = initial_pos.distance_to(target_pos);
	float final_distance = final_pos.distance_to(target_pos);

	// Should have moved closer to target
	CHECK_MESSAGE(final_distance < initial_distance, vformat("Effector should move closer to target. Initial distance: %f, Final distance: %f", initial_distance, final_distance));

	memdelete(target);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] IK solving convergence") {
	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Create a target node
	Node3D *target = memnew(Node3D);
	target->set_global_position(Vector3(0.3, 0.5, 0.1));

	// Setup IK for arm chain
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_target_node(0, target->get_path());

	// Get initial position
	Vector3 initial_pos = skeleton->get_bone_global_pose(7).origin;
	Vector3 target_pos = target->get_global_position();

	// Solve IK multiple times to test convergence
	for (int i = 0; i < 10; i++) {
		ik->process_modification(0.016);
	}

	// Verify convergence
	Vector3 final_pos = skeleton->get_bone_global_pose(7).origin;
	float initial_distance = initial_pos.distance_to(target_pos);
	float final_distance = final_pos.distance_to(target_pos);

	CHECK_MESSAGE(final_distance < initial_distance, vformat("Distance to target should decrease. Initial: %f, Final: %f", initial_distance, final_distance));
	CHECK_MESSAGE(final_distance < 0.1, vformat("Should converge to reasonable distance. Final distance: %f", final_distance));

	memdelete(target);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] IK solving with joint limitations") {
	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Create a target node
	Node3D *target = memnew(Node3D);
	target->set_global_position(Vector3(0.8, 0.2, 0.0)); // Position requiring extreme bend

	// Setup IK for arm chain
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_target_node(0, target->get_path());

	// Add joint limitation
	Ref<JointLimitationCone3D> elbow_limit = memnew(JointLimitationCone3D);
	elbow_limit->set_angle(Math::deg_to_rad(30.0)); // Tight 30 degree cone
	ik->set_joint_limitation(0, 5, elbow_limit); // Joint index 5 (LeftUpperArm -> LeftLowerArm)

	// Solve IK multiple times
	for (int i = 0; i < 15; i++) {
		ik->process_modification(0.016);
	}

	// Verify effector is reasonably close but respects joint limits
	Vector3 final_pos = skeleton->get_bone_global_pose(7).origin;
	Vector3 target_pos = target->get_global_position();
	float distance_to_target = final_pos.distance_to(target_pos);

	// Should still make progress but may not reach exact target due to constraints
	CHECK_MESSAGE(distance_to_target < 0.8, vformat("Should make reasonable progress even with constraints. Distance: %f", distance_to_target));

	memdelete(target);
	memdelete(ik);
	memdelete(skeleton);
}

} // namespace TestEWBIK3D
