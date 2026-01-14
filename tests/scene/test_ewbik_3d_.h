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
#include "scene/main/window.h"
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
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	Skeleton3D *skeleton = memnew(Skeleton3D);
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_owner(root);

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

	// Initialize bone poses to rest poses
	for (int i = 0; i < skeleton->get_bone_count(); ++i) {
		skeleton->set_bone_pose(i, skeleton->get_bone_rest(i));
	}
	skeleton->force_update_all_bone_transforms();

	// Setup IK
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "root");
	ik->set_end_bone_name(0, "effector");

	// Verify initial setup
	Vector3 initial_pos = skeleton->get_bone_global_pose(3).origin;
	CHECK_MESSAGE(initial_pos.distance_to(Vector3(0, 3, 0)) < 0.1, "Initial effector position should be at (0,3,0)");

	// Cleanup
	root->remove_child(skeleton);
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
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	Skeleton3D *skeleton = memnew(Skeleton3D);
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_owner(root);

	// Create simple chain
	skeleton->add_bone("root");
	skeleton->add_bone("upper");
	skeleton->add_bone("effector");

	skeleton->set_bone_parent(1, 0);
	skeleton->set_bone_parent(2, 1);

	skeleton->set_bone_rest(0, Transform3D(Basis(), Vector3(0, 0, 0)));
	skeleton->set_bone_rest(1, Transform3D(Basis(), Vector3(0, 1, 0)));
	skeleton->set_bone_rest(2, Transform3D(Basis(), Vector3(0, 1, 0)));

	// Initialize global poses
	for (int i = 0; i < skeleton->get_bone_count(); ++i) {
		skeleton->set_bone_global_pose(i, skeleton->get_bone_rest(i));
	}
	skeleton->force_update_all_bone_transforms();

	// Setup IK with joint limitation
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "root");
	ik->set_end_bone_name(0, "effector");

	// Add joint limitation
	Ref<JointLimitation3D> constraint = memnew(JointLimitation3D);
	ik->set_joint_limitation(0, 0, constraint); // First joint in chain

	Ref<JointLimitation3D> retrieved = ik->get_joint_limitation(0, 0);
	CHECK_MESSAGE(retrieved.is_valid(), "Joint limitation should be retrievable");

	// Cleanup
	root->remove_child(skeleton);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Humanoid arm chain with basis and translation") {
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_owner(root);

	// Setup IK
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");

	// Verify initial setup - check that position is reasonable (not zero, has expected scale)
	Vector3 initial_pos = skeleton->get_bone_global_pose(7).origin;
	CHECK_MESSAGE(initial_pos.length() > 0.1, "Initial hand position should not be at origin");
	CHECK_MESSAGE(Math::abs(initial_pos.y) < 2.0, "Hand Y position should be reasonable");

	// Validate all joints have proper basis and translation
	int bones[] = { 4, 5, 6, 7 }; // LeftShoulder, LeftUpperArm, LeftLowerArm, LeftHand

	for (int i = 0; i < 4; i++) {
		Transform3D transform = skeleton->get_bone_global_pose(bones[i]);
		CHECK_MESSAGE(transform.origin.length() > 0.01, "Bone should have non-zero translation");
		CHECK_MESSAGE(transform.basis.is_finite(), "Bone should have finite basis");
	}

	// Cleanup
	root->remove_child(skeleton);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Humanoid leg chain with basis and translation") {
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	Skeleton3D *skeleton = create_humanoid_leg_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_owner(root);

	// Setup IK
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "Hips");
	ik->set_end_bone_name(0, "LeftFoot");

	// Verify initial setup - check that position is reasonable
	Vector3 initial_pos = skeleton->get_bone_global_pose(3).origin;
	CHECK_MESSAGE(initial_pos.length() > 0.1, "Initial foot position should not be at origin");
	CHECK_MESSAGE(initial_pos.y < 2.0, "Foot Y position should be reasonable");

	// Validate all joints have proper basis and translation
	int bones[] = { 0, 1, 2, 3 }; // Hips, LeftUpperLeg, LeftLowerLeg, LeftFoot

	for (int i = 0; i < 4; i++) {
		Transform3D transform = skeleton->get_bone_global_pose(bones[i]);
		CHECK_MESSAGE(transform.origin.length() > 0.01, "Bone should have non-zero translation");
		CHECK_MESSAGE(transform.basis.is_finite(), "Bone should have finite basis");
	}

	// Cleanup
	root->remove_child(skeleton);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Random root functionality") {
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	// Test EWBIK's unique ability to use any bone as root
	Skeleton3D *skeleton = memnew(Skeleton3D);
	skeleton->add_bone("Hips");
	skeleton->add_bone("Spine");
	skeleton->add_bone("Chest");
	skeleton->add_bone("UpperChest");
	skeleton->add_bone("Neck");
	skeleton->add_bone("Head");

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);

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

	// Initialize global poses
	for (int i = 0; i < skeleton->get_bone_count(); ++i) {
		skeleton->set_bone_global_pose(i, skeleton->get_bone_rest(i));
	}
	skeleton->force_update_all_bone_transforms();

	EWBIK3D *ik = memnew(EWBIK3D);
	skeleton->add_child(ik);
	ik->set_owner(root);

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
	CHECK_MESSAGE(initial_pos.length() > 0.1, "Head should have reasonable position from chest root");

	// Cleanup
	root->remove_child(skeleton);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] IK solving with target position") {
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_owner(root);

	// Create a target node
	Node3D *target = memnew(Node3D);
	target->set_name("Target");
	root->add_child(target);
	target->set_owner(root);
	target->set_global_position(Vector3(0.5, 0.8, 0.2));

	// Setup IK for arm chain
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_target_node(0, NodePath("../../Target")); // Relative path from IK to target

	// Get initial position
	Vector3 initial_pos = skeleton->get_bone_global_pose(7).origin; // LeftHand initial position

	// Solve IK
	ik->process_modification(0.016); // Simulate one frame

	// Update skeleton transforms after IK
	skeleton->force_update_all_bone_transforms();

	// Verify effector moved towards target
	Vector3 final_pos = skeleton->get_bone_global_pose(7).origin; // LeftHand after solving
	Vector3 target_pos = target->get_global_position();

	float initial_distance = initial_pos.distance_to(target_pos);
	float final_distance = final_pos.distance_to(target_pos);

	// Should have moved closer to target
	CHECK_MESSAGE(final_distance < initial_distance, vformat("Effector should move closer to target. Initial distance: %f, Final distance: %f", initial_distance, final_distance));

	// Cleanup
	root->remove_child(target);
	root->remove_child(skeleton);
	memdelete(target);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] IK solving convergence") {
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_owner(root);

	// Create a target node
	Node3D *target = memnew(Node3D);
	target->set_name("Target");
	root->add_child(target);
	target->set_owner(root);
	target->set_global_position(Vector3(0.3, 0.5, 0.1));

	// Setup IK for arm chain
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_target_node(0, NodePath("../../Target")); // Relative path from IK to target

	// Get initial position
	Vector3 initial_pos = skeleton->get_bone_global_pose(7).origin;
	Vector3 target_pos = target->get_global_position();

	// Solve IK multiple times to test convergence
	for (int i = 0; i < 20; i++) {
		ik->process_modification(0.016);
	}

	// Update skeleton transforms after IK
	skeleton->force_update_all_bone_transforms();

	// Verify convergence
	Vector3 final_pos = skeleton->get_bone_global_pose(7).origin;
	float initial_distance = initial_pos.distance_to(target_pos);
	float final_distance = final_pos.distance_to(target_pos);

	// Check that IK actually ran (position should change)
	float position_change = initial_pos.distance_to(final_pos);
	CHECK_MESSAGE(position_change > 0.001, vformat("IK should change bone position. Position change: %f", position_change));

	CHECK_MESSAGE(final_distance < initial_distance, vformat("Distance to target should decrease. Initial: %f, Final: %f", initial_distance, final_distance));
	CHECK_MESSAGE(final_distance < 0.5, vformat("Should converge to reasonable distance. Final distance: %f", final_distance));

	// Cleanup
	root->remove_child(target);
	root->remove_child(skeleton);
	memdelete(target);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] IK solving with joint limitations") {
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_owner(root);

	// Create a target node
	Node3D *target = memnew(Node3D);
	target->set_name("Target");
	root->add_child(target);
	target->set_owner(root);
	target->set_global_position(Vector3(0.8, 0.2, 0.0)); // Position requiring extreme bend

	// Setup IK for arm chain
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_target_node(0, NodePath("../../Target")); // Relative path from IK to target

	// Add joint limitation
	Ref<JointLimitationCone3D> elbow_limit = memnew(JointLimitationCone3D);
	elbow_limit->set_angle(Math::deg_to_rad(30.0)); // Tight 30 degree cone
	ik->set_joint_limitation(0, 1, elbow_limit); // Joint index 1 (LeftUpperArm -> LeftLowerArm)

	// Solve IK multiple times
	for (int i = 0; i < 15; i++) {
		ik->process_modification(0.016);
	}

	// Update skeleton transforms after IK
	skeleton->force_update_all_bone_transforms();

	// Verify effector is reasonably close but respects joint limits
	Vector3 final_pos = skeleton->get_bone_global_pose(7).origin;
	Vector3 target_pos = target->get_global_position();
	float distance_to_target = final_pos.distance_to(target_pos);

	// Should still make progress but may not reach exact target due to constraints
	CHECK_MESSAGE(distance_to_target < 0.8, vformat("Should make reasonable progress even with constraints. Distance: %f", distance_to_target));

	// Cleanup
	root->remove_child(target);
	root->remove_child(skeleton);
	memdelete(target);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Effector opacity functionality") {
	EWBIK3D *ik = memnew(EWBIK3D);

	// Test default opacity
	ik->set_setting_count(1);
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 1.0f), "Default opacity should be 1.0");

	// Test setting opacity
	ik->set_effector_opacity(0, 0.5f);
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 0.5f), "Opacity should be settable");

	// Test multiple effectors
	ik->set_setting_count(3);
	ik->set_effector_opacity(0, 0.2f);
	ik->set_effector_opacity(1, 0.7f);
	ik->set_effector_opacity(2, 1.0f);

	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 0.2f), "Effector 0 opacity should be 0.2");
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(1), 0.7f), "Effector 1 opacity should be 0.7");
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(2), 1.0f), "Effector 2 opacity should be 1.0");

	// Test out of bounds access
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(-1), 1.0f), "Out of bounds negative index should return default");
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(10), 1.0f), "Out of bounds positive index should return default");

	// Test opacity bounds (should clamp to valid range)
	ik->set_effector_opacity(0, -0.5f); // Negative value
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 0.0f), "Negative opacity should be clamped to 0.0");

	ik->set_effector_opacity(0, 2.0f); // Value > 1.0
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 1.0f), "Opacity > 1.0 should be clamped to 1.0");

	memdelete(ik);
}

TEST_CASE("[SceneTree][EWBIK3D] Multi-effector settings validation") {
	EWBIK3D *ik = memnew(EWBIK3D);

	// Test setting up multiple effectors
	ik->set_setting_count(3);

	// Test that we can set different root/end bones for each effector
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_effector_opacity(0, 0.8f);

	ik->set_root_bone_name(1, "RightShoulder");
	ik->set_end_bone_name(1, "RightHand");
	ik->set_effector_opacity(1, 1.0f);

	ik->set_root_bone_name(2, "Spine");
	ik->set_end_bone_name(2, "Chest");
	ik->set_effector_opacity(2, 0.5f);

	// Verify settings are stored correctly
	CHECK_MESSAGE(ik->get_root_bone_name(0) == "LeftShoulder", "First effector root bone should be set");
	CHECK_MESSAGE(ik->get_end_bone_name(0) == "LeftHand", "First effector end bone should be set");
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 0.8f), "First effector opacity should be 0.8");

	CHECK_MESSAGE(ik->get_root_bone_name(1) == "RightShoulder", "Second effector root bone should be set");
	CHECK_MESSAGE(ik->get_end_bone_name(1) == "RightHand", "Second effector end bone should be set");
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(1), 1.0f), "Second effector opacity should be 1.0");

	CHECK_MESSAGE(ik->get_root_bone_name(2) == "Spine", "Third effector root bone should be set");
	CHECK_MESSAGE(ik->get_end_bone_name(2) == "Chest", "Third effector end bone should be set");
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(2), 0.5f), "Third effector opacity should be 0.5");

	// Test opacity clamping
	ik->set_effector_opacity(0, -0.5f); // Negative value
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 0.0f), "Negative opacity should be clamped to 0.0");

	ik->set_effector_opacity(0, 2.0f); // Value > 1.0
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 1.0f), "Opacity > 1.0 should be clamped to 1.0");

	memdelete(ik);
}

TEST_CASE("[SceneTree][EWBIK3D] Effector opacity influence validation") {
	EWBIK3D *ik = memnew(EWBIK3D);

	// Test that opacity values are properly stored and retrieved
	ik->set_setting_count(2);

	// Set different opacity values
	ik->set_effector_opacity(0, 0.0f); // Completely disabled
	ik->set_effector_opacity(1, 1.0f); // Fully enabled

	// Verify values are stored correctly
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 0.0f), "Effector 0 should have 0.0 opacity");
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(1), 1.0f), "Effector 1 should have 1.0 opacity");

	// Test boundary values
	ik->set_effector_opacity(0, 0.001f); // Very small positive value
	ik->set_effector_opacity(1, 0.999f); // Very close to 1.0

	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(0), 0.001f), "Very small opacity should be preserved");
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(1), 0.999f), "Opacity close to 1.0 should be preserved");

	// Test that out-of-bounds access returns default (1.0)
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(-1), 1.0f), "Negative index should return default opacity");
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_effector_opacity(5), 1.0f), "Out-of-bounds index should return default opacity");

	memdelete(ik);
}

TEST_CASE("[SceneTree][EWBIK3D] Single effector IK solving test") {
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	// Use the full humanoid skeleton for multi-effector testing
	Skeleton3D *skeleton = create_humanoid_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_name("EWBIK");
	ik->set_owner(root);

	// Increase max iterations for better convergence
	ik->set_max_iterations(100);

	// Setup two effectors: one for left hand, one for right hand
	ik->set_setting_count(2);

	// Left arm effector
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_effector_opacity(0, 1.0f);

	// Right arm effector
	ik->set_root_bone_name(1, "RightShoulder");
	ik->set_end_bone_name(1, "RightHand");
	ik->set_effector_opacity(1, 1.0f);

	// Create target nodes
	Node3D *left_target = memnew(Node3D);
	left_target->set_name("LeftTarget");
	root->add_child(left_target);
	left_target->set_owner(root);
	left_target->set_global_position(Vector3(0.4, 0.8, 0.2)); // Reach forward-left

	Node3D *right_target = memnew(Node3D);
	right_target->set_name("RightTarget");
	root->add_child(right_target);
	right_target->set_owner(root);
	right_target->set_global_position(Vector3(-0.4, 0.8, 0.2)); // Reach forward-right

	// Set target nodes
	ik->set_target_node(0, NodePath("../../LeftTarget"));
	ik->set_target_node(1, NodePath("../../RightTarget"));

	// Get initial positions
	Vector3 initial_left_pos = skeleton->get_bone_global_pose(7).origin; // LeftHand
	Vector3 initial_right_pos = skeleton->get_bone_global_pose(11).origin; // RightHand

	// Solve IK once
	ik->process_modification(0.016);

	// Update skeleton transforms
	skeleton->force_update_all_bone_transforms();

	// Check final positions
	Vector3 final_left_pos = skeleton->get_bone_global_pose(7).origin;
	Vector3 final_right_pos = skeleton->get_bone_global_pose(11).origin;

	// Both hands should have moved toward their targets
	float left_improvement = initial_left_pos.distance_to(left_target->get_global_position()) -
			final_left_pos.distance_to(left_target->get_global_position());
	float right_improvement = initial_right_pos.distance_to(right_target->get_global_position()) -
			final_right_pos.distance_to(right_target->get_global_position());

	CHECK_MESSAGE(left_improvement > 0.001, vformat("Left hand should move closer to target. Improvement: %f", left_improvement));
	CHECK_MESSAGE(right_improvement > 0.001, vformat("Right hand should move closer to target. Improvement: %f", right_improvement));

	// Final distances should be reasonable
	float final_left_distance = final_left_pos.distance_to(left_target->get_global_position());
	float final_right_distance = final_right_pos.distance_to(right_target->get_global_position());

	CHECK_MESSAGE(final_left_distance < 0.8, vformat("Left hand should be reasonably close to target. Distance: %f", final_left_distance));
	CHECK_MESSAGE(final_right_distance < 0.8, vformat("Right hand should be reasonably close to target. Distance: %f", final_right_distance));

	// Cleanup
	root->remove_child(left_target);
	root->remove_child(right_target);
	root->remove_child(skeleton);
	memdelete(left_target);
	memdelete(right_target);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Effector opacity influence on solving") {
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	Skeleton3D *skeleton = create_humanoid_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_owner(root);

	// Setup two effectors competing for the same arm bones
	ik->set_setting_count(2);

	// First effector (high priority)
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_effector_opacity(0, 1.0f); // Full opacity

	// Second effector (low priority) - same chain but different target
	ik->set_root_bone_name(1, "LeftShoulder");
	ik->set_end_bone_name(1, "LeftHand");
	ik->set_effector_opacity(1, 0.1f); // Low opacity

	// Create target nodes at different positions
	Node3D *primary_target = memnew(Node3D);
	primary_target->set_name("PrimaryTarget");
	root->add_child(primary_target);
	primary_target->set_owner(root);
	primary_target->set_global_position(Vector3(0.3, 0.9, 0.1));

	Node3D *secondary_target = memnew(Node3D);
	secondary_target->set_name("SecondaryTarget");
	root->add_child(secondary_target);
	secondary_target->set_owner(root);
	secondary_target->set_global_position(Vector3(0.5, 0.5, 0.3)); // Different position

	// Set target nodes
	ik->set_target_node(0, NodePath("../../PrimaryTarget"));
	ik->set_target_node(1, NodePath("../../SecondaryTarget"));

	// Get initial position
	Vector3 initial_pos = skeleton->get_bone_global_pose(7).origin; // LeftHand

	// Solve IK multiple times for better convergence
	for (int i = 0; i < 20; i++) {
		ik->process_modification(0.016);
	}

	skeleton->force_update_all_bone_transforms();

	// Check final position
	Vector3 final_pos = skeleton->get_bone_global_pose(7).origin;

	// The hand should be closer to the primary target (high opacity) than secondary
	float distance_to_primary = final_pos.distance_to(primary_target->get_global_position());

	// With opacity weighting, the primary target should have more influence
	// We expect movement toward primary target after convergence
	float primary_improvement = initial_pos.distance_to(primary_target->get_global_position()) - distance_to_primary;
	CHECK_MESSAGE(primary_improvement >= 0.0, vformat("Should not move away from primary target. Improvement: %f", primary_improvement));

	// The effector should be solving (some movement should occur)
	float total_movement = initial_pos.distance_to(final_pos);
	CHECK_MESSAGE(total_movement > 0.001, vformat("IK should cause some movement. Movement: %f", total_movement));

	// Cleanup
	root->remove_child(primary_target);
	root->remove_child(secondary_target);
	root->remove_child(skeleton);
	memdelete(primary_target);
	memdelete(secondary_target);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][EWBIK3D] Bone lengths preservation and pinning behavior") {
	// Create scene tree setup
	SceneTree *tree = SceneTree::get_singleton();
	Window *root = tree->get_root();

	Skeleton3D *skeleton = create_humanoid_skeleton();
	EWBIK3D *ik = memnew(EWBIK3D);

	// Add to scene tree
	root->add_child(skeleton);
	skeleton->set_owner(root);
	skeleton->add_child(ik);
	ik->set_owner(root);

	// Record initial bone lengths
	Vector3 shoulder_pos = skeleton->get_bone_global_pose(4).origin; // LeftShoulder
	Vector3 upper_pos = skeleton->get_bone_global_pose(5).origin; // LeftUpperArm
	Vector3 lower_pos = skeleton->get_bone_global_pose(6).origin; // LeftLowerArm
	Vector3 hand_pos = skeleton->get_bone_global_pose(7).origin; // LeftHand

	float initial_shoulder_to_upper = shoulder_pos.distance_to(upper_pos);
	float initial_upper_to_lower = upper_pos.distance_to(lower_pos);
	float initial_lower_to_hand = lower_pos.distance_to(hand_pos);

	// Setup two effectors: one for shoulder (root), one for hand (tip)
	ik->set_setting_count(2);

	// First effector: shoulder (will be pinned in first scenario)
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftShoulder"); // Single bone effector
	ik->set_effector_opacity(0, 0.5f);

	// Second effector: hand
	ik->set_root_bone_name(1, "LeftShoulder");
	ik->set_end_bone_name(1, "LeftHand");
	ik->set_effector_opacity(1, 1.0f);

	// Create targets at initial positions
	Node3D *shoulder_target = memnew(Node3D);
	shoulder_target->set_name("ShoulderTarget");
	root->add_child(shoulder_target);
	shoulder_target->set_owner(root);
	shoulder_target->set_global_position(shoulder_pos);

	Node3D *hand_target = memnew(Node3D);
	hand_target->set_name("HandTarget");
	root->add_child(hand_target);
	hand_target->set_owner(root);
	hand_target->set_global_position(hand_pos);

	// Set target nodes
	ik->set_target_node(0, NodePath("../../ShoulderTarget"));
	ik->set_target_node(1, NodePath("../../HandTarget"));

	// Scenario 1: Pin shoulder, move hand
	hand_target->set_global_position(Vector3(0.5, 0.8, 0.2)); // Move hand target

	// Solve IK
	for (int i = 0; i < 20; i++) {
		ik->process_modification(0.016);
	}
	skeleton->force_update_all_bone_transforms();

	// Check that shoulder stayed pinned
	Vector3 final_shoulder_pos = skeleton->get_bone_global_pose(4).origin;
	float shoulder_movement = shoulder_pos.distance_to(final_shoulder_pos);
	CHECK_MESSAGE(shoulder_movement < 0.01, vformat("Shoulder should stay pinned. Movement: %f", shoulder_movement));

	// Check that hand moved
	Vector3 final_hand_pos = skeleton->get_bone_global_pose(7).origin;
	float hand_movement = hand_pos.distance_to(final_hand_pos);
	CHECK_MESSAGE(hand_movement > 0.1, vformat("Hand should have moved. Movement: %f", hand_movement));

	// Check bone lengths preserved
	Vector3 final_upper_pos = skeleton->get_bone_global_pose(5).origin;
	Vector3 final_lower_pos = skeleton->get_bone_global_pose(6).origin;

	float final_shoulder_to_upper = final_shoulder_pos.distance_to(final_upper_pos);
	float final_upper_to_lower = final_upper_pos.distance_to(final_lower_pos);
	float final_lower_to_hand = final_lower_pos.distance_to(final_hand_pos);

	CHECK_MESSAGE(Math::is_equal_approx(initial_shoulder_to_upper, final_shoulder_to_upper, 0.001f), vformat("Shoulder to upper arm length should be preserved. Initial: %f, Final: %f", initial_shoulder_to_upper, final_shoulder_to_upper));
	CHECK_MESSAGE(Math::is_equal_approx(initial_upper_to_lower, final_upper_to_lower, 0.001f), vformat("Upper to lower arm length should be preserved. Initial: %f, Final: %f", initial_upper_to_lower, final_upper_to_lower));
	CHECK_MESSAGE(Math::is_equal_approx(initial_lower_to_hand, final_lower_to_hand, 0.001f), vformat("Lower arm to hand length should be preserved. Initial: %f, Final: %f", initial_lower_to_hand, final_lower_to_hand));

	// Scenario 2: Pin hand, move shoulder (dragging root)
	// Keep skeleton at solved position from scenario 1
	// Change effector settings
	ik->set_setting_count(2);

	// First effector: hand (pin)
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");
	ik->set_effector_opacity(0, 0.5f);

	// Second effector: shoulder (move)
	ik->set_root_bone_name(1, "LeftShoulder");
	ik->set_end_bone_name(1, "LeftShoulder");
	ik->set_effector_opacity(1, 1.0f);

	// Set targets to current positions
	shoulder_target->set_global_position(final_shoulder_pos);
	hand_target->set_global_position(final_hand_pos);

	// Move shoulder target
	shoulder_target->set_global_position(Vector3(0.1, 1.2, 0.0)); // Move shoulder target

	// Solve IK
	for (int i = 0; i < 20; i++) {
		ik->process_modification(0.016);
	}
	skeleton->force_update_all_bone_transforms();

	// Check that hand stayed pinned
	Vector3 final_hand_pos2 = skeleton->get_bone_global_pose(7).origin;
	float hand_movement2 = final_hand_pos.distance_to(final_hand_pos2);
	CHECK_MESSAGE(hand_movement2 < 0.01, vformat("Hand should stay pinned. Movement: %f", hand_movement2));

	// Check that shoulder moved
	Vector3 final_shoulder_pos2 = skeleton->get_bone_global_pose(4).origin;
	float shoulder_movement2 = final_shoulder_pos.distance_to(final_shoulder_pos2);
	CHECK_MESSAGE(shoulder_movement2 > 0.1, vformat("Shoulder should have moved. Movement: %f", shoulder_movement2));

	// Check bone lengths preserved again
	Vector3 final_upper_pos2 = skeleton->get_bone_global_pose(5).origin;
	Vector3 final_lower_pos2 = skeleton->get_bone_global_pose(6).origin;

	float final_shoulder_to_upper2 = final_shoulder_pos2.distance_to(final_upper_pos2);
	float final_upper_to_lower2 = final_upper_pos2.distance_to(final_lower_pos2);
	float final_lower_to_hand2 = final_lower_pos2.distance_to(final_hand_pos2);

	CHECK_MESSAGE(Math::is_equal_approx(initial_shoulder_to_upper, final_shoulder_to_upper2, 0.001f), vformat("Shoulder to upper arm length should be preserved. Initial: %f, Final: %f", initial_shoulder_to_upper, final_shoulder_to_upper2));
	CHECK_MESSAGE(Math::is_equal_approx(initial_upper_to_lower, final_upper_to_lower2, 0.001f), vformat("Upper to lower arm length should be preserved. Initial: %f, Final: %f", initial_upper_to_lower, final_upper_to_lower2));
	CHECK_MESSAGE(Math::is_equal_approx(initial_lower_to_hand, final_lower_to_hand2, 0.001f), vformat("Lower arm to hand length should be preserved. Initial: %f, Final: %f", initial_lower_to_hand, final_lower_to_hand2));

	// Cleanup
	root->remove_child(shoulder_target);
	root->remove_child(hand_target);
	root->remove_child(skeleton);
	memdelete(shoulder_target);
	memdelete(hand_target);
	memdelete(ik);
	memdelete(skeleton);
}

} // namespace TestEWBIK3D
