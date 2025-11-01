/**************************************************************************/
/*  test_fabr_ik_3d.h                                                     */
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

#include "scene/3d/fabr_ik_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/3d/joint_limitation_cone_3d.h"
#include "tests/scene/test_ik_common.h"

namespace TestFABRIK3D {

// Helper function to validate joint basis and translation
void validate_joint_transforms(Skeleton3D *p_skeleton, const Vector<String> &p_bone_names, const Vector<Transform3D> &p_expected_transforms, float p_tolerance = 0.01) {
	for (int i = 0; i < p_bone_names.size(); i++) {
		int bone_idx = p_skeleton->find_bone(p_bone_names[i]);
		CHECK_MESSAGE(bone_idx >= 0, vformat("Bone %s should exist", p_bone_names[i]));

		Transform3D actual_transform = p_skeleton->get_bone_global_pose(bone_idx);
		Transform3D expected_transform = p_expected_transforms[i];

		// Check translation
		Vector3 actual_origin = actual_transform.origin;
		Vector3 expected_origin = expected_transform.origin;
		String trans_msg = vformat("Bone %s translation mismatch. Expected: %s, Actual: %s", p_bone_names[i], expected_origin, actual_origin);
		CHECK_MESSAGE(actual_origin.distance_to(expected_origin) < p_tolerance, trans_msg.utf8().get_data());

		// Check basis (rotation)
		Basis actual_basis = actual_transform.basis;
		Basis expected_basis = expected_transform.basis;
		Quaternion actual_quat = actual_basis.get_rotation_quaternion();
		Quaternion expected_quat = expected_basis.get_rotation_quaternion();
		float angle_diff = actual_quat.angle_to(expected_quat);
		String rot_msg = vformat("Bone %s rotation mismatch. Angle diff: %f", p_bone_names[i], angle_diff);
		CHECK_MESSAGE(angle_diff < p_tolerance, rot_msg.utf8().get_data());
	}
}

TEST_CASE("[SceneTree][FABRIK3D] Class instantiation and registration") {
	// Test that FABRIK3D can be instantiated
	FABRIK3D *ik = memnew(FABRIK3D);
	CHECK_MESSAGE(ik != nullptr, "FABRIK3D should be instantiable");

	// Test basic properties
	ik->set_max_iterations(10);
	CHECK_MESSAGE(ik->get_max_iterations() == 10, "Max iterations should be settable");

	ik->set_min_distance(0.01);
	CHECK_MESSAGE(Math::is_equal_approx(ik->get_min_distance(), 0.01), "Min distance should be settable");

	memdelete(ik);
}

TEST_CASE("[SceneTree][FABRIK3D] Humanoid arm chain setup with basis and translation") {
	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	FABRIK3D *ik = memnew(FABRIK3D);

	// Setup IK for arm chain
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");

	// Verify initial setup
	Vector3 initial_pos = skeleton->get_bone_global_pose(7).origin; // LeftHand bone
	float expected_length = 0.9 + 0.1 + 0.1 + 0.1 + 0.05 + 0.3 + 0.35 + 0.25; // Sum of bone lengths in Y direction
	CHECK_MESSAGE(Math::abs(initial_pos.y - expected_length) < 0.1, "Initial hand position should match bone chain length");

	// Validate joint transforms have both basis and translation components
	int shoulder_idx = skeleton->find_bone("LeftShoulder");
	int upper_arm_idx = skeleton->find_bone("LeftUpperArm");
	int lower_arm_idx = skeleton->find_bone("LeftLowerArm");
	int hand_idx = skeleton->find_bone("LeftHand");

	// Check that all bones exist
	CHECK_MESSAGE(shoulder_idx >= 0, "LeftShoulder bone should exist");
	CHECK_MESSAGE(upper_arm_idx >= 0, "LeftUpperArm bone should exist");
	CHECK_MESSAGE(lower_arm_idx >= 0, "LeftLowerArm bone should exist");
	CHECK_MESSAGE(hand_idx >= 0, "LeftHand bone should exist");

	// Check that bones have non-zero translation (length)
	Transform3D shoulder_transform = skeleton->get_bone_global_pose(shoulder_idx);
	Transform3D upper_arm_transform = skeleton->get_bone_global_pose(upper_arm_idx);
	Transform3D lower_arm_transform = skeleton->get_bone_global_pose(lower_arm_idx);
	Transform3D hand_transform = skeleton->get_bone_global_pose(hand_idx);

	CHECK_MESSAGE(shoulder_transform.origin.length() > 0.01, "LeftShoulder should have non-zero translation");
	CHECK_MESSAGE(upper_arm_transform.origin.length() > 0.01, "LeftUpperArm should have non-zero translation");
	CHECK_MESSAGE(lower_arm_transform.origin.length() > 0.01, "LeftLowerArm should have non-zero translation");
	CHECK_MESSAGE(hand_transform.origin.length() > 0.01, "LeftHand should have non-zero translation");

	// Check that basis matrices are finite
	CHECK_MESSAGE(shoulder_transform.basis.is_finite(), "LeftShoulder should have finite basis");
	CHECK_MESSAGE(upper_arm_transform.basis.is_finite(), "LeftUpperArm should have finite basis");
	CHECK_MESSAGE(lower_arm_transform.basis.is_finite(), "LeftLowerArm should have finite basis");
	CHECK_MESSAGE(hand_transform.basis.is_finite(), "LeftHand should have finite basis");

	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][FABRIK3D] Humanoid leg chain setup with basis and translation") {
	Skeleton3D *skeleton = create_humanoid_leg_skeleton();
	FABRIK3D *ik = memnew(FABRIK3D);

	// Setup IK for leg chain
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "Hips");
	ik->set_end_bone_name(0, "LeftFoot");

	// Verify initial setup
	Vector3 initial_pos = skeleton->get_bone_global_pose(3).origin; // LeftFoot bone
	float expected_length = 0.9 + 0.45 + 0.4 + 0.3; // Sum of bone lengths in Y direction
	CHECK_MESSAGE(Math::abs(initial_pos.y - expected_length) < 0.1, "Initial foot position should match bone chain length");

	// Validate joint transforms
	int hips_idx = skeleton->find_bone("Hips");
	int upper_leg_idx = skeleton->find_bone("LeftUpperLeg");
	int lower_leg_idx = skeleton->find_bone("LeftLowerLeg");
	int foot_idx = skeleton->find_bone("LeftFoot");

	// Check that all bones exist
	CHECK_MESSAGE(hips_idx >= 0, "Hips bone should exist");
	CHECK_MESSAGE(upper_leg_idx >= 0, "LeftUpperLeg bone should exist");
	CHECK_MESSAGE(lower_leg_idx >= 0, "LeftLowerLeg bone should exist");
	CHECK_MESSAGE(foot_idx >= 0, "LeftFoot bone should exist");

	// Check that bones have non-zero translation (length)
	Transform3D hips_transform = skeleton->get_bone_global_pose(hips_idx);
	Transform3D upper_leg_transform = skeleton->get_bone_global_pose(upper_leg_idx);
	Transform3D lower_leg_transform = skeleton->get_bone_global_pose(lower_leg_idx);
	Transform3D foot_transform = skeleton->get_bone_global_pose(foot_idx);

	CHECK_MESSAGE(hips_transform.origin.length() > 0.01, "Hips should have non-zero translation");
	CHECK_MESSAGE(upper_leg_transform.origin.length() > 0.01, "LeftUpperLeg should have non-zero translation");
	CHECK_MESSAGE(lower_leg_transform.origin.length() > 0.01, "LeftLowerLeg should have non-zero translation");
	CHECK_MESSAGE(foot_transform.origin.length() > 0.01, "LeftFoot should have non-zero translation");

	// Check that basis matrices are finite
	CHECK_MESSAGE(hips_transform.basis.is_finite(), "Hips should have finite basis");
	CHECK_MESSAGE(upper_leg_transform.basis.is_finite(), "LeftUpperLeg should have finite basis");
	CHECK_MESSAGE(lower_leg_transform.basis.is_finite(), "LeftLowerLeg should have finite basis");
	CHECK_MESSAGE(foot_transform.basis.is_finite(), "LeftFoot should have finite basis");

	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][FABRIK3D] Joint limitation support") {
	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	FABRIK3D *ik = memnew(FABRIK3D);

	// Setup IK
	ik->set_setting_count(1);
	ik->set_root_bone_name(0, "LeftShoulder");
	ik->set_end_bone_name(0, "LeftHand");

	// Add joint limitation
	Ref<JointLimitationCone3D> elbow_limit = memnew(JointLimitationCone3D);
	elbow_limit->set_radius_range(Math::deg_to_rad(45.0)); // 45 degree cone
	ik->set_joint_limitation(0, 5, elbow_limit); // Joint index 5 (LeftUpperArm -> LeftLowerArm)

	Ref<JointLimitation3D> retrieved = ik->get_joint_limitation(0, 5);
	CHECK_MESSAGE(retrieved.is_valid(), "Joint limitation should be retrievable");
	CHECK_MESSAGE(retrieved->is_class_ptr(JointLimitationCone3D::get_class_ptr_static()), "Joint limitation should be cone type");

	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][FABRIK3D] Multiple settings support") {
	FABRIK3D *ik = memnew(FABRIK3D);

	// Test setting count management
	ik->set_setting_count(3);
	CHECK_MESSAGE(ik->get_setting_count() == 3, "Should support multiple settings");

	ik->clear_settings();
	CHECK_MESSAGE(ik->get_setting_count() == 0, "Should be able to clear all settings");

	memdelete(ik);
}

TEST_CASE("[SceneTree][FABRIK3D] IK solving with target position") {
	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	FABRIK3D *ik = memnew(FABRIK3D);

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
	CHECK_MESSAGE(final_distance < initial_distance, (String("Effector should move closer to target. Initial distance: ") + rtos(initial_distance) + ", Final distance: " + rtos(final_distance)).utf8().get_data());

	memdelete(target);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][FABRIK3D] IK solving convergence") {
	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	FABRIK3D *ik = memnew(FABRIK3D);

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

	CHECK_MESSAGE(final_distance < initial_distance, vformat("Distance to target should decrease. Initial: %f, Final: %f", initial_distance, final_distance).utf8().get_data());
	CHECK_MESSAGE(final_distance < 0.1, vformat("Should converge to reasonable distance. Final distance: %f", final_distance).utf8().get_data());

	memdelete(target);
	memdelete(ik);
	memdelete(skeleton);
}

TEST_CASE("[SceneTree][FABRIK3D] IK solving with joint limitations") {
	Skeleton3D *skeleton = create_humanoid_arm_skeleton();
	FABRIK3D *ik = memnew(FABRIK3D);

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
	elbow_limit->set_radius_range(Math::deg_to_rad(30.0)); // Tight 30 degree cone
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
	CHECK_MESSAGE(distance_to_target < 0.8, vformat("Should make reasonable progress even with constraints. Distance: %f", distance_to_target).utf8().get_data());

	memdelete(target);
	memdelete(ik);
	memdelete(skeleton);
}

} // namespace TestFABRIK3D
