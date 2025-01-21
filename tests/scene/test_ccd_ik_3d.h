/**************************************************************************/
/*  test_ccd_ik_3d.h                                                      */
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

#include "scene/3d/ccd_ik_3d.h"

#include "scene/resources/skeleton_profile.h"

#include "tests/test_macros.h"

namespace TestCCDIK3D {

class TestCCDIK3D : public CCDIK3D {
	GDCLASS(TestCCDIK3D, CCDIK3D);

public:
	using CCDIK3D::_process_joints;
};

struct JointTailInfo {
	Vector3 current_tail;
	Vector3 prev_tail;
};

TEST_CASE("[CCDIK3D] solve function with identity rotation") {
	CCDIK3D *ik = memnew(CCDIK3D);
	Vector<Transform3D> targets;
	targets.push_back(Transform3D());
	Vector<Transform3D> moveds;
	moveds.push_back(Transform3D());
	Transform3D result_transform = ik->_solve(targets, moveds, false);
	INFO("Target Transform:", targets[0]);
	INFO("Result Transform:", result_transform);
	INFO("Is target approximately equal to result:", targets[0].is_equal_approx(result_transform));
	CHECK(targets[0].is_equal_approx(result_transform));
	CHECK((Transform3D() * moveds[0]).is_equal_approx(targets[0]));
	memdelete_notnull(ik);
}

TEST_CASE("[CCDIK3D] solve function with translation only") {
	CCDIK3D *ik = memnew(CCDIK3D);
	Vector<Transform3D> targets;
	Transform3D target_transform;
	target_transform.origin = Vector3(10, 20, 30);
	targets.push_back(target_transform);
	Vector<Transform3D> moveds;
	moveds.push_back(Transform3D());
	Transform3D result_transform = ik->_solve(targets, moveds, true);
	INFO("Target Origin:", targets[0].origin);
	INFO("Result Origin:", result_transform.origin);
	INFO("Target Basis:", targets[0].basis);
	INFO("Result Basis:", result_transform.basis);
	INFO("Is origin approximately equal:", targets[0].origin.is_equal_approx(result_transform.origin));
	INFO("Is basis approximately equal:", result_transform.basis.is_equal_approx(target_transform.basis));
	CHECK(targets[0].origin.is_equal_approx(result_transform.origin));
	CHECK(targets[0].basis.is_equal_approx(result_transform.basis)); // Basis should remain unchanged
	CHECK((target_transform * moveds[0]).is_equal_approx(targets[0]));
	memdelete_notnull(ik);
}

TEST_CASE("[CCDIK3D] solve function with rotation only") {
	CCDIK3D *ik = memnew(CCDIK3D);
	Vector<Transform3D> targets;
	Transform3D target_transform;
	target_transform.basis = Basis(Quaternion(0.3, 0.4, 0.5, 0.6).normalized());
	targets.push_back(target_transform);
	Vector<Transform3D> moveds;
	moveds.push_back(Transform3D());
	Transform3D result_transform = ik->_solve(targets, moveds, false);
	INFO("Target Basis:", targets[0].basis);
	INFO("Result Basis:", result_transform.basis);
	INFO("Result Origin:", result_transform.origin);
	INFO("Expected Origin:", Vector3());
	INFO("Is basis approximately equal:", result_transform.basis.is_equal_approx(target_transform.basis));
	INFO("Is origin approximately equal to expected:", result_transform.origin.is_equal_approx(Vector3()));
	CHECK(result_transform.basis.is_equal_approx(target_transform.basis));
	// Origin should remain the same if p_translate is false
	CHECK(result_transform.origin.is_equal_approx(Vector3()));
	CHECK((target_transform * moveds[0]).is_equal_approx(targets[0]));
	memdelete_notnull(ik);
}

TEST_CASE("[CCDIK3D] solve function with multiple transforms") {
	CCDIK3D *ik = memnew(CCDIK3D);
	Vector<Transform3D> targets;

	Transform3D target_transform(Quaternion(0.5, 0.6, 0.7, 0.8).normalized(), Vector3());
	CHECK(target_transform.origin.is_zero_approx());

	Transform3D target_transform_1;
	target_transform_1.origin = Vector3(1, 2, 3);
	target_transform_1.basis = Basis(Quaternion(0.1, 0.2, 0.3, 0.4).normalized());
	CHECK_FALSE(target_transform_1.origin.is_zero_approx());
	CHECK_FALSE(target_transform_1.basis.is_equal_approx(Basis()));
	targets.push_back(target_transform * target_transform_1);

	Transform3D target_transform_2;
	target_transform_2.origin = Vector3(4, 5, 6);
	target_transform_2.basis = Basis(Quaternion(0.1, 0.2, 0.3, 0.4).normalized());
	CHECK_FALSE(target_transform_2.origin.is_zero_approx());
	CHECK_FALSE(target_transform_2.basis.is_equal_approx(Basis()));
	targets.push_back(target_transform * target_transform_2);

	Vector<Transform3D> moveds;
	moveds.push_back(target_transform_1);
	moveds.push_back(target_transform_2);

	Transform3D result_transform = ik->_solve(targets, moveds, false);
	INFO("Target 1: ", targets[0]);
	INFO("Target 2: ", targets[1]);
	INFO("Result Transform: ", result_transform);
	INFO("Result 1: ", result_transform * moveds[0]);
	INFO("Result 2: ", result_transform * moveds[1]);
	CHECK((result_transform * moveds[0]).is_equal_approx(targets[0]));
	CHECK((result_transform * moveds[1]).is_equal_approx(targets[1]));
	memdelete_notnull(ik);
}

TEST_CASE("[SceneTree][CCDIK3D] _process_joints with root to hips joints") {
	TestCCDIK3D *ik = memnew(TestCCDIK3D);
	double delta = 0.016; // Assume 60 FPS.
	Skeleton3D *skeleton = memnew(Skeleton3D);
	skeleton->add_child(ik);
	ik->set_owner(skeleton);

	const String root_bone_name = "Root";
	const String hips_bone_name = "Hips";
	skeleton->add_bone(root_bone_name);
	skeleton->add_bone(hips_bone_name);
	skeleton->set_bone_parent(skeleton->find_bone(hips_bone_name), skeleton->find_bone(root_bone_name));

	Vector<CCDIK3D::ManyBoneIK3DJointSetting *> joints;
	CCDIK3D::ManyBoneIK3DJointSetting *root_joint = memnew(CCDIK3D::ManyBoneIK3DJointSetting);
	root_joint->bone = skeleton->find_bone(root_bone_name);
	root_joint->bone_name = root_bone_name;
	root_joint->twist_limitation = Math_PI;
	root_joint->constraint = nullptr;
	root_joint->constraint_rotation_offset = Quaternion();
	root_joint->solver_info = memnew(CCDIK3D::ManyBoneIK3DSolverInfo);

	JointTailInfo root_tail_info;
	root_tail_info.prev_tail = Vector3(0, 0, 0);
	root_tail_info.current_tail = Vector3(0, 1, 0);
	root_joint->solver_info->forward_vector = Vector3(0, 1, 0);
	root_joint->solver_info->current_rot = Quaternion(0, 1, 0, 1).normalized();
	root_joint->solver_info->length = (root_tail_info.current_tail - root_tail_info.prev_tail).length();
	joints.push_back(root_joint);

	CCDIK3D::ManyBoneIK3DJointSetting *hips_joint = memnew(CCDIK3D::ManyBoneIK3DJointSetting);
	hips_joint->bone = skeleton->find_bone(hips_bone_name);
	hips_joint->bone_name = hips_bone_name;
	hips_joint->twist_limitation = Math_PI;
	hips_joint->constraint = nullptr;
	hips_joint->constraint_rotation_offset = Quaternion();
	hips_joint->solver_info = memnew(CCDIK3D::ManyBoneIK3DSolverInfo);

	JointTailInfo hips_tail_info;
	hips_tail_info.prev_tail = root_tail_info.current_tail;
	hips_tail_info.current_tail = Vector3(0, 0, 0);
	hips_joint->solver_info->forward_vector = Vector3(0, 1, 0);
	hips_joint->solver_info->current_rot = Quaternion(0, 0, 1, 1).normalized();
	hips_joint->solver_info->length = (hips_tail_info.current_tail - hips_tail_info.prev_tail).length();
	joints.push_back(hips_joint);

	Vector<Vector3> p_chain;
	p_chain.push_back(root_tail_info.current_tail);
	p_chain.push_back(hips_tail_info.current_tail);

	Transform3D root_reference_pose;
	root_reference_pose.basis = Basis(Quaternion(0, 1, 0, 1).normalized());
	root_reference_pose.origin = Vector3(0, 1, 0);
	skeleton->set_bone_rest(root_joint->bone, root_reference_pose);

	Transform3D hips_reference_pose;
	hips_reference_pose.basis = Basis(Quaternion(0, 0, 1, 1).normalized());
	skeleton->set_bone_rest(hips_joint->bone, hips_reference_pose);

	skeleton->reset_bone_poses();
	Transform3D space = Transform3D();
	Vector3 target_vector = Vector3(0, 1, 0);
	int max_iterations = 10;
	real_t min_distance = CMP_EPSILON;
	Vector3 destination = Vector3();
	ik->_process_joints(delta, skeleton, joints, p_chain, space, destination, target_vector, max_iterations, min_distance);

	CHECK_EQ(skeleton->get_bone_pose(root_joint->bone).origin, root_reference_pose.origin);
	CHECK_EQ(skeleton->get_bone_pose(hips_joint->bone).origin, hips_reference_pose.origin);

	memdelete_notnull(root_joint->solver_info);
	memdelete_notnull(root_joint);
	memdelete_notnull(hips_joint->solver_info);
	memdelete_notnull(hips_joint);
	memdelete_notnull(ik);
	memdelete_notnull(skeleton);
}
} // namespace TestCCDIK3D
