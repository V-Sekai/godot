/**************************************************************************/
/*  iterate_ik_3d_ewbik.cpp                                               */
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

#include "ewbik_3d.h"

#include "core/math/qcp.h"

void EWBIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) {
	int joint_size = (int)p_setting->joints.size();

	// Get the target transform for directional constraints
	Transform3D target_transform;
	Node3D *target = Object::cast_to<Node3D>(get_node_or_null(p_setting->target_node));
	if (target) {
		target_transform = cached_space.affine_inverse() * target->get_global_transform_interpolated();
	}

	// For each bone in the chain, solve using QCP
	for (int i = 0; i < joint_size; i++) {
		ManyBoneIK3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}

		// Create point correspondences for QCP solving
		PackedVector3Array target_headings;
		PackedVector3Array tip_headings;
		Vector<double> weights;

		_create_point_correspondences(p_skeleton, p_setting, p_destination, target_transform, target_headings, tip_headings, weights);

		// Apply QCP rotation to this bone
		_apply_qcp_rotation(p_skeleton, p_setting, i, target_headings, tip_headings, weights);
	}
}

void EWBIK3D::_create_point_correspondences(Skeleton3D *p_skeleton, const IterateIK3DSetting *p_setting, const Vector3 &p_destination, const Transform3D &p_target_transform,
		PackedVector3Array &r_target_headings, PackedVector3Array &r_tip_headings, Vector<double> &r_weights) {

	int chain_size = (int)p_setting->chain.size();
	if (chain_size < 2) {
		return;
	}

	// Get the current end effector position and target
	Vector3 current_effector = p_setting->chain[chain_size - 1];
	Vector3 target_direction = (p_destination - current_effector).normalized();

	// For each bone, create point correspondences
	for (int i = 0; i < (int)p_setting->joints.size(); i++) {
		ManyBoneIK3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}

		Vector3 bone_origin = p_setting->chain[i];
		// Get the current bone's global transform to access its orientation
		Transform3D current_bone_transform = p_skeleton->get_bone_global_pose(p_setting->joints[i].bone);
		Basis current_bone_basis = current_bone_transform.basis;

		// Weight based on distance from this bone to end effector
		float distance_to_effector = bone_origin.distance_to(current_effector);
		float weight = 1.0f / (1.0f + distance_to_effector); // Closer bones have higher weight

		// Target heading: direction from bone to target (relative to bone origin)
		Vector3 target_heading = target_direction * solver_info->length;
		r_target_headings.push_back(target_heading);
		r_weights.push_back(weight);

		// Tip heading: current direction from bone to end effector (relative to bone origin)
		Vector3 tip_heading = (current_effector - bone_origin).normalized() * solver_info->length;
		r_tip_headings.push_back(tip_heading);
		r_weights.push_back(weight);

		// Add directional constraints for X, Y, Z axes (6 points total: 2 per axis)
		// QCP finds rotation that aligns current bone orientation with target orientation
		for (int axis = 0; axis < 3; axis++) {
			// Get the target's basis column for this axis (desired orientation)
			Vector3 target_axis = p_target_transform.basis.get_column(axis);
			// Get the current bone's basis column for this axis (current orientation)
			Vector3 current_axis = current_bone_basis.get_column(axis);

			// Positive direction constraint: align current bone axis with target axis
			r_target_headings.push_back(target_axis * solver_info->length);
			r_tip_headings.push_back(current_axis * solver_info->length);
			r_weights.push_back(weight * 0.3f); // Medium weight for directional constraints

			// Negative direction constraint: align opposite current bone axis with target axis
			r_target_headings.push_back(target_axis * solver_info->length);
			r_tip_headings.push_back(-current_axis * solver_info->length);
			r_weights.push_back(weight * 0.3f); // Medium weight for directional constraints
		}
	}
}

void EWBIK3D::_apply_qcp_rotation(Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, int p_bone_idx,
		const PackedVector3Array &p_target_headings, const PackedVector3Array &p_tip_headings, const Vector<double> &p_weights) {

	if (p_bone_idx >= (int)p_setting->joints.size()) {
		return;
	}

	ManyBoneIK3DSolverInfo *solver_info = p_setting->solver_info_list[p_bone_idx];
	if (!solver_info || Math::is_zero_approx(solver_info->length)) {
		return;
	}

	// Calculate optimal rotation using QCP
	Quaternion optimal_rotation = _calculate_optimal_rotation(p_target_headings, p_tip_headings, p_weights);

	// Apply joint limitations if they exist
	IterateIK3DJointSetting *joint_setting = p_setting->joint_settings[p_bone_idx];
	if (joint_setting && joint_setting->limitation.is_valid()) {
		// Get the limited rotation
		Vector3 limited_vector = joint_setting->get_limited_rotation(solver_info->current_grest, optimal_rotation.xform(Vector3(1, 0, 0)), solver_info->forward_vector);
		optimal_rotation = Quaternion(solver_info->forward_vector, limited_vector);
	}

	// Apply rotation axis constraints
	if (joint_setting && joint_setting->rotation_axis != ROTATION_AXIS_ALL) {
		// Project rotation onto allowed axis
		Vector3 rot_axis = joint_setting->get_rotation_axis_vector();
		if (rot_axis.is_zero_approx()) {
			return; // No valid rotation axis
		}

		// Extract rotation around the allowed axis
		float angle = optimal_rotation.get_angle();
		Vector3 axis = optimal_rotation.get_axis();
		float projection = axis.dot(rot_axis);
		if (!Math::is_zero_approx(projection)) {
			angle *= projection;
			optimal_rotation = Quaternion(rot_axis, angle);
		}
	}

	// Apply angular delta limiting to prevent oscillation
	Quaternion prev_rotation = solver_info->current_lpose;
	Quaternion new_rotation = solver_info->current_lrest * optimal_rotation;

	double diff = prev_rotation.angle_to(new_rotation);
	if (!Math::is_zero_approx(diff)) {
		new_rotation = prev_rotation.slerp(new_rotation, MIN(1.0, angular_delta_limit / diff));
	}

	// Update the bone pose
	p_skeleton->set_bone_pose_rotation(p_setting->joints[p_bone_idx].bone, new_rotation);
	solver_info->current_lpose = new_rotation;
}

Quaternion EWBIK3D::_calculate_optimal_rotation(const PackedVector3Array &p_target_headings, const PackedVector3Array &p_tip_headings, const Vector<double> &p_weights) {
	if (p_target_headings.size() != p_tip_headings.size() || p_target_headings.size() != p_weights.size()) {
		return Quaternion(); // Invalid input
	}

	if (p_target_headings.is_empty()) {
		return Quaternion(); // No correspondences
	}

	// Use QCP to find optimal rotation
	Array result = QuaternionCharacteristicPolynomial::weighted_superpose(p_tip_headings, p_target_headings, p_weights, false);
	return result[0]; // First element is the rotation quaternion
}

void EWBIK3D::_update_joints(int p_index) {
	IterateIK3DSetting *setting = iterate_settings[p_index];
	Skeleton3D *sk = get_skeleton();

	if (!sk || setting->root_bone.bone < 0 || setting->end_bone.bone < 0) {
		set_joint_count(p_index, 0);
		return;
	}

	setting->joints.clear();

	// If root and end are the same, just return that bone
	if (setting->root_bone.bone == setting->end_bone.bone) {
		BoneJoint joint;
		joint.bone = setting->root_bone.bone;
		joint.name = sk->get_bone_name(setting->root_bone.bone);
		setting->joints.push_back(joint);
		set_joint_count(p_index, setting->joints.size());
		_make_simulation_dirty(p_index);
		return;
	}

	// Build path from end bone up to root
	LocalVector<BoneJoint> temp_joints;
	int current_bone = setting->end_bone.bone;
	bool found_root = false;
	while (current_bone != -1) {
		BoneJoint joint;
		joint.bone = current_bone;
		joint.name = sk->get_bone_name(current_bone);
		temp_joints.push_back(joint);
		if (current_bone == setting->root_bone.bone) {
			found_root = true;
			break;
		}
		current_bone = sk->get_bone_parent(current_bone);
	}

	if (!found_root) {
		set_joint_count(p_index, 0);
		ERR_FAIL_EDMSG("Cannot find a valid bone chain between root and end bones. Root is not an ancestor of end bone.");
		return;
	}

	// Reverse to get order from root to end
	for (int i = temp_joints.size() - 1; i >= 0; --i) {
		setting->joints.push_back(temp_joints[i]);
	}

	set_joint_count(p_index, setting->joints.size());

	// Initialize solver info and other structures
	_make_simulation_dirty(p_index);
}
