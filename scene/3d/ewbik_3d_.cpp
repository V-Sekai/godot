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

#include "ewbik_3d_.h"

#include "core/math/qcp.h"

void EWBIK3D_::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) {
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

void EWBIK3D_::_create_point_correspondences(Skeleton3D *p_skeleton, const IterateIK3DSetting *p_setting, const Vector3 &p_destination, const Transform3D &p_target_transform,
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

void EWBIK3D_::_apply_qcp_rotation(Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, int p_bone_idx,
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
		// Get the desired direction after optimal rotation (where the bone's forward vector would point)
		Vector3 desired_direction = optimal_rotation.xform(solver_info->forward_vector);
		// Apply limitation to constrain this direction
		Vector3 limited_direction = joint_setting->get_limited_rotation(solver_info->current_grest, desired_direction, solver_info->forward_vector);
		// Calculate rotation from current forward to limited direction
		if (!limited_direction.is_zero_approx()) {
			optimal_rotation = Quaternion(solver_info->forward_vector, limited_direction.normalized());
		}
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

Quaternion EWBIK3D_::_calculate_optimal_rotation(const PackedVector3Array &p_target_headings, const PackedVector3Array &p_tip_headings, const Vector<double> &p_weights) {
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

void EWBIK3D_::_update_joints(int p_index) {
	IterateIK3DSetting *setting = iterate_settings[p_index];
	Skeleton3D *sk = get_skeleton();

	if (!sk || setting->root_bone.bone < 0 || setting->end_bone.bone < 0) {
		set_joint_count(p_index, 0);
		return;
	}

	// Find path between root and end bones
	Vector<int> bone_path;
	if (!_find_bone_chain_path(sk, setting->root_bone.bone, setting->end_bone.bone, bone_path)) {
		set_joint_count(p_index, 0);
		ERR_FAIL_EDMSG("Cannot find a valid bone chain between root and end bones.");
		return;
	}

	// Build joints from the path
	_build_chain_from_path(sk, bone_path, setting->joints);
	set_joint_count(p_index, setting->joints.size());

	// Initialize solver info and other structures
	_make_simulation_dirty(p_index);
}

bool EWBIK3D_::_find_bone_chain_path(Skeleton3D *p_skeleton, int p_root_bone, int p_end_bone, Vector<int> &r_chain) const {
	r_chain.clear();

	// If root and end are the same, just return that bone
	if (p_root_bone == p_end_bone) {
		r_chain.push_back(p_root_bone);
		return true;
	}

	// Build path from end bone up to find common ancestor with root
	Vector<int> end_to_root_path;
	int current = p_end_bone;
	while (current >= 0) {
		end_to_root_path.push_back(current);
		if (current == p_root_bone) {
			// Root is an ancestor of end bone - use hierarchical path
			r_chain = end_to_root_path;
			r_chain.reverse();
			return true;
		}
		current = p_skeleton->get_bone_parent(current);
	}

	// If we get here, root is not an ancestor. Find common ancestor.
	// Build path from root up to root
	Vector<int> root_to_root_path;
	current = p_root_bone;
	while (current >= 0) {
		root_to_root_path.push_back(current);
		current = p_skeleton->get_bone_parent(current);
	}

	// Find common ancestor
	int common_ancestor = -1;
	for (int root_ancestor : root_to_root_path) {
		for (int end_ancestor : end_to_root_path) {
			if (root_ancestor == end_ancestor) {
				common_ancestor = root_ancestor;
				goto found_common;
			}
		}
	}

	found_common:
	if (common_ancestor == -1) {
		return false; // No common ancestor - impossible to connect
	}

	// Build the path: root -> common_ancestor -> end
	r_chain.clear();

	// Add path from root to common ancestor
	for (int bone : root_to_root_path) {
		r_chain.push_back(bone);
		if (bone == common_ancestor) {
			break;
		}
	}

	// Add path from common ancestor to end (excluding common ancestor since it's already added)
	bool found_common_in_end_path = false;
	for (int i = end_to_root_path.size() - 1; i >= 0; i--) {
		int bone = end_to_root_path[i];
		if (found_common_in_end_path) {
			r_chain.push_back(bone);
		} else if (bone == common_ancestor) {
			found_common_in_end_path = true;
		}
	}

	return true;
}

void EWBIK3D_::_build_chain_from_path(Skeleton3D *p_skeleton, const Vector<int> &p_path, LocalVector<BoneJoint> &r_joints) const {
	r_joints.clear();
	r_joints.resize(p_path.size());

	for (int i = 0; i < p_path.size(); i++) {
		int bone_idx = p_path[i];
		r_joints[i].bone = bone_idx;
		r_joints[i].name = p_skeleton->get_bone_name(bone_idx);
	}
}
