/**************************************************************************/
/*  ewbik_3d_.cpp                                                         */
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
#include "core/templates/hash_set.h"

void EWBIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) {
	int joint_size = (int)p_setting->joints.size();
	int chain_size = (int)p_setting->chain.size();

	// Find the setting index
	int setting_idx = -1;
	for (int i = 0; i < (int)iterate_settings.size(); i++) {
		if (iterate_settings[i] == p_setting) {
			setting_idx = i;
			break;
		}
	}
	if (setting_idx == -1) {
		return; // Should not happen
	}
	// Build and sort effector groups using Eron's decomposition algorithm
	// Collect effectors from all settings
	Vector<Effector> all_effectors;
	for (int setting_idx = 0; setting_idx < (int)iterate_settings.size(); setting_idx++) {
		IterateIK3D::IterateIK3DSetting *iter_setting = iterate_settings[setting_idx];
		int last_idx = iter_setting->joints.size() - 1;
		if (last_idx >= 0 && last_idx < (int)iter_setting->chain.size()) {
			Effector eff;
			eff.effector_bone = iter_setting->joints[last_idx].bone;
			eff.root_bone = iter_setting->joints[0].bone;
			eff.target_position = iter_setting->chain[last_idx];
			eff.opacity = get_effector_opacity(setting_idx);
			all_effectors.push_back(eff);
		}
	}

	Vector<EffectorGroup> effector_groups;
	_build_effector_groups(p_skeleton, all_effectors, effector_groups);
	effector_groups.sort_custom<EffectorGroupComparator>();

	// Create solve_order by traversing effector groups
	Vector<int> solve_order;
	for (const EffectorGroup &group : effector_groups) {
		for (int bone : group.bones) {
			solve_order.push_back(bone);
		}
	}

	// Get the target transform for directional constraints
	Transform3D target_transform;
	Node3D *target = Object::cast_to<Node3D>(get_node_or_null(p_setting->target_node));
	if (target) {
		target_transform = cached_space.affine_inverse() * target->get_global_transform_interpolated();
	} else {
		return; // Not reached target
	}

	// Create local solve_order for the current setting
	Vector<int> local_solve_order;
	for (int bone : solve_order) {
		for (int j = 0; j < joint_size; j++) {
			if (p_setting->joints[j].bone == bone) {
				local_solve_order.push_back(j);
				break;
			}
		}
	}

	// Solve using the optimized order
	for (int i : local_solve_order) {
		IKModifier3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}

		int HEAD = i;
		int TAIL = i + 1;

		Vector3 current_head = p_setting->chain[HEAD];
		Vector3 current_effector = p_setting->chain[chain_size - 1];
		Vector3 head_to_effector = current_effector - current_head;
		Vector3 head_to_destination = p_destination - current_head;

		if (Math::is_zero_approx(head_to_destination.length_squared() * head_to_effector.length_squared())) {
			continue;
		}

		// Create point correspondences for QCP solving
		PackedVector3Array target_headings;
		PackedVector3Array tip_headings;
		Vector<double> weights;

		_create_point_correspondences(p_skeleton, p_setting, i, p_destination, target_transform, target_headings, tip_headings, weights);

		// Calculate optimal rotation and translation using QCP
		OptimalTransform opt = _calculate_optimal_rotation(target_headings, tip_headings, weights, true);
		Quaternion to_rot = opt.rotation;
		Vector3 translation = opt.translation;

		// Only apply translation for root motion (first bone in chain)
		if (i != 0) {
			translation = Vector3();
		}

		Vector3 new_head = current_head + translation;
		Vector3 to_tail = p_setting->chain[TAIL] - current_head;

		p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, new_head + to_rot.xform(to_tail));
		p_setting->chain[HEAD] = new_head;

		if (p_setting->joint_settings[HEAD]->rotation_axis != ROTATION_AXIS_ALL) {
			p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, p_setting->chain[HEAD] + p_setting->joint_settings[HEAD]->get_projected_rotation(solver_info->current_grest, p_setting->chain[TAIL] - p_setting->chain[HEAD]));
		}
		if (p_setting->joint_settings[HEAD]->limitation.is_valid()) {
			p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, p_setting->chain[HEAD] + p_setting->joint_settings[HEAD]->get_limited_rotation(solver_info->current_grest, p_setting->chain[TAIL] - p_setting->chain[HEAD], solver_info->forward_vector));
		}
	}
}

void EWBIK3D::_create_point_correspondences(Skeleton3D *p_skeleton, const IterateIK3DSetting *p_setting, int p_bone_idx, const Vector3 &p_destination, const Transform3D &p_target_transform,
		PackedVector3Array &r_target_headings, PackedVector3Array &r_tip_headings, Vector<double> &r_weights) {
	int chain_size = (int)p_setting->chain.size();
	if (chain_size < 2 || p_bone_idx < 0 || p_bone_idx >= (int)p_setting->joints.size()) {
		return;
	}

	IKModifier3DSolverInfo *solver_info = p_setting->solver_info_list[p_bone_idx];
	if (!solver_info || Math::is_zero_approx(solver_info->length)) {
		return;
	}

	// Get the current end effector position and target
	Vector3 current_effector = p_setting->chain[chain_size - 1];
	Vector3 bone_origin = p_setting->chain[p_bone_idx];
	Vector3 head_to_effector = current_effector - bone_origin;
	Vector3 head_to_destination = p_destination - bone_origin;

	if (Math::is_zero_approx(head_to_destination.length_squared() * head_to_effector.length_squared())) {
		return;
	}

	// Get the current bone's global transform to access its orientation
	Transform3D current_bone_transform = p_skeleton->get_bone_global_pose(p_setting->joints[p_bone_idx].bone);
	Basis current_bone_basis = current_bone_transform.basis;

	// Add origin correspondence
	r_target_headings.push_back(head_to_destination);
	r_tip_headings.push_back(head_to_effector);
	r_weights.push_back(1.0f);

	// Scale for basis vectors as per design: distance >=1 and >= magnitude of origin
	float distance_to_target = head_to_destination.length();
	float scale = MAX(1.0f, distance_to_target);

	// Add directional constraints for X, Y, Z axes (6 points total: 2 per axis)
	// Representing basis vectors emanating from origin and their opposites
	for (int axis = 0; axis < 3; axis++) {
		// Get the target's basis column for this axis (desired orientation)
		Vector3 target_axis = p_target_transform.basis.get_column(axis);
		// Get the current bone's basis column for this axis (current orientation)
		Vector3 current_axis = current_bone_basis.get_column(axis);

		// Positive direction: emanating basis vector
		r_target_headings.push_back(head_to_destination + target_axis * scale);
		r_tip_headings.push_back(head_to_effector + current_axis * scale);
		r_weights.push_back(1.0f);

		// Negative direction: opposite basis vector
		r_target_headings.push_back(head_to_destination - target_axis * scale);
		r_tip_headings.push_back(head_to_effector - current_axis * scale);
		r_weights.push_back(1.0f);

		// Add origin correspondence
		Vector3 axis_vector = Vector3(0, 0, 0);
		axis_vector[axis] = 1.0;
		r_target_headings.push_back(head_to_destination * axis_vector);
		r_tip_headings.push_back(head_to_effector * axis_vector);
		r_weights.push_back(1.0f);
	}
}

OptimalTransform EWBIK3D::_calculate_optimal_rotation(const PackedVector3Array &p_target_headings, const PackedVector3Array &p_tip_headings, const Vector<double> &p_weights, bool p_calculate_translation) {
	if (p_target_headings.size() != p_tip_headings.size() || p_target_headings.size() != p_weights.size()) {
		return OptimalTransform(); // Invalid input
	}

	if (p_target_headings.is_empty()) {
		return OptimalTransform(); // No correspondences
	}

	// Use QCP to find optimal rotation and optionally translation
	Array result = QuaternionCharacteristicPolynomial::weighted_superpose(p_tip_headings, p_target_headings, p_weights, p_calculate_translation);
	OptimalTransform opt;
	opt.rotation = result[0];
	opt.rotation = opt.rotation.inverse();
	if (p_calculate_translation && result.size() > 1) {
		opt.translation = result[1];
	} else {
		opt.translation = Vector3();
	}
	return opt;
}

void EWBIK3D::_update_joints(int p_index) {
	_make_simulation_dirty(p_index);

#ifdef TOOLS_ENABLED
	update_gizmos(); // To clear invalid setting.
#endif // TOOLS_ENABLED

	Skeleton3D *sk = get_skeleton();
	int root_bone = chain_settings[p_index]->root_bone.bone;
	int end_bone = chain_settings[p_index]->end_bone.bone;
	if (!sk || root_bone < 0 || end_bone < 0) {
		set_joint_count(p_index, 0);
		return;
	}

	// Find path between root and end bones (supports arbitrary paths, not just hierarchical)
	Vector<int> bone_path;
	if (!_find_bone_chain_path(sk, root_bone, end_bone, bone_path)) {
		set_joint_count(p_index, 0);
		ERR_FAIL_EDMSG("Cannot find a valid bone chain between root and end bones.");
		return;
	}

	// Build joints from the path
	LocalVector<BoneJoint> new_joints;
	_build_chain_from_path(sk, bone_path, new_joints);

	set_joint_count(p_index, new_joints.size());
	for (uint32_t i = 0; i < new_joints.size(); i++) {
		_set_joint_bone(p_index, i, new_joints[i].bone);
	}

	if (sk) {
		_validate_axes(sk);
	}

#ifdef TOOLS_ENABLED
	_make_gizmo_dirty();
#endif // TOOLS_ENABLED
}
