/**************************************************************************/
/*  iterate_ik_3d.cpp                                                     */
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

#include "iterate_ik_3d.h"

bool IterateIK3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "target_node") {
			set_target_node(which, p_value);
		} else if (what == "pole_node") {
			set_pole_node(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool IterateIK3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "target_node") {
			r_ret = get_target_node(which);
		} else if (what == "pole_node") {
			r_ret = get_pole_node(which);
		} else {
			return false;
		}
	}
	return true;
}

void IterateIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
	LocalVector<PropertyInfo> props;

	for (int i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, path + "target_node"));
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, path + "pole_node"));
	}
}

void IterateIK3D::set_max_iterations(int p_max_iterations) {
	max_iterations = p_max_iterations;
}

int IterateIK3D::get_max_iterations() const {
	return max_iterations;
}

void IterateIK3D::set_min_distance(real_t p_min_distance) {
	min_distance = p_min_distance;
}

real_t IterateIK3D::get_min_distance() const {
	return min_distance;
}

void IterateIK3D::set_angular_delta_limit(real_t p_angular_delta_limit) {
	angular_delta_limit = p_angular_delta_limit;
}

real_t IterateIK3D::get_angular_delta_limit() const {
	return angular_delta_limit;
}

// Setting.

void IterateIK3D::set_target_node(int p_index, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, settings.size());
	iterate_settings[p_index]->target_node = p_node_path;
}

NodePath IterateIK3D::get_target_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), NodePath());
	return iterate_settings[p_index]->target_node;
}

void IterateIK3D::set_pole_node(int p_index, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, settings.size());
	iterate_settings[p_index]->pole_node = p_node_path;
}

NodePath IterateIK3D::get_pole_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), NodePath());
	return iterate_settings[p_index]->pole_node;
}

// Individual joints.

void IterateIK3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_max_iterations", "max_iterations"), &IterateIK3D::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations"), &IterateIK3D::get_max_iterations);
	ClassDB::bind_method(D_METHOD("set_min_distance", "min_distance"), &IterateIK3D::set_min_distance);
	ClassDB::bind_method(D_METHOD("get_min_distance"), &IterateIK3D::get_min_distance);
	ClassDB::bind_method(D_METHOD("set_angular_delta_limit", "angular_delta_limit"), &IterateIK3D::set_angular_delta_limit);
	ClassDB::bind_method(D_METHOD("get_angular_delta_limit"), &IterateIK3D::get_angular_delta_limit);

	// Setting.
	ClassDB::bind_method(D_METHOD("set_target_node", "index", "target_node"), &IterateIK3D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node", "index"), &IterateIK3D::get_target_node);
	ClassDB::bind_method(D_METHOD("set_pole_node", "index", "pole_node"), &IterateIK3D::set_pole_node);
	ClassDB::bind_method(D_METHOD("get_pole_node", "index"), &IterateIK3D::get_pole_node);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_iterations", PROPERTY_HINT_RANGE, "0,100,or_greater"), "set_max_iterations", "get_max_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_distance", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"), "set_min_distance", "get_min_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_delta_limit", PROPERTY_HINT_RANGE, "0,180,0.01,radians_as_degrees"), "set_angular_delta_limit", "get_angular_delta_limit");
}

void IterateIK3D::_post_init_joints(int p_index) {
	IterateIK3DSetting *setting = iterate_settings[p_index];
	// For pole target.
	setting->joint_size_half = (int)(setting->joints.size() * 0.5);
	int count = 0;
	for (int i = 0; i < setting->joint_size_half; i++) {
		if (!setting->joints[i]->solver_info) {
			continue;
		}
		count++;
	}
	setting->chain_size_half = count;
}

void IterateIK3D::_process_ik(Skeleton3D *p_skeleton, double p_delta) {
	min_distance_squared = min_distance * min_distance;
	for (int i = 0; i < settings.size(); i++) {
		_init_joints(p_skeleton, i);
		Node3D *target = Object::cast_to<Node3D>(get_node_or_null(iterate_settings[i]->target_node));
		if (!target || iterate_settings[i]->joints.is_empty()) {
			continue; // Abort.
		}
		iterate_settings[i]->cache_current_joint_rotations(p_skeleton); // Iterate over first to detect parent (outside of the chain) bone pose changes.

		Node3D *pole = Object::cast_to<Node3D>(get_node_or_null(iterate_settings[i]->pole_node));
		Vector3 destination = iterate_settings[i]->cached_space.affine_inverse().xform(target->get_global_position());
		_process_joints(p_delta, p_skeleton, iterate_settings[i], iterate_settings[i]->joints, iterate_settings[i]->chain, destination, !pole ? destination : iterate_settings[i]->cached_space.affine_inverse().xform(pole->get_global_position()), !!pole);
	}
}

void IterateIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, Vector<ChainIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, const Vector3 &p_pole_destination, bool p_use_pole) {
	real_t distance_to_target_sq = INFINITY;
	int iteration_count = 0;

	if (p_setting->is_penetrated(p_destination)) {
		return;
	}

	while (distance_to_target_sq > min_distance_squared && iteration_count < max_iterations) {
		// Solve the IK for this iteration.
		if (p_use_pole) {
			_solve_iteration_with_pole(p_delta, p_skeleton, p_setting, p_joints, p_chain, p_destination, p_joints.size(), p_chain.size(), p_pole_destination, p_setting->joint_size_half, p_setting->chain_size_half);
		} else {
			_solve_iteration(p_delta, p_skeleton, p_setting, p_joints, p_chain, p_destination, p_joints.size(), p_chain.size());
		}

		// Limitation should be done as the post process to prevent oscillation.
		for (int i = 0; i < p_joints.size(); i++) {
			ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
			if (!solver_info || Math::is_zero_approx(solver_info->length)) {
				continue;
			}

			int HEAD = i;
			int TAIL = i + 1;

			if (p_joints[HEAD]->rotation_axis != ROTATION_AXIS_ALL) {
				p_setting->update_chain_coordinate(p_skeleton, TAIL, p_chain[HEAD] + p_joints[HEAD]->get_projected_rotation(solver_info->current_grest, p_chain[TAIL] - p_chain[HEAD]), false);
			}
			if (p_joints[HEAD]->limitation.is_valid()) {
				p_setting->update_chain_coordinate(p_skeleton, TAIL, p_chain[HEAD] + p_joints[HEAD]->get_limited_rotation(solver_info->current_grest, p_chain[TAIL] - p_chain[HEAD]), false);
			}
		}

		// Update virtual bone rest/poses.
		p_setting->cache_current_joint_rotations(p_skeleton, angular_delta_limit);
		distance_to_target_sq = p_chain[p_chain.size() - 1].distance_squared_to(p_destination);
		iteration_count++;
	}

	// Apply the virtual bone rest/poses to the actual bones.
	for (int i = 0; i < p_joints.size(); i++) {
		ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}
		p_skeleton->set_bone_pose_rotation(p_joints[i]->bone, solver_info->current_lpose);
	}
}

void IterateIK3D::_solve_iteration_with_pole(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, Vector<ChainIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size, const Vector3 &p_pole_destination, int p_joint_size_half, int p_chain_size_half) {
	// Default implementation of the pole target solving is using two passes: root-to-middle and root-to-end chain solving.
	_solve_iteration(p_delta, p_skeleton, p_setting, p_joints, p_chain, p_pole_destination, p_joint_size_half, p_chain_size_half);
	_solve_iteration(p_delta, p_skeleton, p_setting, p_joints, p_chain, p_destination, p_joint_size, p_chain_size);
}

void IterateIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, Vector<ChainIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size) {
	//
}
IterateIK3D::~IterateIK3D() {
	clear_settings();
}
