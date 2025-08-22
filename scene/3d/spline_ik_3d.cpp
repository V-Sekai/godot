/**************************************************************************/
/*  spline_ik_3d.cpp                                                      */
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

#include "spline_ik_3d.h"

bool SplineIK3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "path_3d") {
			set_path_3d(which, p_value);
		} else if (what == "end_vector") {
			set_end_vector(which, static_cast<EndVector>((int)p_value));
		} else if (what == "tilt_enabled") {
			set_tilt_enabled(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool SplineIK3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);

		if (what == "path_3d") {
			r_ret = get_path_3d(which);
		} else if (what == "end_vector") {
			r_ret = (int)get_end_vector(which);
		} else if (what == "tilt_enabled") {
			r_ret = is_tilt_enabled(which);
		} else {
			return false;
		}
	}
	return true;
}

void SplineIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
	LocalVector<PropertyInfo> props;
	for (uint32_t i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::NODE_PATH, path + "path_3d", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Path3D"));
		p_list->push_back(PropertyInfo(Variant::INT, path + "end_vector", PROPERTY_HINT_ENUM, "None,Tangent,FromParent"));
		p_list->push_back(PropertyInfo(Variant::BOOL, path + "tilt_enabled"));
	}

	ChainIK3D::get_property_list(p_list);
}

// Setting.

void SplineIK3D::set_path_3d(int p_index, const NodePath &p_path_3d) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	sp_settings[p_index]->path_3d = p_path_3d;
}

NodePath SplineIK3D::get_path_3d(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), NodePath());
	return sp_settings[p_index]->path_3d;
}

void SplineIK3D::set_end_vector(int p_index, EndVector p_end_vector) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	sp_settings[p_index]->end_vector = p_end_vector;
}

SplineIK3D::EndVector SplineIK3D::get_end_vector(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), END_VECTOR_NONE);
	return sp_settings[p_index]->end_vector;
}

void SplineIK3D::set_tilt_enabled(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	sp_settings[p_index]->tilt_enabled = p_enabled;
}

bool SplineIK3D::is_tilt_enabled(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), false);
	return sp_settings[p_index]->tilt_enabled;
}

// Individual joints.

void SplineIK3D::_bind_methods() {
	// Setting.
	ClassDB::bind_method(D_METHOD("set_path_3d", "index", "path_3d"), &SplineIK3D::set_path_3d);
	ClassDB::bind_method(D_METHOD("get_path_3d", "index"), &SplineIK3D::get_path_3d);
	ClassDB::bind_method(D_METHOD("set_end_vector", "index", "end_vector"), &SplineIK3D::set_end_vector);
	ClassDB::bind_method(D_METHOD("get_end_vector", "index"), &SplineIK3D::get_end_vector);
	ClassDB::bind_method(D_METHOD("set_tilt_enabled", "index", "enabled"), &SplineIK3D::set_tilt_enabled);
	ClassDB::bind_method(D_METHOD("is_tilt_enabled", "index"), &SplineIK3D::is_tilt_enabled);

	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");

	BIND_ENUM_CONSTANT(END_VECTOR_NONE);
	BIND_ENUM_CONSTANT(END_VECTOR_TANGENT);
	BIND_ENUM_CONSTANT(END_VECTOR_FROM_PARENT);
}

void SplineIK3D::_init_joints(Skeleton3D *p_skeleton, int p_index) {
	SplineIK3DSetting *setting = sp_settings[p_index];
	cached_space = p_skeleton->get_global_transform();
	if (!setting->simulation_dirty) {
		return;
	}
	for (uint32_t i = 0; i < setting->solver_info_list.size(); i++) {
		if (setting->solver_info_list[i]) {
			memdelete(setting->solver_info_list[i]);
		}
	}
	setting->solver_info_list.clear();
	setting->solver_info_list.resize_initialized(setting->joints.size());
	setting->chain.clear();
	bool extend_end_bone = setting->extend_end_bone && setting->end_bone_length > 0;
	for (uint32_t i = 0; i < setting->joints.size(); i++) {
		setting->chain.push_back(p_skeleton->get_bone_global_pose(setting->joints[i].bone).origin);
		bool last = i == setting->joints.size() - 1;
		if (last && extend_end_bone && setting->end_bone_length > 0) {
			Vector3 axis = get_bone_axis(setting->end_bone.bone, setting->end_bone_direction);
			if (axis.is_zero_approx()) {
				continue;
			}
			setting->solver_info_list[i] = memnew(ManyBoneIK3DSolverInfo);
			setting->solver_info_list[i]->forward_vector = axis.normalized();
			setting->solver_info_list[i]->length = setting->end_bone_length;
			setting->chain.push_back(p_skeleton->get_bone_global_pose(setting->joints[i].bone).xform(axis * setting->end_bone_length));
		} else if (!last) {
			Vector3 axis = p_skeleton->get_bone_rest(setting->joints[i + 1].bone).origin;
			if (axis.is_zero_approx()) {
				continue; // Means always we need to check solver info, but `!solver_info` means that the bone is zero length, so IK should skip it in the all process.
			}
			setting->solver_info_list[i] = memnew(ManyBoneIK3DSolverInfo);
			setting->solver_info_list[i]->forward_vector = axis.normalized();
			setting->solver_info_list[i]->length = axis.length();
		}
	}

	setting->init_current_joint_rotations(p_skeleton);

	setting->simulation_dirty = false;
}

void SplineIK3D::_process_ik(Skeleton3D *p_skeleton, double p_delta) {
	for (uint32_t i = 0; i < settings.size(); i++) {
		_init_joints(p_skeleton, i);
		if (sp_settings[i]->joints.is_empty()) {
			continue; // Abort.
		}
		Path3D *path_3d = Object::cast_to<Path3D>(get_node_or_null(sp_settings[i]->path_3d));
		if (!path_3d) {
			continue; // Abort.
		}
		Ref<Curve3D> curve = path_3d->get_curve();
		if (curve.is_null() || curve->get_point_count() == 0) {
			continue; // Abort.
		}
		sp_settings[i]->cache_current_joint_rotations(p_skeleton); // Iterate over first to detect parent (outside of the chain) bone pose changes.
		_process_joints(p_delta, p_skeleton, sp_settings[i], curve, cached_space.affine_inverse() * path_3d->get_global_transform());
	}
}

void SplineIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, SplineIK3DSetting *p_setting, Ref<Curve3D> p_curve, const Transform3D &p_curve_space) {
	Vector3 root_to_start = p_curve_space.xform(p_curve->get_point_position(0)) - p_skeleton->get_bone_global_pose(p_setting->joints[0].bone).origin;
	real_t dist_root_to_start = root_to_start.length();
	real_t path_length = p_curve->get_baked_length();

	real_t det_accum = 0.0;
	for (uint32_t i = 0; i < p_setting->solver_info_list.size(); i++) {
		ManyBoneIK3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}
		int HEAD = i;
		int TAIL = i + 1;
		det_accum += solver_info->length;
		real_t offset = det_accum - dist_root_to_start;
		if (offset <= 0) {
			p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, limit_length(p_setting->chain[HEAD], p_setting->chain[HEAD] + root_to_start, solver_info->length));
			continue;
		} else if (offset > path_length) {
			// TODO: extends end.
			continue;
		}
		p_setting->update_chain_coordinate_fw(p_skeleton, TAIL, limit_length(p_setting->chain[HEAD], p_curve->sample_baked(offset, true), solver_info->length));
	}

	// Update virtual bone rest/poses.
	p_setting->cache_current_vectors(p_skeleton);
	p_setting->cache_current_joint_rotations(p_skeleton, 0.0);

	// Apply the virtual bone rest/poses to the actual bones.
	for (uint32_t i = 0; i < p_setting->solver_info_list.size(); i++) {
		ManyBoneIK3DSolverInfo *solver_info = p_setting->solver_info_list[i];
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}
		p_skeleton->set_bone_pose_rotation(p_setting->joints[i].bone, solver_info->current_lpose);
	}
}

SplineIK3D::~SplineIK3D() {
	clear_settings();
}
