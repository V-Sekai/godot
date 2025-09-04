/**************************************************************************/
/*  chain_ik_3d.cpp                                                       */
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

#include "chain_ik_3d.h"

bool ChainIK3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "root_bone_name") {
			set_root_bone_name(which, p_value);
		} else if (what == "root_bone") {
			set_root_bone(which, p_value);
		} else if (what == "end_bone_name") {
			set_end_bone_name(which, p_value);
		} else if (what == "end_bone") {
			String opt = path.get_slicec('/', 3);
			if (opt.is_empty()) {
				set_end_bone(which, p_value);
			} else if (opt == "direction") {
				set_end_bone_direction(which, static_cast<BoneDirection>((int)p_value));
			} else if (opt == "length") {
				set_end_bone_length(which, p_value);
			} else {
				return false;
			}
		} else if (what == "extend_end_bone") {
			set_extend_end_bone(which, p_value);
		} else if (what == "pole_node") {
			set_pole_node(which, p_value);
		} else if (what == "target_node") {
			set_target_node(which, p_value);
		} else if (what == "joint_count") {
			set_joint_count(which, p_value);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "bone_name") {
				set_joint_bone_name(which, idx, p_value);
			} else if (prop == "bone") {
				set_joint_bone(which, idx, p_value);
			} else if (prop == "rotation_axis") {
				set_joint_rotation_axis(which, idx, static_cast<RotationAxis>((int)p_value));
			} else if (prop == "rotation_axis_vector") {
				set_joint_rotation_axis_vector(which, idx, p_value);
			} else if (prop == "limitation") {
				set_joint_limitation(which, idx, p_value);
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

bool ChainIK3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "root_bone_name") {
			r_ret = get_root_bone_name(which);
		} else if (what == "root_bone") {
			r_ret = get_root_bone(which);
		} else if (what == "end_bone_name") {
			r_ret = get_end_bone_name(which);
		} else if (what == "end_bone") {
			String opt = path.get_slicec('/', 3);
			if (opt.is_empty()) {
				r_ret = get_end_bone(which);
			} else if (opt == "direction") {
				r_ret = (int)get_end_bone_direction(which);
			} else if (opt == "length") {
				r_ret = get_end_bone_length(which);
			} else {
				return false;
			}
		} else if (what == "extend_end_bone") {
			r_ret = is_end_bone_extended(which);
		} else if (what == "pole_node") {
			r_ret = get_pole_node(which);
		} else if (what == "target_node") {
			r_ret = get_target_node(which);
		} else if (what == "joint_count") {
			r_ret = get_joint_count(which);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "bone_name") {
				r_ret = get_joint_bone_name(which, idx);
			} else if (prop == "bone") {
				r_ret = get_joint_bone(which, idx);
			} else if (prop == "rotation_axis") {
				r_ret = (int)get_joint_rotation_axis(which, idx);
			} else if (prop == "rotation_axis_vector") {
				r_ret = get_joint_rotation_axis_vector(which, idx);
			} else if (prop == "limitation") {
				r_ret = get_joint_limitation(which, idx);
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

void ChainIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
	String enum_hint;
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton) {
		enum_hint = skeleton->get_concatenated_bone_names();
	}

	LocalVector<PropertyInfo> props;

	for (int i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		props.push_back(PropertyInfo(Variant::STRING, path + "root_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "root_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::STRING, path + "end_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::BOOL, path + "extend_end_bone"));
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone/direction", PROPERTY_HINT_ENUM, "+X,-X,+Y,-Y,+Z,-Z,FromParent"));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "end_bone/length", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"));
		props.push_back(PropertyInfo(Variant::NODE_PATH, path + "pole_node"));
		props.push_back(PropertyInfo(Variant::NODE_PATH, path + "target_node"));
		props.push_back(PropertyInfo(Variant::INT, path + "joint_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Joints," + path + "joints/,static,const"));
		for (int j = 0; j < settings[i]->joints.size(); j++) {
			String joint_path = path + "joints/" + itos(j) + "/";
			props.push_back(PropertyInfo(Variant::STRING, joint_path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint, PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY | PROPERTY_USAGE_STORAGE));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_READ_ONLY));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "rotation_axis", PROPERTY_HINT_ENUM, "X,Y,Z,All,Custom"));
			props.push_back(PropertyInfo(Variant::VECTOR3, joint_path + "rotation_axis_vector"));
			props.push_back(PropertyInfo(Variant::OBJECT, joint_path + "limitation", PROPERTY_HINT_RESOURCE_TYPE, "JointLimitation3D"));
		}
	}

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

void ChainIK3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();

		// Extended end bone option.
		bool force_hide = false;
		if (split[2] == "extend_end_bone" && get_end_bone(which) != -1) {
			p_property.usage = PROPERTY_USAGE_NONE;
			force_hide = true;
		}
		if (force_hide || (split[2] == "end_bone" && !is_end_bone_extended(which) && split.size() > 3)) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
	if (split.size() > 3 && split[0] == "settings") {
		int which = split[1].to_int();
		int joint = split[3].to_int();
		// Joints option.
		if (split[2] == "joints" && split.size() > 4) {
			if (split[4] == "rotation_axis_vector" && get_joint_rotation_axis(which, joint) != ROTATION_AXIS_CUSTOM) {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
		}
	}
}

void ChainIK3D::set_max_iterations(int p_max_iterations) {
	max_iterations = p_max_iterations;
}

int ChainIK3D::get_max_iterations() const {
	return max_iterations;
}

void ChainIK3D::set_min_distance(real_t p_min_distance) {
	min_distance = p_min_distance;
}

real_t ChainIK3D::get_min_distance() const {
	return min_distance;
}

void ChainIK3D::set_angular_delta_limit(real_t p_angular_delta_limit) {
	angular_delta_limit = p_angular_delta_limit;
}

real_t ChainIK3D::get_angular_delta_limit() const {
	return angular_delta_limit;
}

// Setting.

void ChainIK3D::set_root_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->root_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_root_bone(p_index, sk->find_bone(settings[p_index]->root_bone_name));
	}
}

String ChainIK3D::get_root_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return settings[p_index]->root_bone_name;
}

void ChainIK3D::set_root_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, settings.size());
	bool changed = settings[p_index]->root_bone != p_bone;
	settings[p_index]->root_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->root_bone <= -1 || settings[p_index]->root_bone >= sk->get_bone_count()) {
			WARN_PRINT("Root bone index out of range!");
			settings[p_index]->root_bone = -1;
		} else {
			settings[p_index]->root_bone_name = sk->get_bone_name(settings[p_index]->root_bone);
		}
	}
	if (changed) {
		_update_joints(p_index);
	}
}

int ChainIK3D::get_root_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return settings[p_index]->root_bone;
}

void ChainIK3D::set_end_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->end_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_end_bone(p_index, sk->find_bone(settings[p_index]->end_bone_name));
	}
}

String ChainIK3D::get_end_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return settings[p_index]->end_bone_name;
}

void ChainIK3D::set_end_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, settings.size());
	bool changed = settings[p_index]->end_bone != p_bone;
	settings[p_index]->end_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->end_bone <= -1 || settings[p_index]->end_bone >= sk->get_bone_count()) {
			WARN_PRINT("End bone index out of range!");
			settings[p_index]->end_bone = -1;
		} else {
			settings[p_index]->end_bone_name = sk->get_bone_name(settings[p_index]->end_bone);
		}
	}
	if (changed) {
		_update_joints(p_index);
	}
	notify_property_list_changed();
}

int ChainIK3D::get_end_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return settings[p_index]->end_bone;
}

void ChainIK3D::set_extend_end_bone(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->extend_end_bone = p_enabled;
	settings[p_index]->simulation_dirty = true;
	Skeleton3D *sk = get_skeleton();
	if (sk && !settings[p_index]->joints.is_empty()) {
		_validate_rotation_axis(sk, p_index, settings[p_index]->joints.size() - 1);
	}
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

bool ChainIK3D::is_end_bone_extended(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), false);
	return settings[p_index]->extend_end_bone;
}

void ChainIK3D::set_end_bone_direction(int p_index, BoneDirection p_bone_direction) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->end_bone_direction = p_bone_direction;
	settings[p_index]->simulation_dirty = true;
	Skeleton3D *sk = get_skeleton();
	if (sk && !settings[p_index]->joints.is_empty()) {
		_validate_rotation_axis(sk, p_index, settings[p_index]->joints.size() - 1);
	}
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

ChainIK3D::BoneDirection ChainIK3D::get_end_bone_direction(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), BONE_DIRECTION_FROM_PARENT);
	return settings[p_index]->end_bone_direction;
}

void ChainIK3D::set_end_bone_length(int p_index, float p_length) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->end_bone_length = p_length;
	settings[p_index]->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

float ChainIK3D::get_end_bone_length(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	return settings[p_index]->end_bone_length;
}

void ChainIK3D::set_pole_node(int p_index, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->pole_node = p_node_path;
}

NodePath ChainIK3D::get_pole_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), NodePath());
	return settings[p_index]->pole_node;
}

void ChainIK3D::set_target_node(int p_index, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->target_node = p_node_path;
}

NodePath ChainIK3D::get_target_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), NodePath());
	return settings[p_index]->target_node;
}

void ChainIK3D::set_setting_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);

	int delta = p_count - settings.size();
	if (delta < 0) {
		for (int i = delta; i < 0; i++) {
			memdelete(settings[settings.size() + i]);
		}
	}
	settings.resize(p_count);
	delta++;
	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			settings.write[p_count - i] = memnew(ChainIK3DSetting);
		}
	}
	notify_property_list_changed();
}

int ChainIK3D::get_setting_count() const {
	return settings.size();
}

void ChainIK3D::clear_settings() {
	set_setting_count(0);
}

// Individual joints.

void ChainIK3D::set_joint_bone_name(int p_index, int p_joint, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_joint_bone(p_index, p_joint, sk->find_bone(joints[p_joint]->bone_name));
	}
}

String ChainIK3D::get_joint_bone_name(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), String());
	return joints[p_joint]->bone_name;
}

void ChainIK3D::set_joint_bone(int p_index, int p_joint, int p_bone) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (joints[p_joint]->bone <= -1 || joints[p_joint]->bone >= sk->get_bone_count()) {
			WARN_PRINT("Joint bone index out of range!");
			joints[p_joint]->bone = -1;
		} else {
			joints[p_joint]->bone_name = sk->get_bone_name(joints[p_joint]->bone);
		}
	}
}

int ChainIK3D::get_joint_bone(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), -1);
	return joints[p_joint]->bone;
}

void ChainIK3D::set_joint_rotation_axis(int p_index, int p_joint, RotationAxis p_axis) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->rotation_axis = p_axis;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_rotation_axis(sk, p_index, p_joint);
	}
	notify_property_list_changed();
	settings[p_index]->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

ChainIK3D::RotationAxis ChainIK3D::get_joint_rotation_axis(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), ROTATION_AXIS_ALL);
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), ROTATION_AXIS_ALL);
	return joints[p_joint]->rotation_axis;
}

void ChainIK3D::set_joint_rotation_axis_vector(int p_index, int p_joint, Vector3 p_vector) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->rotation_axis_vector = p_vector;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_rotation_axis(sk, p_index, p_joint);
	}
	settings[p_index]->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Vector3 ChainIK3D::get_joint_rotation_axis_vector(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Vector3());
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), Vector3());
	return joints[p_joint]->get_rotation_axis_vector();
}

void ChainIK3D::set_joint_limitation(int p_index, int p_joint, const Ref<JointLimitation3D> &p_limitation) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->limitation = p_limitation;
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Ref<JointLimitation3D> ChainIK3D::get_joint_limitation(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Ref<JointLimitation3D>());
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), Ref<JointLimitation3D>());
	return joints[p_joint]->limitation;
}

void ChainIK3D::set_joint_count(int p_index, int p_count) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ERR_FAIL_COND(p_count < 0);
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	int delta = p_count - joints.size() + 1;
	joints.resize(p_count);
	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			joints.write[p_count - i] = memnew(ManyBoneIK3DJointSetting);
		}
	}
	notify_property_list_changed();
}

int ChainIK3D::get_joint_count(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	return joints.size();
}

void ChainIK3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_max_iterations", "max_iterations"), &ChainIK3D::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations"), &ChainIK3D::get_max_iterations);
	ClassDB::bind_method(D_METHOD("set_min_distance", "min_distance"), &ChainIK3D::set_min_distance);
	ClassDB::bind_method(D_METHOD("get_min_distance"), &ChainIK3D::get_min_distance);
	ClassDB::bind_method(D_METHOD("set_angular_delta_limit", "angular_delta_limit"), &ChainIK3D::set_angular_delta_limit);
	ClassDB::bind_method(D_METHOD("get_angular_delta_limit"), &ChainIK3D::get_angular_delta_limit);

	// Setting.
	ClassDB::bind_method(D_METHOD("set_root_bone_name", "index", "bone_name"), &ChainIK3D::set_root_bone_name);
	ClassDB::bind_method(D_METHOD("get_root_bone_name", "index"), &ChainIK3D::get_root_bone_name);
	ClassDB::bind_method(D_METHOD("set_root_bone", "index", "bone"), &ChainIK3D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone", "index"), &ChainIK3D::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_end_bone_name", "index", "bone_name"), &ChainIK3D::set_end_bone_name);
	ClassDB::bind_method(D_METHOD("get_end_bone_name", "index"), &ChainIK3D::get_end_bone_name);
	ClassDB::bind_method(D_METHOD("set_end_bone", "index", "bone"), &ChainIK3D::set_end_bone);
	ClassDB::bind_method(D_METHOD("get_end_bone", "index"), &ChainIK3D::get_end_bone);

	ClassDB::bind_method(D_METHOD("set_extend_end_bone", "index", "enabled"), &ChainIK3D::set_extend_end_bone);
	ClassDB::bind_method(D_METHOD("is_end_bone_extended", "index"), &ChainIK3D::is_end_bone_extended);
	ClassDB::bind_method(D_METHOD("set_end_bone_direction", "index", "bone_direction"), &ChainIK3D::set_end_bone_direction);
	ClassDB::bind_method(D_METHOD("get_end_bone_direction", "index"), &ChainIK3D::get_end_bone_direction);
	ClassDB::bind_method(D_METHOD("set_end_bone_length", "index", "length"), &ChainIK3D::set_end_bone_length);
	ClassDB::bind_method(D_METHOD("get_end_bone_length", "index"), &ChainIK3D::get_end_bone_length);

	ClassDB::bind_method(D_METHOD("set_pole_node", "index", "pole_node"), &ChainIK3D::set_pole_node);
	ClassDB::bind_method(D_METHOD("get_pole_node", "index"), &ChainIK3D::get_pole_node);

	ClassDB::bind_method(D_METHOD("set_target_node", "index", "target_node"), &ChainIK3D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node", "index"), &ChainIK3D::get_target_node);

	ClassDB::bind_method(D_METHOD("set_setting_count", "count"), &ChainIK3D::set_setting_count);
	ClassDB::bind_method(D_METHOD("get_setting_count"), &ChainIK3D::get_setting_count);
	ClassDB::bind_method(D_METHOD("clear_settings"), &ChainIK3D::clear_settings);

	// Individual joints.
	ClassDB::bind_method(D_METHOD("get_joint_bone_name", "index", "joint"), &ChainIK3D::get_joint_bone_name);
	ClassDB::bind_method(D_METHOD("get_joint_bone", "index", "joint"), &ChainIK3D::get_joint_bone);
	ClassDB::bind_method(D_METHOD("set_joint_rotation_axis", "index", "joint", "axis"), &ChainIK3D::set_joint_rotation_axis);
	ClassDB::bind_method(D_METHOD("get_joint_rotation_axis", "index", "joint"), &ChainIK3D::get_joint_rotation_axis);
	ClassDB::bind_method(D_METHOD("set_joint_rotation_axis_vector", "index", "joint", "axis_vector"), &ChainIK3D::set_joint_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("get_joint_rotation_axis_vector", "index", "joint"), &ChainIK3D::get_joint_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("set_joint_limitation", "index", "joint", "limitation"), &ChainIK3D::set_joint_limitation);
	ClassDB::bind_method(D_METHOD("get_joint_limitation", "index", "joint"), &ChainIK3D::get_joint_limitation);

	ClassDB::bind_method(D_METHOD("get_joint_count", "index"), &ChainIK3D::get_joint_count);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_iterations", PROPERTY_HINT_RANGE, "0,100,or_greater"), "set_max_iterations", "get_max_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_distance", PROPERTY_HINT_RANGE, "0,1,0.01,or_greater"), "set_min_distance", "get_min_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_delta_limit", PROPERTY_HINT_RANGE, "0,180,0.01,radians_as_degrees"), "set_angular_delta_limit", "get_angular_delta_limit");
	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");
}

void ChainIK3D::_validate_bone_names() {
	for (int i = 0; i < settings.size(); i++) {
		// Prior bone name.
		if (!settings[i]->root_bone_name.is_empty()) {
			set_root_bone_name(i, settings[i]->root_bone_name);
		} else if (settings[i]->root_bone != -1) {
			set_root_bone(i, settings[i]->root_bone);
		}
		// Prior bone name.
		if (!settings[i]->end_bone_name.is_empty()) {
			set_end_bone_name(i, settings[i]->end_bone_name);
		} else if (settings[i]->end_bone != -1) {
			set_end_bone(i, settings[i]->end_bone);
		}
	}
}

void ChainIK3D::_validate_rotation_axes(Skeleton3D *p_skeleton) const {
	for (int i = 0; i < settings.size(); i++) {
		for (int j = 0; j < settings[i]->joints.size(); j++) {
			_validate_rotation_axis(p_skeleton, i, j);
		}
	}
}

void ChainIK3D::_validate_rotation_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const {
	RotationAxis axis = settings[p_index]->joints[p_joint]->rotation_axis;
	if (axis == ROTATION_AXIS_ALL) {
		return;
	}
	Vector3 rot = get_joint_rotation_axis_vector(p_index, p_joint).normalized();
	Vector3 fwd;
	if (p_joint < settings[p_index]->joints.size() - 1) {
		fwd = p_skeleton->get_bone_rest(settings[p_index]->joints[p_joint + 1]->bone).origin;
	} else if (settings[p_index]->extend_end_bone) {
		fwd = get_bone_axis(settings[p_index]->end_bone, settings[p_index]->end_bone_direction);
		if (fwd.is_zero_approx()) {
			return;
		}
	}
	fwd.normalize();
	if (Math::is_equal_approx(Math::abs(rot.dot(fwd)), 1)) {
		WARN_PRINT_ED("Setting: " + itos(p_index) + " Joint: " + itos(p_joint) + ": Rotation axis and forward vectors are colinear. This is not advised as it may cause unwanted rotation.");
	}
}

void ChainIK3D::_make_all_joints_dirty() {
	for (int i = 0; i < settings.size(); i++) {
		_update_joints(i);
	}
}

void ChainIK3D::_init_joints(Skeleton3D *p_skeleton, ChainIK3DSetting *setting) {
	setting->cached_space = p_skeleton->get_global_transform();
	if (!setting->simulation_dirty) {
		return;
	}
	setting->chain.clear();
	bool extend_end_bone = setting->extend_end_bone && setting->end_bone_length > 0;
	for (int i = 0; i < setting->joints.size(); i++) {
		if (setting->joints[i]->solver_info) {
			memdelete(setting->joints[i]->solver_info);
			setting->joints[i]->solver_info = nullptr;
		}
		setting->chain.push_back(p_skeleton->get_bone_global_pose(setting->joints[i]->bone).origin);
		bool last = i == setting->joints.size() - 1;
		if (last && extend_end_bone && setting->end_bone_length > 0) {
			Vector3 axis = get_bone_axis(setting->end_bone, setting->end_bone_direction);
			if (axis.is_zero_approx()) {
				continue;
			}
			setting->joints[i]->solver_info = memnew(ManyBoneIK3DSolverInfo);
			setting->joints[i]->solver_info->forward_vector = snap_vector_to_plane(setting->joints[i]->get_rotation_axis_vector(), axis.normalized());
			setting->joints[i]->solver_info->length = setting->end_bone_length;
			setting->chain.push_back(p_skeleton->get_bone_global_pose(setting->joints[i]->bone).xform(axis * setting->end_bone_length));
		} else if (!last) {
			Vector3 axis = p_skeleton->get_bone_rest(setting->joints[i + 1]->bone).origin;
			if (axis.is_zero_approx()) {
				continue; // Means always we need to check solver info, but `!solver_info` means that the bone is zero length, so IK should skip it in the all process.
			}
			setting->joints[i]->solver_info = memnew(ManyBoneIK3DSolverInfo);
			setting->joints[i]->solver_info->forward_vector = snap_vector_to_plane(setting->joints[i]->get_rotation_axis_vector(), axis.normalized());
			setting->joints[i]->solver_info->length = axis.length();
		}
	}

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

	setting->init_current_joint_rotations(p_skeleton);

	setting->simulation_dirty = false;
}

void ChainIK3D::_update_joints(int p_index) {
	settings[p_index]->simulation_dirty = true;

#ifdef TOOLS_ENABLED
	update_gizmos(); // To clear invalid setting.
#endif // TOOLS_ENABLED

	Skeleton3D *sk = get_skeleton();
	int current_bone = settings[p_index]->end_bone;
	int root_bone = settings[p_index]->root_bone;
	if (!sk || current_bone < 0 || root_bone < 0) {
		set_joint_count(p_index, 0);
		return;
	}

	// Validation.
	bool valid = false;
	while (current_bone >= 0) {
		if (current_bone == root_bone) {
			valid = true;
			break;
		}
		current_bone = sk->get_bone_parent(current_bone);
	}

	if (!valid) {
		set_joint_count(p_index, 0);
		ERR_FAIL_EDMSG("End bone must be the same as or a child of root bone.");
	}

	Vector<int> new_joints;
	current_bone = settings[p_index]->end_bone;
	while (current_bone != root_bone) {
		new_joints.push_back(current_bone);
		current_bone = sk->get_bone_parent(current_bone);
	}
	new_joints.push_back(current_bone);
	new_joints.reverse();

	set_joint_count(p_index, new_joints.size());
	for (int i = 0; i < new_joints.size(); i++) {
		set_joint_bone(p_index, i, new_joints[i]);
	}

	if (sk) {
		_validate_rotation_axes(sk);
	}

#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

void ChainIK3D::_process_ik(Skeleton3D *p_skeleton, double p_delta) {
	min_distance_squared = min_distance * min_distance;
	for (int i = 0; i < settings.size(); i++) {
		_init_joints(p_skeleton, settings[i]);
		Node3D *target = Object::cast_to<Node3D>(get_node_or_null(settings[i]->target_node));
		if (!target || settings[i]->joints.is_empty()) {
			continue; // Abort.
		}
		settings[i]->cache_current_joint_rotations(p_skeleton); // Iterate over first to detect parent (outside of the chain) bone pose changes.

		Node3D *pole = Object::cast_to<Node3D>(get_node_or_null(settings[i]->pole_node));
		Vector3 destination = settings[i]->cached_space.affine_inverse().xform(target->get_global_position());
		_process_joints(p_delta, p_skeleton, settings[i], settings[i]->joints, settings[i]->chain, destination, !pole ? destination : settings[i]->cached_space.affine_inverse().xform(pole->get_global_position()), !!pole);
	}
}

void ChainIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, ChainIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, const Vector3 &p_pole_destination, bool p_use_pole) {
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

void ChainIK3D::_solve_iteration_with_pole(double p_delta, Skeleton3D *p_skeleton, ChainIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size, const Vector3 &p_pole_destination, int p_joint_size_half, int p_chain_size_half) {
	// Default implementation of the pole target solving is using two passes: root-to-middle and root-to-end chain solving.
	_solve_iteration(p_delta, p_skeleton, p_setting, p_joints, p_chain, p_pole_destination, p_joint_size_half, p_chain_size_half);
	_solve_iteration(p_delta, p_skeleton, p_setting, p_joints, p_chain, p_destination, p_joint_size, p_chain_size);
}

void ChainIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, ChainIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size) {
	//
}

void ChainIK3D::reset() {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	for (int i = 0; i < settings.size(); i++) {
		settings[i]->simulation_dirty = true;
		_init_joints(skeleton, settings[i]);
	}
}

ChainIK3D::~ChainIK3D() {
	clear_settings();
}
