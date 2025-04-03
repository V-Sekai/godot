/**************************************************************************/
/*  many_bone_ik_3d.cpp                                                   */
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

#include "many_bone_ik_3d.h"

bool ManyBoneIK3D::_set(const StringName &p_path, const Variant &p_value) {
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
		} else if (what == "target_node") {
			set_target_node(which, p_value);
		} else if (what == "use_target_axis") {
			set_use_target_axis(which, p_value);
		} else if (what == "target_axis") {
			set_target_axis(which, static_cast<BoneAxis>((int)p_value));
		} else if (what == "max_iterations") {
			set_max_iterations(which, p_value);
		} else if (what == "min_distance") {
			set_min_distance(which, p_value);
		} else if (what == "joint_count") {
			set_joint_count(which, p_value);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "bone_name") {
				set_joint_bone_name(which, idx, p_value);
			} else if (prop == "bone") {
				set_joint_bone(which, idx, p_value);
			} else if (prop == "twist_limitation") {
				set_joint_twist_limitation(which, idx, p_value);
			} else if (prop == "limitation") {
				set_joint_limitation(which, idx, p_value);
			} else if (prop == "limitation_rotation_offset") {
				set_joint_limitation_rotation_offset(which, idx, p_value);
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

bool ManyBoneIK3D::_get(const StringName &p_path, Variant &r_ret) const {
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
		} else if (what == "target_node") {
			r_ret = get_target_node(which);
		} else if (what == "use_target_axis") {
			r_ret = is_using_target_axis(which);
		} else if (what == "target_axis") {
			r_ret = (int)get_target_axis(which);
		} else if (what == "max_iterations") {
			r_ret = get_max_iterations(which);
		} else if (what == "min_distance") {
			r_ret = get_min_distance(which);
		} else if (what == "joint_count") {
			r_ret = get_joint_count(which);
		} else if (what == "joints") {
			int idx = path.get_slicec('/', 3).to_int();
			String prop = path.get_slicec('/', 4);
			if (prop == "bone_name") {
				r_ret = get_joint_bone_name(which, idx);
			} else if (prop == "bone") {
				r_ret = get_joint_bone(which, idx);
			} else if (prop == "twist_limitation") {
				r_ret = get_joint_twist_limitation(which, idx);
			} else if (prop == "limitation") {
				r_ret = get_joint_limitation(which, idx);
			} else if (prop == "limitation_rotation_offset") {
				r_ret = get_joint_limitation_rotation_offset(which, idx);
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

void ManyBoneIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
	String enum_hint;
	Skeleton3D *skeleton = get_skeleton();
	if (skeleton) {
		enum_hint = skeleton->get_concatenated_bone_names();
	}

	for (int i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";
		p_list->push_back(PropertyInfo(Variant::STRING, path + "root_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		p_list->push_back(PropertyInfo(Variant::INT, path + "root_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::STRING, path + "end_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		p_list->push_back(PropertyInfo(Variant::INT, path + "end_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		p_list->push_back(PropertyInfo(Variant::BOOL, path + "extend_end_bone"));
		p_list->push_back(PropertyInfo(Variant::INT, path + "end_bone/direction", PROPERTY_HINT_ENUM, "+X,-X,+Y,-Y,+Z,-Z,FromParent"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, path + "end_bone/length", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"));

		p_list->push_back(PropertyInfo(Variant::NODE_PATH, path + "target_node"));
		p_list->push_back(PropertyInfo(Variant::BOOL, path + "use_target_axis"));
		p_list->push_back(PropertyInfo(Variant::INT, path + "target_axis", PROPERTY_HINT_ENUM, "+X,-X,+Y,-Y,+Z,-Z"));
		p_list->push_back(PropertyInfo(Variant::INT, path + "max_iterations", PROPERTY_HINT_RANGE, "0,100,1"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, path + "min_distance", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"));

		p_list->push_back(PropertyInfo(Variant::INT, path + "joint_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Joints," + path + "joints/,static,const"));
		for (int j = 0; j < settings[i]->joints.size(); j++) {
			String joint_path = path + "joints/" + itos(j) + "/";
			p_list->push_back(PropertyInfo(Variant::STRING, joint_path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint, PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY | PROPERTY_USAGE_STORAGE));
			p_list->push_back(PropertyInfo(Variant::INT, joint_path + "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_READ_ONLY));
			p_list->push_back(PropertyInfo(Variant::FLOAT, joint_path + "twist_limitation", PROPERTY_HINT_RANGE, "0,180,0.01,radians_as_degrees"));
			p_list->push_back(PropertyInfo(Variant::OBJECT, joint_path + "limitation", PROPERTY_HINT_RESOURCE_TYPE, "JointLimitation3D"));
			p_list->push_back(PropertyInfo(Variant::QUATERNION, joint_path + "limitation_rotation_offset"));
		}
	}

	for (PropertyInfo &E : *p_list) {
		_validate_property(E);
	}
}

void ManyBoneIK3D::_validate_property(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();

		// Extended end bone option.
		if (split[2] == "end_bone" && !is_end_bone_extended(which) && split.size() > 3) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}

		// Target axis option.
		if ((split[2] == "use_target_axis") && get_target_node(which).is_empty()) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
		if (split[2] == "target_axis" && (get_target_node(which).is_empty() || !is_using_target_axis(which))) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}

		// Joints option.
		if (split[2] == "joints" && split.size() > 4) {
			int joint = split[3].to_int();
			if (split[4] == "limitation_rotation_offset" && get_joint_limitation(which, joint).is_null()) {
				p_property.usage = PROPERTY_USAGE_NONE;
			}
		}
	}
}

void ManyBoneIK3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				set_notify_local_transform(true); // Used for updating gizmo in editor.
			}
#endif // TOOLS_ENABLED
			_make_all_joints_dirty();
		} break;
#ifdef TOOLS_ENABLED
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			update_gizmos();
		} break;
#endif // TOOLS_ENABLED
	}
}

// Setting.

void ManyBoneIK3D::set_root_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->root_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_root_bone(p_index, sk->find_bone(settings[p_index]->root_bone_name));
	}
}

String ManyBoneIK3D::get_root_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return settings[p_index]->root_bone_name;
}

void ManyBoneIK3D::set_root_bone(int p_index, int p_bone) {
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
		_update_joint_array(p_index);
	}
}

int ManyBoneIK3D::get_root_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return settings[p_index]->root_bone;
}

void ManyBoneIK3D::set_end_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->end_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_end_bone(p_index, sk->find_bone(settings[p_index]->end_bone_name));
	}
}

String ManyBoneIK3D::get_end_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return settings[p_index]->end_bone_name;
}

void ManyBoneIK3D::set_end_bone(int p_index, int p_bone) {
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
		_update_joint_array(p_index);
	}
}

int ManyBoneIK3D::get_end_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return settings[p_index]->end_bone;
}

void ManyBoneIK3D::set_extend_end_bone(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->extend_end_bone = p_enabled;
	settings[p_index]->simulation_dirty = true;
	notify_property_list_changed();
}

bool ManyBoneIK3D::is_end_bone_extended(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), false);
	return settings[p_index]->extend_end_bone;
}

void ManyBoneIK3D::set_end_bone_direction(int p_index, BoneDirection p_bone_direction) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->end_bone_direction = p_bone_direction;
	settings[p_index]->simulation_dirty = true;
}

ManyBoneIK3D::BoneDirection ManyBoneIK3D::get_end_bone_direction(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), BONE_DIRECTION_FROM_PARENT);
	return settings[p_index]->end_bone_direction;
}

void ManyBoneIK3D::set_end_bone_length(int p_index, float p_length) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->end_bone_length = p_length;
	settings[p_index]->simulation_dirty = true;
}

float ManyBoneIK3D::get_end_bone_length(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	return settings[p_index]->end_bone_length;
}

Vector3 ManyBoneIK3D::get_end_bone_axis(int p_end_bone, BoneDirection p_direction) const {
	Vector3 axis;
	if (p_direction == BONE_DIRECTION_FROM_PARENT) {
		Skeleton3D *sk = get_skeleton();
		if (sk) {
			axis = sk->get_bone_rest(p_end_bone).basis.xform_inv(sk->get_bone_rest(p_end_bone).origin);
			axis.normalize();
		}
	} else {
		axis = get_vector_from_bone_axis(static_cast<BoneAxis>((int)p_direction));
	}
	return axis;
}

void ManyBoneIK3D::set_target_node(int p_index, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->target_node = p_node_path;
	notify_property_list_changed();
}

NodePath ManyBoneIK3D::get_target_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), NodePath());
	return settings[p_index]->target_node;
}

void ManyBoneIK3D::set_use_target_axis(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->use_target_axis = p_enabled;
	notify_property_list_changed();
}

bool ManyBoneIK3D::is_using_target_axis(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), false);
	return settings[p_index]->use_target_axis;
}

void ManyBoneIK3D::set_target_axis(int p_index, BoneAxis p_axis) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->target_axis = p_axis;
}

SkeletonModifier3D::BoneAxis ManyBoneIK3D::get_target_axis(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), BONE_AXIS_PLUS_Y);
	return settings[p_index]->target_axis;
}

void ManyBoneIK3D::set_max_iterations(int p_index, int p_max_iterations) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->max_iterations = p_max_iterations;
}

int ManyBoneIK3D::get_max_iterations(int p_index) const {
	return settings[p_index]->max_iterations;
}

void ManyBoneIK3D::set_min_distance(int p_index, real_t p_min_distance) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->min_distance = p_min_distance;
}

real_t ManyBoneIK3D::get_min_distance(int p_index) const {
	return settings[p_index]->min_distance;
}

void ManyBoneIK3D::set_setting_count(int p_count) {
	ERR_FAIL_COND(p_count < 0);
	int delta = p_count - settings.size() + 1;
	settings.resize(p_count);
	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			settings.write[p_count - i] = memnew(ManyBoneIK3DSetting);
		}
	}
	notify_property_list_changed();
}

int ManyBoneIK3D::get_setting_count() const {
	return settings.size();
}

void ManyBoneIK3D::clear_settings() {
	set_setting_count(0);
}

// Individual joints.

void ManyBoneIK3D::set_joint_bone_name(int p_index, int p_joint, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_joint_bone(p_index, p_joint, sk->find_bone(joints[p_joint]->bone_name));
	}
}

String ManyBoneIK3D::get_joint_bone_name(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), String());
	return joints[p_joint]->bone_name;
}

void ManyBoneIK3D::set_joint_bone(int p_index, int p_joint, int p_bone) {
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

int ManyBoneIK3D::get_joint_bone(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), -1);
	return joints[p_joint]->bone;
}

void ManyBoneIK3D::set_joint_twist_limitation(int p_index, int p_joint, const real_t &p_angle) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->twist_limitation = p_angle;
}

real_t ManyBoneIK3D::get_joint_twist_limitation(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), 0);
	return joints[p_joint]->twist_limitation;
}

void ManyBoneIK3D::set_joint_limitation(int p_index, int p_joint, const Ref<JointLimitation3D> &p_limitation) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->limitation = p_limitation;
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Ref<JointLimitation3D> ManyBoneIK3D::get_joint_limitation(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Ref<JointLimitation3D>());
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), Ref<JointLimitation3D>());
	return joints[p_joint]->limitation;
}

void ManyBoneIK3D::set_joint_limitation_rotation_offset(int p_index, int p_joint, const Quaternion &p_rotation_offset) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ManyBoneIK3DJointSetting *> &joints = settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->limitation_rotation_offset = p_rotation_offset;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Quaternion ManyBoneIK3D::get_joint_limitation_rotation_offset(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Quaternion());
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), Quaternion());
	return joints[p_joint]->limitation_rotation_offset;
}

void ManyBoneIK3D::set_joint_count(int p_index, int p_count) {
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

int ManyBoneIK3D::get_joint_count(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	Vector<ManyBoneIK3DJointSetting *> joints = settings[p_index]->joints;
	return joints.size();
}

void ManyBoneIK3D::_bind_methods() {
	// Setting.
	ClassDB::bind_method(D_METHOD("set_root_bone_name", "index", "bone_name"), &ManyBoneIK3D::set_root_bone_name);
	ClassDB::bind_method(D_METHOD("get_root_bone_name", "index"), &ManyBoneIK3D::get_root_bone_name);
	ClassDB::bind_method(D_METHOD("set_root_bone", "index", "bone"), &ManyBoneIK3D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone", "index"), &ManyBoneIK3D::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_end_bone_name", "index", "bone_name"), &ManyBoneIK3D::set_end_bone_name);
	ClassDB::bind_method(D_METHOD("get_end_bone_name", "index"), &ManyBoneIK3D::get_end_bone_name);
	ClassDB::bind_method(D_METHOD("set_end_bone", "index", "bone"), &ManyBoneIK3D::set_end_bone);
	ClassDB::bind_method(D_METHOD("get_end_bone", "index"), &ManyBoneIK3D::get_end_bone);

	ClassDB::bind_method(D_METHOD("set_extend_end_bone", "index", "enabled"), &ManyBoneIK3D::set_extend_end_bone);
	ClassDB::bind_method(D_METHOD("is_end_bone_extended", "index"), &ManyBoneIK3D::is_end_bone_extended);
	ClassDB::bind_method(D_METHOD("set_end_bone_direction", "index", "bone_direction"), &ManyBoneIK3D::set_end_bone_direction);
	ClassDB::bind_method(D_METHOD("get_end_bone_direction", "index"), &ManyBoneIK3D::get_end_bone_direction);
	ClassDB::bind_method(D_METHOD("set_end_bone_length", "index", "length"), &ManyBoneIK3D::set_end_bone_length);
	ClassDB::bind_method(D_METHOD("get_end_bone_length", "index"), &ManyBoneIK3D::get_end_bone_length);

	ClassDB::bind_method(D_METHOD("set_target_node", "index", "target_node"), &ManyBoneIK3D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node", "index"), &ManyBoneIK3D::get_target_node);
	ClassDB::bind_method(D_METHOD("set_max_iterations", "index", "max_iterations"), &ManyBoneIK3D::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations", "index"), &ManyBoneIK3D::get_max_iterations);
	ClassDB::bind_method(D_METHOD("set_min_distance", "index", "min_distance"), &ManyBoneIK3D::set_min_distance);
	ClassDB::bind_method(D_METHOD("get_min_distance", "index"), &ManyBoneIK3D::get_min_distance);

	ClassDB::bind_method(D_METHOD("set_setting_count", "count"), &ManyBoneIK3D::set_setting_count);
	ClassDB::bind_method(D_METHOD("get_setting_count"), &ManyBoneIK3D::get_setting_count);
	ClassDB::bind_method(D_METHOD("clear_settings"), &ManyBoneIK3D::clear_settings);

	// Individual joints.
	ClassDB::bind_method(D_METHOD("get_joint_bone_name", "index", "joint"), &ManyBoneIK3D::get_joint_bone_name);
	ClassDB::bind_method(D_METHOD("get_joint_bone", "index", "joint"), &ManyBoneIK3D::get_joint_bone);
	ClassDB::bind_method(D_METHOD("set_joint_twist_limitation", "index", "joint", "angle"), &ManyBoneIK3D::set_joint_twist_limitation);
	ClassDB::bind_method(D_METHOD("get_joint_twist_limitation", "index", "joint"), &ManyBoneIK3D::get_joint_twist_limitation);
	ClassDB::bind_method(D_METHOD("set_joint_limitation", "index", "joint", "limitation"), &ManyBoneIK3D::set_joint_limitation);
	ClassDB::bind_method(D_METHOD("get_joint_limitation", "index", "joint"), &ManyBoneIK3D::get_joint_limitation);
	ClassDB::bind_method(D_METHOD("set_joint_limitation_rotation_offset", "index", "joint", "limitation_rotation_offset"), &ManyBoneIK3D::set_joint_limitation_rotation_offset);
	ClassDB::bind_method(D_METHOD("get_joint_limitation_rotation_offset", "index", "joint"), &ManyBoneIK3D::get_joint_limitation_rotation_offset);

	ClassDB::bind_method(D_METHOD("get_joint_count", "index"), &ManyBoneIK3D::get_joint_count);

	// To process manually.
	ClassDB::bind_method(D_METHOD("reset"), &ManyBoneIK3D::reset);

	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");

	BIND_ENUM_CONSTANT(BONE_DIRECTION_PLUS_X);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_MINUS_X);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_PLUS_Y);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_MINUS_Y);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_PLUS_Z);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_MINUS_Z);
	BIND_ENUM_CONSTANT(BONE_DIRECTION_FROM_PARENT);
}

void ManyBoneIK3D::_make_all_joints_dirty() {
	for (int i = 0; i < settings.size(); i++) {
		_update_joint_array(i);
	}
}

void ManyBoneIK3D::_update_joint_array(int p_index) {
	settings[p_index]->simulation_dirty = true;

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
}

void ManyBoneIK3D::_set_active(bool p_active) {
	if (p_active) {
		reset();
	}
}

void ManyBoneIK3D::_process_modification(double p_delta) {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	for (int i = 0; i < settings.size(); i++) {
		_init_joints(skeleton, settings[i]);
		Node3D *target = Object::cast_to<Node3D>(get_node_or_null(settings[i]->target_node));
		if (!target) {
			return;
		}
		Vector3 target_vector;
		if (settings[i]->use_target_axis) {
			int axis = (int)settings[i]->target_axis;
			target_vector = target->get_global_basis().get_column(axis / 2) * (axis % 2 == 0 ? 1 : -1);
			target_vector.normalize();
			ERR_CONTINUE_MSG(target_vector.is_zero_approx(), "Target axis must not be zero.");
		} else {
			if (settings[i]->joints.is_empty()) {
				continue; // Abort.
			}
			Transform3D root_joint_transform = skeleton->get_bone_global_pose(settings[i]->joints[0]->bone);
			target_vector = target->get_global_position() - root_joint_transform.origin;
			target_vector.normalize();
			ERR_CONTINUE_MSG(target_vector.is_zero_approx(), "Target position must not be the same with root bone joint position.");
		}

		_process_joints(p_delta, skeleton, settings[i]->joints, settings[i]->chain, settings[i]->cached_space, target->get_global_position(), target_vector, settings[i]->max_iterations, settings[i]->min_distance);
	}
}

Quaternion ManyBoneIK3D::get_local_pose_rotation(Skeleton3D *p_skeleton, int p_bone, const Quaternion &p_global_pose_rotation) {
	int parent = p_skeleton->get_bone_parent(p_bone);
	if (parent < 0) {
		return p_global_pose_rotation;
	}
	return p_skeleton->get_bone_global_pose(parent).basis.orthonormalized().inverse() * p_global_pose_rotation;
}

// TODO: coding.
ManyBoneIK3D::TwistSwing ManyBoneIK3D::decompose_rotation_to_twist_and_swing(const Quaternion &p_rest, const Quaternion &p_rotation) {
	return TwistSwing();
}

// TODO: coding.
Quaternion ManyBoneIK3D::compose_rotation_from_twist_and_swing(const Quaternion &p_rest, const TwistSwing &p_twist_and_swing) {
	return Quaternion();
}

void ManyBoneIK3D::reset() {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	for (int i = 0; i < settings.size(); i++) {
		settings[i]->simulation_dirty = true;
		_init_joints(skeleton, settings[i]);
	}
}

void ManyBoneIK3D::_init_joints(Skeleton3D *p_skeleton, ManyBoneIK3DSetting *setting) {
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
		if (last && extend_end_bone) {
			Vector3 axis = get_end_bone_axis(setting->end_bone, setting->end_bone_direction);
			if (axis.is_zero_approx()) {
				continue;
			}
			setting->joints[i]->solver_info = memnew(ManyBoneIK3DSolverInfo);
			setting->joints[i]->solver_info->forward_vector = axis;
			setting->joints[i]->solver_info->length = setting->end_bone_length;
			setting->joints[i]->solver_info->current_rot = Quaternion(0, 0, 0, 1);
			setting->chain.push_back(p_skeleton->get_bone_global_pose(setting->joints[i]->bone).xform(axis * setting->end_bone_length));
		} else if (!last) {
			setting->joints[i]->solver_info = memnew(ManyBoneIK3DSolverInfo);
			Vector3 axis = p_skeleton->get_bone_rest(setting->joints[i + 1]->bone).origin;
			setting->joints[i]->solver_info->forward_vector = axis.normalized();
			setting->joints[i]->solver_info->length = axis.length();
			setting->joints[i]->solver_info->current_rot = Quaternion(0, 0, 0, 1);
		}
	}

	setting->simulation_dirty = false;
}

void ManyBoneIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Transform3D &p_space, const Vector3 &p_destination, const Vector3 &p_target_vector, int p_max_iterations, real_t p_min_distance) {
	// Solve IK here in extended class. Show example for iterating parent to child below.
	/*
	for (int i = 0; i < p_joints.size(); i++) {
		ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
		if (!solver_info) {
			continue; // Means not extended end bone.
		}
		Vector3 destination;
		Ref<JointLimitation3D> limitation = p_joints[i]->limitation;
		if (limitation.is_valid()) {
			destination = limitation.solve(destination,  p_joints[i]->cached_space * p_skeleton->get_bone_global_pose(p_joints[i]->bone) * p_joints[i]->limitation_rotation_offset);
		}
	}
	*/
}
