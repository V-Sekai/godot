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
				String opt = path.get_slicec('/', 5);
				if (opt.is_empty()) {
					set_joint_limitation(which, idx, p_value);
				} else if (opt == "right_axis") {
					set_joint_limitation_right_axis(which, idx, p_value);
				} else if (opt == "right_axis_vector") {
					set_joint_limitation_right_axis_vector(which, idx, p_value);
				} else if (opt == "rotation_offset") {
					set_joint_limitation_rotation_offset(which, idx, p_value);
				} else {
					return false;
				}
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
				String opt = path.get_slicec('/', 5);
				if (opt.is_empty()) {
					r_ret = get_joint_limitation(which, idx);
				} else if (opt == "right_axis") {
					r_ret = get_joint_limitation_right_axis(which, idx);
				} else if (opt == "right_axis_vector") {
					r_ret = get_joint_limitation_right_axis_vector(which, idx);
				} else if (opt == "rotation_offset") {
					r_ret = get_joint_limitation_rotation_offset(which, idx);
				} else {
					return false;
				}
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	return true;
}

void ChainIK3D::get_property_list(List<PropertyInfo> *p_list) const {
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
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone/direction", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_bone_direction()));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "end_bone/length", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"));
		props.push_back(PropertyInfo(Variant::INT, path + "joint_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Joints," + path + "joints/,static,const"));
		for (int j = 0; j < chain_settings[i]->joints.size(); j++) {
			String joint_path = path + "joints/" + itos(j) + "/";
			props.push_back(PropertyInfo(Variant::STRING, joint_path + "bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint, PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY | PROPERTY_USAGE_STORAGE));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_READ_ONLY));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "rotation_axis", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_rotation_axis()));
			props.push_back(PropertyInfo(Variant::VECTOR3, joint_path + "rotation_axis_vector"));
			props.push_back(PropertyInfo(Variant::OBJECT, joint_path + "limitation", PROPERTY_HINT_RESOURCE_TYPE, "JointLimitation3D"));
			props.push_back(PropertyInfo(Variant::INT, joint_path + "limitation/right_axis", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_secondary_direction()));
			props.push_back(PropertyInfo(Variant::VECTOR3, joint_path + "limitation/right_axis_vector"));
			props.push_back(PropertyInfo(Variant::QUATERNION, joint_path + "limitation/rotation_offset"));
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
		if (split[2] == "extend_end_bone" && get_end_bone(which) == -1) {
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
			if (split[4] == "limitation" && split.size() > 5) {
				if (get_joint_limitation(which, joint).is_null()) {
					p_property.usage = PROPERTY_USAGE_NONE;
				} else if (split[5] == "right_axis_vector" && get_joint_limitation_right_axis(which, joint) != SECONDARY_DIRECTION_CUSTOM) {
					p_property.usage = PROPERTY_USAGE_NONE;
				}
			}
		}
	}
}

// Setting.

void ChainIK3D::set_root_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	chain_settings[p_index]->root_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_root_bone(p_index, sk->find_bone(chain_settings[p_index]->root_bone_name));
	}
}

String ChainIK3D::get_root_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return chain_settings[p_index]->root_bone_name;
}

void ChainIK3D::set_root_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, settings.size());
	bool changed = chain_settings[p_index]->root_bone != p_bone;
	chain_settings[p_index]->root_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (chain_settings[p_index]->root_bone <= -1 || chain_settings[p_index]->root_bone >= sk->get_bone_count()) {
			WARN_PRINT("Root bone index out of range!");
			chain_settings[p_index]->root_bone = -1;
		} else {
			chain_settings[p_index]->root_bone_name = sk->get_bone_name(chain_settings[p_index]->root_bone);
		}
	}
	if (changed) {
		_update_joints(p_index);
	}
}

int ChainIK3D::get_root_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return chain_settings[p_index]->root_bone;
}

void ChainIK3D::set_end_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	chain_settings[p_index]->end_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_end_bone(p_index, sk->find_bone(chain_settings[p_index]->end_bone_name));
	}
}

String ChainIK3D::get_end_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return chain_settings[p_index]->end_bone_name;
}

void ChainIK3D::set_end_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, settings.size());
	bool changed = chain_settings[p_index]->end_bone != p_bone;
	chain_settings[p_index]->end_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (chain_settings[p_index]->end_bone <= -1 || chain_settings[p_index]->end_bone >= sk->get_bone_count()) {
			WARN_PRINT("End bone index out of range!");
			chain_settings[p_index]->end_bone = -1;
		} else {
			chain_settings[p_index]->end_bone_name = sk->get_bone_name(chain_settings[p_index]->end_bone);
		}
	}
	if (changed) {
		_update_joints(p_index);
	}
	notify_property_list_changed();
}

int ChainIK3D::get_end_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return chain_settings[p_index]->end_bone;
}

void ChainIK3D::set_extend_end_bone(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	chain_settings[p_index]->extend_end_bone = p_enabled;
	chain_settings[p_index]->simulation_dirty = true;
	Skeleton3D *sk = get_skeleton();
	if (sk && !chain_settings[p_index]->joints.is_empty()) {
		_validate_rotation_axis(sk, p_index, chain_settings[p_index]->joints.size() - 1);
	}
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

bool ChainIK3D::is_end_bone_extended(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), false);
	return chain_settings[p_index]->extend_end_bone;
}

void ChainIK3D::set_end_bone_direction(int p_index, BoneDirection p_bone_direction) {
	ERR_FAIL_INDEX(p_index, settings.size());
	chain_settings[p_index]->end_bone_direction = p_bone_direction;
	chain_settings[p_index]->simulation_dirty = true;
	Skeleton3D *sk = get_skeleton();
	if (sk && !chain_settings[p_index]->joints.is_empty()) {
		_validate_rotation_axis(sk, p_index, chain_settings[p_index]->joints.size() - 1);
	}
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

SkeletonModifier3D::BoneDirection ChainIK3D::get_end_bone_direction(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), BONE_DIRECTION_FROM_PARENT);
	return chain_settings[p_index]->end_bone_direction;
}

void ChainIK3D::set_end_bone_length(int p_index, float p_length) {
	ERR_FAIL_INDEX(p_index, settings.size());
	chain_settings[p_index]->end_bone_length = p_length;
	chain_settings[p_index]->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

float ChainIK3D::get_end_bone_length(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	return chain_settings[p_index]->end_bone_length;
}

// Individual joints.

void ChainIK3D::set_joint_bone_name(int p_index, int p_joint, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ChainIK3DJointSetting *> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_joint_bone(p_index, p_joint, sk->find_bone(joints[p_joint]->bone_name));
	}
}

String ChainIK3D::get_joint_bone_name(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), String());
	return joints[p_joint]->bone_name;
}

void ChainIK3D::set_joint_bone(int p_index, int p_joint, int p_bone) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ChainIK3DJointSetting *> &joints = chain_settings[p_index]->joints;
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
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), -1);
	return joints[p_joint]->bone;
}

void ChainIK3D::set_joint_rotation_axis(int p_index, int p_joint, RotationAxis p_axis) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ChainIK3DJointSetting *> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->rotation_axis = p_axis;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_rotation_axis(sk, p_index, p_joint);
	}
	notify_property_list_changed();
	chain_settings[p_index]->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

SkeletonModifier3D::RotationAxis ChainIK3D::get_joint_rotation_axis(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), ROTATION_AXIS_ALL);
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), ROTATION_AXIS_ALL);
	return joints[p_joint]->rotation_axis;
}

void ChainIK3D::set_joint_rotation_axis_vector(int p_index, int p_joint, Vector3 p_vector) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ChainIK3DJointSetting *> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->rotation_axis_vector = p_vector;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_rotation_axis(sk, p_index, p_joint);
	}
	chain_settings[p_index]->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Vector3 ChainIK3D::get_joint_rotation_axis_vector(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Vector3());
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), Vector3());
	return joints[p_joint]->get_rotation_axis_vector();
}

Quaternion ChainIK3D::get_joint_limitation_space(int p_index, int p_joint, const Vector3 &p_forward) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Quaternion());
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), Quaternion());
	return joints[p_joint]->get_limitation_space(p_forward);
}

void ChainIK3D::set_joint_limitation(int p_index, int p_joint, const Ref<JointLimitation3D> &p_limitation) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ChainIK3DJointSetting *> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->limitation = p_limitation;
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	update_gizmos();
	_bind_limitations();
#endif // TOOLS_ENABLED
}

Ref<JointLimitation3D> ChainIK3D::get_joint_limitation(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Ref<JointLimitation3D>());
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), Ref<JointLimitation3D>());
	return joints[p_joint]->limitation;
}

void ChainIK3D::set_joint_limitation_right_axis(int p_index, int p_joint, SecondaryDirection p_direction) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ChainIK3DJointSetting *> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->limitation_right_axis = p_direction;
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

ManyBoneIK3D::SecondaryDirection ChainIK3D::get_joint_limitation_right_axis(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), SECONDARY_DIRECTION_NONE);
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), SECONDARY_DIRECTION_NONE);
	return joints[p_joint]->limitation_right_axis;
}

void ChainIK3D::set_joint_limitation_right_axis_vector(int p_index, int p_joint, const Vector3 &p_vector) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ChainIK3DJointSetting *> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->limitation_right_axis_vector = p_vector;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Vector3 ChainIK3D::get_joint_limitation_right_axis_vector(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Vector3());
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), Vector3());
	return joints[p_joint]->get_limitation_right_axis_vector();
}

void ChainIK3D::set_joint_limitation_rotation_offset(int p_index, int p_joint, const Quaternion &p_offset) {
	ERR_FAIL_INDEX(p_index, settings.size());
	Vector<ChainIK3DJointSetting *> &joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX(p_joint, joints.size());
	joints[p_joint]->limitation_rotation_offset = p_offset;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Quaternion ChainIK3D::get_joint_limitation_rotation_offset(int p_index, int p_joint) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Quaternion());
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	ERR_FAIL_INDEX_V(p_joint, joints.size(), Quaternion());
	return joints[p_joint]->limitation_rotation_offset;
}

void ChainIK3D::set_joint_count(int p_index, int p_count) {
	ERR_FAIL_INDEX(p_index, settings.size());
	ERR_FAIL_COND(p_count < 0);
	Vector<ChainIK3DJointSetting *> &joints = chain_settings[p_index]->joints;
	int delta = p_count - joints.size();
	if (delta < 0) {
		for (int i = delta; i < 0; i++) {
			memdelete(joints[joints.size() + i]);
		}
	}
	joints.resize(p_count);
	delta++;
	if (delta > 1) {
		for (int i = 1; i < delta; i++) {
			joints.write[p_count - i] = memnew(ChainIK3DJointSetting);
		}
	}
	notify_property_list_changed();
}

int ChainIK3D::get_joint_count(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	Vector<ChainIK3DJointSetting *> joints = chain_settings[p_index]->joints;
	return joints.size();
}

void ChainIK3D::_bind_methods() {
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

	// Individual joints.
	ClassDB::bind_method(D_METHOD("get_joint_bone_name", "index", "joint"), &ChainIK3D::get_joint_bone_name);
	ClassDB::bind_method(D_METHOD("get_joint_bone", "index", "joint"), &ChainIK3D::get_joint_bone);
	ClassDB::bind_method(D_METHOD("set_joint_rotation_axis", "index", "joint", "axis"), &ChainIK3D::set_joint_rotation_axis);
	ClassDB::bind_method(D_METHOD("get_joint_rotation_axis", "index", "joint"), &ChainIK3D::get_joint_rotation_axis);
	ClassDB::bind_method(D_METHOD("set_joint_rotation_axis_vector", "index", "joint", "axis_vector"), &ChainIK3D::set_joint_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("get_joint_rotation_axis_vector", "index", "joint"), &ChainIK3D::get_joint_rotation_axis_vector);
	ClassDB::bind_method(D_METHOD("set_joint_limitation", "index", "joint", "limitation"), &ChainIK3D::set_joint_limitation);
	ClassDB::bind_method(D_METHOD("get_joint_limitation", "index", "joint"), &ChainIK3D::get_joint_limitation);
	ClassDB::bind_method(D_METHOD("set_joint_limitation_right_axis", "index", "joint", "direction"), &ChainIK3D::set_joint_limitation_right_axis);
	ClassDB::bind_method(D_METHOD("get_joint_limitation_right_axis", "index", "joint"), &ChainIK3D::get_joint_limitation_right_axis);
	ClassDB::bind_method(D_METHOD("set_joint_limitation_right_axis_vector", "index", "joint", "vector"), &ChainIK3D::set_joint_limitation_right_axis_vector);
	ClassDB::bind_method(D_METHOD("get_joint_limitation_right_axis_vector", "index", "joint"), &ChainIK3D::get_joint_limitation_right_axis_vector);
	ClassDB::bind_method(D_METHOD("set_joint_limitation_rotation_offset", "index", "joint", "offset"), &ChainIK3D::set_joint_limitation_rotation_offset);
	ClassDB::bind_method(D_METHOD("get_joint_limitation_rotation_offset", "index", "joint"), &ChainIK3D::get_joint_limitation_rotation_offset);

	ClassDB::bind_method(D_METHOD("get_joint_count", "index"), &ChainIK3D::get_joint_count);
}

void ChainIK3D::_validate_bone_names() {
	for (int i = 0; i < settings.size(); i++) {
		// Prior bone name.
		if (!chain_settings[i]->root_bone_name.is_empty()) {
			set_root_bone_name(i, chain_settings[i]->root_bone_name);
		} else if (chain_settings[i]->root_bone != -1) {
			set_root_bone(i, chain_settings[i]->root_bone);
		}
		// Prior bone name.
		if (!chain_settings[i]->end_bone_name.is_empty()) {
			set_end_bone_name(i, chain_settings[i]->end_bone_name);
		} else if (chain_settings[i]->end_bone != -1) {
			set_end_bone(i, chain_settings[i]->end_bone);
		}
	}
}

void ChainIK3D::_validate_rotation_axes(Skeleton3D *p_skeleton) const {
	for (int i = 0; i < settings.size(); i++) {
		for (int j = 0; j < chain_settings[i]->joints.size(); j++) {
			_validate_rotation_axis(p_skeleton, i, j);
		}
	}
}

void ChainIK3D::_validate_rotation_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const {
	RotationAxis axis = chain_settings[p_index]->joints[p_joint]->rotation_axis;
	if (axis == ROTATION_AXIS_ALL) {
		return;
	}
	Vector3 rot = get_joint_rotation_axis_vector(p_index, p_joint).normalized();
	Vector3 fwd;
	if (p_joint < chain_settings[p_index]->joints.size() - 1) {
		fwd = p_skeleton->get_bone_rest(chain_settings[p_index]->joints[p_joint + 1]->bone).origin;
	} else if (chain_settings[p_index]->extend_end_bone) {
		fwd = get_bone_axis(chain_settings[p_index]->end_bone, chain_settings[p_index]->end_bone_direction);
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

void ChainIK3D::_init_joints(Skeleton3D *p_skeleton, int p_index) {
	ChainIK3DSetting *setting = chain_settings[p_index];
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

	_post_init_joints(p_index);

	setting->init_current_joint_rotations(p_skeleton);

	setting->simulation_dirty = false;
}

void ChainIK3D::_post_init_joints(int p_index) {
	//
}

void ChainIK3D::_update_joints(int p_index) {
	chain_settings[p_index]->simulation_dirty = true;

#ifdef TOOLS_ENABLED
	update_gizmos(); // To clear invalid setting.
#endif // TOOLS_ENABLED

	Skeleton3D *sk = get_skeleton();
	int current_bone = chain_settings[p_index]->end_bone;
	int root_bone = chain_settings[p_index]->root_bone;
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
	current_bone = chain_settings[p_index]->end_bone;
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
	//
}

#ifdef TOOLS_ENABLED
void ChainIK3D::_bind_limitations() {
	for (int i = 0; i < settings.size(); i++) {
		for (int j = 0; j < chain_settings[i]->joints.size(); j++) {
			if (chain_settings[i]->joints[j]->limitation.is_valid()) {
				chain_settings[i]->joints[j]->limitation->disconnect_changed(callable_mp((Node3D *)this, &Node3D::update_gizmos));
			}
		}
	}
	for (int i = 0; i < settings.size(); i++) {
		for (int j = 0; j < chain_settings[i]->joints.size(); j++) {
			if (chain_settings[i]->joints[j]->limitation.is_valid()) {
				chain_settings[i]->joints[j]->limitation->connect_changed(callable_mp((Node3D *)this, &Node3D::update_gizmos));
			}
		}
	}
}
#endif // TOOLS_ENABLED

ChainIK3D::~ChainIK3D() {
	clear_settings();
}
