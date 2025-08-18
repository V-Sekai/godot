/**************************************************************************/
/*  two_bone_ik_3d.cpp                                                    */
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

#include "two_bone_ik_3d.h"

bool TwoBoneIK3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "root_bone_name") {
			set_root_bone_name(which, p_value);
		} else if (what == "root_bone") {
			set_root_bone(which, p_value);
		} else if (what == "middle_bone_name") {
			set_middle_bone_name(which, p_value);
		} else if (what == "middle_bone") {
			set_middle_bone(which, p_value);
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
		} else if (what == "use_virtual_end") {
			set_use_virtual_end(which, p_value);
		} else if (what == "extend_end_bone") {
			set_extend_end_bone(which, p_value);
		} else if (what == "pole_node") {
			set_pole_node(which, p_value);
		} else if (what == "knuckle_direction") {
			set_knuckle_direction(which, static_cast<SecondaryDirection>((int)p_value));
		} else if (what == "knuckle_direction_vector") {
			set_knuckle_direction_vector(which, p_value);
		} else if (what == "target_node") {
			set_target_node(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool TwoBoneIK3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, settings.size(), false);

		if (what == "root_bone_name") {
			r_ret = get_root_bone_name(which);
		} else if (what == "root_bone") {
			r_ret = get_root_bone(which);
		} else if (what == "middle_bone_name") {
			r_ret = get_middle_bone_name(which);
		} else if (what == "middle_bone") {
			r_ret = get_middle_bone(which);
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
		} else if (what == "use_virtual_end") {
			r_ret = is_using_virtual_end(which);
		} else if (what == "extend_end_bone") {
			r_ret = is_end_bone_extended(which);
		} else if (what == "pole_node") {
			r_ret = get_pole_node(which);
		} else if (what == "knuckle_direction") {
			r_ret = (int)get_knuckle_direction(which);
		} else if (what == "knuckle_direction_vector") {
			r_ret = get_knuckle_direction_vector(which);
		} else if (what == "target_node") {
			r_ret = get_target_node(which);
		} else {
			return false;
		}
	}
	return true;
}

void TwoBoneIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
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
		props.push_back(PropertyInfo(Variant::STRING, path + "middle_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "middle_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::STRING, path + "end_bone_name", PROPERTY_HINT_ENUM_SUGGESTION, enum_hint));
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
		props.push_back(PropertyInfo(Variant::BOOL, path + "use_virtual_end"));
		props.push_back(PropertyInfo(Variant::BOOL, path + "extend_end_bone"));
		props.push_back(PropertyInfo(Variant::INT, path + "end_bone/direction", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_bone_direction()));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "end_bone/length", PROPERTY_HINT_RANGE, "0,1,0.001,or_greater,suffix:m"));
		props.push_back(PropertyInfo(Variant::NODE_PATH, path + "pole_node"));
		props.push_back(PropertyInfo(Variant::INT, path + "knuckle_direction", PROPERTY_HINT_ENUM, SkeletonModifier3D::get_hint_secondary_direction()));
		props.push_back(PropertyInfo(Variant::VECTOR3, path + "knuckle_direction_vector"));
		props.push_back(PropertyInfo(Variant::NODE_PATH, path + "target_node"));
	}

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

void TwoBoneIK3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();

		bool force_hide = false;
		if ((split[2] == "end_bone" || split[2] == "end_bone_name") && split.size() == 3 && is_using_virtual_end(which)) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
		if (split[2] == "use_virtual_end" && get_middle_bone(which) == -1) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
		if (split[2] == "extend_end_bone") {
			if (is_using_virtual_end(which)) {
				p_property.usage = PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_READ_ONLY;
			} else if (get_end_bone(which) == -1) {
				p_property.usage = PROPERTY_USAGE_NONE;
				force_hide = true;
			}
		}
		if (force_hide || (split[2] == "end_bone" && !is_end_bone_extended(which) && split.size() > 3)) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}

		if (split[2] == "knuckle_direction_vector" && get_knuckle_direction(which) != SECONDARY_DIRECTION_CUSTOM) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

// Setting.

void TwoBoneIK3D::set_root_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->root_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_root_bone(p_index, sk->find_bone(settings[p_index]->root_bone_name));
	}
}

String TwoBoneIK3D::get_root_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return settings[p_index]->root_bone_name;
}

void TwoBoneIK3D::set_root_bone(int p_index, int p_bone) {
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

int TwoBoneIK3D::get_root_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return settings[p_index]->root_bone;
}

void TwoBoneIK3D::set_middle_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->middle_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_middle_bone(p_index, sk->find_bone(settings[p_index]->middle_bone_name));
	}
}

String TwoBoneIK3D::get_middle_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return settings[p_index]->middle_bone_name;
}

void TwoBoneIK3D::set_middle_bone(int p_index, int p_bone) {
	ERR_FAIL_INDEX(p_index, settings.size());
	bool changed = settings[p_index]->middle_bone != p_bone;
	settings[p_index]->middle_bone = p_bone;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		if (settings[p_index]->middle_bone <= -1 || settings[p_index]->middle_bone >= sk->get_bone_count()) {
			WARN_PRINT("Middle bone index out of range!");
			settings[p_index]->middle_bone = -1;
			settings[p_index]->use_virtual_end = false; // To sync inspector.
		} else {
			settings[p_index]->middle_bone_name = sk->get_bone_name(settings[p_index]->middle_bone);
		}
	}
	if (changed) {
		_update_joints(p_index);
	}
	notify_property_list_changed();
}

int TwoBoneIK3D::get_middle_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return settings[p_index]->middle_bone;
}

void TwoBoneIK3D::set_end_bone_name(int p_index, const String &p_bone_name) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->end_bone_name = p_bone_name;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		set_end_bone(p_index, sk->find_bone(settings[p_index]->end_bone_name));
	}
}

String TwoBoneIK3D::get_end_bone_name(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), String());
	return settings[p_index]->end_bone_name;
}

void TwoBoneIK3D::set_end_bone(int p_index, int p_bone) {
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

int TwoBoneIK3D::get_end_bone(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), -1);
	return settings[p_index]->get_end_bone();
}

void TwoBoneIK3D::set_use_virtual_end(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	bool changed = settings[p_index]->use_virtual_end != p_enabled;
	settings[p_index]->use_virtual_end = p_enabled;
	if (p_enabled) {
		// To sync inspector.
		settings[p_index]->extend_end_bone = true;
	}
	settings[p_index]->simulation_dirty = true;
	if (changed) {
		_update_joints(p_index);
	}
	notify_property_list_changed();
}

bool TwoBoneIK3D::is_using_virtual_end(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), false);
	return settings[p_index]->use_virtual_end;
}

void TwoBoneIK3D::set_extend_end_bone(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->extend_end_bone = p_enabled;
	settings[p_index]->simulation_dirty = true;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_knuckle_direction(sk, p_index);
	}
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

bool TwoBoneIK3D::is_end_bone_extended(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), false);
	return settings[p_index]->extend_end_bone;
}

void TwoBoneIK3D::set_end_bone_direction(int p_index, BoneDirection p_bone_direction) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->end_bone_direction = p_bone_direction;
	settings[p_index]->simulation_dirty = true;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_knuckle_direction(sk, p_index);
	}
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

SkeletonModifier3D::BoneDirection TwoBoneIK3D::get_end_bone_direction(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), BONE_DIRECTION_FROM_PARENT);
	return settings[p_index]->end_bone_direction;
}

void TwoBoneIK3D::set_end_bone_length(int p_index, float p_length) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->end_bone_length = p_length;
	settings[p_index]->simulation_dirty = true;
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

float TwoBoneIK3D::get_end_bone_length(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), 0);
	return settings[p_index]->end_bone_length;
}

void TwoBoneIK3D::set_target_node(int p_index, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->target_node = p_node_path;
}

NodePath TwoBoneIK3D::get_target_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), NodePath());
	return settings[p_index]->target_node;
}

void TwoBoneIK3D::set_pole_node(int p_index, const NodePath &p_node_path) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->pole_node = p_node_path;
}

NodePath TwoBoneIK3D::get_pole_node(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), NodePath());
	return settings[p_index]->pole_node;
}

void TwoBoneIK3D::set_knuckle_direction(int p_index, SecondaryDirection p_direction) {
	ERR_FAIL_INDEX(p_index, settings.size());
	settings[p_index]->knuckle_direction = p_direction;
	settings[p_index]->simulation_dirty = true;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_knuckle_direction(sk, p_index);
	}
	notify_property_list_changed();
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

SkeletonModifier3D::SecondaryDirection TwoBoneIK3D::get_knuckle_direction(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), SECONDARY_DIRECTION_NONE);
	return settings[p_index]->knuckle_direction;
}

void TwoBoneIK3D::set_knuckle_direction_vector(int p_index, const Vector3 &p_vector) {
	ERR_FAIL_INDEX(p_index, settings.size());
	if (settings[p_index]->knuckle_direction != SECONDARY_DIRECTION_CUSTOM) {
		return;
	}
	settings[p_index]->knuckle_direction_vector = p_vector;
	settings[p_index]->simulation_dirty = true;
	Skeleton3D *sk = get_skeleton();
	if (sk) {
		_validate_knuckle_direction(sk, p_index);
	}
#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

Vector3 TwoBoneIK3D::get_knuckle_direction_vector(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), Vector3());
	return settings[p_index]->get_knuckle_direction_vector();
}

void TwoBoneIK3D::set_setting_count(int p_count) {
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
			settings.write[p_count - i] = memnew(TwoBoneIK3DSetting);
		}
	}
	notify_property_list_changed();
}

int TwoBoneIK3D::get_setting_count() const {
	return settings.size();
}

void TwoBoneIK3D::clear_settings() {
	set_setting_count(0);
}

bool TwoBoneIK3D::is_valid(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, settings.size(), false);
	return settings[p_index]->root_bone != -1 && settings[p_index]->middle_bone != -1 && settings[p_index]->is_end_valid() && settings[p_index]->root_joint.bone != -1 && settings[p_index]->mid_joint.bone != -1;
}

void TwoBoneIK3D::_bind_methods() {
	// Setting.
	ClassDB::bind_method(D_METHOD("set_root_bone_name", "index", "bone_name"), &TwoBoneIK3D::set_root_bone_name);
	ClassDB::bind_method(D_METHOD("get_root_bone_name", "index"), &TwoBoneIK3D::get_root_bone_name);
	ClassDB::bind_method(D_METHOD("set_root_bone", "index", "bone"), &TwoBoneIK3D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone", "index"), &TwoBoneIK3D::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_middle_bone_name", "index", "bone_name"), &TwoBoneIK3D::set_middle_bone_name);
	ClassDB::bind_method(D_METHOD("get_middle_bone_name", "index"), &TwoBoneIK3D::get_middle_bone_name);
	ClassDB::bind_method(D_METHOD("set_middle_bone", "index", "bone"), &TwoBoneIK3D::set_middle_bone);
	ClassDB::bind_method(D_METHOD("get_middle_bone", "index"), &TwoBoneIK3D::get_middle_bone);

	ClassDB::bind_method(D_METHOD("set_end_bone_name", "index", "bone_name"), &TwoBoneIK3D::set_end_bone_name);
	ClassDB::bind_method(D_METHOD("get_end_bone_name", "index"), &TwoBoneIK3D::get_end_bone_name);
	ClassDB::bind_method(D_METHOD("set_end_bone", "index", "bone"), &TwoBoneIK3D::set_end_bone);
	ClassDB::bind_method(D_METHOD("get_end_bone", "index"), &TwoBoneIK3D::get_end_bone);

	ClassDB::bind_method(D_METHOD("set_use_virtual_end", "index", "enabled"), &TwoBoneIK3D::set_use_virtual_end);
	ClassDB::bind_method(D_METHOD("is_using_virtual_end", "index"), &TwoBoneIK3D::is_using_virtual_end);
	ClassDB::bind_method(D_METHOD("set_extend_end_bone", "index", "enabled"), &TwoBoneIK3D::set_extend_end_bone);
	ClassDB::bind_method(D_METHOD("is_end_bone_extended", "index"), &TwoBoneIK3D::is_end_bone_extended);
	ClassDB::bind_method(D_METHOD("set_end_bone_direction", "index", "bone_direction"), &TwoBoneIK3D::set_end_bone_direction);
	ClassDB::bind_method(D_METHOD("get_end_bone_direction", "index"), &TwoBoneIK3D::get_end_bone_direction);
	ClassDB::bind_method(D_METHOD("set_end_bone_length", "index", "length"), &TwoBoneIK3D::set_end_bone_length);
	ClassDB::bind_method(D_METHOD("get_end_bone_length", "index"), &TwoBoneIK3D::get_end_bone_length);

	ClassDB::bind_method(D_METHOD("set_pole_node", "index", "pole_node"), &TwoBoneIK3D::set_pole_node);
	ClassDB::bind_method(D_METHOD("get_pole_node", "index"), &TwoBoneIK3D::get_pole_node);

	ClassDB::bind_method(D_METHOD("set_knuckle_direction", "index", "direction"), &TwoBoneIK3D::set_knuckle_direction);
	ClassDB::bind_method(D_METHOD("get_knuckle_direction", "index"), &TwoBoneIK3D::get_knuckle_direction);
	ClassDB::bind_method(D_METHOD("set_knuckle_direction_vector", "index", "vector"), &TwoBoneIK3D::set_knuckle_direction_vector);
	ClassDB::bind_method(D_METHOD("get_knuckle_direction_vector", "index"), &TwoBoneIK3D::get_knuckle_direction_vector);

	ClassDB::bind_method(D_METHOD("set_target_node", "index", "target_node"), &TwoBoneIK3D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node", "index"), &TwoBoneIK3D::get_target_node);

	ClassDB::bind_method(D_METHOD("set_setting_count", "count"), &TwoBoneIK3D::set_setting_count);
	ClassDB::bind_method(D_METHOD("get_setting_count"), &TwoBoneIK3D::get_setting_count);
	ClassDB::bind_method(D_METHOD("clear_settings"), &TwoBoneIK3D::clear_settings);

	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");
}

void TwoBoneIK3D::_validate_bone_names() {
	for (int i = 0; i < settings.size(); i++) {
		// Prior bone name.
		if (!settings[i]->root_bone_name.is_empty()) {
			set_root_bone_name(i, settings[i]->root_bone_name);
		} else if (settings[i]->root_bone != -1) {
			set_root_bone(i, settings[i]->root_bone);
		}
		// Prior bone name.
		if (!settings[i]->middle_bone_name.is_empty()) {
			set_middle_bone_name(i, settings[i]->middle_bone_name);
		} else if (settings[i]->middle_bone != -1) {
			set_middle_bone(i, settings[i]->middle_bone);
		}
		// Prior bone name.
		if (!settings[i]->end_bone_name.is_empty()) {
			set_end_bone_name(i, settings[i]->end_bone_name);
		} else if (settings[i]->end_bone != -1) {
			set_end_bone(i, settings[i]->end_bone);
		}
	}
}

void TwoBoneIK3D::_validate_knuckle_directions(Skeleton3D *p_skeleton) const {
	for (int i = 0; i < settings.size(); i++) {
		_validate_knuckle_direction(p_skeleton, i);
	}
}

void TwoBoneIK3D::_validate_knuckle_direction(Skeleton3D *p_skeleton, int p_index) const {
	TwoBoneIK3DSetting *setting = settings[p_index];
	SecondaryDirection dir = setting->knuckle_direction;
	if (!is_valid(p_index) || dir == SECONDARY_DIRECTION_NONE) {
		return;
	}
	Vector3 kv = get_knuckle_direction_vector(p_index).normalized();
	Vector3 fwd;

	// End bone.
	int valid_end_bone = setting->get_end_bone();
	Vector3 axis = get_bone_axis(valid_end_bone, setting->end_bone_direction);
	Vector3 global_rest_origin;
	if (setting->extend_end_bone && setting->end_bone_length > 0 && !axis.is_zero_approx()) {
		global_rest_origin = p_skeleton->get_bone_global_rest(valid_end_bone).xform(axis * setting->end_bone_length);
	} else {
		// Shouldn't be using virtual end.
		global_rest_origin = p_skeleton->get_bone_global_rest(valid_end_bone).origin;
	}

	// Middle bone.
	axis = global_rest_origin - p_skeleton->get_bone_global_rest(setting->middle_bone).origin;
	if (!axis.is_zero_approx()) {
		fwd = p_skeleton->get_bone_global_rest(setting->middle_bone).basis.get_rotation_quaternion().xform_inv(axis).normalized();
	} else {
		return;
	}

	if (Math::is_equal_approx(Math::abs(kv.dot(fwd)), 1)) {
		WARN_PRINT_ED("Setting: " + itos(p_index) + ": Knuckle direction and forward vectors are colinear. This is not advised as it may cause unwanted rotation.");
	}
}

void TwoBoneIK3D::_make_all_joints_dirty() {
	for (int i = 0; i < settings.size(); i++) {
		_update_joints(i);
	}
}

void TwoBoneIK3D::_init_joints(Skeleton3D *p_skeleton, TwoBoneIK3DSetting *setting) {
	setting->cached_space = p_skeleton->get_global_transform();
	if (!setting->simulation_dirty) {
		return;
	}
	setting->root_pos = Vector3();
	setting->mid_pos = Vector3();
	setting->end_pos = Vector3();
	if (setting->root_joint.solver_info) {
		memdelete(setting->root_joint.solver_info);
		setting->root_joint.solver_info = nullptr;
	}
	if (setting->mid_joint.solver_info) {
		memdelete(setting->mid_joint.solver_info);
		setting->mid_joint.solver_info = nullptr;
	}
	if (setting->root_bone == -1 || setting->middle_bone == -1 || !setting->is_end_valid() || setting->root_joint.bone == -1 || setting->mid_joint.bone == -1) {
		return;
	}
	bool extend_end_bone = setting->extend_end_bone && setting->end_bone_length > 0;

	// End bone.
	int valid_end_bone = setting->get_end_bone();
	Vector3 axis = get_bone_axis(valid_end_bone, setting->end_bone_direction);
	Vector3 global_rest_origin;
	if (extend_end_bone && setting->end_bone_length > 0 && !axis.is_zero_approx()) {
		setting->end_pos = p_skeleton->get_bone_global_pose(valid_end_bone).xform(axis * setting->end_bone_length);
		global_rest_origin = p_skeleton->get_bone_global_rest(valid_end_bone).xform(axis * setting->end_bone_length);
	} else {
		// Shouldn't be using virtual end.
		setting->end_pos = p_skeleton->get_bone_global_pose(valid_end_bone).origin;
		global_rest_origin = p_skeleton->get_bone_global_rest(valid_end_bone).origin;
	}

	// Middle bone.
	axis = global_rest_origin - p_skeleton->get_bone_global_rest(setting->middle_bone).origin;
	global_rest_origin = p_skeleton->get_bone_global_rest(setting->middle_bone).origin;
	if (!axis.is_zero_approx()) {
		setting->mid_pos = p_skeleton->get_bone_global_pose(setting->middle_bone).origin;
		setting->mid_joint.solver_info = memnew(ManyBoneIK3DSolverInfo);
		setting->mid_joint.solver_info->forward_vector = p_skeleton->get_bone_global_rest(setting->middle_bone).basis.get_rotation_quaternion().xform_inv(axis).normalized();
		setting->mid_joint.solver_info->length = axis.length();
	} else {
		return;
	}

	// Root bone.
	axis = global_rest_origin - p_skeleton->get_bone_global_rest(setting->root_bone).origin;
	global_rest_origin = p_skeleton->get_bone_global_rest(setting->root_bone).origin;
	if (!axis.is_zero_approx()) {
		setting->root_pos = p_skeleton->get_bone_global_pose(setting->root_bone).origin;
		setting->root_joint.solver_info = memnew(ManyBoneIK3DSolverInfo);
		setting->root_joint.solver_info->forward_vector = p_skeleton->get_bone_global_rest(setting->root_bone).basis.get_rotation_quaternion().xform_inv(axis).normalized();
		setting->root_joint.solver_info->length = axis.length();
	} else if (setting->mid_joint.solver_info) {
		memdelete(setting->mid_joint.solver_info);
		setting->mid_joint.solver_info = nullptr;
		return;
	}

	setting->init_current_joint_rotations(p_skeleton);

	real_t total_length = setting->root_joint.solver_info->length + setting->mid_joint.solver_info->length;
	setting->cached_length_sq = total_length * total_length;

	setting->simulation_dirty = false;
}

void TwoBoneIK3D::_update_joints(int p_index) {
	settings[p_index]->simulation_dirty = true;
	settings[p_index]->root_joint.bone = -1;
	settings[p_index]->mid_joint.bone = -1;

#ifdef TOOLS_ENABLED
	update_gizmos(); // To clear invalid setting.
#endif // TOOLS_ENABLED

	Skeleton3D *sk = get_skeleton();
	if (!sk || settings[p_index]->root_bone == -1 || settings[p_index]->middle_bone == -1 || !settings[p_index]->is_end_valid()) {
		return;
	}

	// Validation for middle bone.
	int parent_bone = settings[p_index]->root_bone;
	int current_bone = settings[p_index]->middle_bone;
	bool valid = false;
	while (current_bone >= 0) {
		if (current_bone == parent_bone) {
			valid = true;
			break;
		}
		current_bone = sk->get_bone_parent(current_bone);
	}
	if (!valid) {
		ERR_FAIL_EDMSG("Middle bone must be a child of root bone.");
	}

	// Validation for end bone.
	if (!settings[p_index]->use_virtual_end) {
		parent_bone = settings[p_index]->middle_bone;
		current_bone = settings[p_index]->end_bone;
		valid = false;
		while (current_bone >= 0) {
			if (current_bone == parent_bone) {
				valid = true;
				break;
			}
			current_bone = sk->get_bone_parent(current_bone);
		}
		if (!valid) {
			ERR_FAIL_EDMSG("End bone must be a child of middle bone.");
		}
	}

	// Copy bone indices to the joint settings, but name is not used in TwoBoneIK.
	settings[p_index]->root_joint.bone = settings[p_index]->root_bone;
	settings[p_index]->mid_joint.bone = settings[p_index]->middle_bone;

	if (sk) {
		_validate_knuckle_directions(sk);
	}

#ifdef TOOLS_ENABLED
	update_gizmos();
#endif // TOOLS_ENABLED
}

void TwoBoneIK3D::_process_ik(Skeleton3D *p_skeleton, double p_delta) {
	for (int i = 0; i < settings.size(); i++) {
		_init_joints(p_skeleton, settings[i]);
		Node3D *target = Object::cast_to<Node3D>(get_node_or_null(settings[i]->target_node));
		Node3D *pole = Object::cast_to<Node3D>(get_node_or_null(settings[i]->pole_node));
		if (!target || !pole || !settings[i]->is_valid()) {
			continue; // Abort.
		}
		Vector3 destination = settings[i]->cached_space.affine_inverse().xform(target->get_global_position());
		Vector3 pole_destination = settings[i]->cached_space.affine_inverse().xform(pole->get_global_position());
		settings[i]->cache_current_joint_rotations(p_skeleton, pole_destination); // Iterate over first to detect parent (outside of the chain) bone pose changes.
		_process_joints(p_delta, p_skeleton, settings[i], destination, pole_destination);
	}
}

void TwoBoneIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, TwoBoneIK3DSetting *p_setting, const Vector3 &p_destination, const Vector3 &p_pole_destination) {
	// Solve the IK for this iteration.
	Vector3 destination = p_destination;

	// Make vector from root to destination.
	p_setting->root_pos = p_skeleton->get_bone_global_pose(p_setting->root_joint.bone).origin; // New root position.
	Vector3 root_to_destination = destination - p_setting->root_pos;
	if (root_to_destination.is_zero_approx()) {
		return; // Abort.
	}

	real_t rd_len_sq = root_to_destination.length_squared();
	// Compare the distance to the target with the length of the bones.
	if (rd_len_sq >= p_setting->cached_length_sq) {
		// Result is straight.
		Vector3 rd_nrm = root_to_destination.normalized();
		p_setting->mid_pos = p_setting->root_pos + rd_nrm * p_setting->root_joint.solver_info->length;
		p_setting->end_pos = p_setting->mid_pos + rd_nrm * p_setting->mid_joint.solver_info->length;
	} else {
		// Check if the target can be reached by subtracting the lengths of the bones.
		// If not, push out target to normal of the root bone sphere.
		real_t sub = p_setting->root_joint.solver_info->length - p_setting->mid_joint.solver_info->length;
		if (rd_len_sq < sub * sub) {
			Vector3 push_nrm = (destination - p_setting->root_pos).normalized();
			destination = p_setting->root_pos + push_nrm * Math::abs(sub);
			root_to_destination = destination - p_setting->root_pos;
		}

		// End is snapped to the target.
		p_setting->end_pos = destination;

		// Result is bent, determine the mid position to respect the pole target.
		// Mid-position should be a point of intersection of two circles.
		real_t l_chain = root_to_destination.length();
		Vector3 u = root_to_destination.normalized();
		Vector3 pole_vec = get_normal(p_setting->root_pos, p_setting->end_pos, p_pole_destination);

		// Circle1: center is the root, radius is the length of the root bone.
		real_t r_root = p_setting->root_joint.solver_info->length;
		// Circle2: center is the target, radius is the length of the middle bone.
		real_t r_mid = p_setting->mid_joint.solver_info->length;

		real_t a = (l_chain * l_chain + r_root * r_root - r_mid * r_mid) / (2.0 * l_chain);
		real_t h2 = r_root * r_root - a * a;
		if (h2 < 0) {
			h2 = 0;
		}
		real_t h = Math::sqrt(h2);

		Vector3 det_plus = (p_setting->root_pos + u * a) + pole_vec * h;
		Vector3 det_minus = (p_setting->root_pos + u * a) - pole_vec * h;

		// Pick the intersection that is closest to the pole target.
		p_setting->mid_pos = p_pole_destination.distance_squared_to(det_plus) < p_pole_destination.distance_squared_to(det_minus) ? det_plus : det_minus;
	}

	// Update virtual bone rest/poses.
	p_setting->cache_current_vectors(p_skeleton);
	p_setting->cache_current_joint_rotations(p_skeleton, p_pole_destination);

	// Apply the virtual bone rest/poses to the actual bones.
	p_skeleton->set_bone_pose_rotation(p_setting->root_joint.bone, p_setting->root_joint.solver_info->current_lpose);
	// Mid joint pose is relative to the root joint pose for the case root-mid or mid-end have more than 1 joints.
	p_skeleton->set_bone_pose_rotation(p_setting->mid_joint.bone, get_local_pose_rotation(p_skeleton, p_setting->mid_joint.bone, p_setting->mid_joint.solver_info->current_gpose));
}

void TwoBoneIK3D::reset() {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}
	for (int i = 0; i < settings.size(); i++) {
		settings[i]->simulation_dirty = true;
		_init_joints(skeleton, settings[i]);
	}
}

TwoBoneIK3D::~TwoBoneIK3D() {
	clear_settings();
}
