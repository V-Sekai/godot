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
#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/string/string_name.h"
#include "ik_bone_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"

void ManyBoneIK3D::set_pin_count(int32_t p_value) {
	int32_t old_count = pins.size();
	pin_count = p_value;
	pins.resize(p_value);
	for (int32_t pin_i = p_value; pin_i-- > old_count;) {
		pins.write[pin_i].instantiate();
	}
	set_dirty();
	notify_property_list_changed();
}

int32_t ManyBoneIK3D::get_pin_count() const {
	return pin_count;
}

void ManyBoneIK3D::set_pin_target_node_path(int32_t p_pin_index, const NodePath &p_target_node) {
	ERR_FAIL_INDEX(p_pin_index, pins.size());
	Ref<IKEffectorTemplate3D> effector_template = pins[p_pin_index];
	if (effector_template.is_null()) {
		effector_template.instantiate();
		pins.write[p_pin_index] = effector_template;
	}
	effector_template->set_target_node(p_target_node);
	set_dirty();
}

NodePath ManyBoneIK3D::get_pin_target_node_path(int32_t p_pin_index) {
	ERR_FAIL_INDEX_V(p_pin_index, pins.size(), NodePath());
	const Ref<IKEffectorTemplate3D> effector_template = pins[p_pin_index];
	return effector_template->get_target_node();
}

Vector<Ref<IKEffectorTemplate3D>> ManyBoneIK3D::_get_bone_effectors() const {
	return pins;
}

void ManyBoneIK3D::_remove_pin(int32_t p_index) {
	ERR_FAIL_INDEX(p_index, pins.size());
	pins.remove_at(p_index);
	pin_count--;
	pins.resize(pin_count);
	set_dirty();
}

void ManyBoneIK3D::_update_ik_bones_transform() {
	for (int32_t bone_i = bone_list.size(); bone_i-- > 0;) {
		Ref<IKBone3D> bone = bone_list[bone_i];
		if (bone.is_null()) {
			continue;
		}
		bone->set_initial_pose(get_skeleton());
		if (bone->is_pinned()) {
			bone->get_pin()->update_target_global_transform(get_skeleton(), this);
		}
	}
}

void ManyBoneIK3D::_update_skeleton_bones_transform() {
	for (int32_t bone_i = bone_list.size(); bone_i-- > 0;) {
		Ref<IKBone3D> bone = bone_list[bone_i];
		if (bone.is_null()) {
			continue;
		}
		if (bone->get_bone_id() == -1) {
			continue;
		}
		bone->set_skeleton_bone_pose(get_skeleton());
	}
}

void ManyBoneIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
	const Vector<Ref<IKBone3D>> ik_bones = get_bone_list();
	RBSet<StringName> existing_pins;
	for (int32_t pin_i = 0; pin_i < get_pin_count(); pin_i++) {
		const String bone_name = get_pin_bone_name(pin_i);
		existing_pins.insert(bone_name);
	}
	const uint32_t pin_usage = PROPERTY_USAGE_DEFAULT;
	p_list->push_back(
			PropertyInfo(Variant::INT, "pin_count",
					PROPERTY_HINT_RANGE, "0,65536,or_greater", pin_usage | PROPERTY_USAGE_ARRAY | PROPERTY_USAGE_READ_ONLY,
					"Pins,pins/"));
	for (int pin_i = 0; pin_i < get_pin_count(); pin_i++) {
		PropertyInfo effector_name;
		effector_name.type = Variant::STRING_NAME;
		effector_name.name = "pins/" + itos(pin_i) + "/bone_name";
		effector_name.usage = pin_usage;
		if (get_skeleton()) {
			String names;
			for (int bone_i = 0; bone_i < get_skeleton()->get_bone_count(); bone_i++) {
				String name = get_skeleton()->get_bone_name(bone_i);
				StringName string_name = StringName(name);
				if (existing_pins.has(string_name)) {
					continue;
				}
				name += ",";
				names += name;
				existing_pins.insert(name);
			}
			effector_name.hint = PROPERTY_HINT_ENUM_SUGGESTION;
			effector_name.hint_string = names;
		} else {
			effector_name.hint = PROPERTY_HINT_NONE;
			effector_name.hint_string = "";
		}
		p_list->push_back(effector_name);
		p_list->push_back(
				PropertyInfo(Variant::NODE_PATH, "pins/" + itos(pin_i) + "/target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", pin_usage));
		p_list->push_back(
				PropertyInfo(Variant::BOOL, "pins/" + itos(pin_i) + "/target_static", PROPERTY_HINT_NONE, "", pin_usage));
		p_list->push_back(
				PropertyInfo(Variant::FLOAT, "pins/" + itos(pin_i) + "/motion_propagation_factor", PROPERTY_HINT_RANGE, "0,1,0.1,or_greater", pin_usage));
		p_list->push_back(
				PropertyInfo(Variant::FLOAT, "pins/" + itos(pin_i) + "/weight", PROPERTY_HINT_RANGE, "0,1,0.1,or_greater", pin_usage));
		p_list->push_back(
				PropertyInfo(Variant::VECTOR3, "pins/" + itos(pin_i) + "/direction_priorities", PROPERTY_HINT_RANGE, "0,1,0.1,or_greater", pin_usage));
	}
}

bool ManyBoneIK3D::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	if (name == "pin_count") {
		r_ret = get_pin_count();
		return true;
	} else if (name == "bone_count") {
		r_ret = get_bone_count();
		return true;
	} else if (name.begins_with("pins/")) {
		int index = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(index, pins.size(), false);
		Ref<IKEffectorTemplate3D> effector_template = pins[index];
		ERR_FAIL_COND_V(effector_template.is_null(), false);
		if (what == "bone_name") {
			r_ret = effector_template->get_name();
			return true;
		} else if (what == "target_node") {
			r_ret = effector_template->get_target_node();
			return true;
		} else if (what == "target_static") {
			r_ret = effector_template->get_target_node().is_empty();
			return true;
		} else if (what == "motion_propagation_factor") {
			r_ret = get_pin_motion_propagation_factor(index);
			return true;
		} else if (what == "weight") {
			r_ret = get_pin_weight(index);
			return true;
		} else if (what == "direction_priorities") {
			r_ret = get_pin_direction_priorities(index);
			return true;
		}
	}
	return false;
}

bool ManyBoneIK3D::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	if (name == "pin_count") {
		set_pin_count(p_value);
		return true;
	} else if (name.begins_with("pins/")) {
		int index = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);
		if (index >= pins.size()) {
			set_pin_count(constraint_count);
		}
		if (what == "bone_name") {
			set_pin_bone_name(index, p_value);
			return true;
		} else if (what == "target_node") {
			set_pin_target_node_path(index, p_value);
			return true;
		} else if (what == "target_static") {
			if (p_value) {
				set_pin_target_node_path(index, NodePath());
			}
			return true;
		} else if (what == "motion_propagation_factor") {
			set_pin_motion_propagation_factor(index, p_value);
			return true;
		} else if (what == "weight") {
			set_pin_weight(index, p_value);
			return true;
		} else if (what == "direction_priorities") {
			set_pin_direction_priorities(index, p_value);
			return true;
		}
	}
	return false;
}

void ManyBoneIK3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constraint_name_at_index", "index", "name"), &ManyBoneIK3D::set_constraint_name_at_index);
	ClassDB::bind_method(D_METHOD("set_total_effector_count", "count"), &ManyBoneIK3D::set_pin_count);
	ClassDB::bind_method(D_METHOD("remove_constraint_at_index", "index"), &ManyBoneIK3D::remove_pin_at_index);
	ClassDB::bind_method(D_METHOD("set_dirty"), &ManyBoneIK3D::set_dirty);
	ClassDB::bind_method(D_METHOD("set_pin_motion_propagation_factor", "index", "falloff"), &ManyBoneIK3D::set_pin_motion_propagation_factor);
	ClassDB::bind_method(D_METHOD("get_pin_motion_propagation_factor", "index"), &ManyBoneIK3D::get_pin_motion_propagation_factor);
	ClassDB::bind_method(D_METHOD("get_pin_count"), &ManyBoneIK3D::get_pin_count);
	ClassDB::bind_method(D_METHOD("set_pin_count", "count"), &ManyBoneIK3D::set_pin_count);

	ClassDB::bind_method(D_METHOD("get_effector_bone_name", "index"), &ManyBoneIK3D::get_pin_bone_name);
	ClassDB::bind_method(D_METHOD("get_pin_direction_priorities", "index"), &ManyBoneIK3D::get_pin_direction_priorities);
	ClassDB::bind_method(D_METHOD("set_pin_direction_priorities", "index", "priority"), &ManyBoneIK3D::set_pin_direction_priorities);
	ClassDB::bind_method(D_METHOD("get_effector_pin_node_path", "index"), &ManyBoneIK3D::get_pin_node_path);
	ClassDB::bind_method(D_METHOD("set_effector_pin_node_path", "index", "nodepath"), &ManyBoneIK3D::set_pin_node_path);
	ClassDB::bind_method(D_METHOD("set_pin_weight", "index", "weight"), &ManyBoneIK3D::set_pin_weight);
	ClassDB::bind_method(D_METHOD("get_pin_weight", "index"), &ManyBoneIK3D::get_pin_weight);
	ClassDB::bind_method(D_METHOD("get_pin_enabled", "index"), &ManyBoneIK3D::get_pin_enabled);
	ClassDB::bind_method(D_METHOD("get_iterations_per_frame"), &ManyBoneIK3D::get_iterations_per_frame);
	ClassDB::bind_method(D_METHOD("set_iterations_per_frame", "count"), &ManyBoneIK3D::set_iterations_per_frame);
	ClassDB::bind_method(D_METHOD("find_pin", "name"), &ManyBoneIK3D::find_pin);
	ClassDB::bind_method(D_METHOD("get_default_damp"), &ManyBoneIK3D::get_default_damp);
	ClassDB::bind_method(D_METHOD("set_default_damp", "damp"), &ManyBoneIK3D::set_default_damp);
	ClassDB::bind_method(D_METHOD("get_bone_count"), &ManyBoneIK3D::get_bone_count);
	ClassDB::bind_method(D_METHOD("set_stabilization_passes", "passes"), &ManyBoneIK3D::set_stabilization_passes);
	ClassDB::bind_method(D_METHOD("get_stabilization_passes"), &ManyBoneIK3D::get_stabilization_passes);
	ClassDB::bind_method(D_METHOD("set_effector_bone_name", "index", "name"), &ManyBoneIK3D::set_pin_bone_name);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "iterations_per_frame", PROPERTY_HINT_RANGE, "1,150,1,or_greater"), "set_iterations_per_frame", "get_iterations_per_frame");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_damp", PROPERTY_HINT_RANGE, "0.01,180.0,0.1,radians,exp", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_default_damp", "get_default_damp");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stabilization_passes"), "set_stabilization_passes", "get_stabilization_passes");
}

ManyBoneIK3D::ManyBoneIK3D() {
}

ManyBoneIK3D::~ManyBoneIK3D() {
}

float ManyBoneIK3D::get_pin_motion_propagation_factor(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, pins.size(), 0.0f);
	const Ref<IKEffectorTemplate3D> effector_template = pins[p_effector_index];
	return effector_template->get_motion_propagation_factor();
}

void ManyBoneIK3D::set_pin_motion_propagation_factor(int32_t p_effector_index, const float p_motion_propagation_factor) {
	ERR_FAIL_INDEX(p_effector_index, pins.size());
	Ref<IKEffectorTemplate3D> effector_template = pins[p_effector_index];
	ERR_FAIL_COND(effector_template.is_null());
	effector_template->set_motion_propagation_factor(p_motion_propagation_factor);
	set_dirty();
}

int32_t ManyBoneIK3D::find_pin_id(StringName p_bone_name) {
	for (int32_t constraint_i = 0; constraint_i < constraint_count; constraint_i++) {
		if (constraint_names[constraint_i] == p_bone_name) {
			return constraint_i;
		}
	}
	return -1;
}

real_t ManyBoneIK3D::get_default_damp() const {
	return default_damp;
}

void ManyBoneIK3D::set_default_damp(float p_default_damp) {
	default_damp = p_default_damp;
	set_dirty();
}

StringName ManyBoneIK3D::get_pin_bone_name(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, pins.size(), "");
	Ref<IKEffectorTemplate3D> effector_template = pins[p_effector_index];
	return effector_template->get_name();
}

void ManyBoneIK3D::set_constraint_name_at_index(int32_t p_index, String p_name) {
	ERR_FAIL_INDEX(p_index, constraint_names.size());
	constraint_names.write[p_index] = p_name;
	set_dirty();
}

Vector<Ref<IKBoneSegment3D>> ManyBoneIK3D::get_segmented_skeletons() {
	return segmented_skeletons;
}

float ManyBoneIK3D::get_iterations_per_frame() const {
	return iterations_per_frame;
}

void ManyBoneIK3D::set_iterations_per_frame(const float &p_iterations_per_frame) {
	iterations_per_frame = p_iterations_per_frame;
}

void ManyBoneIK3D::set_pin_node_path(int32_t p_effector_index, NodePath p_node_path) {
	ERR_FAIL_INDEX(p_effector_index, pins.size());
	Node *node = get_node_or_null(p_node_path);
	if (!node) {
		return;
	}
	Ref<IKEffectorTemplate3D> effector_template = pins[p_effector_index];
	effector_template->set_target_node(p_node_path);
}

NodePath ManyBoneIK3D::get_pin_node_path(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, pins.size(), NodePath());
	Ref<IKEffectorTemplate3D> effector_template = pins[p_effector_index];
	return effector_template->get_target_node();
}

void ManyBoneIK3D::_process_modification() {
	if (!get_skeleton()) {
		return;
	}
	if (get_pin_count() == 0) {
		return;
	}
	if (!segmented_skeletons.size()) {
		set_dirty();
	}
	if (is_dirty) {
		is_dirty = false;
		_bone_list_changed();
	}
	if (bone_list.size()) {
		Ref<IKNode3D> root_ik_bone = bone_list.write[0]->get_ik_transform();
		if (root_ik_bone.is_null()) {
			return;
		}
		Skeleton3D *skeleton = get_skeleton();
		godot_skeleton_transform.instantiate();
		godot_skeleton_transform->set_transform(skeleton->get_transform());
		godot_skeleton_transform_inverse = skeleton->get_transform().affine_inverse();
	}
	bool has_pins = false;
	for (Ref<IKEffectorTemplate3D> pin : pins) {
		if (pin.is_valid() && !pin->get_name().is_empty()) {
			has_pins = true;
			break;
		}
	}
	if (!has_pins) {
		return;
	}
	if (!is_enabled()) {
		return;
	}
	if (!is_visible()) {
		return;
	}
	for (int32_t i = 0; i < get_iterations_per_frame(); i++) {
		for (Ref<IKBoneSegment3D> segmented_skeleton : segmented_skeletons) {
			if (segmented_skeleton.is_null()) {
				continue;
			}
			segmented_skeleton->segment_solver(bone_damp, get_default_damp(), false, i, get_iterations_per_frame());
		}
	}
	_update_skeleton_bones_transform();
}

real_t ManyBoneIK3D::get_pin_weight(int32_t p_pin_index) const {
	ERR_FAIL_INDEX_V(p_pin_index, pins.size(), 0.0);
	const Ref<IKEffectorTemplate3D> effector_template = pins[p_pin_index];
	return effector_template->get_weight();
}

void ManyBoneIK3D::set_pin_weight(int32_t p_pin_index, const real_t &p_weight) {
	ERR_FAIL_INDEX(p_pin_index, pins.size());
	Ref<IKEffectorTemplate3D> effector_template = pins[p_pin_index];
	if (effector_template.is_null()) {
		effector_template.instantiate();
		pins.write[p_pin_index] = effector_template;
	}
	effector_template->set_weight(p_weight);
	set_dirty();
}

Vector3 ManyBoneIK3D::get_pin_direction_priorities(int32_t p_pin_index) const {
	ERR_FAIL_INDEX_V(p_pin_index, pins.size(), Vector3(0, 0, 0));
	const Ref<IKEffectorTemplate3D> effector_template = pins[p_pin_index];
	return effector_template->get_direction_priorities();
}

void ManyBoneIK3D::set_pin_direction_priorities(int32_t p_pin_index, const Vector3 &p_priority_direction) {
	ERR_FAIL_INDEX(p_pin_index, pins.size());
	Ref<IKEffectorTemplate3D> effector_template = pins[p_pin_index];
	if (effector_template.is_null()) {
		effector_template.instantiate();
		pins.write[p_pin_index] = effector_template;
	}
	effector_template->set_direction_priorities(p_priority_direction);
	set_dirty();
}

void ManyBoneIK3D::set_dirty() {
	is_dirty = true;
}

void ManyBoneIK3D::remove_pin_at_index(int32_t p_index) {
	ERR_FAIL_INDEX(p_index, constraint_count);

	constraint_names.remove_at(p_index);
	kusudama_open_cone_count.remove_at(p_index);
	kusudama_open_cones.remove_at(p_index);
	joint_twist.remove_at(p_index);

	constraint_count--;

	set_dirty();
}

void ManyBoneIK3D::_set_bone_count(int32_t p_count) {
	bone_damp.resize(p_count);
	for (int32_t bone_i = p_count; bone_i-- > bone_count;) {
		bone_damp.write[bone_i] = get_default_damp();
	}
	bone_count = p_count;
	set_dirty();
	notify_property_list_changed();
}

int32_t ManyBoneIK3D::get_bone_count() const {
	return bone_count;
}

Vector<Ref<IKBone3D>> ManyBoneIK3D::get_bone_list() const {
	return bone_list;
}

bool ManyBoneIK3D::get_pin_enabled(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, pins.size(), false);
	Ref<IKEffectorTemplate3D> effector_template = pins[p_effector_index];
	if (effector_template->get_target_node().is_empty()) {
		return true;
	}
	return !effector_template->get_target_node().is_empty();
}

void ManyBoneIK3D::set_stabilization_passes(int32_t p_passes) {
	stabilize_passes = p_passes;
	set_dirty();
}

int32_t ManyBoneIK3D::get_stabilization_passes() {
	return stabilize_passes;
}

Transform3D ManyBoneIK3D::get_godot_skeleton_transform_inverse() {
	return godot_skeleton_transform_inverse;
}

Ref<IKNode3D> ManyBoneIK3D::get_godot_skeleton_transform() {
	return godot_skeleton_transform;
}

int32_t ManyBoneIK3D::find_pin(String p_string) const {
	for (int32_t pin_i = 0; pin_i < pin_count; pin_i++) {
		if (get_pin_bone_name(pin_i) == p_string) {
			return pin_i;
		}
	}
	return -1;
}

bool ManyBoneIK3D::get_pin_target_fixed(int32_t p_effector_index) {
	ERR_FAIL_INDEX_V(p_effector_index, pins.size(), false);
	Ref<IKEffectorTemplate3D> effector_template = pins[p_effector_index];
	return get_pin_node_path(p_effector_index).is_empty();
}

void ManyBoneIK3D::set_pin_target_fixed(int32_t p_effector_index, bool p_force_ignore) {
	ERR_FAIL_INDEX(p_effector_index, pins.size());
	if (!p_force_ignore) {
		return;
	}
	Ref<IKEffectorTemplate3D> effector_template = pins[p_effector_index];
	effector_template->set_target_node(NodePath());
	set_dirty();
}

void ManyBoneIK3D::_bone_list_changed() {
	Skeleton3D *skeleton = get_skeleton();
	Vector<int32_t> roots = skeleton->get_parentless_bones();
	if (roots.is_empty()) {
		return;
	}
	bone_list.clear();
	segmented_skeletons.clear();
	for (BoneId root_bone_index : roots) {
		String parentless_bone = skeleton->get_bone_name(root_bone_index);
		Ref<IKBoneSegment3D> segmented_skeleton = Ref<IKBoneSegment3D>(memnew(IKBoneSegment3D(skeleton, parentless_bone, pins, this, nullptr, root_bone_index, -1, stabilize_passes)));
		ik_origin.instantiate();
		segmented_skeleton->get_root()->get_ik_transform()->set_parent(ik_origin);
		segmented_skeleton->generate_default_segments(pins, root_bone_index, -1, this);
		Vector<Ref<IKBone3D>> new_bone_list;
		segmented_skeleton->create_bone_list(new_bone_list, true);
		bone_list.append_array(new_bone_list);
		Vector<Vector<double>> weight_array;
		segmented_skeleton->update_pinned_list(weight_array);
		segmented_skeleton->recursive_create_headings_arrays_for(segmented_skeleton);
		segmented_skeletons.push_back(segmented_skeleton);
	}
	_update_ik_bones_transform();
	for (Ref<IKBone3D> &ik_bone_3d : bone_list) {
		ik_bone_3d->update_default_bone_direction_transform(skeleton);
	}
}

void ManyBoneIK3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	if (p_old) {
		if (p_old->is_connected(SNAME("bone_list_changed"), callable_mp(this, &ManyBoneIK3D::_bone_list_changed))) {
			p_old->disconnect(SNAME("bone_list_changed"), callable_mp(this, &ManyBoneIK3D::_bone_list_changed));
		}
	}
	if (p_new) {
		if (!p_new->is_connected(SNAME("bone_list_changed"), callable_mp(this, &ManyBoneIK3D::_bone_list_changed))) {
			p_new->connect(SNAME("bone_list_changed"), callable_mp(this, &ManyBoneIK3D::_bone_list_changed));
		}
	}
	if (is_connected(SNAME("modification_processed"), callable_mp(this, &ManyBoneIK3D::_update_ik_bones_transform))) {
		disconnect(SNAME("modification_processed"), callable_mp(this, &ManyBoneIK3D::_update_ik_bones_transform));
	}
	connect(SNAME("modification_processed"), callable_mp(this, &ManyBoneIK3D::_update_ik_bones_transform));
	_bone_list_changed();
}

void ManyBoneIK3D::set_pin_bone_name(int32_t p_pin_index, const String &p_bone) {
	ERR_FAIL_INDEX(p_pin_index, pins.size());
	Ref<IKEffectorTemplate3D> effector_template = pins[p_pin_index];
	if (effector_template.is_null()) {
		effector_template.instantiate();
		pins.write[p_pin_index] = effector_template;
	}
	effector_template->set_name(p_bone);
	set_dirty();
}
