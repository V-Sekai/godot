/**************************************************************************/
/*  ewbik_3d.cpp                                                          */
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
#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/string/string_name.h"
#include "ik_bone_3d.h"
#include "ik_bone_segment_3d.h"
#include "many_bone_ik_3d_state.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"

Vector<Ref<IKEffectorTemplate3D>> EWBIK3D::_get_bone_effectors() const {
	// Convert settings to effector templates for internal use
	Vector<Ref<IKEffectorTemplate3D>> effectors;
	for (EWBIK3DSetting *setting : ewbik_settings) {
		if (setting) {
			Ref<IKEffectorTemplate3D> effector;
			effector.instantiate();
			effector->set_name(setting->bone_name);
			effector->set_target_node(setting->target_node);
			effector->set_weight(setting->weight);
			effector->set_direction_priorities(setting->direction_priorities);
			effector->set_motion_propagation_factor(setting->motion_propagation_factor);
			effectors.push_back(effector);
		}
	}
	return effectors;
}

void EWBIK3D::_update_ik_bones_transform() {
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

void EWBIK3D::_update_skeleton_bones_transform() {
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
	update_gizmos();
}

void EWBIK3D::_get_property_list(List<PropertyInfo> *p_list) const {
	// Call parent to get base properties
	ManyBoneIK3D::_get_property_list(p_list);

	// Add settings properties
	RBSet<StringName> existing_bones;
	for (uint32_t i = 0; i < ewbik_settings.size(); i++) {
		if (ewbik_settings[i] && !ewbik_settings[i]->bone_name.is_empty()) {
			existing_bones.insert(ewbik_settings[i]->bone_name);
		}
	}

	const uint32_t setting_usage = PROPERTY_USAGE_DEFAULT;

	for (uint32_t i = 0; i < ewbik_settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";

		PropertyInfo bone_name_prop;
		bone_name_prop.type = Variant::STRING_NAME;
		bone_name_prop.name = path + "bone_name";
		bone_name_prop.usage = setting_usage;
		if (get_skeleton()) {
			String names;
			for (int bone_i = 0; bone_i < get_skeleton()->get_bone_count(); bone_i++) {
				String name = get_skeleton()->get_bone_name(bone_i);
				StringName string_name = StringName(name);
				if (existing_bones.has(string_name)) {
					continue;
				}
				name += ",";
				names += name;
				existing_bones.insert(name);
			}
			bone_name_prop.hint = PROPERTY_HINT_ENUM_SUGGESTION;
			bone_name_prop.hint_string = names;
		} else {
			bone_name_prop.hint = PROPERTY_HINT_NONE;
			bone_name_prop.hint_string = "";
		}
		p_list->push_back(bone_name_prop);

		p_list->push_back(
				PropertyInfo(Variant::NODE_PATH, path + "target_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node3D", setting_usage));
		p_list->push_back(
				PropertyInfo(Variant::FLOAT, path + "weight", PROPERTY_HINT_RANGE, "0,1,0.1,or_greater", setting_usage));
		p_list->push_back(
				PropertyInfo(Variant::VECTOR3, path + "direction_priorities", PROPERTY_HINT_RANGE, "0,1,0.1,or_greater", setting_usage));
		p_list->push_back(
				PropertyInfo(Variant::FLOAT, path + "motion_propagation_factor", PROPERTY_HINT_RANGE, "0,1,0.1,or_greater", setting_usage));
	}
}

bool EWBIK3D::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;

	// Handle settings properties
	if (name.begins_with("settings/")) {
		int index = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);

		if (index >= (int32_t)ewbik_settings.size()) {
			return false;
		}

		EWBIK3DSetting *setting = ewbik_settings[index];
		if (!setting) {
			return false;
		}

		if (what == "bone_name") {
			r_ret = setting->bone_name;
			return true;
		} else if (what == "target_node") {
			r_ret = setting->target_node;
			return true;
		} else if (what == "weight") {
			r_ret = setting->weight;
			return true;
		} else if (what == "direction_priorities") {
			r_ret = setting->direction_priorities;
			return true;
		} else if (what == "motion_propagation_factor") {
			r_ret = setting->motion_propagation_factor;
			return true;
		}
	}

	return ManyBoneIK3D::_get(p_name, r_ret);
}

bool EWBIK3D::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;

	// Handle settings properties
	if (name.begins_with("settings/")) {
		int index = name.get_slicec('/', 1).to_int();
		String what = name.get_slicec('/', 2);

		if (index >= (int32_t)ewbik_settings.size()) {
			return false;
		}

		EWBIK3DSetting *setting = ewbik_settings[index];
		if (!setting) {
			return false;
		}

		if (what == "bone_name") {
			setting->bone_name = p_value;
			set_dirty();
			return true;
		} else if (what == "target_node") {
			setting->target_node = p_value;
			set_dirty();
			return true;
		} else if (what == "weight") {
			setting->weight = p_value;
			set_dirty();
			return true;
		} else if (what == "direction_priorities") {
			setting->direction_priorities = p_value;
			set_dirty();
			return true;
		} else if (what == "motion_propagation_factor") {
			setting->motion_propagation_factor = p_value;
			set_dirty();
			return true;
		}
	}

	return ManyBoneIK3D::_set(p_name, p_value);
}

void EWBIK3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_total_effector_count", "count"), &EWBIK3D::set_pin_count);
	ClassDB::bind_method(D_METHOD("register_skeleton"), &EWBIK3D::register_skeleton);
	ClassDB::bind_method(D_METHOD("set_dirty"), &EWBIK3D::set_dirty);
	ClassDB::bind_method(D_METHOD("set_pin_motion_propagation_factor", "index", "falloff"), &EWBIK3D::set_pin_motion_propagation_factor);
	ClassDB::bind_method(D_METHOD("get_pin_motion_propagation_factor", "index"), &EWBIK3D::get_pin_motion_propagation_factor);
	ClassDB::bind_method(D_METHOD("get_pin_count"), &EWBIK3D::get_pin_count);
	ClassDB::bind_method(D_METHOD("set_pin_count", "count"), &EWBIK3D::set_pin_count);

	ClassDB::bind_method(D_METHOD("get_effector_bone_name", "index"), &EWBIK3D::get_pin_bone_name);
	ClassDB::bind_method(D_METHOD("get_pin_direction_priorities", "index"), &EWBIK3D::get_pin_direction_priorities);
	ClassDB::bind_method(D_METHOD("set_pin_direction_priorities", "index", "priority"), &EWBIK3D::set_pin_direction_priorities);
	ClassDB::bind_method(D_METHOD("get_effector_pin_node_path", "index"), &EWBIK3D::get_pin_node_path);
	ClassDB::bind_method(D_METHOD("set_effector_pin_node_path", "index", "nodepath"), &EWBIK3D::set_pin_node_path);
	ClassDB::bind_method(D_METHOD("set_pin_weight", "index", "weight"), &EWBIK3D::set_pin_weight);
	ClassDB::bind_method(D_METHOD("get_pin_weight", "index"), &EWBIK3D::get_pin_weight);
	ClassDB::bind_method(D_METHOD("get_pin_enabled", "index"), &EWBIK3D::get_pin_enabled);
	ClassDB::bind_method(D_METHOD("get_iterations_per_frame"), &EWBIK3D::get_iterations_per_frame);
	ClassDB::bind_method(D_METHOD("set_iterations_per_frame", "count"), &EWBIK3D::set_iterations_per_frame);
	ClassDB::bind_method(D_METHOD("find_pin", "name"), &EWBIK3D::find_pin);
	ClassDB::bind_method(D_METHOD("get_default_damp"), &EWBIK3D::get_default_damp);
	ClassDB::bind_method(D_METHOD("set_default_damp", "damp"), &EWBIK3D::set_default_damp);
	ClassDB::bind_method(D_METHOD("get_bone_count"), &EWBIK3D::get_bone_count);
	ClassDB::bind_method(D_METHOD("set_stabilization_passes", "passes"), &EWBIK3D::set_stabilization_passes);
	ClassDB::bind_method(D_METHOD("get_stabilization_passes"), &EWBIK3D::get_stabilization_passes);
	ClassDB::bind_method(D_METHOD("set_effector_bone_name", "index", "name"), &EWBIK3D::set_pin_bone_name);

	// IterateIK3D compatible methods
	ClassDB::bind_method(D_METHOD("set_angular_delta_limit", "limit"), &EWBIK3D::set_angular_delta_limit);
	ClassDB::bind_method(D_METHOD("get_angular_delta_limit"), &EWBIK3D::get_angular_delta_limit);
	ClassDB::bind_method(D_METHOD("set_max_iterations", "iterations"), &EWBIK3D::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations"), &EWBIK3D::get_max_iterations);
	ClassDB::bind_method(D_METHOD("set_min_distance", "distance"), &EWBIK3D::set_min_distance);
	ClassDB::bind_method(D_METHOD("get_min_distance"), &EWBIK3D::get_min_distance);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "iterations_per_frame", PROPERTY_HINT_RANGE, "1,150,1,or_greater"), "set_iterations_per_frame", "get_iterations_per_frame");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_damp", PROPERTY_HINT_RANGE, "0.01,180.0,0.1,radians,exp", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_default_damp", "get_default_damp");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stabilization_passes"), "set_stabilization_passes", "get_stabilization_passes");

	// IterateIK3D compatible properties
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_delta_limit", PROPERTY_HINT_RANGE, "0.001,3.14159,0.001,radians"), "set_angular_delta_limit", "get_angular_delta_limit");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_iterations", PROPERTY_HINT_RANGE, "1,100,1"), "set_max_iterations", "get_max_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_distance", PROPERTY_HINT_RANGE, "0.0001,1.0,0.0001"), "set_min_distance", "get_min_distance");
}

Ref<ManyBoneIK3DState> EWBIK3D::get_state() const {
	Ref<ManyBoneIK3DState> state;
	state.instantiate();

	state->set_iterations_per_frame(get_iterations_per_frame());
	state->set_default_damp(get_default_damp());
	state->set_stabilization_passes(get_stabilization_passes());

	// Copy effector templates
	Vector<Ref<IKEffectorTemplate3D>> effector_templates;
	for (int32_t i = 0; i < get_pin_count(); ++i) {
		Ref<IKEffectorTemplate3D> template_ref;
		template_ref.instantiate();
		template_ref->set_name(get_pin_bone_name(i));
		template_ref->set_target_node(get_pin_target_node_path(i));
		template_ref->set_weight(get_pin_weight(i));
		template_ref->set_direction_priorities(get_pin_direction_priorities(i));
		template_ref->set_motion_propagation_factor(get_pin_motion_propagation_factor(i));
		effector_templates.push_back(template_ref);
	}
	state->set_effector_templates(effector_templates);

	// Copy original bone poses if available
	if (get_skeleton()) {
		Vector<Transform3D> original_poses;
		for (int32_t i = 0; i < get_skeleton()->get_bone_count(); ++i) {
			original_poses.push_back(get_skeleton()->get_bone_pose(i));
		}
		state->set_original_bone_poses(original_poses);
	}

	return state;
}

void EWBIK3D::set_state(Ref<ManyBoneIK3DState> p_state) {
	ERR_FAIL_COND(p_state.is_null());

	set_iterations_per_frame(p_state->get_iterations_per_frame());
	set_default_damp(p_state->get_default_damp());
	set_stabilization_passes(p_state->get_stabilization_passes());

	// Copy effector templates
	const Vector<Ref<IKEffectorTemplate3D>> &effector_templates = p_state->get_effector_templates();
	set_pin_count(effector_templates.size());
	for (int32_t i = 0; i < effector_templates.size(); ++i) {
		const Ref<IKEffectorTemplate3D> &template_ref = effector_templates[i];
		if (template_ref.is_valid()) {
			set_pin_bone_name(i, template_ref->get_name());
			set_pin_target_node_path(i, template_ref->get_target_node());
			set_pin_weight(i, template_ref->get_weight());
			set_pin_direction_priorities(i, template_ref->get_direction_priorities());
			set_pin_motion_propagation_factor(i, template_ref->get_motion_propagation_factor());
		}
	}

	set_dirty();
}

EWBIK3D::EWBIK3D() {
	// Initialize with no settings initially
	set_setting_count(0);
}

int32_t EWBIK3D::get_pin_count() const {
	return pin_count;
}

void EWBIK3D::set_pin_count(int32_t p_pin_count) {
	pin_count = p_pin_count;
	set_setting_count(p_pin_count);
	set_dirty();
}

void EWBIK3D::add_segment(Ref<IKBoneSegment3D> p_segment) {
	if (p_segment.is_valid()) {
		segmented_skeletons.push_back(p_segment);
	}
}

// IterateIK3D compatible parameter implementations
void EWBIK3D::set_angular_delta_limit(float p_limit) {
	angular_delta_limit = p_limit;
	set_dirty();
}

float EWBIK3D::get_angular_delta_limit() const {
	return angular_delta_limit;
}

void EWBIK3D::set_max_iterations(int32_t p_iterations) {
	max_iterations = p_iterations;
	set_dirty();
}

int32_t EWBIK3D::get_max_iterations() const {
	return max_iterations;
}

void EWBIK3D::set_min_distance(float p_distance) {
	min_distance = p_distance;
	set_dirty();
}

float EWBIK3D::get_min_distance() const {
	return min_distance;
}

StringName EWBIK3D::get_pin_bone_name(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, (int32_t)ewbik_settings.size(), StringName());
	EWBIK3DSetting *setting = ewbik_settings[p_effector_index];
	if (!setting) {
		return StringName();
	}
	return setting->bone_name;
}

real_t EWBIK3D::get_default_damp() const {
	return default_damp;
}

void EWBIK3D::set_default_damp(float p_default_damp) {
	default_damp = p_default_damp;
	set_dirty();
}

void EWBIK3D::cleanup() {
	// Force immediate cleanup of all resources
	set_active(false); // Stop any processing

	// Clear all collections to break cycles
	bone_list.clear();
	segmented_skeletons.clear();

	is_dirty = true;
}

EWBIK3D::~EWBIK3D() {
	// Clear all collections - Ref<> objects handle their own cleanup automatically
	segmented_skeletons.clear();
	bone_list.clear();

	// Clear transform references - Ref<> will handle cleanup automatically
	godot_skeleton_transform.unref();
	ik_origin.unref();
}

float EWBIK3D::get_pin_motion_propagation_factor(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, (int32_t)ewbik_settings.size(), 0.0f);
	EWBIK3DSetting *setting = ewbik_settings[p_effector_index];
	if (!setting) {
		return 0.0f;
	}
	return setting->motion_propagation_factor;
}

void EWBIK3D::set_pin_motion_propagation_factor(int32_t p_effector_index, const float p_motion_propagation_factor) {
	ERR_FAIL_INDEX(p_effector_index, (int32_t)ewbik_settings.size());
	EWBIK3DSetting *setting = ewbik_settings[p_effector_index];
	if (setting) {
		setting->motion_propagation_factor = p_motion_propagation_factor;
		set_dirty();
	}
}



TypedArray<IKBoneSegment3D> EWBIK3D::get_segmented_skeletons() {
	TypedArray<IKBoneSegment3D> result;
	for (const Ref<IKBoneSegment3D> &segment : segmented_skeletons) {
		result.push_back(segment);
	}
	return result;
}

float EWBIK3D::get_iterations_per_frame() const {
	return iterations_per_frame;
}

void EWBIK3D::set_iterations_per_frame(const float &p_iterations_per_frame) {
	iterations_per_frame = p_iterations_per_frame;
}

void EWBIK3D::set_pin_node_path(int32_t p_effector_index, NodePath p_node_path) {
	ERR_FAIL_INDEX(p_effector_index, (int32_t)ewbik_settings.size());
	EWBIK3DSetting *setting = ewbik_settings[p_effector_index];
	if (setting) {
		setting->target_node = p_node_path;
		set_dirty();
	}
}

NodePath EWBIK3D::get_pin_node_path(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, (int32_t)ewbik_settings.size(), NodePath());
	EWBIK3DSetting *setting = ewbik_settings[p_effector_index];
	if (!setting) {
		return NodePath();
	}
	return setting->target_node;
}

NodePath EWBIK3D::get_pin_target_node_path(int32_t p_pin_index) const {
	return get_pin_node_path(p_pin_index);
}

void EWBIK3D::set_pin_target_node_path(int32_t p_pin_index, const NodePath &p_target_node) {
	set_pin_node_path(p_pin_index, p_target_node);
}

void EWBIK3D::_process_modification(double p_delta) {
	if (!get_skeleton()) {
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
			segmented_skeleton->segment_solver(get_default_damp(), i, get_iterations_per_frame());
		}
	}
	_update_skeleton_bones_transform();

	// Apply standard Godot bone constraints after IK solving
	_apply_bone_constraints();
}

real_t EWBIK3D::get_pin_weight(int32_t p_pin_index) const {
	ERR_FAIL_INDEX_V(p_pin_index, (int32_t)ewbik_settings.size(), 0.0);
	EWBIK3DSetting *setting = ewbik_settings[p_pin_index];
	if (!setting) {
		return 0.0;
	}
	return setting->weight;
}

void EWBIK3D::set_pin_weight(int32_t p_pin_index, const real_t &p_weight) {
	ERR_FAIL_INDEX(p_pin_index, (int32_t)ewbik_settings.size());
	EWBIK3DSetting *setting = ewbik_settings[p_pin_index];
	if (setting) {
		setting->weight = p_weight;
		set_dirty();
	}
}

Vector3 EWBIK3D::get_pin_direction_priorities(int32_t p_pin_index) const {
	ERR_FAIL_INDEX_V(p_pin_index, (int32_t)ewbik_settings.size(), Vector3(0, 0, 0));
	EWBIK3DSetting *setting = ewbik_settings[p_pin_index];
	if (!setting) {
		return Vector3(0, 0, 0);
	}
	return setting->direction_priorities;
}

void EWBIK3D::set_pin_direction_priorities(int32_t p_pin_index, const Vector3 &p_priority_direction) {
	ERR_FAIL_INDEX(p_pin_index, (int32_t)ewbik_settings.size());
	EWBIK3DSetting *setting = ewbik_settings[p_pin_index];
	if (setting) {
		setting->direction_priorities = p_priority_direction;
		set_dirty();
	}
}

void EWBIK3D::set_dirty() {
	is_dirty = true;
}

int32_t EWBIK3D::get_bone_count() const {
	return bone_count;
}

TypedArray<IKBone3D> EWBIK3D::get_bone_list() const {
	TypedArray<IKBone3D> result;
	for (const Ref<IKBone3D> &bone : bone_list) {
		result.push_back(bone);
	}
	return result;
}

bool EWBIK3D::get_pin_enabled(int32_t p_effector_index) const {
	ERR_FAIL_INDEX_V(p_effector_index, (int32_t)ewbik_settings.size(), false);
	EWBIK3DSetting *setting = ewbik_settings[p_effector_index];
	if (!setting) {
		return false;
	}
	return !setting->target_node.is_empty();
}

void EWBIK3D::register_skeleton() {
	set_dirty();
}

void EWBIK3D::set_stabilization_passes(int32_t p_passes) {
	stabilize_passes = p_passes;
	set_dirty();
}

int32_t EWBIK3D::get_stabilization_passes() const {
	return stabilize_passes;
}

Transform3D EWBIK3D::get_godot_skeleton_transform_inverse() {
	return godot_skeleton_transform_inverse;
}

Ref<IKNode3D> EWBIK3D::get_godot_skeleton_transform() {
	return godot_skeleton_transform;
}



int32_t EWBIK3D::find_pin(String p_string) const {
	for (int32_t i = 0; i < (int32_t)ewbik_settings.size(); i++) {
		if (ewbik_settings[i] && ewbik_settings[i]->bone_name == p_string) {
			return i;
		}
	}
	return -1;
}

void EWBIK3D::_bone_list_changed() {
	Skeleton3D *skeleton = get_skeleton();
	Vector<int32_t> roots = skeleton->get_parentless_bones();
	if (roots.is_empty()) {
		return;
	}
	bone_list.clear();
	segmented_skeletons.clear();

	// Get effectors from settings
	Vector<Ref<IKEffectorTemplate3D>> effectors = _get_bone_effectors();

	// Create ik_origin once outside the loop
	if (ik_origin.is_null()) {
		ik_origin.instantiate();
	}

	for (BoneId root_bone_index : roots) {
		String parentless_bone = skeleton->get_bone_name(root_bone_index);
		Ref<IKBoneSegment3D> segmented_skeleton = Ref<IKBoneSegment3D>(memnew(IKBoneSegment3D(skeleton, parentless_bone, effectors, this, nullptr, root_bone_index, -1, stabilize_passes)));
		segmented_skeleton->get_root()->get_ik_transform()->set_parent(ik_origin);
		segmented_skeleton->generate_default_segments(effectors, root_bone_index, -1, this);
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

void EWBIK3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
	// Call base class method first
	ManyBoneIK3D::_skeleton_changed(p_old, p_new);

	// EWBIK-specific skeleton change handling
	if (p_old) {
		if (p_old->is_connected(SNAME("bone_list_changed"), callable_mp(this, &EWBIK3D::_bone_list_changed))) {
			p_old->disconnect(SNAME("bone_list_changed"), callable_mp(this, &EWBIK3D::_bone_list_changed));
		}
	}
	if (p_new) {
		if (!p_new->is_connected(SNAME("bone_list_changed"), callable_mp(this, &EWBIK3D::_bone_list_changed))) {
			p_new->connect(SNAME("bone_list_changed"), callable_mp(this, &EWBIK3D::_bone_list_changed));
		}
	}
	if (is_connected(SNAME("modification_processed"), callable_mp(this, &EWBIK3D::_update_ik_bones_transform))) {
		disconnect(SNAME("modification_processed"), callable_mp(this, &EWBIK3D::_update_ik_bones_transform));
	}
	connect(SNAME("modification_processed"), callable_mp(this, &EWBIK3D::_update_ik_bones_transform));
	_bone_list_changed();
}

void EWBIK3D::set_pin_bone_name(int32_t p_pin_index, const String &p_bone) {
	ERR_FAIL_INDEX(p_pin_index, (int32_t)ewbik_settings.size());
	EWBIK3DSetting *setting = ewbik_settings[p_pin_index];
	if (setting) {
		setting->bone_name = StringName(p_bone);
		set_dirty();
	}
}

void EWBIK3D::_apply_bone_constraints() {
	Skeleton3D *skeleton = get_skeleton();
	if (!skeleton) {
		return;
	}

	// Find and apply all BoneConstraint3D nodes that are children of the skeleton
	for (int i = 0; i < skeleton->get_child_count(); i++) {
		BoneConstraint3D *constraint = Object::cast_to<BoneConstraint3D>(skeleton->get_child(i));
		if (constraint && constraint->is_active()) {
			// Process the constraint with a small delta (constraints are typically processed per-frame)
			constraint->process_modification(0.0);
		}
	}
}
