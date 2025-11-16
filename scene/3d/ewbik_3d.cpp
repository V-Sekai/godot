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

#include "scene/3d/ewbik_3d.h"

// Include these before class_db.h to ensure types are fully defined for method binding
#include "scene/3d/ik_bone_3d.h"
#include "scene/3d/ik_bone_segment_3d.h"

#include "core/error/error_macros.h"
#include "core/math/math_defs.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/string/string_name.h"
#include "core/variant/type_info.h"
#include "scene/3d/ik_kusudama_3d.h"
#include "scene/3d/ik_open_cone_3d.h"
#include "scene/3d/iterate_ik_3d.h"
#include "scene/3d/marker_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/3d/joint_limitation_kusudama_3d.h"

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
	// Use IterateIK3D's property system for settings
	IterateIK3D::_get_property_list(p_list);

	// Add EWBIK-specific properties
	p_list->push_back(PropertyInfo(Variant::INT, "iterations_per_frame", PROPERTY_HINT_RANGE, "1,150,1,or_greater"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "default_damp", PROPERTY_HINT_RANGE, "0.01,180.0,0.1,radians,exp", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED));
	p_list->push_back(PropertyInfo(Variant::INT, "stabilization_passes", PROPERTY_HINT_RANGE, "0,100,1"));
	p_list->push_back(PropertyInfo(Variant::INT, "ui_selected_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR));
}

bool EWBIK3D::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	
	// Handle EWBIK-specific properties
	if (name == "iterations_per_frame") {
		r_ret = iterations_per_frame;
		return true;
	}
	if (name == "default_damp") {
		r_ret = default_damp;
		return true;
	}
	if (name == "stabilization_passes") {
		r_ret = stabilize_passes;
		return true;
	}
	if (name == "ui_selected_bone") {
		r_ret = ui_selected_bone;
		return true;
	}
	if (name == "bone_count") {
		r_ret = get_bone_count();
		return true;
	}
	
	// Handle settings-based properties directly (avoid delegation)
	if (!name.begins_with("settings/")) {
		return false;
	}
	
	String path = name;
	
	// Guard: Check if settings array is empty
	if (settings.is_empty()) {
		return false;
	}
	
	int which = path.get_slicec('/', 1).to_int();
	String what = path.get_slicec('/', 2);
	
	// Guard: Check if setting index is valid
	if (which >= (int)settings.size()) {
		return false;
	}
	
	// Handle ChainIK3D properties
	if (what == "root_bone_name") {
		r_ret = get_root_bone_name(which);
		return true;
	}
	if (what == "root_bone") {
		r_ret = get_root_bone(which);
		return true;
	}
	if (what == "end_bone_name") {
		r_ret = get_end_bone_name(which);
		return true;
	}
	if (what == "end_bone") {
		String opt = path.get_slicec('/', 3);
		if (opt.is_empty()) {
			r_ret = get_end_bone(which);
			return true;
		}
		if (opt == "direction") {
			r_ret = (int)get_end_bone_direction(which);
			return true;
		}
		if (opt == "length") {
			r_ret = get_end_bone_length(which);
			return true;
		}
		return false;
	}
	if (what == "extend_end_bone") {
		r_ret = is_end_bone_extended(which);
		return true;
	}
	if (what == "joint_count") {
		r_ret = get_joint_count(which);
		return true;
	}
	if (what == "target_node") {
		r_ret = get_target_node(which);
		return true;
	}
	if (what == "motion_propagation_factor") {
		r_ret = get_motion_propagation_factor(which);
		return true;
	}
	if (what == "weight") {
		r_ret = get_weight(which);
		return true;
	}
	if (what == "direction_priorities") {
		r_ret = get_direction_priorities(which);
		return true;
	}
	if (what != "joints") {
		return false;
	}
	
	// Handle joints
	int idx = path.get_slicec('/', 3).to_int();
	String prop = path.get_slicec('/', 4);
	
	// Guard: Check if joint_settings is valid
	const IterateIK3DSetting *setting = static_cast<const IterateIK3DSetting *>(settings[which]);
	if (!setting || (uint32_t)idx >= setting->joint_settings.size()) {
		return false;
	}
	
	// Handle ChainIK3D joint properties
	if (prop == "bone_name") {
		r_ret = get_joint_bone_name(which, idx);
		return true;
	}
	if (prop == "bone") {
		r_ret = get_joint_bone(which, idx);
		return true;
	}
	
	// Handle IterateIK3D joint properties
	if (prop == "rotation_axis") {
		r_ret = (int)get_joint_rotation_axis(which, idx);
		return true;
	}
	if (prop == "rotation_axis_vector") {
		r_ret = get_joint_rotation_axis_vector(which, idx);
		return true;
	}
	if (prop != "limitation") {
		return false;
	}
	
	// Handle limitation properties
	String opt = path.get_slicec('/', 5);
	if (opt.is_empty()) {
		r_ret = get_joint_limitation(which, idx);
		return true;
	}
	if (opt == "right_axis") {
		r_ret = get_joint_limitation_right_axis(which, idx);
		return true;
	}
	if (opt == "right_axis_vector") {
		r_ret = get_joint_limitation_right_axis_vector(which, idx);
		return true;
	}
	if (opt == "rotation_offset") {
		r_ret = get_joint_limitation_rotation_offset(which, idx);
		return true;
	}
	
	return false;
}

bool EWBIK3D::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	
	// Handle EWBIK-specific properties
	if (name == "iterations_per_frame") {
		iterations_per_frame = p_value;
		return true;
	}
	if (name == "default_damp") {
		default_damp = p_value;
		set_dirty();
		return true;
	}
	if (name == "stabilization_passes") {
		stabilize_passes = p_value;
		set_dirty();
		return true;
	}
	if (name == "ui_selected_bone") {
		ui_selected_bone = p_value;
		return true;
	}
	
	// Handle settings-based properties directly (avoid delegation)
	if (!name.begins_with("settings/")) {
		return false;
	}
	
	String path = name;
	
	// Guard: Check if settings array is empty
	if (settings.is_empty()) {
		return false;
	}
	
	int which = path.get_slicec('/', 1).to_int();
	String what = path.get_slicec('/', 2);
	
	// Guard: Check if setting index is valid
	if (which >= (int)settings.size()) {
		return false;
	}
	
	// Handle ChainIK3D properties
	if (what == "root_bone_name") {
		set_root_bone_name(which, p_value);
		return true;
	}
	if (what == "root_bone") {
		set_root_bone(which, p_value);
		return true;
	}
	if (what == "end_bone_name") {
		set_end_bone_name(which, p_value);
		return true;
	}
	if (what == "end_bone") {
		String opt = path.get_slicec('/', 3);
		if (opt.is_empty()) {
			set_end_bone(which, p_value);
			return true;
		}
		if (opt == "direction") {
			set_end_bone_direction(which, static_cast<BoneDirection>((int)p_value));
			return true;
		}
		if (opt == "length") {
			set_end_bone_length(which, p_value);
			return true;
		}
		return false;
	}
	if (what == "extend_end_bone") {
		set_extend_end_bone(which, p_value);
		return true;
	}
	if (what == "joint_count") {
		set_joint_count(which, p_value);
		return true;
	}
	if (what == "target_node") {
		set_target_node(which, p_value);
		return true;
	}
	if (what == "motion_propagation_factor") {
		set_motion_propagation_factor(which, p_value);
		return true;
	}
	if (what == "weight") {
		set_weight(which, p_value);
		return true;
	}
	if (what == "direction_priorities") {
		set_direction_priorities(which, p_value);
		return true;
	}
	if (what != "joints") {
		return false;
	}
	
	// Handle joints
	int idx = path.get_slicec('/', 3).to_int();
	String prop = path.get_slicec('/', 4);
	
	// Guard: Check if joint_settings is valid
	IterateIK3DSetting *setting = static_cast<IterateIK3DSetting *>(settings[which]);
	if (!setting || (uint32_t)idx >= setting->joint_settings.size()) {
		return false;
	}
	
	// Handle ChainIK3D joint properties
	if (prop == "bone_name") {
		set_joint_bone_name(which, idx, p_value);
		return true;
	}
	if (prop == "bone") {
		set_joint_bone(which, idx, p_value);
		return true;
	}
	
	// Handle IterateIK3D joint properties
	if (prop == "rotation_axis") {
		set_joint_rotation_axis(which, idx, static_cast<RotationAxis>((int)p_value));
		return true;
	}
	if (prop == "rotation_axis_vector") {
		set_joint_rotation_axis_vector(which, idx, p_value);
		return true;
	}
	if (prop != "limitation") {
		return false;
	}
	
	// Handle limitation properties
	String opt = path.get_slicec('/', 5);
	if (opt.is_empty()) {
		set_joint_limitation(which, idx, p_value);
		return true;
	}
	if (opt == "right_axis") {
		set_joint_limitation_right_axis(which, idx, p_value);
		return true;
	}
	if (opt == "right_axis_vector") {
		set_joint_limitation_right_axis_vector(which, idx, p_value);
		return true;
	}
	if (opt == "rotation_offset") {
		set_joint_limitation_rotation_offset(which, idx, p_value);
		return true;
	}
	
	return false;
}

void EWBIK3D::_bind_methods() {
	// EWBIK-specific methods
	ClassDB::bind_method(D_METHOD("register_skeleton"), &EWBIK3D::register_skeleton);
	ClassDB::bind_method(D_METHOD("set_dirty"), &EWBIK3D::set_dirty);
	ClassDB::bind_method(D_METHOD("get_iterations_per_frame"), &EWBIK3D::get_iterations_per_frame);
	ClassDB::bind_method(D_METHOD("set_iterations_per_frame", "count"), &EWBIK3D::set_iterations_per_frame);
	ClassDB::bind_method(D_METHOD("get_default_damp"), &EWBIK3D::get_default_damp);
	ClassDB::bind_method(D_METHOD("set_default_damp", "damp"), &EWBIK3D::set_default_damp);
	ClassDB::bind_method(D_METHOD("get_bone_count"), &EWBIK3D::get_bone_count);
	ClassDB::bind_method(D_METHOD("set_ui_selected_bone", "bone"), &EWBIK3D::set_ui_selected_bone);
	ClassDB::bind_method(D_METHOD("get_ui_selected_bone"), &EWBIK3D::get_ui_selected_bone);
	ClassDB::bind_method(D_METHOD("set_stabilization_passes", "passes"), &EWBIK3D::set_stabilization_passes);
	ClassDB::bind_method(D_METHOD("get_stabilization_passes"), &EWBIK3D::get_stabilization_passes);
	ClassDB::bind_method(D_METHOD("get_bone_list"), &EWBIK3D::get_bone_list);
	ClassDB::bind_method(D_METHOD("get_segmented_skeletons"), &EWBIK3D::get_segmented_skeletons);
	ClassDB::bind_method(D_METHOD("queue_print_skeleton"), &EWBIK3D::queue_print_skeleton);
	ClassDB::bind_method(D_METHOD("cleanup"), &EWBIK3D::cleanup);
	ClassDB::bind_method(D_METHOD("get_godot_skeleton_transform"), &EWBIK3D::get_godot_skeleton_transform);
	ClassDB::bind_method(D_METHOD("get_godot_skeleton_transform_inverse"), &EWBIK3D::get_godot_skeleton_transform_inverse);
	ClassDB::bind_method(D_METHOD("set_state", "state"), &EWBIK3D::set_state);
	ClassDB::bind_method(D_METHOD("get_state"), &EWBIK3D::get_state);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "iterations_per_frame", PROPERTY_HINT_RANGE, "1,150,1,or_greater"), "set_iterations_per_frame", "get_iterations_per_frame");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "default_damp", PROPERTY_HINT_RANGE, "0.01,180.0,0.1,radians,exp", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_default_damp", "get_default_damp");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ui_selected_bone", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_ui_selected_bone", "get_ui_selected_bone");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "stabilization_passes"), "set_stabilization_passes", "get_stabilization_passes");
}

EWBIK3D::EWBIK3D() :
		IterateIK3D() {
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

real_t EWBIK3D::get_default_damp() const {
	return default_damp;
}

void EWBIK3D::set_default_damp(float p_default_damp) {
	default_damp = p_default_damp;
	set_dirty();
}

TypedArray<IKBoneSegment3D> EWBIK3D::get_segmented_skeletons() {
	TypedArray<IKBoneSegment3D> result;
	result.resize(segmented_skeletons.size());
	for (int i = 0; i < segmented_skeletons.size(); i++) {
		result[i] = segmented_skeletons[i];
	}
	return result;
}

float EWBIK3D::get_iterations_per_frame() const {
	return iterations_per_frame;
}

void EWBIK3D::set_iterations_per_frame(const float &p_iterations_per_frame) {
	iterations_per_frame = p_iterations_per_frame;
}

void EWBIK3D::_build_internal_data_from_settings(Skeleton3D *p_skeleton) {
	// Build internal pins from settings
	internal_pins.clear();
	internal_constraint_names.clear();
	internal_joint_twist.clear();
	internal_kusudama_open_cones.clear();
	internal_kusudama_open_cone_count.clear();

	for (uint32_t i = 0; i < iterate_settings.size(); i++) {
		IterateIK3DSetting *setting = iterate_settings[i];
		if (!setting || setting->target_node.is_empty()) {
			continue;
		}
		Ref<IKEffectorTemplate3D> pin;
		pin.instantiate();
		pin->set_target_node(setting->target_node);
		// Map end_bone to pin bone name
		if (setting->end_bone.bone >= 0) {
			pin->set_name(p_skeleton->get_bone_name(setting->end_bone.bone));
		}
		// Copy EWBIK-compatible properties from settings to pin
		pin->set_motion_propagation_factor(setting->motion_propagation_factor);
		pin->set_weight(setting->weight);
		pin->set_direction_priorities(setting->direction_priorities);
		internal_pins.push_back(pin);

		// Build constraints from joint_settings
		// Map joint limitations to internal constraint data structures
		for (uint32_t joint_i = 0; joint_i < setting->joint_settings.size(); joint_i++) {
			IterateIK3DJointSetting *joint_setting = setting->joint_settings[joint_i];
			if (!joint_setting || joint_i >= setting->joints.size()) {
				continue;
			}
			int bone_id = setting->joints[joint_i].bone;
			if (bone_id < 0) {
				continue;
			}
			StringName bone_name = p_skeleton->get_bone_name(bone_id);

			// Check if we already have a constraint for this bone
			int constraint_index = -1;
			for (int j = 0; j < internal_constraint_names.size(); j++) {
				if (internal_constraint_names[j] == bone_name) {
					constraint_index = j;
					break;
				}
			}

			if (constraint_index < 0) {
				// Create new constraint entry
				constraint_index = internal_constraint_names.size();
				internal_constraint_names.push_back(bone_name);
				internal_joint_twist.push_back(Vector2(0, Math::PI));
				internal_kusudama_open_cone_count.push_back(0);
				internal_kusudama_open_cones.push_back(Vector<Vector4>());
			}

			// Extract limitation data from JointLimitation3D if present
			Ref<JointLimitation3D> limitation = joint_setting->limitation;
			Ref<JointLimitationKusudama3D> kusudama_lim = limitation;

			if (kusudama_lim.is_valid()) {
				// Extract kusudama data from the limitation
				Vector<Vector4> cones = kusudama_lim->get_open_cones();
				internal_kusudama_open_cone_count.write[constraint_index] = cones.size();
				Vector<Vector4> &internal_cones = internal_kusudama_open_cones.write[constraint_index];
				internal_cones = cones;

				// Extract axial limits
				if (kusudama_lim->is_axially_constrained()) {
					real_t min_angle = kusudama_lim->get_min_axial_angle();
					real_t range_angle = kusudama_lim->get_range_angle();
					internal_joint_twist.write[constraint_index] = Vector2(min_angle, min_angle + range_angle);
				}
			} else if (limitation.is_valid()) {
				// Other limitation types - store that this joint has a limitation
				// The actual kusudama cone data extraction happens in _bone_list_changed()
				// when IKBone3D structures are created and constraints are applied
			}
		}
	}
}

void EWBIK3D::_process_ik(Skeleton3D *p_skeleton, double p_delta) {
	// Build internal data from settings
	_build_internal_data_from_settings(p_skeleton);

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
		godot_skeleton_transform.instantiate();
		godot_skeleton_transform->set_transform(p_skeleton->get_transform());
		godot_skeleton_transform_inverse = p_skeleton->get_transform().affine_inverse();
	}
	if (!is_enabled()) {
		return;
	}
	if (!is_visible()) {
		return;
	}

	// Use EWBIK algorithm with iterations_per_frame
	bone_damp.resize(bone_list.size());
	for (int32_t bone_i = bone_list.size(); bone_i-- > 0;) {
		bone_damp.write[bone_i] = get_default_damp();
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

void EWBIK3D::_validate_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const {
	// Validate axis for the joint in the setting
	ERR_FAIL_INDEX(p_index, (int)iterate_settings.size());
	IterateIK3DSetting *setting = iterate_settings[p_index];
	ERR_FAIL_NULL(setting);
	ERR_FAIL_INDEX(p_joint, (int)setting->joint_settings.size());
	// Validation is handled by IterateIK3D base class
}

void EWBIK3D::_init_joints(Skeleton3D *p_skeleton, int p_index) {
	ERR_FAIL_INDEX(p_index, (int)iterate_settings.size());
	IterateIK3DSetting *setting = iterate_settings[p_index];
	ERR_FAIL_NULL(setting);

	// Initialize joints from chain data
	// This will be called by IterateIK3D::_process_ik before solving
	// For EWBIK, we need to ensure the chain is properly set up
	if (setting->simulation_dirty) {
		setting->init_current_joint_rotations(p_skeleton);
		setting->simulation_dirty = false;
	}
}

void EWBIK3D::_make_simulation_dirty(int p_index) {
	ERR_FAIL_INDEX(p_index, (int)iterate_settings.size());
	IterateIK3DSetting *setting = iterate_settings[p_index];
	if (!setting) {
		return;
	}
	setting->simulation_dirty = true;
	is_dirty = true;
}

void EWBIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) {
	// Note: EWBIK3D overrides _process_ik() completely, so this method is not typically called
	// However, it's implemented here for completeness in case the base class behavior changes

	// EWBIK uses a segmented skeleton approach that processes the entire skeleton at once
	// rather than iterating on individual chains. The actual solving happens in _process_ik()
	// via segmented_skeleton->segment_solver() calls.

	// If this method is called, we could implement a chain-based solver here,
	// but for now EWBIK's algorithm is fully handled in _process_ik()
}

void EWBIK3D::_set_joint_count(int p_index, int p_count) {
	ERR_FAIL_INDEX(p_index, (int)iterate_settings.size());
	IterateIK3DSetting *setting = iterate_settings[p_index];
	ERR_FAIL_NULL(setting);

	// Resize joint_settings to match joint count
	int old_count = setting->joint_settings.size();
	setting->joint_settings.resize(p_count);

	// Initialize new joint settings
	for (int i = old_count; i < p_count; i++) {
		if (!setting->joint_settings[i]) {
			setting->joint_settings[i] = memnew(IterateIK3DJointSetting);
		}
	}

	setting->simulation_dirty = true;
	is_dirty = true;
}

void EWBIK3D::set_dirty() {
	is_dirty = true;
}

int32_t EWBIK3D::get_bone_count() const {
	return bone_list.size();
}

TypedArray<IKBone3D> EWBIK3D::get_bone_list() const {
	TypedArray<IKBone3D> result;
	result.resize(bone_list.size());
	for (int i = 0; i < bone_list.size(); i++) {
		result[i] = bone_list[i];
	}
	return result;
}

void EWBIK3D::register_skeleton() {
	// Settings are managed by IterateIK3D base class
	set_dirty();
}

int32_t EWBIK3D::get_ui_selected_bone() const {
	return ui_selected_bone;
}

void EWBIK3D::set_ui_selected_bone(int32_t p_ui_selected_bone) {
	ui_selected_bone = p_ui_selected_bone;
}

void EWBIK3D::set_stabilization_passes(int32_t p_passes) {
	stabilize_passes = p_passes;
	set_dirty();
}

int32_t EWBIK3D::get_stabilization_passes() {
	return stabilize_passes;
}

Transform3D EWBIK3D::get_godot_skeleton_transform_inverse() {
	return godot_skeleton_transform_inverse;
}

Ref<IKNode3D> EWBIK3D::get_godot_skeleton_transform() {
	return godot_skeleton_transform;
}

void EWBIK3D::_bone_list_changed() {
	Skeleton3D *skeleton = get_skeleton();
	Vector<int32_t> roots = skeleton->get_parentless_bones();
	if (roots.is_empty()) {
		return;
	}
	bone_list.clear();
	segmented_skeletons.clear();

	// Create ik_origin once outside the loop
	if (ik_origin.is_null()) {
		ik_origin.instantiate();
	}

	for (BoneId root_bone_index : roots) {
		String parentless_bone = skeleton->get_bone_name(root_bone_index);
		Ref<IKBoneSegment3D> segmented_skeleton = Ref<IKBoneSegment3D>(memnew(IKBoneSegment3D(skeleton, parentless_bone, internal_pins, this, nullptr, root_bone_index, -1, stabilize_passes)));
		segmented_skeleton->get_root()->get_ik_transform()->set_parent(ik_origin);
		segmented_skeleton->generate_default_segments(internal_pins, root_bone_index, -1, this);
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
	// Apply constraints from internal data (built from settings)
	for (int constraint_i = 0; constraint_i < internal_constraint_names.size(); ++constraint_i) {
		String bone = internal_constraint_names[constraint_i];
		BoneId bone_id = skeleton->find_bone(bone);
		for (Ref<IKBone3D> &ik_bone_3d : bone_list) {
			if (ik_bone_3d->get_bone_id() != bone_id) {
				continue;
			}

			// Check if bone already has a constraint (from previous build or settings)
			Ref<IKKusudama3D> existing_constraint = ik_bone_3d->get_constraint();
			Ref<IKKusudama3D> constraint;

			if (existing_constraint.is_valid()) {
				// Use existing constraint and update it
				constraint = existing_constraint;
				constraint->clear_open_cones();
			} else {
				// Create new constraint
				constraint.instantiate();
				constraint->enable_orientational_limits();
			}

			// Apply kusudama cone data from internal structures
			int32_t cone_count = internal_kusudama_open_cone_count[constraint_i];
			const Vector<Vector4> &cones = internal_kusudama_open_cones[constraint_i];
			for (int32_t cone_i = 0; cone_i < cone_count; ++cone_i) {
				const Vector4 &cone = cones[cone_i];
				Ref<IKLimitCone3D> new_cone;
				new_cone.instantiate();
				new_cone->set_attached_to(constraint);
				new_cone->set_radius(MAX(1.0e-38, cone.w));
				new_cone->set_control_point(Vector3(cone.x, cone.y, cone.z).normalized());
				constraint->add_open_cone(new_cone);
			}

			// Apply axial limits (twist)
			const Vector2 axial_limit = internal_joint_twist[constraint_i];
			constraint->enable_axial_limits();
			constraint->set_axial_limits(axial_limit.x, axial_limit.y);

			// Attach constraint to bone if not already attached
			if (existing_constraint.is_null()) {
				ik_bone_3d->add_constraint(constraint);
			}

			// Update constraint with bone's transform
			if (ik_bone_3d->get_constraint_twist_transform().is_valid()) {
				constraint->_update_constraint(ik_bone_3d->get_constraint_twist_transform());
			}
			break;
		}
	}

	// Extract kusudama data from existing IKBone3D constraints and sync to internal structures
	// This ensures settings stay in sync with actual constraint data
	for (Ref<IKBone3D> &ik_bone_3d : bone_list) {
		if (ik_bone_3d.is_null()) {
			continue;
		}
		Ref<IKKusudama3D> constraint = ik_bone_3d->get_constraint();
		if (constraint.is_null()) {
			continue;
		}

		int bone_id = ik_bone_3d->get_bone_id();
		if (bone_id < 0) {
			continue;
		}
		StringName bone_name = skeleton->get_bone_name(bone_id);

		// Find or create constraint entry
		int constraint_index = -1;
		for (int j = 0; j < internal_constraint_names.size(); j++) {
			if (internal_constraint_names[j] == bone_name) {
				constraint_index = j;
				break;
			}
		}

		if (constraint_index < 0) {
			// Create new entry
			constraint_index = internal_constraint_names.size();
			internal_constraint_names.push_back(bone_name);
			internal_joint_twist.push_back(Vector2(0, Math::PI));
			internal_kusudama_open_cone_count.push_back(0);
			internal_kusudama_open_cones.push_back(Vector<Vector4>());
		}

		// Extract kusudama data from constraint
		TypedArray<IKLimitCone3D> open_cones = constraint->get_open_cones();
		int32_t cone_count = open_cones.size();
		internal_kusudama_open_cone_count.write[constraint_index] = cone_count;
		Vector<Vector4> &cones = internal_kusudama_open_cones.write[constraint_index];
		cones.resize(cone_count);

		for (int32_t cone_i = 0; cone_i < cone_count; ++cone_i) {
			Ref<IKLimitCone3D> cone = open_cones[cone_i];
			if (cone.is_null()) {
				continue;
			}
			Vector3 center = cone->get_control_point();
			float radius = cone->get_radius();
			cones.write[cone_i] = Vector4(center.x, center.y, center.z, radius);
		}

		// Extract axial limits
		if (constraint->is_axially_constrained()) {
			real_t min_angle = constraint->get_min_axial_angle();
			real_t range_angle = constraint->get_range_angle();
			internal_joint_twist.write[constraint_index] = Vector2(min_angle, min_angle + range_angle);
		}
	}
}

void EWBIK3D::_skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) {
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

void EWBIK3D::set_state(const Variant &p_state) {
	// State management is now handled by IterateIK3D settings system
	// This method is kept for compatibility but does nothing
}

Variant EWBIK3D::get_state() const {
	// State management is now handled by IterateIK3D settings system
	// Return null as state is no longer used
	return Variant();
}

void EWBIK3D::queue_print_skeleton() {
	// Debug method - can be implemented if needed
}
