/**************************************************************************/
/*  convert_transform_modifier_3d.cpp                                     */
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

#include "convert_transform_modifier_3d.h"

constexpr const char *HINT_POSITION = "-10,10,0.01,or_greater,or_less,suffix:m";
constexpr const char *HINT_ROTATION = "-360,360,0.01,radians_as_degrees";
constexpr const char *HINT_SCALE = "0,10,0.01,or_greater";

bool ConvertTransformModifier3D::_set(const StringName &p_path, const Variant &p_value) {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String where = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);
		String what = path.get_slicec('/', 3);

		if (where == "apply") {
			if (what == "transform_mode") {
				set_apply_transform_mode(which, static_cast<TransformMode>((int)p_value));
			} else if (what == "axis") {
				set_apply_axis(which, static_cast<Vector3::Axis>((int)p_value));
			} else if (what == "range_min") {
				set_apply_range_min(which, p_value);
			} else if (what == "range_max") {
				set_apply_range_max(which, p_value);
			} else {
				return false;
			}
		} else if (where == "reference") {
			if (what == "transform_mode") {
				set_reference_transform_mode(which, static_cast<TransformMode>((int)p_value));
			} else if (what == "axis") {
				set_reference_axis(which, static_cast<Vector3::Axis>((int)p_value));
			} else if (what == "range_min") {
				set_reference_range_min(which, p_value);
			} else if (what == "range_max") {
				set_reference_range_max(which, p_value);
			} else {
				return false;
			}
		} else if (where == "relative") {
			set_relative(which, p_value);
		} else if (where == "additive") {
			set_additive(which, p_value);
		} else {
			return false;
		}
	}
	return true;
}

bool ConvertTransformModifier3D::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("settings/")) {
		int which = path.get_slicec('/', 1).to_int();
		String where = path.get_slicec('/', 2);
		ERR_FAIL_INDEX_V(which, (int)settings.size(), false);
		String what = path.get_slicec('/', 3);

		if (where == "apply") {
			if (what == "transform_mode") {
				r_ret = (int)get_apply_transform_mode(which);
			} else if (what == "axis") {
				r_ret = (int)get_apply_axis(which);
			} else if (what == "range_min") {
				r_ret = get_apply_range_min(which);
			} else if (what == "range_max") {
				r_ret = get_apply_range_max(which);
			} else {
				return false;
			}
		} else if (where == "reference") {
			if (what == "transform_mode") {
				r_ret = (int)get_reference_transform_mode(which);
			} else if (what == "axis") {
				r_ret = (int)get_reference_axis(which);
			} else if (what == "range_min") {
				r_ret = get_reference_range_min(which);
			} else if (what == "range_max") {
				r_ret = get_reference_range_max(which);
			} else {
				return false;
			}
		} else if (where == "relative") {
			r_ret = is_relative(which);
		} else if (where == "additive") {
			r_ret = is_additive(which);
		} else {
			return false;
		}
	}
	return true;
}

void ConvertTransformModifier3D::_get_property_list(List<PropertyInfo> *p_list) const {
	BoneConstraint3D::get_property_list(p_list);

	LocalVector<PropertyInfo> props;

	for (uint32_t i = 0; i < settings.size(); i++) {
		String path = "settings/" + itos(i) + "/";

		String hint_apply_range;
		if (get_apply_transform_mode(i) == TRANSFORM_MODE_POSITION) {
			hint_apply_range = HINT_POSITION;
		} else if (get_apply_transform_mode(i) == TRANSFORM_MODE_ROTATION) {
			hint_apply_range = HINT_ROTATION;
		} else {
			hint_apply_range = HINT_SCALE;
		}
		props.push_back(PropertyInfo(Variant::INT, path + "apply/transform_mode", PROPERTY_HINT_ENUM, "Position,Rotation,Scale"));
		props.push_back(PropertyInfo(Variant::INT, path + "apply/axis", PROPERTY_HINT_ENUM, "X,Y,Z"));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "apply/range_min", PROPERTY_HINT_RANGE, hint_apply_range));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "apply/range_max", PROPERTY_HINT_RANGE, hint_apply_range));

		String hint_reference_range;
		if (get_reference_transform_mode(i) == TRANSFORM_MODE_POSITION) {
			hint_reference_range = HINT_POSITION;
		} else if (get_reference_transform_mode(i) == TRANSFORM_MODE_ROTATION) {
			hint_reference_range = HINT_ROTATION;
		} else {
			hint_reference_range = HINT_SCALE;
		}
		props.push_back(PropertyInfo(Variant::INT, path + "reference/transform_mode", PROPERTY_HINT_ENUM, "Position,Rotation,Scale"));
		props.push_back(PropertyInfo(Variant::INT, path + "reference/axis", PROPERTY_HINT_ENUM, "X,Y,Z"));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "reference/range_min", PROPERTY_HINT_RANGE, hint_reference_range));
		props.push_back(PropertyInfo(Variant::FLOAT, path + "reference/range_max", PROPERTY_HINT_RANGE, hint_reference_range));

		props.push_back(PropertyInfo(Variant::BOOL, path + "relative"));
		props.push_back(PropertyInfo(Variant::BOOL, path + "additive"));
	}

	for (PropertyInfo &p : props) {
		_validate_dynamic_prop(p);
		p_list->push_back(p);
	}
}

void ConvertTransformModifier3D::_validate_dynamic_prop(PropertyInfo &p_property) const {
	PackedStringArray split = p_property.name.split("/");
	if (split.size() > 2 && split[0] == "settings") {
		int which = split[1].to_int();
		if (split[2].begins_with("relative") && get_reference_type(which) != REFERENCE_TYPE_BONE) {
			p_property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

void ConvertTransformModifier3D::_validate_setting(int p_index) {
	settings[p_index] = memnew(ConvertTransform3DSetting);
}

void ConvertTransformModifier3D::set_apply_transform_mode(int p_index, TransformMode p_transform_mode) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->apply_transform_mode = p_transform_mode;
	notify_property_list_changed();
}

ConvertTransformModifier3D::TransformMode ConvertTransformModifier3D::get_apply_transform_mode(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), TRANSFORM_MODE_POSITION);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->apply_transform_mode;
}

void ConvertTransformModifier3D::set_apply_axis(int p_index, Vector3::Axis p_axis) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->apply_axis = p_axis;
}

Vector3::Axis ConvertTransformModifier3D::get_apply_axis(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3::AXIS_X);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->apply_axis;
}

void ConvertTransformModifier3D::set_apply_range_min(int p_index, float p_range_min) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->apply_range_min = p_range_min;
}

float ConvertTransformModifier3D::get_apply_range_min(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->apply_range_min;
}

void ConvertTransformModifier3D::set_apply_range_max(int p_index, float p_range_max) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->apply_range_max = p_range_max;
}

float ConvertTransformModifier3D::get_apply_range_max(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->apply_range_max;
}

void ConvertTransformModifier3D::set_reference_transform_mode(int p_index, TransformMode p_transform_mode) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->reference_transform_mode = p_transform_mode;
	notify_property_list_changed();
}

ConvertTransformModifier3D::TransformMode ConvertTransformModifier3D::get_reference_transform_mode(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), TRANSFORM_MODE_POSITION);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->reference_transform_mode;
}

void ConvertTransformModifier3D::set_reference_axis(int p_index, Vector3::Axis p_axis) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->reference_axis = p_axis;
}

Vector3::Axis ConvertTransformModifier3D::get_reference_axis(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), Vector3::AXIS_X);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->reference_axis;
}

void ConvertTransformModifier3D::set_reference_range_min(int p_index, float p_range_min) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->reference_range_min = p_range_min;
}

float ConvertTransformModifier3D::get_reference_range_min(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->reference_range_min;
}

void ConvertTransformModifier3D::set_reference_range_max(int p_index, float p_range_max) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->reference_range_max = p_range_max;
}

float ConvertTransformModifier3D::get_reference_range_max(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->reference_range_max;
}

void ConvertTransformModifier3D::set_relative(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->relative = p_enabled;
}

bool ConvertTransformModifier3D::is_relative(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->is_relative();
}

void ConvertTransformModifier3D::set_additive(int p_index, bool p_enabled) {
	ERR_FAIL_INDEX(p_index, (int)settings.size());
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	setting->additive = p_enabled;
}

bool ConvertTransformModifier3D::is_additive(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, (int)settings.size(), 0);
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	return setting->additive;
}

void ConvertTransformModifier3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_apply_transform_mode", "index", "transform_mode"), &ConvertTransformModifier3D::set_apply_transform_mode);
	ClassDB::bind_method(D_METHOD("get_apply_transform_mode", "index"), &ConvertTransformModifier3D::get_apply_transform_mode);
	ClassDB::bind_method(D_METHOD("set_apply_axis", "index", "axis"), &ConvertTransformModifier3D::set_apply_axis);
	ClassDB::bind_method(D_METHOD("get_apply_axis", "index"), &ConvertTransformModifier3D::get_apply_axis);
	ClassDB::bind_method(D_METHOD("set_apply_range_min", "index", "range_min"), &ConvertTransformModifier3D::set_apply_range_min);
	ClassDB::bind_method(D_METHOD("get_apply_range_min", "index"), &ConvertTransformModifier3D::get_apply_range_min);
	ClassDB::bind_method(D_METHOD("set_apply_range_max", "index", "range_max"), &ConvertTransformModifier3D::set_apply_range_max);
	ClassDB::bind_method(D_METHOD("get_apply_range_max", "index"), &ConvertTransformModifier3D::get_apply_range_max);

	ClassDB::bind_method(D_METHOD("set_reference_transform_mode", "index", "transform_mode"), &ConvertTransformModifier3D::set_reference_transform_mode);
	ClassDB::bind_method(D_METHOD("get_reference_transform_mode", "index"), &ConvertTransformModifier3D::get_reference_transform_mode);
	ClassDB::bind_method(D_METHOD("set_reference_axis", "index", "axis"), &ConvertTransformModifier3D::set_reference_axis);
	ClassDB::bind_method(D_METHOD("get_reference_axis", "index"), &ConvertTransformModifier3D::get_reference_axis);
	ClassDB::bind_method(D_METHOD("set_reference_range_min", "index", "range_min"), &ConvertTransformModifier3D::set_reference_range_min);
	ClassDB::bind_method(D_METHOD("get_reference_range_min", "index"), &ConvertTransformModifier3D::get_reference_range_min);
	ClassDB::bind_method(D_METHOD("set_reference_range_max", "index", "range_max"), &ConvertTransformModifier3D::set_reference_range_max);
	ClassDB::bind_method(D_METHOD("get_reference_range_max", "index"), &ConvertTransformModifier3D::get_reference_range_max);

	ClassDB::bind_method(D_METHOD("set_relative", "index", "enabled"), &ConvertTransformModifier3D::set_relative);
	ClassDB::bind_method(D_METHOD("is_relative", "index"), &ConvertTransformModifier3D::is_relative);
	ClassDB::bind_method(D_METHOD("set_additive", "index", "enabled"), &ConvertTransformModifier3D::set_additive);
	ClassDB::bind_method(D_METHOD("is_additive", "index"), &ConvertTransformModifier3D::is_additive);

	ADD_ARRAY_COUNT("Settings", "setting_count", "set_setting_count", "get_setting_count", "settings/");

	BIND_ENUM_CONSTANT(TRANSFORM_MODE_POSITION);
	BIND_ENUM_CONSTANT(TRANSFORM_MODE_ROTATION);
	BIND_ENUM_CONSTANT(TRANSFORM_MODE_SCALE);
}

void ConvertTransformModifier3D::_process_constraint_by_bone(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, int p_reference_bone, float p_amount) {
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);
	Transform3D destination = p_skeleton->get_bone_pose(p_reference_bone);
	if (setting->is_relative()) {
		Vector3 scl_relative = destination.basis.get_scale() / p_skeleton->get_bone_rest(p_reference_bone).basis.get_scale();
		destination.basis = p_skeleton->get_bone_rest(p_reference_bone).basis.get_rotation_quaternion().inverse() * destination.basis.get_rotation_quaternion();
		destination.basis.scale_local(scl_relative);
		destination.origin = destination.origin - p_skeleton->get_bone_rest(p_reference_bone).origin;
	}
	_process_convert(p_index, p_skeleton, p_apply_bone, destination, p_amount);
}

void ConvertTransformModifier3D::_process_constraint_by_node(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, const NodePath &p_reference_node, float p_amount) {
	Node3D *nd = Object::cast_to<Node3D>(get_node_or_null(p_reference_node));
	if (!nd) {
		return;
	}
	Transform3D skel_tr = p_skeleton->get_global_transform_interpolated();
	int parent = p_skeleton->get_bone_parent(p_apply_bone);
	if (parent >= 0) {
		skel_tr = skel_tr * p_skeleton->get_bone_global_pose(parent);
	}
	Transform3D dest_tr = nd->get_global_transform_interpolated();
	Transform3D reference_dest = skel_tr.affine_inverse() * dest_tr;
	_process_convert(p_index, p_skeleton, p_apply_bone, reference_dest, p_amount);
}

void ConvertTransformModifier3D::_process_convert(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, const Transform3D &p_destination, float p_amount) {
	ConvertTransform3DSetting *setting = static_cast<ConvertTransform3DSetting *>(settings[p_index]);

	Transform3D destination = p_destination;

	// Retrieve point from reference.
	double point = 0.0;
	int axis = (int)setting->reference_axis;
	switch (setting->reference_transform_mode) {
		case TRANSFORM_MODE_POSITION: {
			point = destination.origin[axis];
		} break;
		case TRANSFORM_MODE_ROTATION: {
			// Work entirely in quaternion space to avoid angle extraction.
			Quaternion tgt_rot = destination.basis.get_rotation_quaternion();
			Vector3 ref_axis = get_vector_from_axis(setting->reference_axis);
			Quaternion roll_quat = get_roll_quaternion(tgt_rot, ref_axis);
			
			// Create reference quaternions for min/max ranges.
			Quaternion ref_min_quat = Quaternion(ref_axis, (real_t)setting->reference_range_min).normalized();
			Quaternion ref_max_quat = Quaternion(ref_axis, (real_t)setting->reference_range_max).normalized();
			
			// Find interpolation parameter t using quaternion log space.
			// For rotations around the same axis, we can use log space to find where
			// roll_quat lies between ref_min_quat and ref_max_quat.
			double t = 0.0;
			if (Math::is_equal_approx(setting->reference_range_min, setting->reference_range_max)) {
				// Special case: single value range - compare quaternions directly using dot product.
				// If roll_quat is closer to ref_min_quat (dot >= 0), use t=0, else t=1.
				// Note: For quaternions, q and -q represent the same rotation, so we check |dot|.
				double dot_min = Math::abs(roll_quat.dot(ref_min_quat));
				// If dot is close to 1, quaternions are similar (t=0), otherwise t=1.
				t = (dot_min > 0.5) ? 0.0 : 1.0;
			} else {
				// Find where roll_quat lies between ref_min_quat and ref_max_quat.
				// Use quaternion log space: log(q) represents the rotation in a linear space.
				Quaternion ref_min_inv = ref_min_quat.inverse();
				Quaternion rel_to_min = (ref_min_inv * roll_quat).normalized();
				Quaternion rel_min_to_max = (ref_min_inv * ref_max_quat).normalized();
				
				// Convert to log space for linear interpolation.
				Quaternion log_rel_to_min = rel_to_min.log();
				Quaternion log_rel_min_to_max = rel_min_to_max.log();
				
				// For rotations around the same axis, the log quaternions have vector parts
				// pointing in the same direction. Project one onto the other to find t.
				double log_min_mag_sq = log_rel_min_to_max.length_squared();
				if (log_min_mag_sq > CMP_EPSILON) {
					// Dot product in log space gives us the projection.
					double dot = log_rel_to_min.dot(log_rel_min_to_max);
					t = CLAMP(dot / log_min_mag_sq, 0.0, 1.0);
				} else {
					// Degenerate case: min and max are very close.
					t = 0.0;
				}
			}
			
			// Store t for later use in apply step (will be used to slerp between apply ranges).
			point = t;
		} break;
		case TRANSFORM_MODE_SCALE: {
			point = destination.basis.get_scale()[axis];
		} break;
	}
	// Convert point to apply.
	destination = p_skeleton->get_bone_pose(p_apply_bone);
	// For rotation mode, point is already the interpolation parameter t (0-1).
	// For other modes, perform the standard inverse_lerp/lerp.
	if (setting->reference_transform_mode != TRANSFORM_MODE_ROTATION) {
		if (Math::is_equal_approx(setting->reference_range_min, setting->reference_range_max)) {
			point = point <= (double)setting->reference_range_min ? 0 : 1;
		} else {
			point = Math::inverse_lerp((double)setting->reference_range_min, (double)setting->reference_range_max, point);
		}
		point = Math::lerp((double)setting->apply_range_min, (double)setting->apply_range_max, CLAMP(point, 0, 1));
	}
	// For rotation mode, point is t (0-1), which we'll use directly for quaternion slerp.
	axis = (int)setting->apply_axis;
	switch (setting->apply_transform_mode) {
		case TRANSFORM_MODE_POSITION: {
			if (setting->additive) {
				point = p_skeleton->get_bone_pose(p_apply_bone).origin[axis] + point;
			} else if (setting->is_relative()) {
				point = p_skeleton->get_bone_rest(p_apply_bone).origin[axis] + point;
			}
			destination.origin[axis] = point;
		} break;
		case TRANSFORM_MODE_ROTATION: {
			Vector3 rot_axis = get_vector_from_axis(setting->apply_axis);
			Vector3 dest_scl = destination.basis.get_scale();
			
			// Use quaternion slerp to interpolate between apply_range_min and apply_range_max.
			// point is already the interpolation parameter t (0-1) from the reference step.
			Quaternion apply_min_quat = Quaternion(rot_axis, (real_t)setting->apply_range_min).normalized();
			Quaternion apply_max_quat = Quaternion(rot_axis, (real_t)setting->apply_range_max).normalized();
			real_t t = CLAMP((real_t)point, 0.0, 1.0);
			Quaternion rot = apply_min_quat.slerp(apply_max_quat, t);
			
			if (setting->additive) {
				destination.basis = p_skeleton->get_bone_pose(p_apply_bone).basis.get_rotation_quaternion() * rot;
			} else if (setting->is_relative()) {
				destination.basis = p_skeleton->get_bone_rest(p_apply_bone).basis.get_rotation_quaternion() * rot;
			} else {
				destination.basis = rot;
			}
			// Scale may not have meaning, but it might affect when it is negative.
			destination.basis.scale_local(dest_scl);
		} break;
		case TRANSFORM_MODE_SCALE: {
			Vector3 dest_scl = Vector3(1, 1, 1);
			if (setting->additive) {
				dest_scl = p_skeleton->get_bone_pose(p_apply_bone).basis.get_scale();
				dest_scl[axis] = dest_scl[axis] * point;
			} else if (setting->is_relative()) {
				dest_scl = p_skeleton->get_bone_rest(p_apply_bone).basis.get_scale();
				dest_scl[axis] = dest_scl[axis] * point;
			} else {
				dest_scl = p_skeleton->get_bone_pose(p_apply_bone).basis.get_scale();
				dest_scl[axis] = point;
			}
			destination.basis = destination.basis.orthonormalized().scaled_local(dest_scl);
		} break;
	}
	// Process interpolation depends on the amount.
	destination = p_skeleton->get_bone_pose(p_apply_bone).interpolate_with(destination, p_amount);
	// Apply transform depends on the mode.
	switch (setting->apply_transform_mode) {
		case TRANSFORM_MODE_POSITION: {
			p_skeleton->set_bone_pose_position(p_apply_bone, destination.origin);
		} break;
		case TRANSFORM_MODE_ROTATION: {
			p_skeleton->set_bone_pose_rotation(p_apply_bone, destination.basis.get_rotation_quaternion());
		} break;
		case TRANSFORM_MODE_SCALE: {
			p_skeleton->set_bone_pose_scale(p_apply_bone, destination.basis.get_scale());
		} break;
	}
}

ConvertTransformModifier3D::~ConvertTransformModifier3D() {
	clear_settings();
}
