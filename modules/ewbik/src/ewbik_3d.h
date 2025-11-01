/**************************************************************************/
/*  ewbik_3d.h                                                            */
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

#pragma once

#include "core/math/math_defs.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/variant/typed_array.h"
#include "ik_bone_3d.h"
#include "ik_effector_template_3d.h"
#include "math/ik_node_3d.h"
#include "scene/3d/many_bone_ik_3d.h"

class IKBoneSegment3D;
class ManyBoneIK3DState;
class EWBIK3D : public ManyBoneIK3D {
public:
	struct EWBIK3DSetting : public ManyBoneIK3DSetting {
		StringName bone_name;
		NodePath target_node;
		float weight = 1.0f;
		Vector3 direction_priorities = Vector3(0.25f, 0.25f, 0.25f);
		float motion_propagation_factor = 0.5f;
	};
	GDCLASS(EWBIK3D, ManyBoneIK3D);

	NodePath skeleton_path;
	Vector<Ref<IKBoneSegment3D>> segmented_skeletons;
	int32_t bone_count = 0;
	Vector<Ref<IKBone3D>> bone_list;

	// Settings system
	LocalVector<EWBIK3DSetting *> ewbik_settings;
	int32_t pin_count = 0; // Keep for compatibility but use settings system
	int32_t iterations_per_frame = 15;
	float default_damp = Math::deg_to_rad(5.0f);
	Ref<IKNode3D> godot_skeleton_transform;
	Transform3D godot_skeleton_transform_inverse;
	Ref<IKNode3D> ik_origin;
	bool is_dirty = true;
	NodePath skeleton_node_path = NodePath("..");
	int32_t stabilize_passes = 0;

	// IterateIK3D compatible parameters
	float angular_delta_limit = Math::deg_to_rad(2.0f); // ~2 degrees per iteration
	int32_t max_iterations = 4;
	float min_distance = 0.001f;

	void _on_timer_timeout();
	void _update_ik_bones_transform();
	void _update_skeleton_bones_transform();
	Vector<Ref<IKEffectorTemplate3D>> _get_bone_effectors() const;
	void _remove_pin(int32_t p_index);
	void _set_bone_count(int32_t p_count);
	void _set_pin_root_bone(int32_t p_pin_index, const String &p_root_bone);
	String _get_pin_root_bone(int32_t p_pin_index) const;
	void _bone_list_changed();
	void _pose_updated();
	void _update_ik_bone_pose(int32_t p_bone_idx);
	void _apply_bone_constraints();

	virtual void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) override;

	virtual void set_setting_count(int p_count) override {
		_set_setting_count<EWBIK3DSetting>(p_count);
		ewbik_settings = _cast_settings<EWBIK3DSetting>();
	}
	virtual void clear_settings() override {
		_set_setting_count<EWBIK3DSetting>(0);
		ewbik_settings.clear();
	}

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	static void _bind_methods();
	virtual void _process_modification(double p_delta) override;

public:
	void set_state(Ref<ManyBoneIK3DState> p_state);
	Ref<ManyBoneIK3DState> get_state() const;
	void set_stabilization_passes(int32_t p_passes);
	int32_t get_stabilization_passes() const;
	Transform3D get_godot_skeleton_transform_inverse();
	Ref<IKNode3D> get_godot_skeleton_transform();
	bool get_pin_enabled(int32_t p_effector_index) const;
	void register_skeleton();
	TypedArray<IKBone3D> get_bone_list() const;
	TypedArray<IKBoneSegment3D> get_segmented_skeletons();
	float get_iterations_per_frame() const;
	void set_iterations_per_frame(const float &p_iterations_per_frame);
	void queue_print_skeleton();
	int32_t get_pin_count() const;
	void set_pin_count(int32_t p_pin_count);
	void remove_pin_at_index(int32_t p_index);
	void set_pin_bone_name(int32_t p_pin_index, const String &p_bone);
	StringName get_pin_bone_name(int32_t p_effector_index) const;
	void set_pin_node_path(int32_t p_effector_index, NodePath p_node_path);
	NodePath get_pin_node_path(int32_t p_effector_index) const;
	int32_t find_pin_id(StringName p_bone_name);
	void set_pin_target_node_path(int32_t p_effector_index, const NodePath &p_target_node);
	void set_pin_weight(int32_t p_pin_index, const real_t &p_weight);
	real_t get_pin_weight(int32_t p_pin_index) const;
	void set_pin_direction_priorities(int32_t p_pin_index, const Vector3 &p_priority_direction);
	Vector3 get_pin_direction_priorities(int32_t p_pin_index) const;
	NodePath get_pin_target_node_path(int32_t p_pin_index) const;
	void set_pin_motion_propagation_factor(int32_t p_effector_index, const float p_motion_propagation_factor);
	float get_pin_motion_propagation_factor(int32_t p_effector_index) const;
	real_t get_default_damp() const;
	void set_default_damp(float p_default_damp);
	int32_t find_pin(String p_string) const;
	int32_t get_bone_count() const;
	void cleanup();
	void add_segment(Ref<IKBoneSegment3D> p_segment);

	// IterateIK3D compatible parameter getters/setters
	void set_angular_delta_limit(float p_limit);
	float get_angular_delta_limit() const;
	void set_max_iterations(int32_t p_iterations);
	int32_t get_max_iterations() const;
	void set_min_distance(float p_distance);
	float get_min_distance() const;

	EWBIK3D();
	~EWBIK3D();
	void set_dirty();
};
