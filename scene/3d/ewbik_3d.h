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
#include "core/object/ref_counted.h"
#include "scene/3d/ik_bone_3d.h"
#include "scene/3d/ik_bone_segment_3d.h"
#include "scene/3d/ik_effector_template_3d.h"
#include "scene/3d/ik_math/ik_node_3d.h"
#include "scene/3d/iterate_ik_3d.h"
#include "scene/3d/skeleton_3d.h"

class EWBIK3D : public IterateIK3D {
	GDCLASS(EWBIK3D, IterateIK3D);

	// Internal IK solving data structures (kept for algorithm, built from settings)
	Vector<Ref<IKBoneSegment3D>> segmented_skeletons;
	Vector<Ref<IKBone3D>> bone_list;
	Ref<IKNode3D> godot_skeleton_transform;
	Transform3D godot_skeleton_transform_inverse;
	Ref<IKNode3D> ik_origin;
	bool is_dirty = true;
	NodePath skeleton_node_path = NodePath("..");

	// Temporary internal data built from settings (not exposed as public API)
	Vector<Ref<IKEffectorTemplate3D>> internal_pins; // Built from iterate_settings
	Vector<StringName> internal_constraint_names; // Built from joint_settings
	Vector<Vector2> internal_joint_twist; // Built from joint_settings limitations
	Vector<Vector<Vector4>> internal_kusudama_open_cones; // Built from joint_settings limitations
	Vector<int> internal_kusudama_open_cone_count; // Built from joint_settings limitations
	Vector<float> bone_damp; // Per-bone damping

	void _build_internal_data_from_settings(Skeleton3D *p_skeleton);

	// EWBIK-specific properties (global to all settings)
	int32_t iterations_per_frame = 15;
	float default_damp = Math::deg_to_rad(5.0f);
	int32_t stabilize_passes = 0;
	int32_t ui_selected_bone = -1;

	void _update_ik_bones_transform();
	void _update_skeleton_bones_transform();
	void _bone_list_changed();
	void _pose_updated();
	void _update_ik_bone_pose(int32_t p_bone_idx);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	static void _bind_methods();
	void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) override;

	// Required IterateIK3D virtual methods
	virtual void _validate_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const override;
	virtual void _init_joints(Skeleton3D *p_skeleton, int p_index) override;
	virtual void _make_simulation_dirty(int p_index) override;
	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta) override;
	virtual void _solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) override;
	virtual void _set_joint_count(int p_index, int p_count) override;

public:
	void set_state(const Variant &p_state);
	Variant get_state() const;
	void set_stabilization_passes(int32_t p_passes);
	int32_t get_stabilization_passes();
	Transform3D get_godot_skeleton_transform_inverse();
	Ref<IKNode3D> get_godot_skeleton_transform();
	void set_ui_selected_bone(int32_t p_ui_selected_bone);
	int32_t get_ui_selected_bone() const;
	void register_skeleton();
	TypedArray<IKBone3D> get_bone_list() const;
	TypedArray<IKBoneSegment3D> get_segmented_skeletons();
	float get_iterations_per_frame() const;
	void set_iterations_per_frame(const float &p_iterations_per_frame);
	void queue_print_skeleton();
	real_t get_default_damp() const;
	void set_default_damp(float p_default_damp);
	int32_t get_bone_count() const;
	void cleanup();
	void set_dirty();

	EWBIK3D();
	~EWBIK3D();
};
