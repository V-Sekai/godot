/**************************************************************************/
/*  many_bone_ik_3d.h                                                     */
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

#include "scene/3d/skeleton_modifier_3d.h"

#include "scene/resources/3d/joint_limitation_3d.h"

class ManyBoneIK3D : public SkeletonModifier3D {
	GDCLASS(ManyBoneIK3D, SkeletonModifier3D);

public:
	enum BoneDirection {
		BONE_DIRECTION_PLUS_X,
		BONE_DIRECTION_MINUS_X,
		BONE_DIRECTION_PLUS_Y,
		BONE_DIRECTION_MINUS_Y,
		BONE_DIRECTION_PLUS_Z,
		BONE_DIRECTION_MINUS_Z,
		BONE_DIRECTION_FROM_PARENT,
	};

	struct ManyBoneIK3DSolverInfo {
		Quaternion current_rot;
		Vector3 forward_vector;
		float length = 0.0;
	};

	struct ManyBoneIK3DJointSetting {
		String bone_name;
		int bone = -1;

		// Limitation for the twist.
		real_t twist_limitation = Math_PI;
		// Limitation for the swing.
		Ref<JointLimitation3D> limitation;
		Quaternion limitation_rotation_offset;

		// To process.
		ManyBoneIK3DSolverInfo *solver_info = nullptr;
	};

	struct ManyBoneIK3DSetting {
		String root_bone_name;
		int root_bone = -1;

		String end_bone_name;
		int end_bone = -1;

		// To make virtual end joint.
		bool extend_end_bone = false;
		BoneDirection end_bone_direction = BONE_DIRECTION_FROM_PARENT;
		float end_bone_length = 0.0;

		NodePath target_node;
		bool use_target_axis = false;
		BoneAxis target_axis = BONE_AXIS_PLUS_Y;
		int max_iterations = 4;
		real_t min_distance = 0.01; // If distance between end joint and target is less than min_distance, finish iteration.

		Vector<ManyBoneIK3DJointSetting *> joints;
		Vector<Vector3> chain;

		// To process.
		bool simulation_dirty = false;
		Transform3D cached_space;
	};

	struct TwistSwing {
		Quaternion twist;
		Quaternion swing;
	};

protected:
	Vector<ManyBoneIK3DSetting *> settings;

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_property(PropertyInfo &p_property) const;

	void _notification(int p_what);

	static void _bind_methods();

	virtual void _set_active(bool p_active) override;
	virtual void _process_modification(double p_delta) override;
	void _init_joints(Skeleton3D *p_skeleton, ManyBoneIK3DSetting *p_setting);

	virtual void _process_joints(double p_delta, Skeleton3D *p_skeleton, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Transform3D &p_space, const Vector3 &p_destination, int p_max_iterations, real_t p_min_distance_squared);

	void _make_joints_dirty(int p_index);
	void _make_all_joints_dirty();

	void _update_joint_array(int p_index);

public:
	// Setting.
	void set_root_bone_name(int p_index, const String &p_bone_name);
	String get_root_bone_name(int p_index) const;
	void set_root_bone(int p_index, int p_bone);
	int get_root_bone(int p_index) const;

	void set_end_bone_name(int p_index, const String &p_bone_name);
	String get_end_bone_name(int p_index) const;
	void set_end_bone(int p_index, int p_bone);
	int get_end_bone(int p_index) const;

	void set_extend_end_bone(int p_index, bool p_enabled);
	bool is_end_bone_extended(int p_index) const;
	void set_end_bone_direction(int p_index, BoneDirection p_bone_direction);
	BoneDirection get_end_bone_direction(int p_index) const;
	void set_end_bone_length(int p_index, float p_length);
	float get_end_bone_length(int p_index) const;
	Vector3 get_end_bone_axis(int p_end_bone, BoneDirection p_direction) const; // Helper.

	void set_target_node(int p_index, const NodePath &p_target_node);
	NodePath get_target_node(int p_index) const;

	void set_max_iterations(int p_index, int p_max_iterations);
	int get_max_iterations(int p_index) const;
	void set_min_distance(int p_index, real_t p_min_distance);
	real_t get_min_distance(int p_index) const;

	void set_setting_count(int p_count);
	int get_setting_count() const;
	void clear_settings();

	// Individual joints.
	void set_joint_bone_name(int p_index, int p_joint, const String &p_bone_name);
	String get_joint_bone_name(int p_index, int p_joint) const;
	void set_joint_bone(int p_index, int p_joint, int p_bone);
	int get_joint_bone(int p_index, int p_joint) const;

	void set_joint_twist_limitation(int p_index, int p_joint, const real_t &p_angle);
	real_t get_joint_twist_limitation(int p_index, int p_joint) const;
	void set_joint_limitation(int p_index, int p_joint, const Ref<JointLimitation3D> &p_limitation);
	Ref<JointLimitation3D> get_joint_limitation(int p_index, int p_joint) const;
	void set_joint_limitation_rotation_offset(int p_index, int p_joint, const Quaternion &p_rotation_offset);
	Quaternion get_joint_limitation_rotation_offset(int p_index, int p_joint) const;

	void set_joint_count(int p_index, int p_count);
	int get_joint_count(int p_index) const;

	// Helper.
	static Quaternion get_local_pose_rotation(Skeleton3D *p_skeleton, int p_bone, const Quaternion &p_global_pose_rotation);
	static TwistSwing decompose_rotation_to_twist_and_swing(const Vector3 &p_forward_axis, const Quaternion &p_rotation);
	static Quaternion compose_rotation_from_twist_and_swing(const TwistSwing &p_twist_and_swing);

	// To process manually.
	void reset();
};

VARIANT_ENUM_CAST(ManyBoneIK3D::BoneDirection);
