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

	enum RotationAxis {
		ROTATION_AXIS_X,
		ROTATION_AXIS_Y,
		ROTATION_AXIS_Z,
		ROTATION_AXIS_ALL,
		ROTATION_AXIS_OPTIONAL,
	};

	struct ManyBoneIK3DSolverInfo {
		Quaternion current_gpose;
		Quaternion current_grest;
		Vector3 current_vector;
		Vector3 forward_vector;
		float length = 0.0;
	};

	struct ManyBoneIK3DJointSetting {
		String bone_name;
		int bone = -1;

		// Pole vector.
		RotationAxis rotation_axis = ROTATION_AXIS_ALL;
		Vector3 rotation_axis_vector = Vector3(1, 0, 0);
		Vector3 get_rotation_axis_vector() const {
			Vector3 ret;
			switch (rotation_axis) {
				case ROTATION_AXIS_X:
					ret = Vector3(1, 0, 0);
					break;
				case ROTATION_AXIS_Y:
					ret = Vector3(0, 1, 0);
					break;
				case ROTATION_AXIS_Z:
					ret = Vector3(0, 0, 1);
					break;
				case ROTATION_AXIS_ALL:
					ret = Vector3(0, 0, 0);
					break;
				case ROTATION_AXIS_OPTIONAL:
					ret = rotation_axis_vector;
					break;
			}
			return ret;
		}

		// To limit rotation.
		Ref<JointLimitation3D> limitation;

		// To process.
		ManyBoneIK3DSolverInfo *solver_info = nullptr;

		Vector3 get_projected_rotation(const Quaternion &p_offset, const Vector3 &p_vector) const {
			ERR_FAIL_COND_V(rotation_axis == ROTATION_AXIS_ALL, p_vector);
			Vector3 axis = get_rotation_axis_vector();
			Vector3 local_vector = p_offset.xform_inv(p_vector);
			Vector3 projected = snap_vector_to_plane(axis, local_vector);
			if (!Math::is_zero_approx(local_vector.length())) {
				projected = projected.normalized() * local_vector.length();
			}
			return p_offset.xform(projected);
		}

		Vector3 get_limited_rotation(const Quaternion &p_offset, const Vector3 &p_vector) const {
			ERR_FAIL_COND_V(limitation.is_null(), p_vector);
			Vector3 local_vector = p_offset.xform_inv(p_vector);
			float length = local_vector.length();
			if (Math::is_zero_approx(length)) {
				return p_vector;
			}
			Vector3 limited = limitation->solve(solver_info->forward_vector, local_vector.normalized()) * length;
			return p_offset.xform(limited);
		}
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

		bool is_penetrated(const Vector3 &p_destination) {
			bool ret = false;
			Vector3 chain_dir = (chain[chain.size() - 1] - chain[0]).normalized();
			bool is_straight = true;
			for (int i = 1; i < chain.size() - 1; i++) {
				Vector3 dir = (chain[i] - chain[0]).normalized();
				if (!dir.is_equal_approx(chain_dir)) {
					is_straight = false;
					break;
				}
			}
			if (is_straight) {
				Vector3 to_target = (p_destination - chain[0]);
				float proj = to_target.dot(chain_dir);
				float total_length = 0.0f;
				for (int i = 0; i < joints.size(); i++) {
					if (joints[i]->solver_info) {
						total_length += joints[i]->solver_info->length;
					}
				}
				ret = proj >= 0.0f && proj <= total_length && (to_target.normalized().is_equal_approx(chain_dir));
			}
			return ret;
		}

		void update_chain_coordinate(Skeleton3D *p_skeleton, int p_index, const Vector3 &p_position, bool p_backward = true) {
			// Don't update if the position is same as the current position.
			if (Math::is_zero_approx(chain[p_index].distance_squared_to(p_position))) {
				return;
			}
			// Prevent flipping.
			if (p_backward) {
				const int PREV = p_index - 1;
				if (PREV >= 0 && PREV < joints.size()) {
					ManyBoneIK3DSolverInfo *solver_info = joints[PREV]->solver_info;
					if (solver_info && Math::is_equal_approx((float)solver_info->current_vector.dot((p_position - chain[PREV]).normalized()), -1.0f)) {
						chain.write[p_index] = chain[PREV] + solver_info->current_vector * solver_info->length;
						return;
					}
				}
			} else {
				const int PREV = p_index + 1;
				if (PREV >= 0 && PREV < joints.size()) {
					ManyBoneIK3DSolverInfo *solver_info = joints[p_index]->solver_info;
					if (solver_info && Math::is_equal_approx((float)solver_info->current_vector.dot((chain[PREV] - p_position).normalized()), -1.0f)) {
						chain.write[p_index] = chain[PREV] - solver_info->current_vector * solver_info->length;
						return;
					}
				}
			}
			chain.write[p_index] = p_position;
		}

		void init_current_joint_rotations(Skeleton3D *p_skeleton) {
			Quaternion parent_gpose;
			int parent = p_skeleton->get_bone_parent(root_bone);
			if (parent >= 0) {
				parent_gpose = p_skeleton->get_bone_global_pose(parent).basis.get_rotation_quaternion();
			}

			for (int i = 0; i < joints.size(); i++) {
				ManyBoneIK3DSolverInfo *solver_info = joints[i]->solver_info;
				if (!solver_info) {
					continue;
				}
				solver_info->current_grest = parent_gpose * p_skeleton->get_bone_rest(joints[i]->bone).basis.get_rotation_quaternion();
				solver_info->current_gpose = p_skeleton->get_bone_global_pose(joints[i]->bone).basis.get_rotation_quaternion();
				parent_gpose = solver_info->current_gpose;
			}

			for (int i = 0; i < joints.size(); i++) {
				const int HEAD = i;
				const int TAIL = i + 1;
				ManyBoneIK3DSolverInfo *solver_info = joints[HEAD]->solver_info;
				if (!solver_info) {
					continue;
				}
				solver_info->current_vector = (chain[TAIL] - chain[HEAD]).normalized();
			}
		}

		void cache_current_joint_rotations(Skeleton3D *p_skeleton, int p_first_index = 0) {
			Quaternion parent_gpose;
			int parent = p_skeleton->get_bone_parent(root_bone);
			if (parent >= 0) {
				parent_gpose = p_skeleton->get_bone_global_pose(parent).basis.get_rotation_quaternion();
			}

			for (int i = p_first_index; i < joints.size(); i++) {
				const int HEAD = i;
				const int TAIL = i + 1;
				ManyBoneIK3DSolverInfo *solver_info = joints[HEAD]->solver_info;
				if (!solver_info) {
					continue;
				}
				solver_info->current_grest = parent_gpose * p_skeleton->get_bone_rest(joints[HEAD]->bone).basis.get_rotation_quaternion();
				Quaternion current_wrest = solver_info->current_grest;
				Vector3 from = solver_info->forward_vector;
				Vector3 to = current_wrest.xform_inv((chain[TAIL] - chain[HEAD])).normalized();
				solver_info->current_gpose = solver_info->current_grest * get_from_to_rotation(
						from,
						to,
						solver_info->current_grest.inverse() * solver_info->current_gpose);
				parent_gpose = solver_info->current_gpose;
			}

			for (int i = 0; i < joints.size(); i++) {
				const int HEAD = i;
				const int TAIL = i + 1;
				ManyBoneIK3DSolverInfo *solver_info = joints[HEAD]->solver_info;
				if (!solver_info) {
					continue;
				}
				solver_info->current_vector = (chain[TAIL] - chain[HEAD]).normalized();
			}
		}
	};

protected:
	Vector<ManyBoneIK3DSetting *> settings;

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_property(PropertyInfo &p_property) const;

	void _notification(int p_what);

	virtual void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) override;
	virtual void _validate_bone_names() override;

	static void _bind_methods();

	virtual void _set_active(bool p_active) override;
	virtual void _process_modification(double p_delta) override;
	void _init_joints(Skeleton3D *p_skeleton, ManyBoneIK3DSetting *p_setting);

	virtual void _process_joints(double p_delta, Skeleton3D *p_skeleton, ManyBoneIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_max_iterations, real_t p_min_distance_squared);

	void _make_joints_dirty(int p_index);
	void _make_all_joints_dirty();

	void _update_joint_array(int p_index);

	void _validate_rotation_axes(Skeleton3D *p_skeleton) const;
	void _validate_rotation_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const;

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

	void set_joint_rotation_axis(int p_index, int p_joint, RotationAxis p_axis);
	RotationAxis get_joint_rotation_axis(int p_index, int p_joint) const;
	void set_joint_rotation_axis_vector(int p_index, int p_joint, Vector3 p_vector);
	Vector3 get_joint_rotation_axis_vector(int p_index, int p_joint) const;

	void set_joint_limitation(int p_index, int p_joint, const Ref<JointLimitation3D> &p_limitation);
	Ref<JointLimitation3D> get_joint_limitation(int p_index, int p_joint) const;

	void set_joint_count(int p_index, int p_count);
	int get_joint_count(int p_index) const;

	// Helper.
	static Quaternion get_local_pose_rotation(Skeleton3D *p_skeleton, int p_bone, const Quaternion &p_global_pose_rotation);

	// To process manually.
	void reset();

	~ManyBoneIK3D();
};

VARIANT_ENUM_CAST(ManyBoneIK3D::BoneDirection);
VARIANT_ENUM_CAST(ManyBoneIK3D::RotationAxis);
