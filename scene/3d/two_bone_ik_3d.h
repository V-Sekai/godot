/**************************************************************************/
/*  two_bone_ik_3d.h                                                      */
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

#include "scene/3d/many_bone_ik_3d.h"

class TwoBoneIK3D : public ManyBoneIK3D {
	GDCLASS(TwoBoneIK3D, ManyBoneIK3D);

public:
	enum KnuckleDirection {
		KNUCKLE_DIRECTION_NONE,
		KNUCKLE_DIRECTION_PLUS_X,
		KNUCKLE_DIRECTION_MINUS_X,
		KNUCKLE_DIRECTION_PLUS_Y,
		KNUCKLE_DIRECTION_MINUS_Y,
		KNUCKLE_DIRECTION_PLUS_Z,
		KNUCKLE_DIRECTION_MINUS_Z,
		KNUCKLE_DIRECTION_CUSTOM,
	};

	struct TwoBoneIK3DSetting {
		bool joints_dirty = false;

		String root_bone_name;
		int root_bone = -1;

		String mid_bone_name;
		int mid_bone = -1;

		String end_bone_name;
		int end_bone = -1;

		// To make virtual end joint.
		bool use_virtual_end = false;
		bool extend_end_bone = false;
		BoneDirection end_bone_direction = BONE_DIRECTION_FROM_PARENT;
		float end_bone_length = 0.0;

		NodePath pole_node;
		KnuckleDirection knuckle_direction = KNUCKLE_DIRECTION_NONE; // Sort to pole target plane.
		Vector3 knuckle_direction_vector = Vector3(0, 0, 0); // Custom vector.
		NodePath target_node;

		ManyBoneIK3DJointSetting root_joint;
		ManyBoneIK3DJointSetting mid_joint;
		Vector3 root_pos;
		Vector3 mid_pos;
		Vector3 end_pos;

		// To process.
		bool simulation_dirty = true;
		Transform3D cached_space;
		real_t cached_length_sq = 0.0;

		bool is_valid() const {
			return root_joint.solver_info && mid_joint.solver_info;
		}
		bool is_end_valid() const {
			return (!use_virtual_end && end_bone != -1) || (use_virtual_end && !Math::is_zero_approx(end_bone_length));
		}
		int get_end_bone() const {
			return use_virtual_end ? mid_bone : end_bone; // Hack, but useful for external class such as TwoBoneIK3DGizmoPlugin.
		}

		Vector3 get_knuckle_direction_vector() const {
			Vector3 ret;
			switch (knuckle_direction) {
				case KNUCKLE_DIRECTION_NONE:
					ret = Vector3(0, 0, 0);
					break;
				case KNUCKLE_DIRECTION_PLUS_X:
					ret = Vector3(1, 0, 0);
					break;
				case KNUCKLE_DIRECTION_MINUS_X:
					ret = Vector3(-1, 0, 0);
					break;
				case KNUCKLE_DIRECTION_PLUS_Y:
					ret = Vector3(0, 1, 0);
					break;
				case KNUCKLE_DIRECTION_MINUS_Y:
					ret = Vector3(0, -1, 0);
					break;
				case KNUCKLE_DIRECTION_PLUS_Z:
					ret = Vector3(0, 0, 1);
					break;
				case KNUCKLE_DIRECTION_MINUS_Z:
					ret = Vector3(0, 0, -1);
					break;
				case KNUCKLE_DIRECTION_CUSTOM:
					ret = knuckle_direction_vector;
					break;
			}
			return ret;
		}

		void cache_current_vectors(Skeleton3D *p_skeleton) {
			if (!is_valid()) {
				return;
			}
			root_joint.solver_info->current_vector = (mid_pos - root_pos).normalized();
			mid_joint.solver_info->current_vector = (end_pos - mid_pos).normalized();
		}

		void init_current_joint_rotations(Skeleton3D *p_skeleton) {
			if (!is_valid()) {
				return;
			}

			Quaternion parent_gpose;
			int parent = p_skeleton->get_bone_parent(root_bone);
			if (parent >= 0) {
				parent_gpose = p_skeleton->get_bone_global_pose(parent).basis.get_rotation_quaternion();
			}
			root_joint.solver_info->current_lrest = p_skeleton->get_bone_rest(root_joint.bone).basis.get_rotation_quaternion();
			root_joint.solver_info->current_grest = parent_gpose * root_joint.solver_info->current_lrest;
			root_joint.solver_info->current_grest.normalize();
			root_joint.solver_info->current_lpose = p_skeleton->get_bone_pose(root_joint.bone).basis.get_rotation_quaternion();
			root_joint.solver_info->current_gpose = parent_gpose * root_joint.solver_info->current_lpose;
			root_joint.solver_info->current_gpose.normalize();
			parent_gpose = root_joint.solver_info->current_gpose;

			// Mid joint pose is relative to the root joint pose.
			mid_joint.solver_info->current_lrest = p_skeleton->get_bone_global_rest(root_joint.bone).basis.get_rotation_quaternion().inverse() * p_skeleton->get_bone_global_rest(mid_joint.bone).basis.get_rotation_quaternion();
			mid_joint.solver_info->current_grest = parent_gpose * mid_joint.solver_info->current_lrest;
			mid_joint.solver_info->current_grest.normalize();
			mid_joint.solver_info->current_lpose = p_skeleton->get_bone_global_pose(root_joint.bone).basis.get_rotation_quaternion().inverse() * p_skeleton->get_bone_global_pose(mid_joint.bone).basis.get_rotation_quaternion();
			mid_joint.solver_info->current_gpose = parent_gpose * mid_joint.solver_info->current_lpose;
			mid_joint.solver_info->current_gpose.normalize();

			cache_current_vectors(p_skeleton);
		}

		// Make rotation as bone pose from chain coordinates.
		void cache_current_joint_rotations(Skeleton3D *p_skeleton, Vector3 p_pole_destination) {
			if (!is_valid()) {
				return;
			}

			Quaternion parent_gpose;
			int parent = p_skeleton->get_bone_parent(root_bone);
			if (parent >= 0) {
				parent_gpose = p_skeleton->get_bone_global_pose(parent).basis.get_rotation_quaternion();
			}

			root_joint.solver_info->current_lrest = p_skeleton->get_bone_rest(root_joint.bone).basis.get_rotation_quaternion();
			root_joint.solver_info->current_grest = parent_gpose * root_joint.solver_info->current_lrest;
			root_joint.solver_info->current_grest.normalize();

			Vector3 from = root_joint.solver_info->forward_vector;
			Vector3 to = root_joint.solver_info->current_grest.xform_inv(root_joint.solver_info->current_vector).normalized();
			root_joint.solver_info->current_lpose = root_joint.solver_info->current_lrest * get_swing(Quaternion(from, to), from);

			root_joint.solver_info->current_gpose = parent_gpose * root_joint.solver_info->current_lpose;
			root_joint.solver_info->current_gpose.normalize();
			Quaternion root_gpose = root_joint.solver_info->current_gpose;

			// Mid joint pose is relative to the root joint pose for the case root-mid or mid-end have more than 1 joints.
			mid_joint.solver_info->current_lrest = p_skeleton->get_bone_global_rest(root_joint.bone).basis.get_rotation_quaternion().inverse() * p_skeleton->get_bone_global_rest(mid_joint.bone).basis.get_rotation_quaternion();
			mid_joint.solver_info->current_grest = root_gpose * mid_joint.solver_info->current_lrest;
			mid_joint.solver_info->current_grest.normalize();

			from = mid_joint.solver_info->forward_vector;
			to = mid_joint.solver_info->current_grest.xform_inv(mid_joint.solver_info->current_vector).normalized();
			mid_joint.solver_info->current_lpose = mid_joint.solver_info->current_lrest * get_swing(Quaternion(from, to), from);

			mid_joint.solver_info->current_gpose = root_gpose * mid_joint.solver_info->current_lpose;
			mid_joint.solver_info->current_gpose.normalize();

			bool is_knuckle_defined = knuckle_direction != KNUCKLE_DIRECTION_NONE && (knuckle_direction != KNUCKLE_DIRECTION_CUSTOM || !knuckle_direction_vector.is_zero_approx());
			// Fix roll to align knuckle vector to plane.
			if (is_knuckle_defined) {
				// Calc roll angles.
				Quaternion root_roll_rot = Quaternion();
				Quaternion mid_roll_rot = Quaternion();

				// Make roll to align knuckle_vector onto plane with selecting the point nearer pole_destination.
				Vector3 pole_dir = get_normal(root_pos, end_pos, p_pole_destination);
				if (pole_dir.is_zero_approx()) {
					return;
				}
				Vector3 a = mid_joint.solver_info->current_vector.normalized(); // Global roll axis (mid forward in current pose).
				Vector3 k = mid_joint.solver_info->current_gpose.xform(get_knuckle_direction_vector()).normalized(); // Global knuckle vector.
				Vector3 n = pole_dir.cross((mid_pos - root_pos).normalized()).normalized(); // Global plane normal.

				// Guard: degenerate cases (zero or already parallel)
				if (a.is_zero_approx() || k.is_zero_approx() || n.is_zero_approx() || Math::is_zero_approx(n.dot(k))) {
					return;
				}
				// c0 cosθ + c1 sinθ + c2 = 0
				real_t c0 = n.dot(k - a * k.dot(a)); // n·(k⊥a)
				real_t c1 = n.dot(a.cross(k)); // n·(a×k)
				real_t c2 = n.dot(a) * k.dot(a); // (n·a)(k·a)
				real_t r = Math::sqrt(c0 * c0 + c1 * c1);
				real_t cos_arg = CLAMP(-c2 / r, (real_t)-1.0, (real_t)1.0);
				real_t phi = Math::atan2(c1, c0);
				real_t acosv = Math::acos(cos_arg);

				// Two candidate angles.
				real_t t1 = phi + acosv;
				real_t t2 = phi - acosv;
				Quaternion q1(a, t1);
				Quaternion q2(a, t2);

				// Choose the one whose projected knuckle points closer to pole side.
				Vector3 pole_proj = snap_vector_to_plane(n, pole_dir).normalized();
				Vector3 k1p = snap_vector_to_plane(n, q1.xform(k)).normalized();
				Vector3 k2p = snap_vector_to_plane(n, q2.xform(k)).normalized();
				real_t s1 = pole_proj.is_zero_approx() ? Math::abs(t1) : k1p.dot(pole_proj);
				real_t s2 = pole_proj.is_zero_approx() ? Math::abs(t2) : k2p.dot(pole_proj);

				real_t t = s1 >= s2 ? t1 : t2;
				root_roll_rot = Quaternion(root_joint.solver_info->forward_vector, t);
				mid_roll_rot = Quaternion(mid_joint.solver_info->forward_vector, t);

				root_joint.solver_info->current_lpose = root_joint.solver_info->current_lpose * root_roll_rot;
				root_joint.solver_info->current_gpose = parent_gpose * root_joint.solver_info->current_lpose;
				root_joint.solver_info->current_gpose.normalize();
				root_gpose = root_joint.solver_info->current_gpose;

				mid_joint.solver_info->current_lpose = root_roll_rot.inverse() * mid_joint.solver_info->current_lpose * mid_roll_rot;
				mid_joint.solver_info->current_gpose = root_gpose * mid_joint.solver_info->current_lpose;
				mid_joint.solver_info->current_gpose.normalize();
			}
		}
	};

protected:
	Vector<TwoBoneIK3DSetting *> settings;

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_dynamic_prop(PropertyInfo &p_property) const;

	static void _bind_methods();

	virtual void _validate_bone_names() override;
	void _validate_knuckle_directions(Skeleton3D *p_skeleton) const;
	void _validate_knuckle_direction(Skeleton3D *p_skeleton, int p_index) const;

	void _make_all_joints_dirty() override;
	void _init_joints(Skeleton3D *p_skeleton, TwoBoneIK3DSetting *p_setting);
	void _update_joints(int p_index);

	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta) override;
	void _process_joints(double p_delta, Skeleton3D *p_skeleton, TwoBoneIK3DSetting *p_setting, const Vector3 &p_destination, const Vector3 &p_pole_destination);

public:
	void set_max_iterations(int p_max_iterations);
	int get_max_iterations() const;
	void set_min_distance(real_t p_min_distance);
	real_t get_min_distance() const;
	void set_angular_delta_limit(real_t p_angular_delta_limit);
	real_t get_angular_delta_limit() const;

	// Setting.
	void set_root_bone_name(int p_index, const String &p_bone_name);
	String get_root_bone_name(int p_index) const;
	void set_root_bone(int p_index, int p_bone);
	int get_root_bone(int p_index) const;

	void set_mid_bone_name(int p_index, const String &p_bone_name);
	String get_mid_bone_name(int p_index) const;
	void set_mid_bone(int p_index, int p_bone);
	int get_mid_bone(int p_index) const;

	void set_end_bone_name(int p_index, const String &p_bone_name);
	String get_end_bone_name(int p_index) const;
	void set_end_bone(int p_index, int p_bone);
	int get_end_bone(int p_index) const;

	void set_use_virtual_end(int p_index, bool p_enabled);
	bool is_using_virtual_end(int p_index) const;
	void set_extend_end_bone(int p_index, bool p_enabled);
	bool is_end_bone_extended(int p_index) const;
	void set_end_bone_direction(int p_index, BoneDirection p_bone_direction);
	BoneDirection get_end_bone_direction(int p_index) const;
	void set_end_bone_length(int p_index, float p_length);
	float get_end_bone_length(int p_index) const;

	void set_pole_node(int p_index, const NodePath &p_pole_node);
	NodePath get_pole_node(int p_index) const;

	void set_target_node(int p_index, const NodePath &p_target_node);
	NodePath get_target_node(int p_index) const;

	void set_knuckle_direction(int p_index, KnuckleDirection p_axis);
	KnuckleDirection get_knuckle_direction(int p_index) const;
	void set_knuckle_direction_vector(int p_index, const Vector3 &p_vector);
	Vector3 get_knuckle_direction_vector(int p_index) const;

	void set_setting_count(int p_count);
	int get_setting_count() const;
	void clear_settings();

	bool is_valid(int p_index) const; // Helper for editor and validation.

	// To process manually.
	virtual void reset() override;

	// Helper.
	static Vector3 get_normal(Vector3 p_a, Vector3 p_b, Vector3 p_point) {
		const Vector3 dir = p_b - p_a;
		const real_t denom = dir.length_squared();
		if (Math::is_zero_approx(denom)) {
			return Vector3();
		}
		const Vector3 w = p_point - p_a;
		const real_t t = w.dot(dir) / denom;
		const Vector3 h = p_a + dir * t;
		return (p_point - h).normalized();
	}

	~TwoBoneIK3D();
};

VARIANT_ENUM_CAST(TwoBoneIK3D::KnuckleDirection);
