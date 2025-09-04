/**************************************************************************/
/*  chain_ik_3d.h                                                         */
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

class ChainIK3D : public ManyBoneIK3D {
	GDCLASS(ChainIK3D, ManyBoneIK3D);

public:
	struct ChainIK3DSetting {
		bool joints_dirty = false;

		String root_bone_name;
		int root_bone = -1;

		String end_bone_name;
		int end_bone = -1;

		// To make virtual end joint.
		bool extend_end_bone = false;
		BoneDirection end_bone_direction = BONE_DIRECTION_FROM_PARENT;
		float end_bone_length = 0.0;

		NodePath pole_node;
		NodePath target_node;

		Vector<ManyBoneIK3DJointSetting *> joints;
		Vector<Vector3> chain;
		int joint_size_half = -1;
		int chain_size_half = -1;

		// To process.
		bool simulation_dirty = true;
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
				real_t proj = to_target.dot(chain_dir);
				real_t total_length = 0;
				for (int i = 0; i < joints.size(); i++) {
					if (joints[i]->solver_info) {
						total_length += joints[i]->solver_info->length;
					}
				}
				ret = proj >= 0 && proj <= total_length && (to_target.normalized().is_equal_approx(chain_dir));
			}
			return ret;
		}

		// Only update chain coordinates to avoid to override previous result (bone poses).
		// Chain coordinates will be converted to bone pose by cache_current_joint_rotations() in the end of iterating.
		void update_chain_coordinate(Skeleton3D *p_skeleton, int p_index, const Vector3 &p_position, bool p_backward = true) {
			// Don't update if the position is same as the current position.
			if (Math::is_zero_approx(chain[p_index].distance_squared_to(p_position))) {
				return;
			}

			// Prevent flipping.
			Vector3 result = p_position;
			if (p_backward) {
				int HEAD = p_index - 1;
				int TAIL = p_index;
				if (HEAD >= 0 && HEAD < joints.size()) {
					ManyBoneIK3DSolverInfo *solver_info = joints[HEAD]->solver_info;
					if (solver_info) {
						Vector3 old_head_to_tail = solver_info->current_vector;
						Vector3 new_head_to_tail = (result - chain[HEAD]).normalized();
						if (Math::is_equal_approx((double)old_head_to_tail.dot(new_head_to_tail), -1.0)) {
							chain.write[TAIL] = chain[HEAD] + old_head_to_tail * solver_info->length; // Revert.
							return; // No change, cache is not updated.
						}
					}
				}
			} else {
				int HEAD = p_index;
				int TAIL = p_index + 1;
				if (TAIL >= 0 && TAIL < joints.size()) {
					ManyBoneIK3DSolverInfo *solver_info = joints[HEAD]->solver_info;
					if (solver_info) {
						Vector3 old_head_to_tail = solver_info->current_vector;
						Vector3 new_head_to_tail = (chain[TAIL] - result).normalized();
						if (Math::is_equal_approx((double)old_head_to_tail.dot(new_head_to_tail), -1.0)) {
							chain.write[HEAD] = chain[TAIL] - old_head_to_tail * solver_info->length; // Revert.
							return; // No change, cache is not updated.
						}
					}
				}
			}

			chain.write[p_index] = result;
			cache_current_vector(p_skeleton, p_index);
		}

		void cache_current_vector(Skeleton3D *p_skeleton, int p_index) {
			int cur_head = p_index - 1;
			int cur_tail = p_index;
			if (cur_head >= 0) {
				joints[cur_head]->solver_info->current_vector = (chain[cur_tail] - chain[cur_head]).normalized();
			}
			cur_head = p_index;
			cur_tail = p_index + 1;
			if (cur_tail < chain.size()) {
				joints[cur_head]->solver_info->current_vector = (chain[cur_tail] - chain[cur_head]).normalized();
			}
		}

		void cache_current_vectors(Skeleton3D *p_skeleton) {
			for (int i = 0; i < joints.size(); i++) {
				int HEAD = i;
				int TAIL = i + 1;
				ManyBoneIK3DSolverInfo *solver_info = joints[HEAD]->solver_info;
				if (!solver_info) {
					continue;
				}
				solver_info->current_vector = (chain[TAIL] - chain[HEAD]).normalized();
			}
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
				solver_info->current_lrest = p_skeleton->get_bone_rest(joints[i]->bone).basis.get_rotation_quaternion();
				solver_info->current_grest = parent_gpose * solver_info->current_lrest;
				solver_info->current_grest.normalize();
				solver_info->current_lpose = p_skeleton->get_bone_pose(joints[i]->bone).basis.get_rotation_quaternion();
				solver_info->current_gpose = parent_gpose * solver_info->current_lpose;
				solver_info->current_gpose.normalize();
				parent_gpose = solver_info->current_gpose;
			}

			cache_current_vectors(p_skeleton);
		}

		// Make rotation as bone pose from chain coordinates.
		void cache_current_joint_rotations(Skeleton3D *p_skeleton, real_t p_angular_delta_limit = Math::PI) {
			Transform3D parent_gpose_tr;
			int parent = p_skeleton->get_bone_parent(root_bone);
			if (parent >= 0) {
				parent_gpose_tr = p_skeleton->get_bone_global_pose(parent);
			}
			Quaternion parent_gpose = parent_gpose_tr.basis.get_rotation_quaternion();

			for (int i = 0; i < joints.size(); i++) {
				int HEAD = i;
				ManyBoneIK3DSolverInfo *solver_info = joints[HEAD]->solver_info;
				if (!solver_info) {
					continue;
				}
				solver_info->current_lrest = p_skeleton->get_bone_rest(joints[HEAD]->bone).basis.get_rotation_quaternion();
				solver_info->current_grest = parent_gpose * solver_info->current_lrest;
				solver_info->current_grest.normalize();
				Vector3 from = solver_info->forward_vector;
				Vector3 to = solver_info->current_grest.xform_inv(solver_info->current_vector).normalized();
				Quaternion prev = solver_info->current_lpose;
				if (joints[HEAD]->rotation_axis == ROTATION_AXIS_ALL) {
					solver_info->current_lpose = solver_info->current_lrest * get_swing(Quaternion(from, to), from);
				} else {
					// To stabilize rotation path especially nearely 180deg.
					solver_info->current_lpose = solver_info->current_lrest * get_from_to_rotation_by_axis(from, to, joints[HEAD]->get_rotation_axis_vector().normalized());
				}
				solver_info->current_lpose = prev.slerp(solver_info->current_lpose, MIN(1.0, p_angular_delta_limit / prev.angle_to(solver_info->current_lpose)));
				solver_info->current_gpose = parent_gpose * solver_info->current_lpose;
				solver_info->current_gpose.normalize();
				parent_gpose = solver_info->current_gpose;
			}

			// Apply back angular_delta_limit to chain coordinates.
			if (chain.is_empty()) {
				return;
			}
			chain.write[0] = parent_gpose_tr.origin;
			for (int i = 0; i < joints.size(); i++) {
				int HEAD = i;
				int TAIL = i + 1;
				ManyBoneIK3DSolverInfo *solver_info = joints[HEAD]->solver_info;
				if (!solver_info) {
					continue;
				}
				chain.write[TAIL] = chain[HEAD] + solver_info->current_gpose.xform(solver_info->forward_vector) * solver_info->length;
			}
			cache_current_vectors(p_skeleton);
		}

		~ChainIK3DSetting() {
			for (int i = 0; i < joints.size(); i++) {
				memdelete(joints[i]);
			}
			joints.clear();
			chain.clear();
		}
	};

protected:
	int max_iterations = 4;
	real_t min_distance = 0.01; // If distance between end joint and target is less than min_distance, finish iteration.
	real_t min_distance_squared = min_distance * min_distance; // For cache.
	real_t angular_delta_limit = Math::deg_to_rad(2.0); // If the delta is too large, the results before and after iterating can change significantly, and divergence of calculations can easily occur.

	Vector<ChainIK3DSetting *> settings;

	bool _get(const StringName &p_path, Variant &r_ret) const;
	bool _set(const StringName &p_path, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_dynamic_prop(PropertyInfo &p_property) const;

	static void _bind_methods();

	virtual void _validate_bone_names() override;
	void _validate_rotation_axes(Skeleton3D *p_skeleton) const;
	void _validate_rotation_axis(Skeleton3D *p_skeleton, int p_index, int p_joint) const;

	virtual void _make_all_joints_dirty() override;
	void _init_joints(Skeleton3D *p_skeleton, ChainIK3DSetting *p_setting);
	void _update_joints(int p_index);

	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta) override;
	void _process_joints(double p_delta, Skeleton3D *p_skeleton, ChainIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_target_destination, const Vector3 &p_pole_destination, bool p_use_pole);
	virtual void _solve_iteration_with_pole(double p_delta, Skeleton3D *p_skeleton, ChainIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size, const Vector3 &p_pole_destination, int p_joint_size_half, int p_chain_size_half);
	virtual void _solve_iteration(double p_delta, Skeleton3D *p_skeleton, ChainIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size);

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

	void set_pole_node(int p_index, const NodePath &p_pole_node);
	NodePath get_pole_node(int p_index) const;

	void set_target_node(int p_index, const NodePath &p_target_node);
	NodePath get_target_node(int p_index) const;

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

	// To process manually.
	virtual void reset() override;

	~ChainIK3D();
};
