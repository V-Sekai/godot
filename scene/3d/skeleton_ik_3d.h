/**************************************************************************/
/*  skeleton_ik_3d.h                                                      */
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

#ifndef SKELETON_IK_3D_H
#define SKELETON_IK_3D_H

#ifndef _3D_DISABLED

#include "scene/3d/skeleton_3d.h"
#include "core/math/qcp.h"

struct InverseKinematicChain : public Resource {
	GDCLASS(InverseKinematicChain, Resource);

public:
	struct Joint {
		Quaternion rotation;
		BoneId id;
		Vector3 relative_prev;
		Vector3 relative_next;
		float prev_distance = 0;
		float next_distance = 0;

		float root_influence = 0;
		float leaf_influence = 0;
		float twist_influence = 1;
	};

private:
	BoneId root_bone = -1;
	BoneId first_bone = -1;
	BoneId leaf_bone = -1;

	Vector<Joint> joints;
	float total_length = 0;
	Transform3D rest_leaf;
	void init_chain(Skeleton3D *p_skeleton);
	float root_influence =
			0; // how much the start bone is influenced by the root rotation
	float leaf_influence =
			0; // how much the end bone is influenced by the goal rotation
	float twist_influence =
			1; // How much the chain tries to twist to follow the end when the start
			   // is facing a different direction
	float twist_start = 0; // Where along the chain the twisting starts

public:
	void init(Vector3 p_chain_curve_direction, float p_root_influence,
			float p_leaf_influence, float p_twist_influence,
			float p_twist_start);
	void set_root_bone(Skeleton3D *skeleton, BoneId p_root_bone);
	void set_leaf_bone(Skeleton3D *skeleton, BoneId p_leaf_bone);
	bool is_valid();
	Vector3 chain_curve_direction; // This defines which way to prebend it
	float get_total_length();
	Vector<InverseKinematicChain::Joint> get_joints();
	Transform3D get_relative_rest_leaf();
	BoneId get_first_bone();
	BoneId get_root_bone();
	BoneId get_leaf_bone();

	float get_root_stiffness();
	void set_root_stiffness(Skeleton3D *p_skeleton, float p_stiffness);
	float get_leaf_stiffness();
	void set_leaf_stiffness(Skeleton3D *p_skeleton, float p_stiffness);
	float get_twist();
	void set_twist(Skeleton3D *p_skeleton, float p_twist);
	float get_twist_start();
	void set_twist_start(Skeleton3D *p_skeleton, float p_twist_start);
	bool contains_bone(Skeleton3D *p_skeleton, BoneId p_bone);
};


void InverseKinematicChain::init(Vector3 p_chain_curve_direction, float p_root_influence,
		float p_leaf_influence, float p_twist_influence,
		float p_twist_start) {
	chain_curve_direction = p_chain_curve_direction;
	root_influence = p_root_influence;
	leaf_influence = p_leaf_influence;
	twist_influence = p_twist_influence;
	twist_start = p_twist_start;
}

void InverseKinematicChain::init_chain(Skeleton3D *p_skeleton) {
	joints.clear();
	total_length = 0;
	if (p_skeleton && root_bone >= 0 && leaf_bone >= 0 &&
			root_bone < p_skeleton->get_bone_count() &&
			leaf_bone < p_skeleton->get_bone_count()) {
		BoneId bone = p_skeleton->get_bone_parent(leaf_bone);
		// generate the chain of bones
		Vector<BoneId> chain;
		float last_length = 0.0f;
		rest_leaf = p_skeleton->get_bone_rest(leaf_bone);
		while (bone != root_bone) {
			Transform3D rest_pose = p_skeleton->get_bone_rest(bone);
			rest_leaf = rest_pose * rest_leaf.orthonormalized();
			last_length = rest_pose.origin.length();
			total_length += last_length;
			if (bone < 0) { // invalid chain
				total_length = 0;
				first_bone = -1;
				rest_leaf = Transform3D();
				return;
			}
			chain.push_back(bone);
			first_bone = bone;
			bone = p_skeleton->get_bone_parent(bone);
		}
		total_length -= last_length;
		total_length += p_skeleton->get_bone_rest(leaf_bone).origin.length();

		if (total_length <= 0) { // invalid chain
			total_length = 0;
			first_bone = -1;
			rest_leaf = Transform3D();
			return;
		}

		Basis totalRotation;
		float progress = 0;
		// flip the order and figure out the relative distances of these joints
		for (int i = chain.size() - 1; i >= 0; i--) {
			InverseKinematicChain::Joint j;
			j.id = chain[i];
			Transform3D boneTransform = p_skeleton->get_bone_rest(j.id);
			j.rotation = boneTransform.basis.get_rotation_quaternion();
			j.relative_prev = totalRotation.xform_inv(boneTransform.origin);
			j.prev_distance = j.relative_prev.length();

			// calc influences
			progress += j.prev_distance;
			float percentage = (progress / total_length);
			float effectiveRootInfluence =
					root_influence <= 0 || percentage >= root_influence
					? 0
					: (percentage - root_influence) / -root_influence;
			float effectiveLeafInfluence =
					leaf_influence <= 0 || percentage <= 1 - leaf_influence
					? 0
					: (percentage - (1 - leaf_influence)) / leaf_influence;
			float effectiveTwistInfluence =
					twist_start >= 1 || twist_influence <= 0 || percentage <= twist_start
					? 0
					: (percentage - twist_start) *
							(twist_influence / (1 - twist_start));
			j.root_influence =
					effectiveRootInfluence > 1 ? 1 : effectiveRootInfluence;
			j.leaf_influence =
					effectiveLeafInfluence > 1 ? 1 : effectiveLeafInfluence;
			j.twist_influence =
					effectiveTwistInfluence > 1 ? 1 : effectiveTwistInfluence;

			if (!joints.is_empty()) {
				InverseKinematicChain::Joint oldJ = joints[joints.size() - 1];
				oldJ.relative_next = -j.relative_prev;
				oldJ.next_distance = j.prev_distance;
				joints.set(joints.size() - 1, oldJ);
			}
			joints.push_back(j);
			totalRotation = (totalRotation * boneTransform.basis).orthonormalized();
		}
		if (!joints.is_empty()) {
			InverseKinematicChain::Joint oldJ = joints[joints.size() - 1];
			oldJ.relative_next = -p_skeleton->get_bone_rest(leaf_bone).origin;
			oldJ.next_distance = oldJ.relative_next.length();
			joints.set(joints.size() - 1, oldJ);
		}
	}
}

void InverseKinematicChain::set_root_bone(Skeleton3D *skeleton, BoneId p_root_bone) {
	root_bone = p_root_bone;
	init_chain(skeleton);
}
void InverseKinematicChain::set_leaf_bone(Skeleton3D *skeleton, BoneId p_leaf_bone) {
	leaf_bone = p_leaf_bone;
	init_chain(skeleton);
}

bool InverseKinematicChain::is_valid() { return !joints.is_empty(); }

float InverseKinematicChain::get_total_length() { return total_length; }

Vector<InverseKinematicChain::Joint> InverseKinematicChain::get_joints() { return joints; }

Transform3D InverseKinematicChain::get_relative_rest_leaf() { return rest_leaf; }

BoneId InverseKinematicChain::get_first_bone() { return first_bone; }

BoneId InverseKinematicChain::get_root_bone() { return root_bone; }

BoneId InverseKinematicChain::get_leaf_bone() { return leaf_bone; }

float InverseKinematicChain::get_root_stiffness() { return root_influence; }

void InverseKinematicChain::set_root_stiffness(Skeleton3D *p_skeleton, float p_stiffness) {
	root_influence = p_stiffness;
	init_chain(p_skeleton);
}

float InverseKinematicChain::get_leaf_stiffness() { return leaf_influence; }

void InverseKinematicChain::set_leaf_stiffness(Skeleton3D *p_skeleton, float p_stiffness) {
	leaf_influence = p_stiffness;
	init_chain(p_skeleton);
}

float InverseKinematicChain::get_twist() { return twist_influence; }

void InverseKinematicChain::set_twist(Skeleton3D *p_skeleton, float p_twist) {
	twist_influence = p_twist;
	init_chain(p_skeleton);
}

float InverseKinematicChain::get_twist_start() { return twist_start; }

void InverseKinematicChain::set_twist_start(Skeleton3D *p_skeleton, float p_twist_start) {
	twist_start = p_twist_start;
	init_chain(p_skeleton);
}

bool InverseKinematicChain::contains_bone(Skeleton3D *p_skeleton, BoneId p_bone) {
	if (p_skeleton) {
		BoneId spineBone = leaf_bone;
		while (spineBone >= 0) {
			if (spineBone == p_bone) {
				return true;
			}
			spineBone = p_skeleton->get_bone_parent(spineBone);
		}
	}
	return false;
}

class FabrikInverseKinematic {
	struct EndEffector {
		BoneId tip_bone;
		Transform3D goal_transform;
	};

	struct ChainItem {
		Vector<ChainItem> children;
		ChainItem *parent_item = nullptr;

		// Bone info
		BoneId bone = -1;

		real_t length = 0.0;
		/// Positions relative to root bone
		Transform3D initial_transform;
		Vector3 current_pos;
		// Direction from this bone to child
		Vector3 current_ori;

		ChainItem *find_child(const BoneId p_bone_id);
		ChainItem *add_child(const BoneId p_bone_id);
	};

	struct ChainTip {
		ChainItem *chain_item = nullptr;
		const EndEffector *end_effector = nullptr;

		ChainTip() {}

		ChainTip(ChainItem *p_chain_item, const EndEffector *p_end_effector) :
				chain_item(p_chain_item),
				end_effector(p_end_effector) {}
	};

	struct Chain {
		ChainItem chain_root;
		ChainItem *middle_chain_item = nullptr;
		Vector<ChainTip> tips;
		Vector3 magnet_position;
	};

public:
	struct Task {
		RID self;
		Skeleton3D *skeleton = nullptr;

		Chain chain;

		// Settings
		real_t min_distance = 0.01;
		int max_iterations = 10;

		// Bone data
		BoneId root_bone = -1;
		Vector<EndEffector> end_effectors;

		Transform3D goal_global_transform;

		Task() {}
	};
private:
	/// Init a chain that starts from the root to tip
	static bool build_chain(Task *p_task, bool p_force_simple_chain = true);

	static void solve_simple(Task *p_task, bool p_solve_magnet, Vector3 p_origin_pos);
	/// Special solvers that solve only chains with one end effector
	static void solve_simple_backwards(const Chain &r_chain, bool p_solve_magnet);
	static void solve_simple_forwards(Chain &r_chain, bool p_solve_magnet, Vector3 p_origin_pos);

	Vector<Transform3D> compute_global_transforms(const Vector<InverseKinematicChain::Joint> &joints, const Transform3D &root, const Transform3D &true_root) {
		Vector<Transform3D> global_transforms;
		global_transforms.resize(joints.size());
		Transform3D current_global_transform = true_root;

		for (int i = 0; i < joints.size(); i++) {
			Transform3D local_transform;
			local_transform.basis = Basis(joints[i].rotation);
			if (i == 0) {
				local_transform.origin = root.origin;
			} else {
				local_transform.origin = joints[i - 1].relative_next;
			}
			current_global_transform *= local_transform;
			global_transforms.write[i] = current_global_transform;
		}

		return global_transforms;
	}

	void compute_rest_and_target_positions(const Vector<Transform3D> &p_global_transforms, const Transform3D &p_target, const Vector3 &p_priority, Vector<Vector3> &p_reference_positions, Vector<Vector3> &p_target_positions, Vector<real_t> &r_weights) {
		for (int joint_i = 0; joint_i < p_global_transforms.size(); joint_i++) {
			Transform3D bone_direction_global_transform = p_global_transforms[joint_i];
			real_t pin_weight = r_weights[joint_i];
			int32_t rest_index = joint_i * 7;

			Basis tip_basis = bone_direction_global_transform.basis.orthogonalized();

			Quaternion quaternion = tip_basis.get_rotation_quaternion();
			tip_basis.set_quaternion_scale(quaternion, tip_basis.get_scale());

			p_reference_positions.write[rest_index] = p_target.origin - bone_direction_global_transform.origin;
			rest_index++;

			double epsilon = 1e-6;

			Vector3 target_global_space = p_target.origin;
			if (!quaternion.is_equal_approx(Quaternion())) {
				target_global_space = bone_direction_global_transform.xform(p_target.origin);
			}
			double distance = target_global_space.distance_to(bone_direction_global_transform.origin);
			double scale_by = MAX(1.0, distance);

			for (int axis_i = Vector3::AXIS_X; axis_i <= Vector3::AXIS_Z; ++axis_i) {
				if (p_priority[axis_i] > 0.0) {
					real_t w = r_weights[rest_index];
					Vector3 column = p_target.basis.get_column(axis_i);
					p_reference_positions.write[rest_index] = bone_direction_global_transform.affine_inverse().xform((column + p_target.origin) - bone_direction_global_transform.origin);
					p_reference_positions.write[rest_index] *= Vector3(w, w, w) * scale_by;
					rest_index++;
					p_reference_positions.write[rest_index] = bone_direction_global_transform.affine_inverse().xform((p_target.origin - column) - bone_direction_global_transform.origin);
					p_reference_positions.write[rest_index] *= Vector3(w, w, w) * scale_by;
					rest_index++;
				}
			}

			int32_t target_index = joint_i * 7;
			p_target_positions.write[target_index] = p_target.origin - bone_direction_global_transform.origin;
			target_index++;

			scale_by = pin_weight;

			Vector3 target_local = bone_direction_global_transform.affine_inverse().xform(target_global_space);
			distance = target_local.distance_to(bone_direction_global_transform.origin);

			scale_by *= 1.0 / (distance * distance + epsilon);

			for (int axis_j = Vector3::AXIS_X; axis_j <= Vector3::AXIS_Z; ++axis_j) {
				if (p_priority[axis_j] > 0.0) {
					Vector3 column = tip_basis.get_column(axis_j) * p_priority[axis_j];
					p_target_positions.write[target_index] = bone_direction_global_transform.xform((column + p_target.origin) - bone_direction_global_transform.origin);
					p_target_positions.write[target_index] *= scale_by;
					target_index++;
					p_target_positions.write[target_index] = bone_direction_global_transform.xform((p_target.origin - column) - bone_direction_global_transform.origin);
					p_target_positions.write[target_index] *= scale_by;
					target_index++;
				}
			}
		}
	}

	HashMap<BoneId, Quaternion> solve_ik_qcp(Ref<InverseKinematicChain> chain,
			Transform3D root,
			Transform3D target) {
		HashMap<BoneId, Quaternion> map;

		if (!chain->is_valid()) {
			return map;
		}

		Vector<InverseKinematicChain::Joint> joints = chain->get_joints();
		const Transform3D true_root = root.translated_local(joints[0].relative_prev);
		Vector<Vector3> rest_positions;
		Vector<Vector3> target_positions;
		Vector<double> weights;
		constexpr int TRANSFORM_TO_HEADINGS_COUNT = 7;
		rest_positions.resize(TRANSFORM_TO_HEADINGS_COUNT * joints.size());
		target_positions.resize(TRANSFORM_TO_HEADINGS_COUNT * joints.size());
		weights.resize(TRANSFORM_TO_HEADINGS_COUNT * joints.size());
		weights.fill(1.0);
		const Vector3 priority = Vector3(0.2, 0, 0.2);

		Vector<Transform3D> global_transforms = compute_global_transforms(joints, root, true_root);

		static constexpr double evec_prec = static_cast<double>(1E-6);
		QCP qcp = QCP(evec_prec);

		compute_rest_and_target_positions(global_transforms, target, priority, rest_positions, target_positions, weights);

		for (int joint_i = 0; joint_i < joints.size(); joint_i++) {
			Quaternion solved_global_pose = qcp.weighted_superpose(rest_positions, target_positions, weights, false);

			int parent_index = joint_i > 0 ? joint_i - 1 : 0;
			const Basis new_rot = global_transforms[parent_index].basis;

			const Quaternion local_pose = new_rot.inverse() * solved_global_pose * new_rot;
			map.insert(joints[joint_i].id, local_pose);

			global_transforms.write[joint_i] = global_transforms[parent_index] * Transform3D(Basis(local_pose));
		}

		return map;
	}

public:
	static Task *create_simple_task(Skeleton3D *p_sk, BoneId root_bone, BoneId tip_bone, const Transform3D &goal_transform);
	static void free_task(Task *p_task);
	// The goal of chain should be always in local space
	static void set_goal(Task *p_task, const Transform3D &p_goal);
	static void make_goal(Task *p_task, const Transform3D &p_inverse_transf, real_t blending_delta);
	static void solve(Task *p_task, real_t blending_delta, bool override_tip_basis, bool p_use_magnet, const Vector3 &p_magnet_position);

	static void _update_chain(const Skeleton3D *p_skeleton, ChainItem *p_chain_item);
};

class SkeletonIK3D : public Node {
	GDCLASS(SkeletonIK3D, Node);

	StringName root_bone;
	StringName tip_bone;
	real_t interpolation = 1.0;
	Transform3D target;
	NodePath target_node_path_override;
	bool override_tip_basis = true;
	bool use_magnet = false;
	Vector3 magnet_position;

	real_t min_distance = 0.01;
	int max_iterations = 10;

	Variant skeleton_ref = Variant();
	Variant target_node_override_ref = Variant();
	FabrikInverseKinematic::Task *task = nullptr;

protected:
	void _validate_property(PropertyInfo &p_property) const;

	static void _bind_methods();
	virtual void _notification(int p_what);

public:
	SkeletonIK3D();
	virtual ~SkeletonIK3D();

	void set_root_bone(const StringName &p_root_bone);
	StringName get_root_bone() const;

	void set_tip_bone(const StringName &p_tip_bone);
	StringName get_tip_bone() const;

	void set_interpolation(real_t p_interpolation);
	real_t get_interpolation() const;

	void set_target_transform(const Transform3D &p_target);
	const Transform3D &get_target_transform() const;

	void set_target_node(const NodePath &p_node);
	NodePath get_target_node();

	void set_override_tip_basis(bool p_override);
	bool is_override_tip_basis() const;

	void set_use_magnet(bool p_use);
	bool is_using_magnet() const;

	void set_magnet_position(const Vector3 &p_local_position);
	const Vector3 &get_magnet_position() const;

	void set_min_distance(real_t p_min_distance);
	real_t get_min_distance() const { return min_distance; }

	void set_max_iterations(int p_iterations);
	int get_max_iterations() const { return max_iterations; }

	Skeleton3D *get_parent_skeleton() const;

	bool is_running();

	void start(bool p_one_time = false);
	void stop();

private:
	Transform3D _get_target_transform();
	void reload_chain();
	void reload_goal();
	void _solve_chain();
};

#endif // _3D_DISABLED

#endif // SKELETON_IK_3D_H
