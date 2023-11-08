/**************************************************************************/
/*  skeleton_ik_3d.cpp                                                    */
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

#include "skeleton_ik_3d.h"
#include "core/math/quaternion.h"
#include "scene/3d/skeleton_3d.h"

#ifndef _3D_DISABLED

FabrikInverseKinematic::ChainItem *FabrikInverseKinematic::ChainItem::find_child(const BoneId p_bone_id) {
	for (int i = children.size() - 1; 0 <= i; --i) {
		if (p_bone_id == children[i].bone) {
			return &children.write[i];
		}
	}
	return nullptr;
}

FabrikInverseKinematic::ChainItem *FabrikInverseKinematic::ChainItem::add_child(const BoneId p_bone_id) {
	const int infant_child_id = children.size();
	children.resize(infant_child_id + 1);
	children.write[infant_child_id].bone = p_bone_id;
	children.write[infant_child_id].parent_item = this;
	return &children.write[infant_child_id];
}

/// Build a chain that starts from the root to tip
bool FabrikInverseKinematic::build_chain(Task *p_task, bool p_force_simple_chain) {
	ERR_FAIL_COND_V(-1 == p_task->root_bone, false);

	Chain &chain(p_task->chain);

	chain.tips.resize(p_task->end_effectors.size());
	chain.chain_root.bone = p_task->root_bone;
	chain.chain_root.initial_transform = p_task->skeleton->get_bone_global_pose(chain.chain_root.bone);
	chain.chain_root.current_pos = chain.chain_root.initial_transform.origin;
	chain.middle_chain_item = nullptr;

	// Holds all IDs that are composing a single chain in reverse order
	Vector<BoneId> chain_ids;
	// This is used to know the chain size
	int sub_chain_size;
	// Resize only one time in order to fit all joints for performance reason
	chain_ids.resize(p_task->skeleton->get_bone_count());

	for (int x = p_task->end_effectors.size() - 1; 0 <= x; --x) {
		const EndEffector *ee(&p_task->end_effectors[x]);
		ERR_FAIL_COND_V(p_task->root_bone >= ee->tip_bone, false);
		ERR_FAIL_INDEX_V(ee->tip_bone, p_task->skeleton->get_bone_count(), false);

		sub_chain_size = 0;
		// Picks all IDs that composing a single chain in reverse order (except the root)
		BoneId chain_sub_tip(ee->tip_bone);
		while (chain_sub_tip > p_task->root_bone) {
			chain_ids.write[sub_chain_size++] = chain_sub_tip;
			chain_sub_tip = p_task->skeleton->get_bone_parent(chain_sub_tip);
		}

		BoneId middle_chain_item_id = (BoneId)(sub_chain_size * 0.5);

		// Build chain by reading chain ids in reverse order
		// For each chain item id will be created a ChainItem if doesn't exists
		ChainItem *sub_chain(&chain.chain_root);
		for (int i = sub_chain_size - 1; 0 <= i; --i) {
			ChainItem *child_ci(sub_chain->find_child(chain_ids[i]));
			if (!child_ci) {
				child_ci = sub_chain->add_child(chain_ids[i]);

				child_ci->initial_transform = p_task->skeleton->get_bone_global_pose(child_ci->bone);
				child_ci->current_pos = child_ci->initial_transform.origin;

				if (child_ci->parent_item) {
					child_ci->length = child_ci->parent_item->current_pos.distance_to(child_ci->current_pos);
				}
			}

			sub_chain = child_ci;

			if (middle_chain_item_id == i) {
				chain.middle_chain_item = child_ci;
			}
		}

		if (!middle_chain_item_id) {
			chain.middle_chain_item = nullptr;
		}

		// Initialize current tip
		chain.tips.write[x].chain_item = sub_chain;
		chain.tips.write[x].end_effector = ee;

		if (p_force_simple_chain) {
			// NOTE:
			//	This is a "hack" that force to create only one tip per chain since the solver of multi tip (end effector)
			//	is not yet created.
			//	Remove this code when this is done
			break;
		}
	}
	return true;
}

void FabrikInverseKinematic::solve_simple(Task *p_task, bool p_solve_magnet, Vector3 p_origin_pos) {
	real_t distance_to_goal(1e4);
	real_t previous_distance_to_goal(0);
	int can_solve(p_task->max_iterations);
	Ref<InverseKinematicChain> inverse_kinematic_chain;
	{
		FabrikInverseKinematic::Chain chain = p_task->chain;
		if (!chain.tips.size()) {
			return;
		}
		inverse_kinematic_chain.instantiate();
		inverse_kinematic_chain->set_root_bone(p_task->skeleton, chain.chain_root.bone);
		for (int i = 0; i < chain.tips.size(); ++i) {
			inverse_kinematic_chain->set_leaf_bone(p_task->skeleton, chain.tips[i].chain_item->bone);
		}
		inverse_kinematic_chain->init(Vector3(0, 15, -15), 0.5, 0.5, 1, 0);
	}
	Transform3D root = p_task->skeleton->get_bone_global_pose(inverse_kinematic_chain->get_root_bone());
	Transform3D target = p_task->chain.tips[0].end_effector->goal_transform;
	while (distance_to_goal > p_task->min_distance && Math::abs(previous_distance_to_goal - distance_to_goal) > 0.005 && can_solve) {
		previous_distance_to_goal = distance_to_goal;
		--can_solve;

		HashMap<BoneId, Quaternion> poses = solve_ik_qcp(inverse_kinematic_chain, root, target);
		Vector<InverseKinematicChain::Joint> joints = inverse_kinematic_chain->get_joints();
		for (int i = joints.size(); i-- > 0;) {
			InverseKinematicChain::Joint joint = joints[i];
			Quaternion rotation = poses[joint.id];
			p_task->skeleton->set_bone_pose_rotation(joint.id, rotation);
		}

		distance_to_goal = target.origin.distance_to(p_task->skeleton->get_bone_global_pose(inverse_kinematic_chain->get_root_bone()).origin);
	}
}

FabrikInverseKinematic::Task *FabrikInverseKinematic::create_simple_task(Skeleton3D *p_sk, BoneId root_bone, BoneId tip_bone, const Transform3D &goal_transform) {
	FabrikInverseKinematic::EndEffector ee;
	ee.tip_bone = tip_bone;

	Task *task(memnew(Task));
	task->skeleton = p_sk;
	task->root_bone = root_bone;
	task->end_effectors.push_back(ee);
	task->goal_global_transform = goal_transform;

	if (!build_chain(task)) {
		free_task(task);
		return nullptr;
	}

	return task;
}

void FabrikInverseKinematic::free_task(Task *p_task) {
	if (p_task) {
		memdelete(p_task);
	}
}

void FabrikInverseKinematic::set_goal(Task *p_task, const Transform3D &p_goal) {
	p_task->goal_global_transform = p_goal;
}

void FabrikInverseKinematic::make_goal(Task *p_task, const Transform3D &p_inverse_transf, real_t blending_delta) {
	if (blending_delta >= 0.99f) {
		// Update the end_effector (local transform) without blending
		p_task->end_effectors.write[0].goal_transform = p_inverse_transf * p_task->goal_global_transform;
	} else {
		// End effector in local transform
		const Transform3D end_effector_pose(p_task->skeleton->get_bone_global_pose_no_override(p_task->end_effectors[0].tip_bone));

		// Update the end_effector (local transform) by blending with current pose
		p_task->end_effectors.write[0].goal_transform = end_effector_pose.interpolate_with(p_inverse_transf * p_task->goal_global_transform, blending_delta);
	}
}

void FabrikInverseKinematic::solve(Task *p_task, real_t blending_delta, bool override_tip_basis, bool p_use_magnet, const Vector3 &p_magnet_position) {
	if (blending_delta <= 0.01f) {
		// Before skipping, make sure we undo the global pose overrides
		ChainItem *ci(&p_task->chain.chain_root);
		while (ci) {
			p_task->skeleton->set_bone_global_pose_override(ci->bone, ci->initial_transform, 0.0, false);

			if (!ci->children.is_empty()) {
				ci = &ci->children.write[0];
			} else {
				ci = nullptr;
			}
		}

		return; // Skip solving
	}

	// Update the initial root transform so its synced with any animation changes
	_update_chain(p_task->skeleton, &p_task->chain.chain_root);

	p_task->skeleton->set_bone_global_pose_override(p_task->chain.chain_root.bone, Transform3D(), 0.0, false);
	Vector3 origin_pos = p_task->skeleton->get_bone_global_pose(p_task->chain.chain_root.bone).origin;

	make_goal(p_task, p_task->skeleton->get_global_transform().affine_inverse(), blending_delta);

	if (p_use_magnet && p_task->chain.middle_chain_item) {
		p_task->chain.magnet_position = p_task->chain.middle_chain_item->initial_transform.origin.lerp(p_magnet_position, blending_delta);
		solve_simple(p_task, true, origin_pos);
	}
	solve_simple(p_task, false, origin_pos);

	// Assign new bone position.
	ChainItem *ci(&p_task->chain.chain_root);
	while (ci) {
		Transform3D new_bone_pose(ci->initial_transform);
		new_bone_pose.origin = ci->current_pos;

		if (!ci->children.is_empty()) {
			Vector3 forward_vector = (ci->children[0].initial_transform.origin - ci->initial_transform.origin).normalized();
			// Rotate the bone towards the next bone in the chain:
			new_bone_pose.basis.rotate_to_align(forward_vector, new_bone_pose.origin.direction_to(ci->children[0].current_pos));

		} else {
			// Set target orientation to tip
			if (override_tip_basis) {
				new_bone_pose.basis = p_task->chain.tips[0].end_effector->goal_transform.basis;
			} else {
				new_bone_pose.basis = new_bone_pose.basis * p_task->chain.tips[0].end_effector->goal_transform.basis;
			}
		}

		// IK should not affect scale, so undo any scaling
		new_bone_pose.basis.orthonormalize();
		new_bone_pose.basis.scale(p_task->skeleton->get_bone_global_pose(ci->bone).basis.get_scale());

		p_task->skeleton->set_bone_global_pose_override(ci->bone, new_bone_pose, 1.0, true);

		if (!ci->children.is_empty()) {
			ci = &ci->children.write[0];
		} else {
			ci = nullptr;
		}
	}
}

void FabrikInverseKinematic::_update_chain(const Skeleton3D *p_sk, ChainItem *p_chain_item) {
	if (!p_chain_item) {
		return;
	}

	p_chain_item->initial_transform = p_sk->get_bone_global_pose_no_override(p_chain_item->bone);
	p_chain_item->current_pos = p_chain_item->initial_transform.origin;

	ChainItem *items = p_chain_item->children.ptrw();
	for (int i = 0; i < p_chain_item->children.size(); i += 1) {
		_update_chain(p_sk, items + i);
	}
}

void SkeletonIK3D::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "root_bone" || p_property.name == "tip_bone") {
		Skeleton3D *skeleton = get_parent_skeleton();
		if (skeleton) {
			String names("--,");
			for (int i = 0; i < skeleton->get_bone_count(); i++) {
				if (i > 0) {
					names += ",";
				}
				names += skeleton->get_bone_name(i);
			}

			p_property.hint = PROPERTY_HINT_ENUM;
			p_property.hint_string = names;
		} else {
			p_property.hint = PROPERTY_HINT_NONE;
			p_property.hint_string = "";
		}
	}
}

void SkeletonIK3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_bone", "root_bone"), &SkeletonIK3D::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone"), &SkeletonIK3D::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_tip_bone", "tip_bone"), &SkeletonIK3D::set_tip_bone);
	ClassDB::bind_method(D_METHOD("get_tip_bone"), &SkeletonIK3D::get_tip_bone);

	ClassDB::bind_method(D_METHOD("set_interpolation", "interpolation"), &SkeletonIK3D::set_interpolation);
	ClassDB::bind_method(D_METHOD("get_interpolation"), &SkeletonIK3D::get_interpolation);

	ClassDB::bind_method(D_METHOD("set_target_transform", "target"), &SkeletonIK3D::set_target_transform);
	ClassDB::bind_method(D_METHOD("get_target_transform"), &SkeletonIK3D::get_target_transform);

	ClassDB::bind_method(D_METHOD("set_target_node", "node"), &SkeletonIK3D::set_target_node);
	ClassDB::bind_method(D_METHOD("get_target_node"), &SkeletonIK3D::get_target_node);

	ClassDB::bind_method(D_METHOD("set_override_tip_basis", "override"), &SkeletonIK3D::set_override_tip_basis);
	ClassDB::bind_method(D_METHOD("is_override_tip_basis"), &SkeletonIK3D::is_override_tip_basis);

	ClassDB::bind_method(D_METHOD("set_use_magnet", "use"), &SkeletonIK3D::set_use_magnet);
	ClassDB::bind_method(D_METHOD("is_using_magnet"), &SkeletonIK3D::is_using_magnet);

	ClassDB::bind_method(D_METHOD("set_magnet_position", "local_position"), &SkeletonIK3D::set_magnet_position);
	ClassDB::bind_method(D_METHOD("get_magnet_position"), &SkeletonIK3D::get_magnet_position);

	ClassDB::bind_method(D_METHOD("get_parent_skeleton"), &SkeletonIK3D::get_parent_skeleton);
	ClassDB::bind_method(D_METHOD("is_running"), &SkeletonIK3D::is_running);

	ClassDB::bind_method(D_METHOD("set_min_distance", "min_distance"), &SkeletonIK3D::set_min_distance);
	ClassDB::bind_method(D_METHOD("get_min_distance"), &SkeletonIK3D::get_min_distance);

	ClassDB::bind_method(D_METHOD("set_max_iterations", "iterations"), &SkeletonIK3D::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations"), &SkeletonIK3D::get_max_iterations);

	ClassDB::bind_method(D_METHOD("start", "one_time"), &SkeletonIK3D::start, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("stop"), &SkeletonIK3D::stop);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "root_bone"), "set_root_bone", "get_root_bone");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "tip_bone"), "set_tip_bone", "get_tip_bone");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "interpolation", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_interpolation", "get_interpolation");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "target", PROPERTY_HINT_NONE, "suffix:m"), "set_target_transform", "get_target_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "override_tip_basis"), "set_override_tip_basis", "is_override_tip_basis");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_magnet"), "set_use_magnet", "is_using_magnet");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "magnet", PROPERTY_HINT_NONE, "suffix:m"), "set_magnet_position", "get_magnet_position");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target_node"), "set_target_node", "get_target_node");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_distance", PROPERTY_HINT_NONE, "suffix:m"), "set_min_distance", "get_min_distance");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_iterations"), "set_max_iterations", "get_max_iterations");
}

void SkeletonIK3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			skeleton_ref = Object::cast_to<Skeleton3D>(get_parent());
			set_process_priority(1);
			reload_chain();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (target_node_override_ref) {
				reload_goal();
			}
			_solve_chain();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			stop();
		} break;
	}
}

SkeletonIK3D::SkeletonIK3D() {
}

SkeletonIK3D::~SkeletonIK3D() {
	FabrikInverseKinematic::free_task(task);
	task = nullptr;
}

void SkeletonIK3D::set_root_bone(const StringName &p_root_bone) {
	root_bone = p_root_bone;
	reload_chain();
}

StringName SkeletonIK3D::get_root_bone() const {
	return root_bone;
}

void SkeletonIK3D::set_tip_bone(const StringName &p_tip_bone) {
	tip_bone = p_tip_bone;
	reload_chain();
}

StringName SkeletonIK3D::get_tip_bone() const {
	return tip_bone;
}

void SkeletonIK3D::set_interpolation(real_t p_interpolation) {
	interpolation = p_interpolation;
}

real_t SkeletonIK3D::get_interpolation() const {
	return interpolation;
}

void SkeletonIK3D::set_target_transform(const Transform3D &p_target) {
	target = p_target;
	reload_goal();
}

const Transform3D &SkeletonIK3D::get_target_transform() const {
	return target;
}

void SkeletonIK3D::set_target_node(const NodePath &p_node) {
	target_node_path_override = p_node;
	target_node_override_ref = Variant();
	reload_goal();
}

NodePath SkeletonIK3D::get_target_node() {
	return target_node_path_override;
}

void SkeletonIK3D::set_override_tip_basis(bool p_override) {
	override_tip_basis = p_override;
}

bool SkeletonIK3D::is_override_tip_basis() const {
	return override_tip_basis;
}

void SkeletonIK3D::set_use_magnet(bool p_use) {
	use_magnet = p_use;
}

bool SkeletonIK3D::is_using_magnet() const {
	return use_magnet;
}

void SkeletonIK3D::set_magnet_position(const Vector3 &p_local_position) {
	magnet_position = p_local_position;
}

const Vector3 &SkeletonIK3D::get_magnet_position() const {
	return magnet_position;
}

void SkeletonIK3D::set_min_distance(real_t p_min_distance) {
	min_distance = p_min_distance;
}

void SkeletonIK3D::set_max_iterations(int p_iterations) {
	max_iterations = p_iterations;
}

Skeleton3D *SkeletonIK3D::get_parent_skeleton() const {
	return cast_to<Skeleton3D>(skeleton_ref.get_validated_object());
}

bool SkeletonIK3D::is_running() {
	return is_processing_internal();
}

void SkeletonIK3D::start(bool p_one_time) {
	if (p_one_time) {
		set_process_internal(false);

		if (target_node_override_ref) {
			reload_goal();
		}

		_solve_chain();
	} else {
		set_process_internal(true);
	}
}

void SkeletonIK3D::stop() {
	set_process_internal(false);
	Skeleton3D *skeleton = get_parent_skeleton();
	if (skeleton) {
		skeleton->clear_bones_global_pose_override();
	}
}

Transform3D SkeletonIK3D::_get_target_transform() {
	if (!target_node_override_ref && !target_node_path_override.is_empty()) {
		target_node_override_ref = Object::cast_to<Node3D>(get_node(target_node_path_override));
	}

	Node3D *target_node_override = cast_to<Node3D>(target_node_override_ref.get_validated_object());
	if (target_node_override && target_node_override->is_inside_tree()) {
		return target_node_override->get_global_transform();
	} else {
		return target;
	}
}

void SkeletonIK3D::reload_chain() {
	FabrikInverseKinematic::free_task(task);
	task = nullptr;

	Skeleton3D *skeleton = get_parent_skeleton();
	if (!skeleton) {
		return;
	}

	task = FabrikInverseKinematic::create_simple_task(skeleton, skeleton->find_bone(root_bone), skeleton->find_bone(tip_bone), _get_target_transform());
	if (task) {
		task->max_iterations = max_iterations;
		task->min_distance = min_distance;
	}
}

void SkeletonIK3D::reload_goal() {
	if (!task) {
		return;
	}

	FabrikInverseKinematic::set_goal(task, _get_target_transform());
}

void SkeletonIK3D::_solve_chain() {
	if (!task) {
		return;
	}
	FabrikInverseKinematic::solve(task, interpolation, override_tip_basis, use_magnet, magnet_position);
}

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

#endif // _3D_DISABLED
