/**************************************************************************/
/*  ccd_ik_3d.cpp                                                         */
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

#include "ccd_ik_3d.h"

#include "core/math/qcp.h"

// The basic idea is to create a bunch of ordered sets of bones.
//
// TODO: Solve one effector to one Godot Engine Node3D solve.
//
// Building each set by starting at an effector and traversing all the way down to the skeleton root. (Keep track of each effector you started from)
//
// The effector is the end of the chain, and the root is the beginning of the chain. A root is also the bone rootmost to the `Vector<int> bones_to_process = skeleton->get_parentless_bones()`;
//
// Then you take the intersections of all of the bonesets, giving you the sets of bonesets which solve for the same effectors.
// This allows you to initialize moved/target arrays once for each boneset, since by definition all of the bones inside of them are solving for the same effectors.
//
// Finally, determine the default traversal order for the bonesets, then flatten those into a bone traversal list.
//
// Assume the existence and usage of the disjoint-set data structure. Do not need to implement it here.
//
// In computer science, a disjoint-set data structure, also called a union–find data structure or merge–find set, is a data structure that stores a collection of disjoint (non-overlapping) sets.

void CCDIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Transform3D &p_space, const Vector3 &p_destination, const Vector3 &p_target_vector, int p_max_iterations, real_t p_min_distance) {
	ERR_FAIL_NULL(p_skeleton);
	ERR_FAIL_INDEX(p_joints.size() - 1, p_joints.size());
	ERR_FAIL_INDEX(0, p_joints.size());
	int iteration_count = 0;
	Vector<SolverInfoData> solver_info_data;
	solver_info_data.resize(p_joints.size());
	for (int joints_i = 0; joints_i < p_joints.size(); ++joints_i) {
		if (p_joints[joints_i] == nullptr) {
			continue;
		}
		ManyBoneIK3DSolverInfo *solver_info = p_joints[joints_i]->solver_info;
		if (!solver_info) {
			continue;
		}
		SolverInfoData &current_solver_data = solver_info_data.write[joints_i];
		current_solver_data.target_transform.origin = p_chain[joints_i] + p_destination;
		current_solver_data.moved_transform.origin = p_chain[joints_i];
	}

	while (iteration_count < p_max_iterations) {
		iteration_count++;
		Vector<Transform3D> moveds;
		Vector<Transform3D> targets;
		for (int joints_i = p_joints.size() - 1; joints_i >= 0; --joints_i) {
			moveds.resize(joints_i + 1);
			targets.resize(joints_i + 1);
			for (int inner_joints_i = joints_i; inner_joints_i >= 0; --inner_joints_i) {
				SolverInfoData &inner_solver_info_data = solver_info_data.write[inner_joints_i];
				targets.write[inner_joints_i] = inner_solver_info_data.target_transform;
				moveds.write[inner_joints_i] = inner_solver_info_data.moved_transform;
			}
			bool translate = joints_i == 0;
			Transform3D result = _solve(targets, moveds, translate);
			solver_info_data.write[joints_i].moved_transform.origin = result.origin;
		}
	}
	for (int joints_i = 0; joints_i < p_joints.size(); ++joints_i) {
		if (!p_joints[joints_i]) {
			continue;
		}
		ManyBoneIK3DSolverInfo *solver_info = p_joints[joints_i]->solver_info;
		if (!solver_info) {
			continue;
		}
		p_chain.write[joints_i] = solver_info_data[joints_i].moved_transform.origin;
		bool translate = joints_i == 0;
		if (translate) {
			p_skeleton->set_bone_pose_position(p_joints[joints_i]->bone, solver_info_data[joints_i].moved_transform.origin);
		}
	}
}

Transform3D CCDIK3D::_solve(const Vector<Transform3D> &p_targets, const Vector<Transform3D> &p_moveds, bool p_translate) {
	ERR_FAIL_COND_V(p_targets.size() != p_moveds.size(), Transform3D());

	Vector<Vector3> moved_positions;
	Vector<Vector3> target_positions;
	Vector<double> weights;

	moved_positions.resize(p_moveds.size() * 7);
	target_positions.resize(p_targets.size() * 7);
	weights.resize(p_moveds.size() * 7);
	real_t pin_weight = 1.0;
	weights.fill(pin_weight);

	for (int i = 0; i < p_targets.size(); ++i) {
		moved_positions.write[i * 7] = p_moveds[i].origin;
		int rest_index = 1;
		for (int axis_i = Vector3::AXIS_X; axis_i <= Vector3::AXIS_Z; ++axis_i) {
			Vector3 column = p_moveds[i].basis.get_column(axis_i);
			moved_positions.write[i * 7 + rest_index] = column + p_moveds[i].origin;
			rest_index++;
			moved_positions.write[i * 7 + rest_index] = p_moveds[i].origin - column;
			rest_index++;
		}

		target_positions.write[i * 7] = p_targets[i].origin;
		int current_index = 1;
		for (int axis_j = Vector3::AXIS_X; axis_j <= Vector3::AXIS_Z; ++axis_j) {
			Vector3 column = p_targets[i].basis.get_column(axis_j);
			target_positions.write[i * 7 + current_index] = column + p_targets[i].origin;
			current_index++;
			target_positions.write[i * 7 + current_index] = p_targets[i].origin - column;
			current_index++;
		}
	}
	Quaternion rotation;
	Vector3 translation;
	QuaternionCharacteristicPolynomial::weighted_superpose(
			moved_positions, target_positions, weights, p_translate, rotation, translation);
	return Transform3D(Basis(rotation), translation);
}
