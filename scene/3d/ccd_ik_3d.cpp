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

void CCDIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, ManyBoneIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_max_iterations, real_t p_min_distance_squared) {
	real_t distance_to_target_sq = INFINITY;
	int iteration_count = 0;

	if (p_setting->is_penetrated(p_destination)) {
		return;
	}

	while (distance_to_target_sq > p_min_distance_squared && iteration_count < p_max_iterations) {
		iteration_count++;

		// Backwards.
		for (int ancestor = p_joints.size() - 1; ancestor >= 0; ancestor--) {
			for (int i = ancestor; i < p_joints.size(); i++) {
				ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
				if (!solver_info || Math::is_zero_approx(solver_info->length)) {
					continue;
				}

				const int HEAD = i;
				const int TAIL = i + 1;

				Vector3 to_effector = p_chain[p_chain.size() - 1] - p_chain[HEAD];
				Vector3 to_target = p_destination - p_chain[HEAD];
				Quaternion to_rot = Quaternion(to_effector.normalized(), to_target.normalized());

				Vector3 to_tail = p_chain[TAIL] - p_chain[HEAD];
				p_setting->update_chain_coordinate(p_skeleton, TAIL, limit_length(p_chain[HEAD], p_chain[HEAD] + to_rot.xform(to_tail), solver_info->length));
				Vector3 result = p_chain[TAIL];

				if (ancestor > 0) {
					continue; // Limitation should be processed only in final pass to prevent oscillation.
				}

				if (p_joints[HEAD]->rotation_axis != ROTATION_AXIS_ALL) {
					p_setting->update_chain_coordinate(p_skeleton, TAIL, p_chain[HEAD] + p_joints[HEAD]->get_projected_rotation(solver_info->current_grest, p_chain[TAIL] - p_chain[HEAD]));
				}

				if (p_joints[HEAD]->limitation.is_valid()) {
					p_setting->update_chain_coordinate(p_skeleton, TAIL, p_chain[HEAD] + p_joints[HEAD]->get_limited_rotation(solver_info->current_grest, p_chain[TAIL] - p_chain[HEAD]));
				}

				// This process is high cost, so we only do it if there is limitation.
				if (!Math::is_zero_approx((result - p_chain[TAIL]).length_squared())) {
					// Propagate the rotation to the previous processed joints (child tails).
					Quaternion diff = get_from_to_rotation(to_tail.normalized(), (p_chain[TAIL] - p_chain[HEAD]).normalized(), Quaternion());
					for (int j = TAIL + 1; j < p_chain.size(); j++) {
						Vector3 rel = p_chain[j] - p_chain[HEAD];
						p_setting->update_chain_coordinate(p_skeleton, j, p_chain[HEAD] + diff.xform(rel)); // Joint rotation will be updated in the first step on the next loop.
					}
				}
			}
		}
		p_setting->cache_current_joint_rotations(p_skeleton);

		distance_to_target_sq = p_chain[p_chain.size() - 1].distance_squared_to(p_destination);
	}

	// Apply the rotation to the bones.
	for (int i = 0; i < p_joints.size(); i++) {
		ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}
		p_skeleton->set_bone_pose_rotation(p_joints[i]->bone, get_local_pose_rotation(p_skeleton, p_joints[i]->bone, solver_info->current_gpose));
	}
}
