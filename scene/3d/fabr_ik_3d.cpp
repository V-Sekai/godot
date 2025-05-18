/**************************************************************************/
/*  fabr_ik_3d.cpp                                                        */
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

#include "fabr_ik_3d.h"

void FABRIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, ManyBoneIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_max_iterations, real_t p_min_distance_squared) {
	real_t distance_to_target_sq = INFINITY;
	int iteration_count = 0;

	if (p_setting->is_penetrated(p_destination)) {
		return;
	}

	while (distance_to_target_sq > p_min_distance_squared && iteration_count < p_max_iterations) {
		iteration_count++;

		// Backwards.
		bool first = true;
		for (int i = p_joints.size() - 1; i >= 0; i--) {
			ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
			if (!solver_info || Math::is_zero_approx(solver_info->length)) {
				continue;
			}

			const int HEAD = i;
			const int TAIL = i + 1;

			if (first) {
				p_setting->update_chain_coordinate(p_skeleton, TAIL, p_destination);
				first = false;
			}

			p_setting->update_chain_coordinate(p_skeleton, HEAD, limit_length(p_chain[TAIL], p_chain[HEAD], solver_info->length));
		}

		// Forwards.
		first = true;
		for (int i = 0; i < p_joints.size(); i++) {
			ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
			if (!solver_info || Math::is_zero_approx(solver_info->length)) {
				continue;
			}

			const int HEAD = i;
			const int TAIL = i + 1;

			if (first) {
				p_setting->update_chain_coordinate(p_skeleton, HEAD, p_skeleton->get_bone_global_pose(p_joints[i]->bone).origin, false);
				first = false;
			}

			p_setting->update_chain_coordinate(p_skeleton, TAIL, limit_length(p_chain[HEAD], p_chain[TAIL], solver_info->length), false);
			// Limitation should be processed only in forward pass to prevent oscillation.
			if (p_joints[HEAD]->rotation_axis != ROTATION_AXIS_ALL) {
				p_setting->update_chain_coordinate(p_skeleton, TAIL, p_chain[HEAD] + p_joints[HEAD]->get_projected_rotation(solver_info->current_grest, p_chain[TAIL] - p_chain[HEAD]), false);
			}
			if (p_joints[HEAD]->limitation.is_valid()) {
				p_setting->update_chain_coordinate(p_skeleton, TAIL, p_chain[HEAD] + p_joints[HEAD]->get_limited_rotation(solver_info->current_grest, p_chain[TAIL] - p_chain[HEAD]), false);
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
