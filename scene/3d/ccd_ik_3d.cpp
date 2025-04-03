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

void CCDIK3D::_process_joints(double p_delta, Skeleton3D *p_skeleton, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Transform3D &p_space, const Vector3 &p_destination, const Vector3 &p_target_vector, int p_max_iterations, real_t p_min_distance) {
	double min_distance_sq = p_min_distance * p_min_distance;
	real_t distance_to_target_sq = INFINITY;
	int iteration_count = 0;

	// Quaternion destination_rotation = p_space.basis.get_rotation_quaternion() * p_skeleton->get_bone_global_pose(p_joints[p_joints.size() - 1]->bone).basis.get_rotation_quaternion();

	while (distance_to_target_sq > min_distance_sq && iteration_count < p_max_iterations) {
		iteration_count++;

		// Backwards.
		int end = -1;
		for (int i = p_joints.size() - 1; i >= 0; i--) {
			ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
			if (!solver_info) {
				continue;
			}
			const int HEAD = i;
			if (end == -1) {
				end = i + 1;
			}
			Vector3 to_vec = p_destination - p_chain[HEAD];
			Vector3 head_to_end = p_chain[end] - p_chain[HEAD];
			Quaternion to_rot = Quaternion(head_to_end.normalized(), to_vec.normalized());
			for (int j = p_chain.size() - 1; j > i; j--) {
				Vector3 head_to_any_joint = p_chain[j] - p_chain[HEAD];
				p_chain.write[j] = p_chain[HEAD] + to_rot.xform(head_to_any_joint);

				// For limitation.
				/*
				Vector3 current_head_to_tail = p_skeleton->get_bone_global_pose(p_joints[i]->bone).basis.get_rotation_quaternion().xform_inv(to_vec);
				Quaternion rotation_result = solver_info->current_rot = Quaternion(-solver_info->forward_vector, current_head_to_tail);
				Transform3D rest = p_skeleton->get_bone_global_pose(p_joints[i]->bone) * Basis(p_joints[i]->limitation_rotation_offset);
				Quaternion rest_rotation = rest.basis.get_rotation_quaternion();
				bool rotation_modified = false;
				TwistSwing ts = decompose_rotation_to_twist_and_swing(rest_rotation, rotation_result);
				if (p_joints[i]->twist_limitation < Math_PI) {
					// TODO: coding which limit twist.
					// ts.twist = XXXXX;
					rotation_modified = true;
				}
				Ref<JointLimitation3D> limitation = p_joints[i]->limitation;
				if (limitation.is_valid()) {
					ts.swing = limitation->solve(rest_rotation, ts.swing);
					rotation_modified = true;
				}
				if (rotation_modified) {
					// TODO: coding which fix any joint by limitationated rotation.
					p_chain.write[j] = compose_rotation_from_twist_and_swing(rest_rotation, ts).xform(-solver_info->forward_vector);
				}
				*/
			}
		}

		distance_to_target_sq = p_chain[p_chain.size() - 1].distance_squared_to(p_destination);
	}

	for (int i = 0; i < p_joints.size(); i++) {
		ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
		if (!solver_info) {
			continue;
		}
		Vector3 current_head_to_tail = (p_chain[i + 1] - p_chain[i]).normalized();
		current_head_to_tail = p_skeleton->get_bone_global_pose(p_joints[i]->bone).basis.get_rotation_quaternion().xform_inv(current_head_to_tail);
		if (solver_info->forward_vector.dot(current_head_to_tail) > 1.0f - CMP_EPSILON) {
			continue;
		}
		solver_info->current_rot = Quaternion(solver_info->forward_vector, current_head_to_tail);
		p_skeleton->set_bone_pose_rotation(p_joints[i]->bone,
			get_local_pose_rotation(
				p_skeleton,
				p_joints[i]->bone,
				p_skeleton->get_bone_global_pose(p_joints[i]->bone).basis.get_rotation_quaternion() * solver_info->current_rot));
	}
}
