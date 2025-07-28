/**************************************************************************/
/*  jacob_ik_3d.cpp                                                       */
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

#include "jacob_ik_3d.h"

void JacobIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, ManyBoneIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size) {
	// Forwards.
	for (int i = 0; i < p_joint_size; i++) {
		ManyBoneIK3DSolverInfo *solver_info = p_joints[i]->solver_info;
		if (!solver_info || Math::is_zero_approx(solver_info->length)) {
			continue;
		}

		int HEAD = i;
		int TAIL = i + 1;

		Vector3 current_head = p_chain[HEAD];
		Vector3 current_effector = p_chain[p_chain_size - 1];
		Vector3 head_to_effector = current_effector - current_head;
		Vector3 effector_to_destination = p_destination - current_effector;
		Vector3 axis = head_to_effector.cross(effector_to_destination);

		if (Math::is_zero_approx(axis.length_squared())) {
			continue;
		}

		// Note:
		// Jacobian can calculate (estimate) all joint rotations at once, so we can use limitations here.
		// If we remove angular_delta_limit here, it behaves more similar to FABR/CCD (quickly converges to the target).
		// But it may cause oscillation in some cases, so we keep it here to avoid that.
		Quaternion to_rot = Quaternion(axis.normalized(), MIN(axis.length() / MAX(CMP_EPSILON, head_to_effector.length_squared()), angular_delta_limit));

		for (int j = TAIL; j < p_chain_size; j++) {
			Vector3 to_tail = p_chain[j] - current_head;
			p_setting->update_chain_coordinate(p_skeleton, j, current_head + to_rot.xform(to_tail), false);
		}
	}
}
