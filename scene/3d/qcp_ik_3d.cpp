/**************************************************************************/
/*  qcp_ik_3d.cpp                                                         */
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

#include "qcp_ik_3d.h"

#include "core/math/qcp.h"

Quaternion QCP_IK3D::clamp_to_cos_half_angle(const Quaternion &p_quat, double p_cos_half_angle) {
	Quaternion quat = p_quat;
	if (quat.w < 0.0) {
		quat = quat * -1.0;
	}
	double previous_coefficient = (1.0 - (quat.w * quat.w));
	if (p_cos_half_angle <= quat.w || Math::is_zero_approx(previous_coefficient)) {
		return quat;
	} else {
		double composite_coefficient = Math::sqrt((1.0 - (p_cos_half_angle * p_cos_half_angle)) / previous_coefficient);
		quat.w = p_cos_half_angle;
		quat.x *= composite_coefficient;
		quat.y *= composite_coefficient;
		quat.z *= composite_coefficient;
		return quat;
	}
}

void QCP_IK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) {
	// Apply QCP per bone like many bone IK, updating from tip to root
	for (int i = p_setting->joints.size() - 1; i >= 0; i--) {
		int bone_idx = p_setting->joints[i].bone;

		// Create headings arrays like many bone IK
		PackedVector3Array tip_headings;
		PackedVector3Array target_headings;
		Vector<double> heading_weights;

		// Get the bone's global transform for local space conversion
		Transform3D bone_global = p_skeleton->get_bone_global_pose(bone_idx);

		// For each bone in subchain, create 7 headings in a star pattern
		for (int j = i; j < p_setting->chain.size(); j++) {
			// Current bone positions in global space
			Vector3 bone_start = p_setting->chain[j];
			Vector3 bone_end = (j + 1 < p_setting->chain.size()) ? p_setting->chain[j + 1] : p_destination;
			Vector3 bone_direction = (bone_end - bone_start).normalized();
			float bone_length = bone_start.distance_to(bone_end);

			// Create coordinate frame
			Basis bone_basis = Basis::looking_at(bone_direction, Vector3(0, 1, 0));

			// Star pattern: 1 center + 6 points at 60-degree intervals
			float radius = bone_length * 0.3f; // Star radius relative to bone length

			// Center point
			tip_headings.push_back(bone_global.affine_inverse().xform(bone_start));

			// 6 points in star pattern around the bone
			for (int k = 0; k < 6; k++) {
				float angle = k * Math::PI / 3.0f; // 60-degree intervals
				// Create point in a plane perpendicular to bone direction
				Vector3 star_point = bone_start +
						bone_basis.get_column(0) * radius * cos(angle) +
						bone_basis.get_column(1) * radius * sin(angle);
				tip_headings.push_back(bone_global.affine_inverse().xform(star_point));
			}

			// Target headings with same star pattern
			if (j == p_setting->chain.size() - 1) {
				// For the tip bone, target is the destination with star pattern
				Vector3 target_direction = (p_destination - bone_start).normalized();
				Basis target_basis = Basis::looking_at(target_direction, Vector3(0, 1, 0));
				float target_length = bone_start.distance_to(p_destination);
				float target_radius = target_length * 0.3f;

				// Center point at destination
				target_headings.push_back(bone_global.affine_inverse().xform(bone_start));

				// 6 points in star pattern around destination
				for (int k = 0; k < 6; k++) {
					float angle = k * Math::PI / 3.0f;
					Vector3 star_point = bone_start +
							target_basis.get_column(0) * target_radius * cos(angle) +
							target_basis.get_column(1) * target_radius * sin(angle);
					target_headings.push_back(bone_global.affine_inverse().xform(star_point));
				}
			} else {
				// For other bones, target matches current star pattern
				target_headings.push_back(bone_global.affine_inverse().xform(bone_start));

				for (int k = 0; k < 6; k++) {
					float angle = k * Math::PI / 3.0f;
					Vector3 star_point = bone_start +
							bone_basis.get_column(0) * radius * cos(angle) +
							bone_basis.get_column(1) * radius * sin(angle);
					target_headings.push_back(bone_global.affine_inverse().xform(star_point));
				}
			}

			// Add weights (uniform for all 7 headings per bone)
			for (int k = 0; k < 7; k++) {
				heading_weights.push_back(1.0);
			}
		}

		// Perform QCP like many bone IK
		Array superpose_result = QuaternionCharacteristicPolynomial::weighted_superpose(tip_headings, target_headings, heading_weights, true);
		Quaternion qcp_rotation = superpose_result[0];
		Vector3 qcp_translation = superpose_result[1];

		// Apply damping like many bone IK
		Transform3D current_local = p_skeleton->get_bone_pose(bone_idx);

		// Apply rotation to the local basis
		Basis new_local_basis = qcp_rotation * current_local.basis;

		// Preserve scale
		Vector3 current_scale = current_local.basis.get_scale();
		new_local_basis = new_local_basis.scaled(current_scale);

		// Apply translation to the local origin
		Vector3 new_local_origin = current_local.origin + qcp_translation;

		// Set the new local pose
		p_skeleton->set_bone_pose_position(bone_idx, new_local_origin);
		p_skeleton->set_bone_pose_rotation(bone_idx, new_local_basis.get_rotation_quaternion());
		p_skeleton->set_bone_pose_scale(bone_idx, new_local_basis.get_scale());

		// Update chain coordinate with the new global position
		Transform3D new_global = p_skeleton->get_bone_global_pose(bone_idx);
		p_setting->update_chain_coordinate_fw(p_skeleton, i, new_global.origin);
	}
}
