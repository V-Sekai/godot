/**************************************************************************/
/*  many_bone_ik_3d.h                                                     */
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

#include "scene/3d/skeleton_modifier_3d.h"

#include "scene/resources/3d/joint_limitation_3d.h"

class ManyBoneIK3D : public SkeletonModifier3D {
	GDCLASS(ManyBoneIK3D, SkeletonModifier3D);

#ifdef TOOLS_ENABLED
	bool saving = false;
#endif //TOOLS_ENABLED

	bool joints_dirty = false;

public:
	enum BoneDirection {
		BONE_DIRECTION_PLUS_X,
		BONE_DIRECTION_MINUS_X,
		BONE_DIRECTION_PLUS_Y,
		BONE_DIRECTION_MINUS_Y,
		BONE_DIRECTION_PLUS_Z,
		BONE_DIRECTION_MINUS_Z,
		BONE_DIRECTION_FROM_PARENT,
	};

	enum RotationAxis {
		ROTATION_AXIS_X,
		ROTATION_AXIS_Y,
		ROTATION_AXIS_Z,
		ROTATION_AXIS_ALL,
		ROTATION_AXIS_CUSTOM,
	};

	struct ManyBoneIK3DSolverInfo {
		Quaternion current_lpose;
		Quaternion current_lrest;
		Quaternion current_gpose;
		Quaternion current_grest;
		Vector3 current_vector; // Global so needs xfrom_inv by gpose or grest in the process.
		Vector3 forward_vector; // Local.
		float length = 0.0;

		Quaternion get_current_relative_pose() {
			return (current_lrest.inverse() * current_lpose).normalized();
		}
	};

	struct ManyBoneIK3DJointSetting {
		String bone_name;
		int bone = -1;

		// Rotation axis.
		RotationAxis rotation_axis = ROTATION_AXIS_ALL;
		Vector3 rotation_axis_vector = Vector3(1, 0, 0);
		Vector3 get_rotation_axis_vector() const {
			Vector3 ret;
			switch (rotation_axis) {
				case ROTATION_AXIS_X:
					ret = Vector3(1, 0, 0);
					break;
				case ROTATION_AXIS_Y:
					ret = Vector3(0, 1, 0);
					break;
				case ROTATION_AXIS_Z:
					ret = Vector3(0, 0, 1);
					break;
				case ROTATION_AXIS_ALL:
					ret = Vector3(0, 0, 0);
					break;
				case ROTATION_AXIS_CUSTOM:
					ret = rotation_axis_vector;
					break;
			}
			return ret;
		}

		// To limit rotation.
		Ref<JointLimitation3D> limitation;

		// To process.
		ManyBoneIK3DSolverInfo *solver_info = nullptr;

		// Get rotation around normal vector (normal vector is rotation axis).
		Vector3 get_projected_rotation(const Quaternion &p_offset, const Vector3 &p_vector) const {
			ERR_FAIL_COND_V(rotation_axis == ROTATION_AXIS_ALL, p_vector);
			const double ALMOST_ONE = 1.0 - CMP_EPSILON;
			Vector3 axis = get_rotation_axis_vector().normalized();
			Vector3 local_vector = p_offset.xform_inv(p_vector);
			double length = local_vector.length();
			Vector3 projected = snap_vector_to_plane(axis, local_vector.normalized());
			if (!Math::is_zero_approx(length)) {
				projected = projected.normalized() * length;
			}
			if (Math::abs(local_vector.normalized().dot(axis)) > ALMOST_ONE) {
				return p_vector;
			}
			return p_offset.xform(projected);
		}

		// Get limited rotation from forward axis in local rest space.
		Vector3 get_limited_rotation(const Quaternion &p_offset, const Vector3 &p_vector) const {
			ERR_FAIL_COND_V(limitation.is_null(), p_vector);
			Vector3 local_vector = p_offset.xform_inv(p_vector);
			float length = local_vector.length();
			if (Math::is_zero_approx(length)) {
				return p_vector;
			}
			Vector3 limited = limitation->solve(solver_info->forward_vector, local_vector.normalized()) * length;
			return p_offset.xform(limited);
		}

		~ManyBoneIK3DJointSetting() {
			if (solver_info) {
				memdelete(solver_info);
				solver_info = nullptr;
			}
		}
	};

protected:
	void _notification(int p_what);
	static void _bind_methods();

	virtual void _set_active(bool p_active) override;
	virtual void _skeleton_changed(Skeleton3D *p_old, Skeleton3D *p_new) override;

	virtual void _validate_bone_names() override;

	virtual void _make_all_joints_dirty();

	virtual void _process_modification(double p_delta) override;
	virtual void _process_ik(Skeleton3D *p_skeleton, double p_delta);

public:
	// Helper.
	static Quaternion get_local_pose_rotation(Skeleton3D *p_skeleton, int p_bone, const Quaternion &p_global_pose_rotation);
	Vector3 get_bone_axis(int p_end_bone, BoneDirection p_direction) const;

	// To process manually.
	virtual void reset();
};

VARIANT_ENUM_CAST(ManyBoneIK3D::BoneDirection);
VARIANT_ENUM_CAST(ManyBoneIK3D::RotationAxis);
