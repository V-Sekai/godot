/**************************************************************************/
/*  ewbik_3d_.h                                                           */
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

#include "scene/3d/iterate_ik_3d.h"

struct OptimalTransform {
	Quaternion rotation;
	Vector3 translation;
};

class EWBIK3D : public IterateIK3D {
	GDCLASS(EWBIK3D, IterateIK3D);

protected:
	virtual void _solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) override;

	// Chain building for arbitrary root-to-effector support
	virtual void _update_joints(int p_index) override;

	// Pathfinding helper methods
	bool _find_bone_chain_path(Skeleton3D *p_skeleton, int p_root_bone, int p_end_bone, Vector<int> &r_chain) const;
	void _build_chain_from_path(Skeleton3D *p_skeleton, const Vector<int> &p_path, LocalVector<BoneJoint> &r_joints) const;

	// QCP solver helper methods
	void _create_point_correspondences(Skeleton3D *p_skeleton, const IterateIK3DSetting *p_setting, int p_bone_idx, const Vector3 &p_destination, const Transform3D &p_target_transform,
			PackedVector3Array &r_target_headings, PackedVector3Array &r_tip_headings, Vector<double> &r_weights);
	OptimalTransform _calculate_optimal_rotation(const PackedVector3Array &p_target_headings, const PackedVector3Array &p_tip_headings, const Vector<double> &p_weights, bool p_calculate_translation = false);
};
