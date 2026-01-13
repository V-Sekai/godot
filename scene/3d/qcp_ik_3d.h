/**************************************************************************/
/*  qcp_ik_3d.h                                                           */
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

class QCP_IK3D : public IterateIK3D {
	GDCLASS(QCP_IK3D, IterateIK3D);

public:
	void _solve_iteration(double p_delta, Skeleton3D *p_skeleton, IterateIK3DSetting *p_setting, const Vector3 &p_destination) override;

	Quaternion clamp_to_cos_half_angle(const Quaternion &p_quat, double p_cos_half_angle);

	struct Effector {
		int bone = -1;
		Vector3 target;
		Transform3D target_transform;
		float opacity = 1.0f;
		int setting_index = -1;
	};

	struct BoneWeight {
		int bone = -1;
		float weight = 1.0f;
	};

	struct EffectorGroup {
		Vector<int> bones;
		Vector<Effector> effectors;
	};

	QCP_IK3D() {}
};
