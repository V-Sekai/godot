/**************************************************************************/
/*  many_bone_ik_3d_state.h                                               */
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

#include "core/math/transform_3d.h"
#include "core/object/ref_counted.h"
#include "ik_effector_template_3d.h"

class ManyBoneIK3DState : public RefCounted {
	GDCLASS(ManyBoneIK3DState, RefCounted);

private:
	// Shadow copy of skeleton pose before IK simulation
	Vector<Transform3D> original_bone_poses;

	// IK configuration data
	Vector<Ref<IKEffectorTemplate3D>> effector_templates;

	// Simulation parameters
	int iterations_per_frame = 10;
	float default_damp = 0.1f;
	int stabilization_passes = 1;

protected:
	static void _bind_methods();

public:
	void set_original_bone_poses(const Vector<Transform3D> &p_poses);
	Vector<Transform3D> get_original_bone_poses() const;

	void set_effector_templates(const Vector<Ref<IKEffectorTemplate3D>> &p_templates);
	Vector<Ref<IKEffectorTemplate3D>> get_effector_templates() const;

	void set_iterations_per_frame(int p_iterations);
	int get_iterations_per_frame() const;

	void set_default_damp(float p_damp);
	float get_default_damp() const;

	void set_stabilization_passes(int p_passes);
	int get_stabilization_passes() const;

	void set_original_bone_poses_array(const Array &p_poses);
	Array get_original_bone_poses_array() const;

	void set_effector_templates_array(const Array &p_templates);
	Array get_effector_templates_array() const;
};
