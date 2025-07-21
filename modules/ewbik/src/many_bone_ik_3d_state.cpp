/**************************************************************************/
/*  many_bone_ik_3d_state.cpp                                             */
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

#include "many_bone_ik_3d_state.h"

#include "core/object/class_db.h"

void ManyBoneIK3DState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_original_bone_poses", "poses"), &ManyBoneIK3DState::set_original_bone_poses_array);
	ClassDB::bind_method(D_METHOD("get_original_bone_poses"), &ManyBoneIK3DState::get_original_bone_poses_array);

	ClassDB::bind_method(D_METHOD("set_effector_templates", "templates"), &ManyBoneIK3DState::set_effector_templates_array);
	ClassDB::bind_method(D_METHOD("get_effector_templates"), &ManyBoneIK3DState::get_effector_templates_array);

	ClassDB::bind_method(D_METHOD("set_iterations_per_frame", "iterations"), &ManyBoneIK3DState::set_iterations_per_frame);
	ClassDB::bind_method(D_METHOD("get_iterations_per_frame"), &ManyBoneIK3DState::get_iterations_per_frame);

	ClassDB::bind_method(D_METHOD("set_default_damp", "damp"), &ManyBoneIK3DState::set_default_damp);
	ClassDB::bind_method(D_METHOD("get_default_damp"), &ManyBoneIK3DState::get_default_damp);

	ClassDB::bind_method(D_METHOD("set_stabilization_passes", "passes"), &ManyBoneIK3DState::set_stabilization_passes);
	ClassDB::bind_method(D_METHOD("get_stabilization_passes"), &ManyBoneIK3DState::get_stabilization_passes);
}

void ManyBoneIK3DState::set_original_bone_poses(const Vector<Transform3D> &p_poses) {
	original_bone_poses = p_poses;
}

Vector<Transform3D> ManyBoneIK3DState::get_original_bone_poses() const {
	return original_bone_poses;
}

void ManyBoneIK3DState::set_effector_templates(const Vector<Ref<IKEffectorTemplate3D>> &p_templates) {
	effector_templates = p_templates;
}

Vector<Ref<IKEffectorTemplate3D>> ManyBoneIK3DState::get_effector_templates() const {
	return effector_templates;
}

void ManyBoneIK3DState::set_iterations_per_frame(int p_iterations) {
	iterations_per_frame = p_iterations;
}

int ManyBoneIK3DState::get_iterations_per_frame() const {
	return iterations_per_frame;
}

void ManyBoneIK3DState::set_default_damp(float p_damp) {
	default_damp = p_damp;
}

float ManyBoneIK3DState::get_default_damp() const {
	return default_damp;
}

void ManyBoneIK3DState::set_stabilization_passes(int p_passes) {
	stabilization_passes = p_passes;
}

int ManyBoneIK3DState::get_stabilization_passes() const {
	return stabilization_passes;
}

void ManyBoneIK3DState::set_original_bone_poses_array(const Array &p_poses) {
	original_bone_poses.clear();
	for (int i = 0; i < p_poses.size(); ++i) {
		Transform3D transform = p_poses[i];
		original_bone_poses.push_back(transform);
	}
}

Array ManyBoneIK3DState::get_original_bone_poses_array() const {
	Array result;
	for (const Transform3D &transform : original_bone_poses) {
		result.push_back(transform);
	}
	return result;
}

void ManyBoneIK3DState::set_effector_templates_array(const Array &p_templates) {
	effector_templates.clear();
	for (int i = 0; i < p_templates.size(); ++i) {
		Ref<IKEffectorTemplate3D> template_ref = p_templates[i];
		if (template_ref.is_valid()) {
			effector_templates.push_back(template_ref);
		}
	}
}

Array ManyBoneIK3DState::get_effector_templates_array() const {
	Array result;
	for (const Ref<IKEffectorTemplate3D> &template_ref : effector_templates) {
		result.push_back(template_ref);
	}
	return result;
}
