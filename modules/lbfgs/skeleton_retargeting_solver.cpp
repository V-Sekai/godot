/**************************************************************************/
/*  skeleton_retargeting_solver.cpp                                       */
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

#include "skeleton_retargeting_solver.h"

#include "core/object/class_db.h"

void SkeletonRetargetingSolver::set_source_skeleton(Skeleton3D *p_skel) {
	source_skeleton = p_skel;
}

Skeleton3D *SkeletonRetargetingSolver::get_source_skeleton() const {
	return source_skeleton;
}

void SkeletonRetargetingSolver::set_target_skeleton(Skeleton3D *p_skel) {
	target_skeleton = p_skel;
}

Skeleton3D *SkeletonRetargetingSolver::get_target_skeleton() const {
	return target_skeleton;
}

Dictionary SkeletonRetargetingSolver::optimize_global_transform() {
	// Implementation needed
	return Dictionary();
}

Dictionary SkeletonRetargetingSolver::optimize_bone_local_scales() {
	// Implementation needed
	return Dictionary();
}

double SkeletonRetargetingSolver::call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) {
	// Implementation needed
	return 0.0;
}

void SkeletonRetargetingSolver::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source_skeleton", "skeleton"), &SkeletonRetargetingSolver::set_source_skeleton);
	ClassDB::bind_method(D_METHOD("get_source_skeleton"), &SkeletonRetargetingSolver::get_source_skeleton);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "source_skeleton", PROPERTY_HINT_RESOURCE_TYPE, "Skeleton3D"), "set_source_skeleton", "get_source_skeleton");

	ClassDB::bind_method(D_METHOD("set_target_skeleton", "skeleton"), &SkeletonRetargetingSolver::set_target_skeleton);
	ClassDB::bind_method(D_METHOD("get_target_skeleton"), &SkeletonRetargetingSolver::get_target_skeleton);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "target_skeleton", PROPERTY_HINT_RESOURCE_TYPE, "Skeleton3D"), "set_target_skeleton", "get_target_skeleton");

	ClassDB::bind_method(D_METHOD("optimize_global_transform"), &SkeletonRetargetingSolver::optimize_global_transform);
	ClassDB::bind_method(D_METHOD("optimize_bone_local_scales"), &SkeletonRetargetingSolver::optimize_bone_local_scales);
}

SkeletonRetargetingSolver::SkeletonRetargetingSolver() {}

SkeletonRetargetingSolver::~SkeletonRetargetingSolver() {}
