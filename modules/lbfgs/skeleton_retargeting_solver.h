/**************************************************************************/
/*  skeleton_retargeting_solver.h                                         */
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

#include "core/object/class_db.h" // For GDCLASS
#include "core/variant/dictionary.h"
#include "lbfgsbpp.h"
#include "scene/3d/skeleton_3d.h"

class SkeletonRetargetingSolver : public LBFGSBSolver {
	GDCLASS(SkeletonRetargetingSolver, LBFGSBSolver);

public:
	Skeleton3D *source_skeleton;
	Skeleton3D *target_skeleton;
	// Add properties for joint mapping if needed: Dictionary joint_map;

	void set_source_skeleton(Skeleton3D *p_skel);
	Skeleton3D *get_source_skeleton() const;

	void set_target_skeleton(Skeleton3D *p_skel);
	Skeleton3D *get_target_skeleton() const;
	// void set_joint_map(const Dictionary& p_map) { joint_map = p_map; }

	// Optimizes target_skeleton's root transform (translation, rotation, scale) to match source_skeleton.
	Dictionary optimize_global_transform();

	// Optimizes local scales of target_skeleton bones to match source bone lengths/proportions.
	Dictionary optimize_bone_local_scales();

protected:
	static void _bind_methods();

public:
	virtual double call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) override;
	SkeletonRetargetingSolver();
	~SkeletonRetargetingSolver();
};
