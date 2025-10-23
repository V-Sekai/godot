/**************************************************************************/
/*  test_skeleton_retargeting_solver.h                                    */
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

#include "../skeleton_retargeting_solver.h"
#include "core/math/basis.h"
#include "core/math/vector3.h"
#include "core/variant/dictionary.h"
#include "scene/3d/skeleton_3d.h"
#include "tests/test_macros.h"

namespace TestSkeletonRetargetingSolver {

static bool vector3_is_equal_approx(const Vector3 &v1, const Vector3 &v2, double epsilon = CMP_EPSILON) {
	return Math::is_equal_approx(real_t(v1.x), real_t(v2.x), real_t(epsilon)) &&
			Math::is_equal_approx(real_t(v1.y), real_t(v2.y), real_t(epsilon)) &&
			Math::is_equal_approx(real_t(v1.z), real_t(v2.z), real_t(epsilon));
}

static Skeleton3D *create_simple_test_skeleton(const Vector3 &root_pos, const Vector3 &bone1_tip_pos, float bone1_local_scale = 1.0f) {
	Skeleton3D *skel = memnew(Skeleton3D);

	skel->add_bone("root");
	skel->set_bone_rest(0, Transform3D(Basis(), root_pos));

	skel->add_bone("test_bone");
	skel->set_bone_parent(1, 0);
	// Set a non-identity rest for the bone to make retargeting more meaningful.
	Transform3D bone_rest = Transform3D(Basis().scaled(Vector3(bone1_local_scale, bone1_local_scale, bone1_local_scale)), bone1_tip_pos - root_pos);
	skel->set_bone_rest(1, bone_rest);
	skel->localize_rests();

	return skel;
}

TEST_CASE("[SceneTree][SkeletonRetargetingSolver] Optimize Global Transform for Proportion Matching" * doctest::skip(true)) {
	SkeletonRetargetingSolver *solver = memnew(SkeletonRetargetingSolver);

	Skeleton3D *source_skel = create_simple_test_skeleton(Vector3(0, 0, 0), Vector3(0, 1.0, 0));
	source_skel->set_name("SourceSkeleton");

	Skeleton3D *target_skel = create_simple_test_skeleton(Vector3(5, 5, 5), Vector3(0, 0.5, 0), 0.5f);
	target_skel->set_name("TargetSkeleton");
	target_skel->set_bone_rest(0, Transform3D(Basis(Vector3(0, 1, 0), Math::deg_to_rad(90.0f)), Vector3(5, 5, 5)));
	target_skel->localize_rests();

	solver->set_source_skeleton(source_skel);
	solver->set_target_skeleton(target_skel);

	Dictionary result = solver->call(SNAME("optimize_global_transform"));

	INFO("Global Skeleton Retargeting result: ", result);
	CHECK_MESSAGE(!result.is_empty(), "Result should not be empty.");
	CHECK_MESSAGE(result.has("final_fx"), "Result should contain 'final_fx'.");
	CHECK_MESSAGE(result.has("optimized_root_transform"), "Result should contain 'optimized_root_transform'.");

	Transform3D optimized_transform = result["optimized_root_transform"];
	target_skel->set_global_transform(optimized_transform);
	target_skel->force_update_all_bone_transforms();

	Vector3 source_bone_tip_pos = source_skel->get_bone_global_pose(source_skel->find_bone("test_bone")).origin;
	Vector3 target_bone_tip_pos = target_skel->get_bone_global_pose(target_skel->find_bone("test_bone")).origin;

	MESSAGE("Optimized Target Bone Tip: ", target_bone_tip_pos, " Expected Source Bone Tip: ", source_bone_tip_pos);
	CHECK_MESSAGE(vector3_is_equal_approx(target_bone_tip_pos, source_bone_tip_pos, 0.1), "Optimized target skeleton should align with source.");

	double final_f = result["final_fx"];
	CHECK_MESSAGE(final_f < 0.1, "Final objective function value should be small for a good fit.");

	memdelete(solver);
	memdelete(source_skel);
	memdelete(target_skel);
}

TEST_CASE("[SceneTree][SkeletonRetargetingSolver] Optimize Bone Local Scales for Length Matching" * doctest::skip(true)) {
	SkeletonRetargetingSolver *solver = memnew(SkeletonRetargetingSolver);

	Skeleton3D *source_skel = create_simple_test_skeleton(Vector3(0, 0, 0), Vector3(0, 0.5, 0));
	source_skel->set_name("SourceSkeleton");

	Skeleton3D *target_skel = create_simple_test_skeleton(Vector3(0, 0, 0), Vector3(0, 0.3, 0));
	target_skel->set_name("TargetSkeleton");

	solver->set_source_skeleton(source_skel);
	solver->set_target_skeleton(target_skel);

	Dictionary result = solver->call(SNAME("optimize_bone_local_scales"));

	INFO("Bone Local Scale Retargeting result: ", result);
	CHECK_MESSAGE(!result.is_empty(), "Result should not be empty.");
	CHECK_MESSAGE(result.has("final_fx"), "Result should contain 'final_fx'.");
	CHECK_MESSAGE(result.has("optimized_local_scales"), "Result should contain 'optimized_local_scales'.");

	Dictionary optimized_scales = result["optimized_local_scales"];
	CHECK_MESSAGE(optimized_scales.has("test_bone"), "Should contain optimized scale for 'test_bone'.");
	Vector3 optimized_test_bone_scale = optimized_scales["test_bone"];

	int test_bone_idx = target_skel->find_bone("test_bone");
	Transform3D original_rest = target_skel->get_bone_rest(test_bone_idx);
	target_skel->set_bone_rest(test_bone_idx, Transform3D(original_rest.basis.scaled(optimized_test_bone_scale), original_rest.origin));
	target_skel->localize_rests();
	target_skel->force_update_all_bone_transforms();

	Vector3 target_bone_tip_after_opt = target_skel->get_bone_global_pose(test_bone_idx).origin;
	Vector3 source_bone_tip = source_skel->get_bone_global_pose(source_skel->find_bone("test_bone")).origin;

	MESSAGE("Optimized 'test_bone' length: ", target_bone_tip_after_opt.length(), " Expected length: ", source_bone_tip.length());
	CHECK_MESSAGE(Math::is_equal_approx(double(target_bone_tip_after_opt.length()), double(source_bone_tip.length()), 0.05), "Optimized 'test_bone' length should be close to source bone length.");

	memdelete(solver);
	memdelete(source_skel);
	memdelete(target_skel);
}

} // namespace TestSkeletonRetargetingSolver
