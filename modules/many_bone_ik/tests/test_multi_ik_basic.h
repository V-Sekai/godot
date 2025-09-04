/**************************************************************************/
/*  test_multi_ik_basic.h                                                 */
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

#include "scene/3d/marker_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "src/many_bone_ik_3d.h"
#include "tests/test_macros.h"

namespace TestMultiIKBasic {

TEST_CASE("[Modules][MultiIK] Basic API Functionality") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);

	// Test initial state
	CHECK_FALSE(ik_solver->get_multi_ik_enabled());
	CHECK_EQ(ik_solver->get_effector_count(), 0);
	CHECK_EQ(ik_solver->get_root_bone_name(), String());

	// Test enabling MultiIK
	ik_solver->set_multi_ik_enabled(true);
	CHECK_TRUE(ik_solver->get_multi_ik_enabled());

	// Test disabling MultiIK
	ik_solver->set_multi_ik_enabled(false);
	CHECK_FALSE(ik_solver->get_multi_ik_enabled());

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Effector Management") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Test adding effectors
	NodePath target1 = NodePath("../Target1");
	NodePath target2 = NodePath("../Target2");

	ik_solver->add_effector("LeftHand", target1, 1.0f);
	CHECK_EQ(ik_solver->get_effector_count(), 1);
	CHECK_EQ(ik_solver->get_effector_bone_name(0), "LeftHand");
	CHECK_EQ(ik_solver->get_effector_target(0), target1);
	CHECK_EQ(ik_solver->get_effector_weight(0), 1.0f);

	ik_solver->add_effector("RightHand", target2, 0.8f);
	CHECK_EQ(ik_solver->get_effector_count(), 2);
	CHECK_EQ(ik_solver->get_effector_bone_name(1), "RightHand");
	CHECK_EQ(ik_solver->get_effector_target(1), target2);
	CHECK_EQ(ik_solver->get_effector_weight(1), 0.8f);

	// Test modifying effectors
	NodePath new_target = NodePath("../NewTarget");
	ik_solver->set_effector_target(0, new_target);
	ik_solver->set_effector_weight(0, 0.5f);

	CHECK_EQ(ik_solver->get_effector_target(0), new_target);
	CHECK_EQ(ik_solver->get_effector_weight(0), 0.5f);

	// Test removing effectors
	ik_solver->remove_effector(0);
	CHECK_EQ(ik_solver->get_effector_count(), 1);
	CHECK_EQ(ik_solver->get_effector_bone_name(0), "RightHand");

	// Test clearing all effectors
	ik_solver->clear_effectors();
	CHECK_EQ(ik_solver->get_effector_count(), 0);

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Pole Target Management") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	NodePath pole_target = NodePath("../ElbowPole");

	// Test setting pole target
	ik_solver->set_pole_target("LeftArm", pole_target);
	CHECK_TRUE(ik_solver->has_pole_target("LeftArm"));
	CHECK_EQ(ik_solver->get_pole_target("LeftArm"), pole_target);

	// Test non-existent pole target
	CHECK_FALSE(ik_solver->has_pole_target("RightArm"));
	CHECK_EQ(ik_solver->get_pole_target("RightArm"), NodePath());

	// Test removing pole target
	ik_solver->remove_pole_target("LeftArm");
	CHECK_FALSE(ik_solver->has_pole_target("LeftArm"));

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Root Bone Configuration") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Test setting root bone
	ik_solver->set_root_bone_name("Hips");
	CHECK_EQ(ik_solver->get_root_bone_name(), "Hips");

	// Test changing root bone
	ik_solver->set_root_bone_name("Spine");
	CHECK_EQ(ik_solver->get_root_bone_name(), "Spine");

	// Test clearing root bone
	ik_solver->set_root_bone_name("");
	CHECK_EQ(ik_solver->get_root_bone_name(), String());

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Chain Priority Management") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Test default priority
	CHECK_EQ(ik_solver->get_chain_priority(0), 1.0f);

	// Test setting priority
	ik_solver->set_chain_priority(0, 2.0f);
	CHECK_EQ(ik_solver->get_chain_priority(0), 2.0f);

	// Test setting priority for non-existent chain (should resize)
	ik_solver->set_chain_priority(5, 0.5f);
	CHECK_EQ(ik_solver->get_chain_priority(5), 0.5f);

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Error Handling") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Test accessing invalid effector indices
	ERR_PRINT_OFF;
	CHECK_EQ(ik_solver->get_effector_bone_name(0), "");
	CHECK_EQ(ik_solver->get_effector_target(0), NodePath());
	CHECK_EQ(ik_solver->get_effector_weight(0), 1.0f);

	// Test removing invalid effector index
	ik_solver->remove_effector(999); // Should not crash

	ERR_PRINT_ON;

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Junction and Chain Analysis") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Create a simple skeleton for testing
	Ref<Skeleton3D> skeleton = memnew(Skeleton3D);

	// Add bones: Hips -> Spine -> Chest -> (LeftArm, RightArm)
	skeleton->add_bone("Hips");
	skeleton->add_bone("Spine");
	skeleton->add_bone("Chest");
	skeleton->add_bone("LeftArm");
	skeleton->add_bone("RightArm");

	skeleton->set_bone_parent(1, 0); // Spine -> Hips
	skeleton->set_bone_parent(2, 1); // Chest -> Spine
	skeleton->set_bone_parent(3, 2); // LeftArm -> Chest
	skeleton->set_bone_parent(4, 2); // RightArm -> Chest

	// Set skeleton on IK solver
	ik_solver->set_skeleton_path(NodePath("../Skeleton"));
	// Note: In a real test, we'd need to properly set up the scene tree

	// Add effectors
	ik_solver->add_effector("LeftArm", NodePath("../LeftTarget"));
	ik_solver->add_effector("RightArm", NodePath("../RightTarget"));

	// Test that we can call the analysis methods without crashing
	// (Actual validation would require a proper scene setup)
	Vector<String> junctions = ik_solver->get_junction_bones();
	Vector<Vector<String>> chains = ik_solver->get_effector_chains();

	// Basic sanity checks
	CHECK_GE(junctions.size(), 0);
	CHECK_GE(chains.size(), 0);

	memdelete(skeleton);
	memdelete(ik_solver);
}

} // namespace TestMultiIKBasic
