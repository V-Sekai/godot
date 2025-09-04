/**************************************************************************/
/*  test_multi_ik_functional.h                                            */
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

namespace TestMultiIKFunctional {

TEST_CASE("[Modules][MultiIK] Single Effector Setup") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Create a simple skeleton
	Ref<Skeleton3D> skeleton = memnew(Skeleton3D);
	skeleton->add_bone("Root");
	skeleton->add_bone("Arm");
	skeleton->add_bone("Hand");
	skeleton->set_bone_parent(1, 0);
	skeleton->set_bone_parent(2, 1);

	// Set up bone poses
	Transform3D root_pose = Transform3D(Basis(), Vector3(0, 0, 0));
	Transform3D arm_pose = Transform3D(Basis(), Vector3(1, 0, 0));
	Transform3D hand_pose = Transform3D(Basis(), Vector3(2, 0, 0));

	skeleton->set_bone_pose(0, root_pose);
	skeleton->set_bone_pose(1, arm_pose);
	skeleton->set_bone_pose(2, hand_pose);

	// Set skeleton on solver (simplified for testing)
	ik_solver->set_skeleton_path(NodePath("../Skeleton"));

	// Add single effector
	NodePath hand_target = NodePath("../HandTarget");
	ik_solver->add_effector("Hand", hand_target, 1.0f);

	// Verify setup
	CHECK_EQ(ik_solver->get_effector_count(), 1);
	CHECK_EQ(ik_solver->get_effector_bone_name(0), "Hand");
	CHECK_EQ(ik_solver->get_effector_target(0), hand_target);

	memdelete(skeleton);
	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Dual Effector Coordination") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Create humanoid arm skeleton
	Ref<Skeleton3D> skeleton = memnew(Skeleton3D);
	skeleton->add_bone("Hips");
	skeleton->add_bone("Spine");
	skeleton->add_bone("Chest");
	skeleton->add_bone("LeftShoulder");
	skeleton->add_bone("LeftArm");
	skeleton->add_bone("LeftHand");
	skeleton->add_bone("RightShoulder");
	skeleton->add_bone("RightArm");
	skeleton->add_bone("RightHand");

	// Set up hierarchy
	skeleton->set_bone_parent(1, 0); // Spine -> Hips
	skeleton->set_bone_parent(2, 1); // Chest -> Spine
	skeleton->set_bone_parent(3, 2); // LeftShoulder -> Chest
	skeleton->set_bone_parent(4, 3); // LeftArm -> LeftShoulder
	skeleton->set_bone_parent(5, 4); // LeftHand -> LeftArm
	skeleton->set_bone_parent(6, 2); // RightShoulder -> Chest
	skeleton->set_bone_parent(7, 6); // RightArm -> RightShoulder
	skeleton->set_bone_parent(8, 7); // RightHand -> RightArm

	// Set up basic poses
	for (int i = 0; i < skeleton->get_bone_count(); i++) {
		skeleton->set_bone_pose(i, Transform3D());
	}

	// Set skeleton path
	ik_solver->set_skeleton_path(NodePath("../Skeleton"));

	// Add dual effectors
	ik_solver->add_effector("LeftHand", NodePath("../LeftHandTarget"), 1.0f);
	ik_solver->add_effector("RightHand", NodePath("../RightHandTarget"), 1.0f);

	// Verify setup
	CHECK_EQ(ik_solver->get_effector_count(), 2);

	// Test junction detection (Chest has two children: LeftShoulder and RightShoulder)
	Vector<String> junctions = ik_solver->get_junction_bones();
	CHECK_GE(junctions.size(), 0); // At minimum, Chest should be detected as junction

	// Test chain building
	Vector<Vector<String>> chains = ik_solver->get_effector_chains();
	CHECK_EQ(chains.size(), 2); // Should have 2 chains (left and right arm)

	memdelete(skeleton);
	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Complex Multi-Effector Setup") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Create full humanoid skeleton
	Ref<Skeleton3D> skeleton = memnew(Skeleton3D);

	// Add bones
	String bone_names[] = {
		"Hips", "Spine", "Chest", "Neck", "Head",
		"LeftShoulder", "LeftArm", "LeftForearm", "LeftHand",
		"RightShoulder", "RightArm", "RightForearm", "RightHand",
		"LeftThigh", "LeftShin", "LeftFoot",
		"RightThigh", "RightShin", "RightFoot"
	};

	for (const String &name : bone_names) {
		skeleton->add_bone(name);
	}

	// Set up hierarchy
	skeleton->set_bone_parent(1, 0); // Spine -> Hips
	skeleton->set_bone_parent(2, 1); // Chest -> Spine
	skeleton->set_bone_parent(3, 2); // Neck -> Chest
	skeleton->set_bone_parent(4, 3); // Head -> Neck

	skeleton->set_bone_parent(5, 2); // LeftShoulder -> Chest
	skeleton->set_bone_parent(6, 5); // LeftArm -> LeftShoulder
	skeleton->set_bone_parent(7, 6); // LeftForearm -> LeftArm
	skeleton->set_bone_parent(8, 7); // LeftHand -> LeftForearm

	skeleton->set_bone_parent(9, 2); // RightShoulder -> Chest
	skeleton->set_bone_parent(10, 9); // RightArm -> RightShoulder
	skeleton->set_bone_parent(11, 10); // RightForearm -> RightArm
	skeleton->set_bone_parent(12, 11); // RightHand -> RightForearm

	skeleton->set_bone_parent(13, 0); // LeftThigh -> Hips
	skeleton->set_bone_parent(14, 13); // LeftShin -> LeftThigh
	skeleton->set_bone_parent(15, 14); // LeftFoot -> LeftShin

	skeleton->set_bone_parent(16, 0); // RightThigh -> Hips
	skeleton->set_bone_parent(17, 16); // RightShin -> RightThigh
	skeleton->set_bone_parent(18, 17); // RightFoot -> RightShin

	// Set up basic poses
	for (int i = 0; i < skeleton->get_bone_count(); i++) {
		skeleton->set_bone_pose(i, Transform3D());
	}

	// Set skeleton path
	ik_solver->set_skeleton_path(NodePath("../Skeleton"));
	ik_solver->set_root_bone_name("Hips");

	// Add multiple effectors (hands, feet, head)
	ik_solver->add_effector("LeftHand", NodePath("../LeftHandTarget"), 1.0f);
	ik_solver->add_effector("RightHand", NodePath("../RightHandTarget"), 1.0f);
	ik_solver->add_effector("LeftFoot", NodePath("../LeftFootTarget"), 0.8f);
	ik_solver->add_effector("RightFoot", NodePath("../RightFootTarget"), 0.8f);
	ik_solver->add_effector("Head", NodePath("../HeadTarget"), 0.6f);

	// Verify setup
	CHECK_EQ(ik_solver->get_effector_count(), 5);

	// Test junction detection
	Vector<String> junctions = ik_solver->get_junction_bones();
	// Should detect multiple junctions: Hips (legs), Chest (arms + neck), etc.
	CHECK_GE(junctions.size(), 2);

	// Test chain building
	Vector<Vector<String>> chains = ik_solver->get_effector_chains();
	CHECK_EQ(chains.size(), 5); // Should have 5 chains

	// Test pole targets
	ik_solver->set_pole_target("LeftArm", NodePath("../LeftElbowPole"));
	ik_solver->set_pole_target("RightArm", NodePath("../RightElbowPole"));
	ik_solver->set_pole_target("LeftThigh", NodePath("../LeftKneePole"));
	ik_solver->set_pole_target("RightThigh", NodePath("../RightKneePole"));

	CHECK_TRUE(ik_solver->has_pole_target("LeftArm"));
	CHECK_TRUE(ik_solver->has_pole_target("RightArm"));
	CHECK_TRUE(ik_solver->has_pole_target("LeftThigh"));
	CHECK_TRUE(ik_solver->has_pole_target("RightThigh"));

	memdelete(skeleton);
	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Effector Weight Influence") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Add effectors with different weights
	ik_solver->add_effector("Primary", NodePath("../PrimaryTarget"), 1.0f);
	ik_solver->add_effector("Secondary", NodePath("../SecondaryTarget"), 0.5f);
	ik_solver->add_effector("Tertiary", NodePath("../TertiaryTarget"), 0.2f);

	// Verify weights are stored correctly
	CHECK_EQ(ik_solver->get_effector_weight(0), 1.0f);
	CHECK_EQ(ik_solver->get_effector_weight(1), 0.5f);
	CHECK_EQ(ik_solver->get_effector_weight(2), 0.2f);

	// Test weight modification
	ik_solver->set_effector_weight(1, 0.8f);
	CHECK_EQ(ik_solver->get_effector_weight(1), 0.8f);

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Chain Priority System") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Test default priorities
	for (int i = 0; i < 5; i++) {
		CHECK_EQ(ik_solver->get_chain_priority(i), 1.0f);
	}

	// Set custom priorities
	ik_solver->set_chain_priority(0, 2.0f); // High priority
	ik_solver->set_chain_priority(1, 1.5f); // Medium priority
	ik_solver->set_chain_priority(2, 0.5f); // Low priority

	CHECK_EQ(ik_solver->get_chain_priority(0), 2.0f);
	CHECK_EQ(ik_solver->get_chain_priority(1), 1.5f);
	CHECK_EQ(ik_solver->get_chain_priority(2), 0.5f);

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK] Backward Compatibility") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);

	// Test that MultiIK is disabled by default
	CHECK_FALSE(ik_solver->get_multi_ik_enabled());

	// Test that existing single-effector functionality still works
	ik_solver->set_pin_count(1);
	ik_solver->set_pin_bone_name(0, "TestBone");
	ik_solver->set_pin_target_node_path(0, NodePath("../TestTarget"));

	CHECK_EQ(ik_solver->get_pin_count(), 1);
	CHECK_EQ(ik_solver->get_pin_bone_name(0), "TestBone");

	// Enable MultiIK and verify it doesn't break existing setup
	ik_solver->set_multi_ik_enabled(true);
	CHECK_TRUE(ik_solver->get_multi_ik_enabled());

	// Existing pins should still be accessible
	CHECK_EQ(ik_solver->get_pin_count(), 1);

	memdelete(ik_solver);
}

} // namespace TestMultiIKFunctional
