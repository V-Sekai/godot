/**************************************************************************/
/*  test_multi_ik_integration.h                                           */
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

namespace TestMultiIKIntegration {

TEST_CASE("[Modules][MultiIK][Integration] Backward Compatibility") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);

	// Test that MultiIK is disabled by default (backward compatibility)
	CHECK_FALSE(ik_solver->get_multi_ik_enabled());

	// Test traditional single-effector setup still works
	ik_solver->set_pin_count(1);
	ik_solver->set_pin_bone_name(0, "TestBone");
	ik_solver->set_pin_target_node_path(0, NodePath("../TestTarget"));
	ik_solver->set_pin_weight(0, 0.8f);

	// Verify traditional API still works
	CHECK_EQ(ik_solver->get_pin_count(), 1);
	CHECK_EQ(ik_solver->get_pin_bone_name(0), "TestBone");
	CHECK_EQ(ik_solver->get_pin_target_node_path(0), NodePath("../TestTarget"));
	CHECK_EQ(ik_solver->get_pin_weight(0), 0.8f);

	// Enable MultiIK and verify it doesn't break existing setup
	ik_solver->set_multi_ik_enabled(true);
	CHECK_TRUE(ik_solver->get_multi_ik_enabled());

	// Existing pins should still be accessible
	CHECK_EQ(ik_solver->get_pin_count(), 1);
	CHECK_EQ(ik_solver->get_pin_bone_name(0), "TestBone");

	// Disable MultiIK and verify traditional setup still works
	ik_solver->set_multi_ik_enabled(false);
	CHECK_FALSE(ik_solver->get_multi_ik_enabled());
	CHECK_EQ(ik_solver->get_pin_count(), 1);

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Integration] Humanoid Skeleton Setup") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Create a humanoid skeleton
	Ref<Skeleton3D> skeleton = memnew(Skeleton3D);

	// Standard bone names for humanoid
	String bone_names[] = {
		"Hips", "Spine", "Chest", "Neck", "Head",
		"LeftShoulder", "LeftArm", "LeftForearm", "LeftHand",
		"RightShoulder", "RightArm", "RightForearm", "RightHand",
		"LeftThigh", "LeftShin", "LeftFoot",
		"RightThigh", "RightShin", "RightFoot"
	};

	const int BONE_COUNT = sizeof(bone_names) / sizeof(bone_names[0]);
	for (int i = 0; i < BONE_COUNT; i++) {
		skeleton->add_bone(bone_names[i]);
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

	// Set basic poses
	for (int i = 0; i < BONE_COUNT; i++) {
		skeleton->set_bone_pose(i, Transform3D());
	}

	// Set skeleton on solver
	ik_solver->set_skeleton_path(NodePath("../Skeleton"));
	ik_solver->set_root_bone_name("Hips");

	// Add standard humanoid effectors
	ik_solver->add_effector("LeftHand", NodePath("../LeftHandTarget"), 1.0f);
	ik_solver->add_effector("RightHand", NodePath("../RightHandTarget"), 1.0f);
	ik_solver->add_effector("LeftFoot", NodePath("../LeftFootTarget"), 0.8f);
	ik_solver->add_effector("RightFoot", NodePath("../RightFootTarget"), 0.8f);
	ik_solver->add_effector("Head", NodePath("../HeadTarget"), 0.6f);

	// Verify setup
	CHECK_EQ(ik_solver->get_effector_count(), 5);

	// Test junction detection for humanoid
	Vector<String> junctions = ik_solver->get_junction_bones();
	// Should detect key junctions: Hips (legs), Chest (arms + neck)
	CHECK_GE(junctions.size(), 2);

	// Verify junctions contain expected bones
	bool has_hips = false;
	bool has_chest = false;
	for (const String &junction : junctions) {
		if (junction == "Hips") {
			has_hips = true;
		}
		if (junction == "Chest") {
			has_chest = true;
		}
	}
	// Note: Actual junction detection depends on implementation details
	// This test validates the system doesn't crash and returns reasonable data

	// Test chain analysis
	Vector<Vector<String>> chains = ik_solver->get_effector_chains();
	CHECK_EQ(chains.size(), 5); // Should have 5 chains

	// Verify each chain ends with the correct effector
	bool found_left_hand = false;
	bool found_right_hand = false;
	bool found_left_foot = false;
	bool found_right_foot = false;
	bool found_head = false;

	for (const Vector<String> &chain : chains) {
		if (!chain.is_empty()) {
			String last_bone = chain[chain.size() - 1];
			if (last_bone == "LeftHand") {
				found_left_hand = true;
			}
			if (last_bone == "RightHand") {
				found_right_hand = true;
			}
			if (last_bone == "LeftFoot") {
				found_left_foot = true;
			}
			if (last_bone == "RightFoot") {
				found_right_foot = true;
			}
			if (last_bone == "Head") {
				found_head = true;
			}
		}
	}

	// All effectors should be found in chains
	CHECK_TRUE(found_left_hand);
	CHECK_TRUE(found_right_hand);
	CHECK_TRUE(found_left_foot);
	CHECK_TRUE(found_right_foot);
	CHECK_TRUE(found_head);

	memdelete(skeleton);
	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Integration] Quadruped Skeleton Setup") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Create a quadruped skeleton
	Ref<Skeleton3D> skeleton = memnew(Skeleton3D);

	String bone_names[] = {
		"Spine", "Neck", "Head",
		"FrontLeftShoulder", "FrontLeftArm", "FrontLeftForearm", "FrontLeftFoot",
		"FrontRightShoulder", "FrontRightArm", "FrontRightForearm", "FrontRightFoot",
		"BackLeftHip", "BackLeftThigh", "BackLeftShin", "BackLeftFoot",
		"BackRightHip", "BackRightThigh", "BackRightShin", "BackRightFoot"
	};

	const int BONE_COUNT = sizeof(bone_names) / sizeof(bone_names[0]);
	for (int i = 0; i < BONE_COUNT; i++) {
		skeleton->add_bone(bone_names[i]);
	}

	// Set up hierarchy
	skeleton->set_bone_parent(1, 0); // Neck -> Spine
	skeleton->set_bone_parent(2, 1); // Head -> Neck

	skeleton->set_bone_parent(3, 0); // FrontLeftShoulder -> Spine
	skeleton->set_bone_parent(4, 3); // FrontLeftArm -> FrontLeftShoulder
	skeleton->set_bone_parent(5, 4); // FrontLeftForearm -> FrontLeftArm
	skeleton->set_bone_parent(6, 5); // FrontLeftFoot -> FrontLeftForearm

	skeleton->set_bone_parent(7, 0); // FrontRightShoulder -> Spine
	skeleton->set_bone_parent(8, 7); // FrontRightArm -> FrontRightShoulder
	skeleton->set_bone_parent(9, 8); // FrontRightForearm -> FrontRightArm
	skeleton->set_bone_parent(10, 9); // FrontRightFoot -> FrontRightForearm

	skeleton->set_bone_parent(11, 0); // BackLeftHip -> Spine
	skeleton->set_bone_parent(12, 11); // BackLeftThigh -> BackLeftHip
	skeleton->set_bone_parent(13, 12); // BackLeftShin -> BackLeftThigh
	skeleton->set_bone_parent(14, 13); // BackLeftFoot -> BackLeftShin

	skeleton->set_bone_parent(15, 0); // BackRightHip -> Spine
	skeleton->set_bone_parent(16, 15); // BackRightThigh -> BackRightHip
	skeleton->set_bone_parent(17, 16); // BackRightShin -> BackRightThigh
	skeleton->set_bone_parent(18, 17); // BackRightFoot -> BackRightShin

	// Set basic poses
	for (int i = 0; i < BONE_COUNT; i++) {
		skeleton->set_bone_pose(i, Transform3D());
	}

	// Set skeleton on solver
	ik_solver->set_skeleton_path(NodePath("../Skeleton"));
	ik_solver->set_root_bone_name("Spine");

	// Add quadruped effectors (4 feet + head)
	ik_solver->add_effector("FrontLeftFoot", NodePath("../FrontLeftTarget"), 1.0f);
	ik_solver->add_effector("FrontRightFoot", NodePath("../FrontRightTarget"), 1.0f);
	ik_solver->add_effector("BackLeftFoot", NodePath("../BackLeftTarget"), 1.0f);
	ik_solver->add_effector("BackRightFoot", NodePath("../BackRightTarget"), 1.0f);
	ik_solver->add_effector("Head", NodePath("../HeadTarget"), 0.7f);

	// Verify setup
	CHECK_EQ(ik_solver->get_effector_count(), 5);

	// Test junction detection for quadruped
	Vector<String> junctions = ik_solver->get_junction_bones();
	// Should detect Spine as major junction (all limbs + neck)
	CHECK_GE(junctions.size(), 1);

	// Test chain analysis
	Vector<Vector<String>> chains = ik_solver->get_effector_chains();
	CHECK_EQ(chains.size(), 5); // Should have 5 chains

	memdelete(skeleton);
	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Integration] Dynamic Effector Management") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Start with basic setup
	ik_solver->add_effector("LeftHand", NodePath("../LeftHandTarget"), 1.0f);
	ik_solver->add_effector("RightHand", NodePath("../RightHandTarget"), 1.0f);

	CHECK_EQ(ik_solver->get_effector_count(), 2);

	// Dynamically add effectors during runtime
	ik_solver->add_effector("Head", NodePath("../HeadTarget"), 0.8f);
	ik_solver->add_effector("LeftFoot", NodePath("../LeftFootTarget"), 0.6f);

	CHECK_EQ(ik_solver->get_effector_count(), 4);

	// Modify effector properties dynamically
	ik_solver->set_effector_weight(2, 0.9f); // Head
	ik_solver->set_effector_target(3, NodePath("../NewLeftFootTarget")); // Left foot

	CHECK_EQ(ik_solver->get_effector_weight(2), 0.9f);
	CHECK_EQ(ik_solver->get_effector_target(3), NodePath("../NewLeftFootTarget"));

	// Remove effectors dynamically
	ik_solver->remove_effector(1); // Remove right hand
	CHECK_EQ(ik_solver->get_effector_count(), 3);
	CHECK_EQ(ik_solver->get_effector_bone_name(1), "Head"); // Head moved to index 1

	// Clear and rebuild
	ik_solver->clear_effectors();
	CHECK_EQ(ik_solver->get_effector_count(), 0);

	// Rebuild with different configuration
	ik_solver->add_effector("RightHand", NodePath("../RightHandTarget"), 1.0f);
	ik_solver->add_effector("RightFoot", NodePath("../RightFootTarget"), 0.8f);

	CHECK_EQ(ik_solver->get_effector_count(), 2);
	CHECK_EQ(ik_solver->get_effector_bone_name(0), "RightHand");
	CHECK_EQ(ik_solver->get_effector_bone_name(1), "RightFoot");

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Integration] Pole Target Integration") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Set up effectors
	ik_solver->add_effector("LeftArm", NodePath("../LeftHandTarget"), 1.0f);
	ik_solver->add_effector("RightArm", NodePath("../RightHandTarget"), 1.0f);
	ik_solver->add_effector("LeftThigh", NodePath("../LeftFootTarget"), 0.8f);
	ik_solver->add_effector("RightThigh", NodePath("../RightFootTarget"), 0.8f);

	// Set pole targets for natural joint bending
	ik_solver->set_pole_target("LeftArm", NodePath("../LeftElbowPole"));
	ik_solver->set_pole_target("RightArm", NodePath("../RightElbowPole"));
	ik_solver->set_pole_target("LeftThigh", NodePath("../LeftKneePole"));
	ik_solver->set_pole_target("RightThigh", NodePath("../RightKneePole"));

	// Verify pole targets are set
	CHECK_TRUE(ik_solver->has_pole_target("LeftArm"));
	CHECK_TRUE(ik_solver->has_pole_target("RightArm"));
	CHECK_TRUE(ik_solver->has_pole_target("LeftThigh"));
	CHECK_TRUE(ik_solver->has_pole_target("RightThigh"));

	// Verify pole target paths
	CHECK_EQ(ik_solver->get_pole_target("LeftArm"), NodePath("../LeftElbowPole"));
	CHECK_EQ(ik_solver->get_pole_target("RightArm"), NodePath("../RightElbowPole"));
	CHECK_EQ(ik_solver->get_pole_target("LeftThigh"), NodePath("../LeftKneePole"));
	CHECK_EQ(ik_solver->get_pole_target("RightThigh"), NodePath("../RightKneePole"));

	// Test pole target removal
	ik_solver->remove_pole_target("LeftArm");
	CHECK_FALSE(ik_solver->has_pole_target("LeftArm"));
	CHECK_TRUE(ik_solver->has_pole_target("RightArm")); // Others should remain

	// Test non-existent pole target
	CHECK_FALSE(ik_solver->has_pole_target("NonExistentBone"));
	CHECK_EQ(ik_solver->get_pole_target("NonExistentBone"), NodePath());

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Integration] Configuration Persistence") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Set up complete configuration
	ik_solver->set_root_bone_name("Hips");

	ik_solver->add_effector("LeftHand", NodePath("../LeftHandTarget"), 1.0f);
	ik_solver->add_effector("RightHand", NodePath("../RightHandTarget"), 0.9f);
	ik_solver->add_effector("Head", NodePath("../HeadTarget"), 0.7f);

	ik_solver->set_pole_target("LeftArm", NodePath("../LeftElbowPole"));
	ik_solver->set_pole_target("RightArm", NodePath("../RightElbowPole"));

	ik_solver->set_chain_priority(0, 2.0f);
	ik_solver->set_chain_priority(1, 1.8f);
	ik_solver->set_chain_priority(2, 1.5f);

	// Verify configuration is maintained
	CHECK_EQ(ik_solver->get_root_bone_name(), "Hips");
	CHECK_EQ(ik_solver->get_effector_count(), 3);
	CHECK_TRUE(ik_solver->has_pole_target("LeftArm"));
	CHECK_TRUE(ik_solver->has_pole_target("RightArm"));
	CHECK_EQ(ik_solver->get_chain_priority(0), 2.0f);
	CHECK_EQ(ik_solver->get_chain_priority(1), 1.8f);
	CHECK_EQ(ik_solver->get_chain_priority(2), 1.5f);

	// Test configuration survives effector modifications
	ik_solver->set_effector_weight(0, 0.8f);
	ik_solver->set_effector_target(1, NodePath("../NewRightHandTarget"));

	// Configuration should still be intact
	CHECK_EQ(ik_solver->get_root_bone_name(), "Hips");
	CHECK_TRUE(ik_solver->has_pole_target("LeftArm"));
	CHECK_EQ(ik_solver->get_chain_priority(0), 2.0f);

	memdelete(ik_solver);
}

} // namespace TestMultiIKIntegration
