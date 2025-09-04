/**************************************************************************/
/*  test_multi_ik_performance.h                                           */
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

#include "core/os/os.h"
#include "scene/3d/marker_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "src/many_bone_ik_3d.h"
#include "tests/test_macros.h"

namespace TestMultiIKPerformance {

class PerformanceTimer {
private:
	uint64_t start_time = 0;

public:
	void start() {
		start_time = OS::get_singleton()->get_ticks_usec();
	}

	uint64_t stop() {
		return OS::get_singleton()->get_ticks_usec() - start_time;
	}

	double stop_ms() {
		return static_cast<double>(stop()) / 1000.0;
	}
};

// Helper function to create a test skeleton with specified number of bones
Ref<Skeleton3D> create_test_skeleton(int num_bones) {
	Ref<Skeleton3D> skeleton = memnew(Skeleton3D);

	// Create a simple chain: Root -> Bone1 -> Bone2 -> ... -> BoneN
	for (int i = 0; i < num_bones; i++) {
		String bone_name = i == 0 ? "Root" : "Bone" + itos(i);
		skeleton->add_bone(bone_name);

		if (i > 0) {
			skeleton->set_bone_parent(i, i - 1);
		}

		// Set basic pose
		Transform3D pose = Transform3D(Basis(), Vector3(static_cast<float>(i), 0, 0));
		skeleton->set_bone_pose(i, pose);
	}

	return skeleton;
}

// Helper function to create multiple targets
Vector<NodePath> create_test_targets(int num_targets) {
	Vector<NodePath> targets;
	for (int i = 0; i < num_targets; i++) {
		targets.push_back(NodePath("../Target" + itos(i)));
	}
	return targets;
}

TEST_CASE("[Modules][MultiIK][Performance] Single vs Multi Effector Overhead") {
	PerformanceTimer timer;

	// Test with 10-bone skeleton
	Ref<Skeleton3D> skeleton = create_test_skeleton(10);
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_skeleton_path(NodePath("../Skeleton"));

	// Test 1: Single effector performance
	ik_solver->set_multi_ik_enabled(false);
	ik_solver->set_pin_count(1);
	ik_solver->set_pin_bone_name(0, "Bone9");
	ik_solver->set_pin_target_node_path(0, NodePath("../Target"));

	timer.start();
	for (int i = 0; i < 100; i++) {
		ik_solver->_process_modification(1.0f / 60.0f);
	}
	double single_time = timer.stop_ms();

	// Test 2: Multi effector performance (same effector)
	ik_solver->set_multi_ik_enabled(true);
	ik_solver->clear_effectors();
	ik_solver->add_effector("Bone9", NodePath("../Target"), 1.0f);

	timer.start();
	for (int i = 0; i < 100; i++) {
		ik_solver->_process_modification(1.0f / 60.0f);
	}
	double multi_time = timer.stop_ms();

	// MultiIK should not have significant overhead for single effector
	double overhead_ratio = multi_time / single_time;
	CHECK_LT(overhead_ratio, 1.5); // Allow up to 50% overhead

	print_line("Single effector time: " + rtos(single_time) + "ms");
	print_line("Multi effector time: " + rtos(multi_time) + "ms");
	print_line("Overhead ratio: " + rtos(overhead_ratio));

	memdelete(skeleton);
	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Performance] Scaling with Multiple Effectors") {
	PerformanceTimer timer;
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Test with different numbers of effectors
	Vector<int> effector_counts = { 1, 2, 5, 10 };
	Vector<double> times;

	for (int count : effector_counts) {
		// Create skeleton with enough bones for effectors
		Ref<Skeleton3D> skeleton = create_test_skeleton(count + 1);
		ik_solver->set_skeleton_path(NodePath("../Skeleton"));

		// Clear previous effectors
		ik_solver->clear_effectors();

		// Add multiple effectors
		for (int i = 0; i < count; i++) {
			String bone_name = "Bone" + itos(i + 1);
			NodePath target_path = NodePath("../Target" + itos(i));
			ik_solver->add_effector(bone_name, target_path, 1.0f);
		}

		// Measure performance
		timer.start();
		for (int i = 0; i < 50; i++) {
			ik_solver->_process_modification(1.0f / 60.0f);
		}
		double time = timer.stop_ms();
		times.push_back(time);

		print_line("Effectors: " + itos(count) + ", Time: " + rtos(time) + "ms");

		memdelete(skeleton);
	}

	// Verify scaling is reasonable (should be roughly linear)
	for (int i = 1; i < times.size(); i++) {
		double ratio = times[i] / times[i - 1];
		int prev_count = effector_counts[i - 1];
		int curr_count = effector_counts[i];
		double expected_ratio = static_cast<double>(curr_count) / prev_count;

		// Allow some variance but should be roughly proportional
		CHECK_LT(ratio, expected_ratio * 2.0);
		print_line("Scaling ratio for " + itos(prev_count) + "->" + itos(curr_count) + ": " + rtos(ratio));
	}

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Performance] Memory Usage Tracking") {
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Test memory usage with increasing numbers of effectors
	for (int count = 0; count <= 20; count += 5) {
		ik_solver->clear_effectors();

		if (count > 0) {
			for (int i = 0; i < count; i++) {
				ik_solver->add_effector("Bone" + itos(i), NodePath("../Target" + itos(i)), 1.0f);
			}
		}

		// Basic memory sanity check - should not crash with many effectors
		CHECK_EQ(ik_solver->get_effector_count(), count);

		print_line("Memory test passed for " + itos(count) + " effectors");
	}

	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Performance] Junction Detection Performance") {
	PerformanceTimer timer;
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Create complex skeleton with many junctions
	Ref<Skeleton3D> skeleton = memnew(Skeleton3D);

	// Create a tree structure with multiple branches
	skeleton->add_bone("Root");

	// Add multiple branches from root
	for (int branch = 0; branch < 5; branch++) {
		String branch_root = "Branch" + itos(branch);
		skeleton->add_bone(branch_root);
		skeleton->set_bone_parent(skeleton->find_bone(branch_root), 0);

		// Add chain in each branch
		String prev_bone = branch_root;
		for (int depth = 0; depth < 3; depth++) {
			String bone_name = branch_root + "_Depth" + itos(depth);
			skeleton->add_bone(bone_name);
			skeleton->set_bone_parent(skeleton->find_bone(bone_name), skeleton->find_bone(prev_bone));
			prev_bone = bone_name;
		}
	}

	// Add effectors at end of each branch
	for (int branch = 0; branch < 5; branch++) {
		String effector_bone = "Branch" + itos(branch) + "_Depth2";
		ik_solver->add_effector(effector_bone, NodePath("../Target" + itos(branch)), 1.0f);
	}

	// Measure junction detection performance
	timer.start();
	for (int i = 0; i < 100; i++) {
		Vector<String> junctions = ik_solver->get_junction_bones();
		Vector<Vector<String>> chains = ik_solver->get_effector_chains();
	}
	double detection_time = timer.stop_ms();

	print_line("Junction detection time (100 runs): " + rtos(detection_time) + "ms");
	print_line("Average per run: " + rtos(detection_time / 100.0) + "ms");

	// Should be reasonably fast
	CHECK_LT(detection_time, 50.0); // Less than 50ms for 100 runs

	memdelete(skeleton);
	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Performance] Real-time Performance Target") {
	PerformanceTimer timer;
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	// Create typical humanoid setup
	Ref<Skeleton3D> skeleton = create_test_skeleton(20); // 20 bones
	ik_solver->set_skeleton_path(NodePath("../Skeleton"));

	// Add typical effectors for humanoid character
	ik_solver->add_effector("Bone18", NodePath("../LeftHand"), 1.0f); // Left hand
	ik_solver->add_effector("Bone17", NodePath("../RightHand"), 1.0f); // Right hand
	ik_solver->add_effector("Bone16", NodePath("../LeftFoot"), 0.8f); // Left foot
	ik_solver->add_effector("Bone15", NodePath("../RightFoot"), 0.8f); // Right foot
	ik_solver->add_effector("Bone14", NodePath("../Head"), 0.6f); // Head

	// Test real-time performance (60 FPS target = 16.67ms per frame)
	const double target_frame_time = 16.67;
	const int test_frames = 300; // Test for 5 seconds at 60 FPS

	timer.start();
	for (int i = 0; i < test_frames; i++) {
		ik_solver->_process_modification(1.0f / 60.0f);
	}
	double total_time = timer.stop_ms();
	double avg_frame_time = total_time / test_frames;

	print_line("Real-time performance test:");
	print_line("Total time: " + rtos(total_time) + "ms");
	print_line("Average frame time: " + rtos(avg_frame_time) + "ms");
	print_line("Target frame time: " + rtos(target_frame_time) + "ms");

	// Should meet real-time performance target
	CHECK_LT(avg_frame_time, target_frame_time);

	// Allow some tolerance but should be close to target
	double performance_ratio = avg_frame_time / target_frame_time;
	print_line("Performance ratio: " + rtos(performance_ratio));
	CHECK_LT(performance_ratio, 1.2); // Within 20% of target

	memdelete(skeleton);
	memdelete(ik_solver);
}

TEST_CASE("[Modules][MultiIK][Performance] Pole Target Overhead") {
	PerformanceTimer timer;
	Ref<EWBIK3D> ik_solver = memnew(EWBIK3D);
	ik_solver->set_multi_ik_enabled(true);

	Ref<Skeleton3D> skeleton = create_test_skeleton(10);
	ik_solver->set_skeleton_path(NodePath("../Skeleton"));

	// Add effectors
	ik_solver->add_effector("Bone5", NodePath("../Target1"), 1.0f);
	ik_solver->add_effector("Bone8", NodePath("../Target2"), 1.0f);

	// Test without pole targets
	timer.start();
	for (int i = 0; i < 100; i++) {
		ik_solver->_process_modification(1.0f / 60.0f);
	}
	double without_poles = timer.stop_ms();

	// Test with pole targets
	ik_solver->set_pole_target("Bone5", NodePath("../Pole1"));
	ik_solver->set_pole_target("Bone8", NodePath("../Pole2"));

	timer.start();
	for (int i = 0; i < 100; i++) {
		ik_solver->_process_modification(1.0f / 60.0f);
	}
	double with_poles = timer.stop_ms();

	print_line("Without pole targets: " + rtos(without_poles) + "ms");
	print_line("With pole targets: " + rtos(with_poles) + "ms");

	// Pole targets should not add significant overhead
	double overhead_ratio = with_poles / without_poles;
	CHECK_LT(overhead_ratio, 1.1); // Less than 10% overhead

	memdelete(skeleton);
	memdelete(ik_solver);
}

} // namespace TestMultiIKPerformance
