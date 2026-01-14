/**************************************************************************/
/*  test_ik_common.h                                                      */
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

#include "core/templates/vector.h"
#include "scene/3d/skeleton_3d.h"

// Helper function to create a humanoid arm skeleton with contracted/muscle-contracted rest poses
Skeleton3D *create_humanoid_arm_skeleton() {
	Skeleton3D *skeleton = memnew(Skeleton3D);

	// Create bones: LeftShoulder -> LeftUpperArm -> LeftLowerArm -> LeftHand
	skeleton->add_bone("Hips");
	skeleton->add_bone("Spine");
	skeleton->add_bone("Chest");
	skeleton->add_bone("UpperChest");
	skeleton->add_bone("LeftShoulder");
	skeleton->add_bone("LeftUpperArm");
	skeleton->add_bone("LeftLowerArm");
	skeleton->add_bone("LeftHand");

	// Set hierarchy
	skeleton->set_bone_parent(1, 0); // Spine -> Hips
	skeleton->set_bone_parent(2, 1); // Chest -> Spine
	skeleton->set_bone_parent(3, 2); // UpperChest -> Chest
	skeleton->set_bone_parent(4, 3); // LeftShoulder -> UpperChest
	skeleton->set_bone_parent(5, 4); // LeftUpperArm -> LeftShoulder
	skeleton->set_bone_parent(6, 5); // LeftLowerArm -> LeftUpperArm
	skeleton->set_bone_parent(7, 6); // LeftHand -> LeftLowerArm

	// Set contracted/muscle-contracted rest poses for challenging IK tests
	skeleton->set_bone_rest(0, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0, 0.9, 0))); // Hips
	skeleton->set_bone_rest(1, Transform3D(Basis::from_euler(Vector3(0.087, 0, 0)), Vector3(0, 0.1, 0))); // Spine (5° forward lean)
	skeleton->set_bone_rest(2, Transform3D(Basis::from_euler(Vector3(0.087, 0, 0)), Vector3(0, 0.1, 0))); // Chest (5° forward lean)
	skeleton->set_bone_rest(3, Transform3D(Basis::from_euler(Vector3(0.087, 0, 0)), Vector3(0, 0.1, 0))); // UpperChest (5° forward lean)
	skeleton->set_bone_rest(4, Transform3D(Basis::from_euler(Vector3(0, 0, -0.175)), Vector3(0.05, 0.1, 0))); // LeftShoulder (-10° outward rotation)
	skeleton->set_bone_rest(5, Transform3D(Basis::from_euler(Vector3(-0.349, 0, -0.087)), Vector3(0, -0.3, 0))); // LeftUpperArm (-20° downward with -5° twist)
	skeleton->set_bone_rest(6, Transform3D(Basis::from_euler(Vector3(1.571, 0, 0)), Vector3(0, -0.35, 0))); // LeftLowerArm (90° bend - contracted elbow)
	skeleton->set_bone_rest(7, Transform3D(Basis::from_euler(Vector3(0, 0.175, 0)), Vector3(0, -0.25, 0))); // LeftHand (10° pronation)

	for (int i = 0; i < skeleton->get_bone_count(); ++i) {
		skeleton->set_bone_global_pose(i, skeleton->get_bone_rest(i));
	}

	return skeleton;
}

// Helper function to create a full humanoid skeleton with both arms
Skeleton3D *create_humanoid_skeleton() {
	Skeleton3D *skeleton = memnew(Skeleton3D);

	// Create bones: Hips -> Spine -> Chest -> UpperChest -> Shoulders -> Arms -> Hands
	// Based on Godot's SkeletonProfileHumanoid reference poses
	skeleton->add_bone("Hips");
	skeleton->add_bone("Spine");
	skeleton->add_bone("Chest");
	skeleton->add_bone("UpperChest");
	skeleton->add_bone("LeftShoulder");
	skeleton->add_bone("LeftUpperArm");
	skeleton->add_bone("LeftLowerArm");
	skeleton->add_bone("LeftHand");
	skeleton->add_bone("RightShoulder");
	skeleton->add_bone("RightUpperArm");
	skeleton->add_bone("RightLowerArm");
	skeleton->add_bone("RightHand");

	// Set hierarchy
	skeleton->set_bone_parent(1, 0); // Spine -> Hips
	skeleton->set_bone_parent(2, 1); // Chest -> Spine
	skeleton->set_bone_parent(3, 2); // UpperChest -> Chest
	skeleton->set_bone_parent(4, 3); // LeftShoulder -> UpperChest
	skeleton->set_bone_parent(5, 4); // LeftUpperArm -> LeftShoulder
	skeleton->set_bone_parent(6, 5); // LeftLowerArm -> LeftUpperArm
	skeleton->set_bone_parent(7, 6); // LeftHand -> LeftLowerArm
	skeleton->set_bone_parent(8, 3); // RightShoulder -> UpperChest
	skeleton->set_bone_parent(9, 8); // RightUpperArm -> RightShoulder
	skeleton->set_bone_parent(10, 9); // RightLowerArm -> RightUpperArm
	skeleton->set_bone_parent(11, 10); // RightHand -> RightLowerArm

	// Set rest poses based on Godot SkeletonProfileHumanoid reference poses
	// Bone lengths and positions match the standard humanoid profile
	skeleton->set_bone_rest(0, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0, 0.75, 0))); // Hips (at y=0.75)
	skeleton->set_bone_rest(1, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0, 0.1, 0))); // Spine (0.1 units up from hips)
	skeleton->set_bone_rest(2, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0, 0.1, 0))); // Chest (0.1 units up from spine)
	skeleton->set_bone_rest(3, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0, 0.1, 0))); // UpperChest (0.1 units up from chest)

	// Left arm - T-pose with proper bone lengths from humanoid profile
	skeleton->set_bone_rest(4, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0.05, 0.1, 0))); // LeftShoulder (0.05 right, 0.1 up from upper chest)
	skeleton->set_bone_rest(5, Transform3D(Basis::from_euler(Vector3(0, 0, -1.571)), Vector3(0, 0.05, 0))); // LeftUpperArm (0.05 down from shoulder, 90° outward rotation)
	skeleton->set_bone_rest(6, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0, 0.25, 0))); // LeftLowerArm (0.25 down from upper arm)
	skeleton->set_bone_rest(7, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0, 0.25, 0))); // LeftHand (0.25 down from lower arm)

	// Right arm - symmetric to left arm
	skeleton->set_bone_rest(8, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(-0.05, 0.1, 0))); // RightShoulder (-0.05 left, 0.1 up from upper chest)
	skeleton->set_bone_rest(9, Transform3D(Basis::from_euler(Vector3(0, 0, 1.571)), Vector3(0, 0.05, 0))); // RightUpperArm (0.05 down from shoulder, -90° outward rotation)
	skeleton->set_bone_rest(10, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0, 0.25, 0))); // RightLowerArm (0.25 down from upper arm)
	skeleton->set_bone_rest(11, Transform3D(Basis::from_euler(Vector3(0, 0, 0)), Vector3(0, 0.25, 0))); // RightHand (0.25 down from lower arm)

	for (int i = 0; i < skeleton->get_bone_count(); ++i) {
		skeleton->set_bone_global_pose(i, skeleton->get_bone_rest(i));
	}

	return skeleton;
}

// Helper function to create a humanoid leg skeleton with contracted/muscle-contracted rest poses
Skeleton3D *create_humanoid_leg_skeleton() {
	Skeleton3D *skeleton = memnew(Skeleton3D);

	// Create bones: Hips -> LeftUpperLeg -> LeftLowerLeg -> LeftFoot -> LeftToes
	skeleton->add_bone("Hips");
	skeleton->add_bone("LeftUpperLeg");
	skeleton->add_bone("LeftLowerLeg");
	skeleton->add_bone("LeftFoot");
	skeleton->add_bone("LeftToes");

	// Set hierarchy
	skeleton->set_bone_parent(1, 0); // LeftUpperLeg -> Hips
	skeleton->set_bone_parent(2, 1); // LeftLowerLeg -> LeftUpperLeg
	skeleton->set_bone_parent(3, 2); // LeftFoot -> LeftLowerLeg
	skeleton->set_bone_parent(4, 3); // LeftToes -> LeftFoot

	// Set contracted/muscle-contracted rest poses for challenging IK tests
	skeleton->set_bone_rest(0, Transform3D(Basis(), Vector3(0, 0.9, 0))); // Hips
	skeleton->set_bone_rest(1, Transform3D(Basis(), Vector3(0, -0.45, 0))); // LeftUpperLeg down
	skeleton->set_bone_rest(2, Transform3D(Basis::from_euler(Vector3(1.047, 0, 0)), Vector3(0, -0.4, 0))); // LeftLowerLeg (60° bend - contracted knee)
	skeleton->set_bone_rest(3, Transform3D(Basis(), Vector3(0, -0.3, 0))); // LeftFoot down
	skeleton->set_bone_rest(4, Transform3D(Basis(), Vector3(0, -0.15, 0))); // LeftToes down

	for (int i = 0; i < skeleton->get_bone_count(); ++i) {
		skeleton->set_bone_global_pose(i, skeleton->get_bone_rest(i));
	}

	return skeleton;
}
