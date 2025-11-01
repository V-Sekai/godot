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

	return skeleton;
}
