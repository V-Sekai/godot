/**************************************************************************/
/*  test_ewbik.h                                                          */
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

#include "core/error/error_macros.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

#ifdef TOOLS_ENABLED

#include "core/os/os.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/main/window.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/packed_scene.h"
#include "tests/core/config/test_project_settings.h"

// Include ewbik headers
#include "modules/ewbik/src/ewbik_3d.h"

namespace TestEwbik {

Node *ewbik_create_test_scene() {
	// Create a scene with simple skin like glTF example - mesh with skeleton for IK testing
	Node3D *root = memnew(Node3D);
	root->set_name("ewbik_test_root");

	// Create skeleton with joints (like glTF skin)
	Skeleton3D *skeleton = memnew(Skeleton3D);
	skeleton->set_name("test_skeleton");

	// Root joint (node 1 in glTF example)
	skeleton->add_bone("root_joint");
	skeleton->set_bone_rest(0, Transform3D());

	// Upper arm joint (node 2 in glTF example)
	skeleton->add_bone("upper_arm_joint");
	skeleton->set_bone_rest(1, Transform3D(Basis(), Vector3(0.0, 1.0, 0.0)));
	skeleton->set_bone_parent(1, 0);

	root->add_child(skeleton);
	skeleton->set_owner(root);

	// Create a simple mesh for skinning (like glTF mesh with JOINTS_0/WEIGHTS_0)
	Ref<BoxMesh> mesh = memnew(BoxMesh);
	mesh->set_size(Vector3(1.0, 2.0, 0.5));
	mesh->set_name("test_mesh");

	// Create mesh instance and attach to skeleton
	MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
	mesh_instance->set_mesh(mesh);
	mesh_instance->set_name("test_mesh_instance");
	mesh_instance->set_skeleton_path(mesh_instance->get_path_to(skeleton));

	root->add_child(mesh_instance);
	mesh_instance->set_owner(root);

	// Add EwbikSolver when available
	EWBIK3D *solver = memnew(EWBIK3D);
	// EWBIK3D is a SkeletonModifier3D, so it gets the skeleton automatically when attached as child
	skeleton->add_child(solver);
	solver->set_owner(skeleton);

	return root;
}


void init(const String &p_test) {
	// Setup project settings since it's needed for the import process.
	String project_folder = TestUtils::get_temp_path(p_test.get_file().get_basename());
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	da->make_dir_recursive(project_folder.path_join(".godot"));
	// Initialize res:// to `project_folder`.
	TestProjectSettingsInternalsAccessor::resource_path() = project_folder;
	ProjectSettings::get_singleton()->setup(project_folder, String(), true);
}

TEST_CASE("[EWBik] Test Scene Creation") {
	Node *scene_root = TestEwbik::ewbik_create_test_scene();
	CHECK(scene_root != nullptr);
	memdelete(scene_root);
}

} // namespace TestEwbik

#endif // TOOLS_ENABLED
