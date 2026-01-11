/**************************************************************************/
/*  test_scene_merge_integration_blend_shapes.h                           */
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

#include "tests/test_macros.h"

#include "core/math/color.h"
#include "core/math/vector3.h"
#include "modules/scene_merge/scene_merge.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/main/node.h"
#include "scene/resources/3d/importer_mesh.h"

namespace TestSceneMergeIntegration {

// Helper function to create a simple triangle mesh
static Ref<ImporterMesh> create_simple_triangle(const Color &p_color) {
	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();

	PackedVector3Array vertices;
	vertices.push_back(Vector3(-0.5f, -0.5f, 0.0f));
	vertices.push_back(Vector3(0.5f, -0.5f, 0.0f));
	vertices.push_back(Vector3(0.0f, 0.5f, 0.0f));

	PackedVector3Array normals;
	normals.push_back(Vector3(0.0f, 0.0f, 1.0f));
	normals.push_back(Vector3(0.0f, 0.0f, 1.0f));
	normals.push_back(Vector3(0.0f, 0.0f, 1.0f));

	PackedInt32Array indices;
	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_INDEX] = indices;

	Ref<StandardMaterial3D> material;
	material.instantiate();
	material->set_albedo(p_color);

	importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, arrays);
	importer_mesh->set_surface_material(0, material);

	return importer_mesh;
}

// Helper function to find merged mesh instance
static ImporterMeshInstance3D *find_merged_mesh_blend(Node *p_root) {
	if (!p_root) {
		return nullptr;
	}

	for (int i = 0; i < p_root->get_child_count(); i++) {
		Node *child = p_root->get_child(i);
		ImporterMeshInstance3D *mi = Object::cast_to<ImporterMeshInstance3D>(child);
		if (mi && mi->get_name() == StringName("MergedMesh")) {
			return mi;
		}
	}

	return nullptr;
}

// Test: Blend shape handling (basic test - SceneMerge currently doesn't preserve blend shapes)
TEST_CASE("[SceneTree][Modules][SceneMerge] Blend shape handling") {
	Node *test_root = memnew(Node);
	test_root->set_name("BlendShapeTest");

	// Create 3 mesh instances at different positions
	for (int i = 0; i < 3; i++) {
		ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
		mesh_instance->set_name(vformat("BlendMesh%d", i));
		mesh_instance->set_position(Vector3(i * 2.0f, 0.0f, 0.0f));
		mesh_instance->set_mesh(create_simple_triangle(Color(0.5f, 0.5f, 0.5f)));
		test_root->add_child(mesh_instance);
	}

	Ref<SceneMerge> scene_merge;
	scene_merge.instantiate();
	Node *merge_result = scene_merge->merge(test_root);

	CHECK_EQ(merge_result, test_root);

	ImporterMeshInstance3D *merged_instance = find_merged_mesh_blend(test_root);
	REQUIRE(merged_instance != nullptr);

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	REQUIRE(merged_mesh.is_valid());
	REQUIRE_GT(merged_mesh->get_surface_count(), 0);

	// Verify vertex count matches original (3 vertices per triangle, 3 instances = 9 vertices)
	Array surface_arrays = merged_mesh->get_surface_arrays(0);
	PackedVector3Array vertices = surface_arrays[Mesh::ARRAY_VERTEX];
	CHECK_EQ(vertices.size(), 9); // 3 triangles * 3 vertices each

	// Note: Full blend shape support would require more complex ImporterMesh setup
	// This test verifies basic merging works, blend shape preservation is a future enhancement

	memdelete(test_root);
}

} // namespace TestSceneMergeIntegration
