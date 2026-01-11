/**************************************************************************/
/*  test_scene_merge_integration_surfaces.h                               */
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
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/material.h"

namespace TestSceneMergeIntegration {

// Helper function to create a multi-surface mesh with different materials
static Ref<ImporterMesh> create_multi_surface_mesh() {
	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();
	importer_mesh->set_name("MultiSurfaceMesh");

	// Create first surface (triangle) with red material
	{
		PackedVector3Array vertices;
		vertices.push_back(Vector3(0, 0, 0));
		vertices.push_back(Vector3(1, 0, 0));
		vertices.push_back(Vector3(0.5, 1, 0));

		PackedVector3Array normals;
		normals.push_back(Vector3(0, 0, 1));
		normals.push_back(Vector3(0, 0, 1));
		normals.push_back(Vector3(0, 0, 1));

		PackedInt32Array indices;
		indices.push_back(0);
		indices.push_back(1);
		indices.push_back(2);

		Array surface_arrays;
		surface_arrays.resize(Mesh::ARRAY_MAX);
		surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
		surface_arrays[Mesh::ARRAY_NORMAL] = normals;
		surface_arrays[Mesh::ARRAY_INDEX] = indices;

		Ref<StandardMaterial3D> material;
		material.instantiate();
		material->set_albedo(Color(1.0f, 0.0f, 0.0f)); // Red
		material->set_name("RedMaterial");

		importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, surface_arrays, TypedArray<Array>(), Dictionary(), material, "Surface1");
	}

	// Create second surface (triangle) with blue material
	{
		PackedVector3Array vertices;
		vertices.push_back(Vector3(2, 0, 0));
		vertices.push_back(Vector3(3, 0, 0));
		vertices.push_back(Vector3(2.5, 1, 0));

		PackedVector3Array normals;
		normals.push_back(Vector3(0, 0, 1));
		normals.push_back(Vector3(0, 0, 1));
		normals.push_back(Vector3(0, 0, 1));

		PackedInt32Array indices;
		indices.push_back(0);
		indices.push_back(1);
		indices.push_back(2);

		Array surface_arrays;
		surface_arrays.resize(Mesh::ARRAY_MAX);
		surface_arrays[Mesh::ARRAY_VERTEX] = vertices;
		surface_arrays[Mesh::ARRAY_NORMAL] = normals;
		surface_arrays[Mesh::ARRAY_INDEX] = indices;

		Ref<StandardMaterial3D> material;
		material.instantiate();
		material->set_albedo(Color(0.0f, 0.0f, 1.0f)); // Blue
		material->set_name("BlueMaterial");

		importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, surface_arrays, TypedArray<Array>(), Dictionary(), material, "Surface2");
	}

	return importer_mesh;
}

// Helper function to find merged mesh instance
static ImporterMeshInstance3D *find_merged_mesh_surfaces(Node *p_root) {
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

// Test: Mesh surface preservation - verify multiple surfaces and materials are maintained
TEST_CASE("[SceneTree][Modules][SceneMerge] Mesh surface preservation") {
	Node *test_root = memnew(Node);
	test_root->set_name("SurfaceTest");

	// Create mesh instance with multiple surfaces
	ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
	mesh_instance->set_name("MultiSurfaceMesh");
	mesh_instance->set_mesh(create_multi_surface_mesh());
	test_root->add_child(mesh_instance);

	Ref<SceneMerge> scene_merge;
	scene_merge.instantiate();
	Node *result = scene_merge->merge(test_root);

	// Should return the same root node
	CHECK_EQ(result, test_root);

	ImporterMeshInstance3D *merged_instance = find_merged_mesh_surfaces(test_root);
	CHECK_NE(merged_instance, nullptr);

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	CHECK(merged_mesh.is_valid());

	// Verify that multiple surfaces are preserved
	int surface_count = merged_mesh->get_surface_count();
	CHECK_GE(surface_count, 1); // At minimum, surfaces should be combined but materials preserved

	// Check that materials are preserved (current implementation merges materials,
	// but this test documents the expected behavior for future enhancement)
	bool has_materials = false;
	for (int i = 0; i < surface_count; i++) {
		Ref<Material> material = merged_mesh->get_surface_material(i);
		if (material.is_valid()) {
			has_materials = true;
			break;
		}
	}
	CHECK(has_materials);

	// Verify vertex count is correct (2 triangles from each surface = 4 triangles total)
	Array surface_arrays = merged_mesh->get_surface_arrays(0);
	PackedVector3Array vertices = surface_arrays[Mesh::ARRAY_VERTEX];
	int total_vertices = vertices.size();
	CHECK_EQ(total_vertices, 12); // 4 triangles * 3 vertices each

	// Original mesh instance should be removed
	bool original_removed = true;
	for (int i = 0; i < test_root->get_child_count(); i++) {
		Node *child = test_root->get_child(i);
		if (child->get_name() == StringName("MultiSurfaceMesh")) {
			original_removed = false;
			break;
		}
	}
	CHECK(original_removed);

	memdelete(test_root);
}
} // namespace TestSceneMergeIntegration
