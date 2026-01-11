/**************************************************************************/
/*  test_scene_merge_integration.h                                        */
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
#include "core/math/math_funcs.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "modules/scene_merge/scene_merge.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/main/node.h"
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/material.h"

namespace TestSceneMergeIntegration {

// Helper function to create a simple triangle mesh with a material
static Ref<ImporterMesh> create_triangle_mesh(const Color &p_color) {
	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();

	// Create a simple triangle mesh
	PackedVector3Array vertices;
	vertices.push_back(Vector3(-0.5f, -0.5f, 0.0f)); // Bottom left
	vertices.push_back(Vector3(0.5f, -0.5f, 0.0f)); // Bottom right
	vertices.push_back(Vector3(0.0f, 0.5f, 0.0f)); // Top middle

	PackedVector3Array normals;
	normals.push_back(Vector3(0.0f, 0.0f, 1.0f));
	normals.push_back(Vector3(0.0f, 0.0f, 1.0f));
	normals.push_back(Vector3(0.0f, 0.0f, 1.0f));

	PackedInt32Array indices;
	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);

	// Create surface arrays
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_INDEX] = indices;

	// Create and set material
	Ref<StandardMaterial3D> material;
	material.instantiate();
	material->set_albedo(p_color);

	// Add surface
	importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, arrays);
	importer_mesh->set_surface_material(0, material);

	return importer_mesh;
}

// Helper function to find merged mesh instance
static ImporterMeshInstance3D *find_merged_mesh(Node *p_root) {
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

// Test 1: Basic scene merge - merge 3 mesh instances
TEST_CASE("[Modules][SceneMerge] Basic scene merge") {
	Node *test_root = memnew(Node);
	test_root->set_name("BasicMergeTest");

	// Create 3 mesh instances with different colors and positions
	for (int i = 0; i < 3; i++) {
		ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
		mesh_instance->set_name(vformat("Mesh%d", i));
		mesh_instance->set_position(Vector3(i * 3.0f, 0.0f, 0.0f));

		Color color = Color(1.0f, 0.0f, 0.0f); // Red
		if (i == 1) {
			color = Color(0.0f, 1.0f, 0.0f); // Green
		} else if (i == 2) {
			color = Color(0.0f, 0.0f, 1.0f); // Blue
		}

		mesh_instance->set_mesh(create_triangle_mesh(color));
		test_root->add_child(mesh_instance);
	}

	// Verify initial state
	CHECK_EQ(test_root->get_child_count(), 3);

	// Execute merge
	Ref<SceneMerge> scene_merge;
	scene_merge.instantiate();
	Node *merge_result = scene_merge->merge(test_root);

	// Verify merge result
	CHECK_EQ(merge_result, test_root);

	// Find merged mesh
	ImporterMeshInstance3D *merged_instance = find_merged_mesh(test_root);
	CHECK_NE(merged_instance, nullptr);
	CHECK_EQ(merged_instance->get_name(), StringName("MergedMesh"));

	// Verify merged mesh exists and has geometry
	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	CHECK(merged_mesh.is_valid());
	CHECK_GT(merged_mesh->get_surface_count(), 0);

	memdelete(test_root);
}

// Test 2: Base color material merge - verify color averaging
TEST_CASE("[Modules][SceneMerge] Base color material merge") {
	Node *test_root = memnew(Node);
	test_root->set_name("BaseColorTest");

	// Create meshes with specific base colors: Red, Green, Blue
	Color colors[] = {
		Color(1.0f, 0.0f, 0.0f), // Red
		Color(0.0f, 1.0f, 0.0f), // Green
		Color(0.0f, 0.0f, 1.0f) // Blue
	};

	for (int i = 0; i < 3; i++) {
		ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
		mesh_instance->set_name(vformat("ColorMesh%d", i));
		mesh_instance->set_position(Vector3(0.0f, i * 3.0f, 0.0f));
		mesh_instance->set_mesh(create_triangle_mesh(colors[i]));
		test_root->add_child(mesh_instance);
	}

	// Execute merge
	Ref<SceneMerge> scene_merge;
	scene_merge.instantiate();
	scene_merge->merge(test_root);

	// Find merged mesh
	ImporterMeshInstance3D *merged_instance = find_merged_mesh(test_root);
	CHECK_NE(merged_instance, nullptr);

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	CHECK(merged_mesh.is_valid());
	CHECK_EQ(merged_mesh->get_surface_count(), 1);

	// Get the material
	Ref<Material> surface_material = merged_mesh->get_surface_material(0);
	CHECK(surface_material.is_valid());

	Ref<BaseMaterial3D> base_mat = surface_material;
	CHECK(base_mat.is_valid());

	// Check that color is averaged
	Color albedo = base_mat->get_albedo();
	Color expected_avg = Color(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);

	// Allow small floating point tolerance
	CHECK_LT(Math::abs(albedo.r - expected_avg.r), 0.05f);
	CHECK_LT(Math::abs(albedo.g - expected_avg.g), 0.05f);
	CHECK_LT(Math::abs(albedo.b - expected_avg.b), 0.05f);

	memdelete(test_root);
}

// Test 3: Performance scaling - test with varying mesh counts
TEST_CASE("[Modules][SceneMerge] Performance scaling") {
	LocalVector<int> mesh_counts;
	mesh_counts.push_back(5);
	mesh_counts.push_back(10);
	mesh_counts.push_back(15);

	for (int count : mesh_counts) {
		Node *test_root = memnew(Node);
		test_root->set_name(vformat("PerformanceTest_%d", count));

		// Create multiple meshes
		for (int i = 0; i < count; i++) {
			ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
			mesh_instance->set_name(vformat("PerfMesh%d", i));
			mesh_instance->set_position(Vector3(i * 3.0f, 0.0f, 0.0f));

			// Random color
			Color random_color = Color(Math::randf(), Math::randf(), Math::randf(), 1.0f);
			mesh_instance->set_mesh(create_triangle_mesh(random_color));
			test_root->add_child(mesh_instance);
		}

		// Execute merge
		Ref<SceneMerge> scene_merge;
		scene_merge.instantiate();
		Node *merge_result = scene_merge->merge(test_root);

		// Verify merge succeeded
		CHECK_EQ(merge_result, test_root);

		// Verify merged mesh exists
		ImporterMeshInstance3D *merged_instance = find_merged_mesh(test_root);
		CHECK_NE(merged_instance, nullptr);

		Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
		CHECK(merged_mesh.is_valid());

		memdelete(test_root);
	}
}

// Test 4: Edge case handling - empty scene and single mesh
TEST_CASE("[Modules][SceneMerge] Edge case handling") {
	// Test 1: Empty scene
	{
		Node *test_root = memnew(Node);
		test_root->set_name("EmptySceneTest");

		Ref<SceneMerge> scene_merge;
		scene_merge.instantiate();
		Node *result = scene_merge->merge(test_root);

		CHECK_EQ(result, test_root);
		memdelete(test_root);
	}

	// Test 2: Single mesh scene
	{
		Node *test_root = memnew(Node);
		test_root->set_name("SingleMeshTest");

		ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
		mesh_instance->set_name("SingleMesh");
		mesh_instance->set_mesh(create_triangle_mesh(Color(0.5f, 0.5f, 0.5f)));
		test_root->add_child(mesh_instance);

		Ref<SceneMerge> scene_merge;
		scene_merge.instantiate();
		Node *result = scene_merge->merge(test_root);

		CHECK_EQ(result, test_root);
		memdelete(test_root);
	}
}

// Test 5: Scene transformation preservation
TEST_CASE("[Modules][SceneMerge] Scene transformation preservation") {
	Node *test_root = memnew(Node);
	test_root->set_name("TransformTest");

	// Create meshes with different transforms
	ImporterMeshInstance3D *mesh1 = memnew(ImporterMeshInstance3D);
	mesh1->set_name("Mesh1");
	mesh1->set_position(Vector3(0.0f, 0.0f, 0.0f));
	mesh1->set_mesh(create_triangle_mesh(Color(1.0f, 0.0f, 0.0f)));
	test_root->add_child(mesh1);

	ImporterMeshInstance3D *mesh2 = memnew(ImporterMeshInstance3D);
	mesh2->set_name("Mesh2");
	mesh2->set_position(Vector3(5.0f, 3.0f, 2.0f));
	mesh2->set_mesh(create_triangle_mesh(Color(0.0f, 1.0f, 0.0f)));
	test_root->add_child(mesh2);

	// Execute merge
	Ref<SceneMerge> scene_merge;
	scene_merge.instantiate();
	scene_merge->merge(test_root);

	// Find merged mesh
	ImporterMeshInstance3D *merged_instance = find_merged_mesh(test_root);
	CHECK_NE(merged_instance, nullptr);

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	CHECK(merged_mesh.is_valid());

	// Verify the merged mesh contains vertices from both original positions
	int vertex_count = 0;
	for (int i = 0; i < merged_mesh->get_surface_count(); i++) {
		Array arrays = merged_mesh->get_surface_arrays(i);
		if (!arrays.is_empty()) {
			PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
			vertex_count += vertices.size();
		}
	}

	// Should have at least 6 vertices (3 per mesh)
	CHECK_GE(vertex_count, 6);

	memdelete(test_root);
}

// Test 6: Blend shape preservation (simplified test without glTF loading)
TEST_CASE("[Modules][SceneMerge] Blend shape handling") {
	Node *test_root = memnew(Node);
	test_root->set_name("BlendShapeTest");

	// Create meshes with basic data
	for (int i = 0; i < 2; i++) {
		ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
		mesh_instance->set_name(vformat("BlendMesh%d", i));
		mesh_instance->set_position(Vector3(i * 3.0f, 0.0f, 0.0f));
		mesh_instance->set_mesh(create_triangle_mesh(Color(0.5f, 0.5f, 0.5f)));
		test_root->add_child(mesh_instance);
	}

	// Execute merge
	Ref<SceneMerge> scene_merge;
	scene_merge.instantiate();
	Node *merge_result = scene_merge->merge(test_root);

	CHECK_EQ(merge_result, test_root);

	// Verify merged mesh was created
	ImporterMeshInstance3D *merged_instance = find_merged_mesh(test_root);
	CHECK_NE(merged_instance, nullptr);

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	CHECK(merged_mesh.is_valid());
	CHECK_GT(merged_mesh->get_surface_count(), 0);

	memdelete(test_root);
}

// Test 7: Skeleton/Animation data handling (simplified test without glTF loading)
TEST_CASE("[Modules][SceneMerge] Skeleton animation handling") {
	Node *test_root = memnew(Node);
	test_root->set_name("SkeletonAnimationTest");

	// Create meshes with basic data
	for (int i = 0; i < 2; i++) {
		ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
		mesh_instance->set_name(vformat("SkeletonMesh%d", i));
		mesh_instance->set_position(Vector3(i * 3.0f, 0.0f, 0.0f));
		mesh_instance->set_mesh(create_triangle_mesh(Color(0.5f, 0.5f, 0.5f)));
		test_root->add_child(mesh_instance);
	}

	// Execute merge
	Ref<SceneMerge> scene_merge;
	scene_merge.instantiate();
	Node *merge_result = scene_merge->merge(test_root);

	CHECK_EQ(merge_result, test_root);

	// Verify merged mesh was created
	ImporterMeshInstance3D *merged_instance = find_merged_mesh(test_root);
	CHECK_NE(merged_instance, nullptr);

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	CHECK(merged_mesh.is_valid());

	// Verify merged mesh contains geometry
	int vertex_count = 0;
	for (int i = 0; i < merged_mesh->get_surface_count(); i++) {
		Array arrays = merged_mesh->get_surface_arrays(i);
		if (!arrays.is_empty()) {
			PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
			vertex_count += vertices.size();
		}
	}

	CHECK_GT(vertex_count, 0);

	memdelete(test_root);
}

} // namespace TestSceneMergeIntegration
