/**************************************************************************/
/*  test_scene_merge_integration_hierarchy.h                              */
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
#include "scene/3d/node_3d.h"
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/material.h"

namespace TestSceneMergeIntegration {

// Helper function to create a simple triangle mesh with a material
static Ref<ImporterMesh> create_triangle_mesh_hierarchy(const Color &p_color) {
	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();

	// Create a simple triangle mesh
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
	material->set_albedo(p_color);
	material->set_name("TestMaterial");

	importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, surface_arrays, TypedArray<Array>(), Dictionary(), material, "Surface");

	return importer_mesh;
}

// Helper function to find merged mesh instance
static ImporterMeshInstance3D *find_merged_mesh_hierarchy(Node *p_root) {
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

// Test: Hierarchy preservation - verify scene structure is maintained after merging
TEST_CASE("[SceneTree][Modules][SceneMerge] Hierarchy preservation") {
	Node *test_root = memnew(Node);
	test_root->set_name("HierarchyTest");

	// Create a complex scene hierarchy with both mesh and non-mesh nodes
	Node3D *group_node = memnew(Node3D);
	group_node->set_name("MeshGroup");
	test_root->add_child(group_node);

	// Add mesh instances under the group node
	ImporterMeshInstance3D *mesh1 = memnew(ImporterMeshInstance3D);
	mesh1->set_name("Mesh1");
	mesh1->set_mesh(create_triangle_mesh_hierarchy(Color(1.0f, 0.0f, 0.0f)));
	group_node->add_child(mesh1);

	ImporterMeshInstance3D *mesh2 = memnew(ImporterMeshInstance3D);
	mesh2->set_name("Mesh2");
	mesh2->set_mesh(create_triangle_mesh_hierarchy(Color(0.0f, 1.0f, 0.0f)));
	group_node->add_child(mesh2);

	// Add a non-mesh node that should be preserved
	Node3D *empty_node = memnew(Node3D);
	empty_node->set_name("EmptyNode");
	test_root->add_child(empty_node);

	// Add another non-mesh node under the group
	Node3D *helper_node = memnew(Node3D);
	helper_node->set_name("HelperNode");
	group_node->add_child(helper_node);

	Ref<SceneMerge> scene_merge;
	scene_merge.instantiate();
	Node *result = scene_merge->merge(test_root);

	// Should return the same root node
	CHECK_EQ(result, test_root);

	// Verify merged mesh was created
	ImporterMeshInstance3D *merged_instance = find_merged_mesh_hierarchy(test_root);
	CHECK_NE(merged_instance, nullptr);

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	CHECK(merged_mesh.is_valid());
	CHECK_EQ(merged_mesh->get_surface_count(), 1);

	// Verify hierarchy preservation
	// Root should still have its children
	int root_child_count = test_root->get_child_count();
	CHECK_EQ(root_child_count, 2); // EmptyNode + MergedMesh

	// Check that EmptyNode is preserved
	bool empty_node_found = false;
	for (int i = 0; i < root_child_count; i++) {
		Node *child = test_root->get_child(i);
		if (child->get_name() == StringName("EmptyNode")) {
			empty_node_found = true;
			break;
		}
	}
	CHECK(empty_node_found);

	// Check that MeshGroup was removed (since it only contained meshes)
	bool group_node_removed = true;
	for (int i = 0; i < root_child_count; i++) {
		Node *child = test_root->get_child(i);
		if (child->get_name() == StringName("MeshGroup")) {
			group_node_removed = false;
			break;
		}
	}
	CHECK(group_node_removed);

	// Verify merged mesh has correct vertex count (2 triangles * 3 vertices each = 6)
	Array surface_arrays = merged_mesh->get_surface_arrays(0);
	PackedVector3Array vertices = surface_arrays[Mesh::ARRAY_VERTEX];
	int total_vertices = vertices.size();
	CHECK_EQ(total_vertices, 6);

	// Verify original mesh instances were removed
	bool mesh1_removed = true;
	bool mesh2_removed = true;
	test_root->get_children(); // Refresh children list
	for (int i = 0; i < test_root->get_child_count(); i++) {
		Node *child = test_root->get_child(i);
		if (child->get_name() == StringName("Mesh1")) {
			mesh1_removed = false;
		}
		if (child->get_name() == StringName("Mesh2")) {
			mesh2_removed = false;
		}
	}
	CHECK(mesh1_removed);
	CHECK(mesh2_removed);

	memdelete(test_root);
}
} // namespace TestSceneMergeIntegration
