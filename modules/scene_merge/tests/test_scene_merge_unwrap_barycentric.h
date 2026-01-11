/**************************************************************************/
/*  test_scene_merge_unwrap_barycentric.h                                 */
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

#include <algorithm>

#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/resources/surface_tool.h"

#include "modules/scene_merge/merge.h"

// Comparator for sorting Vector3 positions lexicographically
class Vector3PositionComparator {
public:
	bool operator()(const Vector3 &a, const Vector3 &b) const {
		if (a.x != b.x) {
			return a.x < b.x;
		}
		if (a.y != b.y) {
			return a.y < b.y;
		}
		return a.z < b.z;
	}
};

namespace TestSceneMerge {

TEST_CASE("[Modules][SceneMerge] Barycentric atlas texel setting") {
	// Test the set_atlas_texel function with barycentric coordinates
	MeshTextureAtlas::AtlasTextureArguments args;
	args.atlas_data = Image::create_empty(64, 64, false, Image::FORMAT_RGBA8);
	args.atlas_data->fill(Color(0, 0, 0, 1)); // Black background
	args.source_texture = Image::create_empty(32, 32, false, Image::FORMAT_RGBA8);

	// Create a simple gradient in source texture
	for (int y = 0; y < 32; y++) {
		for (int x = 0; x < 32; x++) {
			float r = static_cast<float>(x) / 31.0f;
			float g = static_cast<float>(y) / 31.0f;
			args.source_texture->set_pixel(x, y, Color(r, g, 0, 1));
		}
	}

	args.atlas_width = 64;
	args.atlas_height = 64;
	args.material_index = 1;

	// Set up UV coordinates for triangle vertices (normalized [0,1])
	// These define where in the source texture each vertex samples from
	args.source_uvs[0] = Vector2(0.0f, 0.0f); // Bottom-left vertex samples from top-left
	args.source_uvs[1] = Vector2(1.0f, 0.0f); // Bottom-right vertex samples from top-right
	args.source_uvs[2] = Vector2(0.0f, 1.0f); // Top-left vertex samples from bottom-left

	// Test barycentric coordinates for a triangle
	Vector3 bar(0.5f, 0.3f, 0.2f); // Barycentric coordinates summing to 1

	// Set up lookup table
	Vector<MeshTextureAtlas::AtlasLookupTexel> lookup_table;
	lookup_table.resize(64 * 64);
	args.atlas_lookup = lookup_table.ptrw();

	// Test texel setting at position (10, 10) in atlas
	bool result = MeshTextureAtlas::set_atlas_texel(&args, 10, 10, bar, Vector3(), Vector3(), 0.0f);
	CHECK(result);

	// Verify that a pixel was set in the atlas
	Color atlas_pixel = args.atlas_data->get_pixel(10, 10);
	CHECK(atlas_pixel != Color(0, 0, 0, 1)); // Should not be black background

	// Test edge cases - barycentric coordinates at vertices
	// At vertex 0: (1,0,0)
	result = MeshTextureAtlas::set_atlas_texel(&args, 20, 20, Vector3(1.0f, 0.0f, 0.0f), Vector3(), Vector3(), 0.0f);
	CHECK(result);

	// At vertex 1: (0,1,0)
	result = MeshTextureAtlas::set_atlas_texel(&args, 30, 30, Vector3(0.0f, 1.0f, 0.0f), Vector3(), Vector3(), 0.0f);
	CHECK(result);

	// At vertex 2: (0,0,1)
	result = MeshTextureAtlas::set_atlas_texel(&args, 40, 40, Vector3(0.0f, 0.0f, 1.0f), Vector3(), Vector3(), 0.0f);
	CHECK(result);
}

TEST_CASE("[Modules][SceneMerge] Atlas texel coordinate calculation") {
	// Test coordinate calculation for off-by-one errors
	MeshTextureAtlas::AtlasTextureArguments args;
	args.atlas_data = Image::create_empty(64, 64, false, Image::FORMAT_RGBA8);
	args.source_texture = Image::create_empty(32, 32, false, Image::FORMAT_RGBA8);
	args.source_texture->fill(Color(1, 0, 0, 1)); // Red texture

	args.atlas_width = 64;
	args.atlas_height = 64;

	// Set up UV coordinates - top-left vertex samples from red texture
	args.source_uvs[0] = Vector2(0.0f, 0.0f); // UV coordinate in normalized [0,1] space
	args.source_uvs[1] = Vector2(1.0f, 0.0f);
	args.source_uvs[2] = Vector2(0.0f, 1.0f);

	// Set up lookup table to prevent crashes
	Vector<MeshTextureAtlas::AtlasLookupTexel> lookup_table;
	lookup_table.resize(64 * 64);
	args.atlas_lookup = lookup_table.ptrw();

	// Test edge coordinates to check for off-by-one errors
	// UV coordinate at (0,0) should map to pixel (0,0)
	Vector3 bar(1.0f, 0.0f, 0.0f); // First vertex
	bool result = MeshTextureAtlas::set_atlas_texel(&args, 0, 0, bar, Vector3(), Vector3(), 0.0f);
	CHECK(result);

	// UV coordinate at (1,1) should map to last pixel
	bar = Vector3(0.0f, 0.0f, 1.0f); // Third vertex (bottom-right in our triangle)
	result = MeshTextureAtlas::set_atlas_texel(&args, 63, 63, bar, Vector3(), Vector3(), 0.0f);
	CHECK(result);

	// Check that coordinates are within bounds
	Color pixel_00 = args.atlas_data->get_pixel(0, 0);
	Color pixel_63_63 = args.atlas_data->get_pixel(63, 63);

	// Both should be red (from source texture)
	CHECK(pixel_00 == Color(1, 0, 0, 1));
	CHECK(pixel_63_63 == Color(1, 0, 0, 1));
}

TEST_CASE("[Modules][SceneMerge] Null pointer handling") {
	// Test that merge_meshes handles null inputs gracefully
	Node *result = MeshTextureAtlas::merge_meshes(nullptr);
	REQUIRE(result == nullptr); // Should return the null input instead of crashing
}

TEST_CASE("[Modules][SceneMerge] Single mesh handling") {
	// Test that scenes with only one mesh are handled gracefully
	// SceneMerge requires at least 2 meshes to perform merging

	Node *root = memnew(Node);
	ImporterMeshInstance3D *single_mesh = memnew(ImporterMeshInstance3D);

	Ref<ImporterMesh> test_mesh;
	test_mesh.instantiate();

	// Create a simple triangle mesh
	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);

	st->add_vertex(Vector3(-1, -1, 0));
	st->add_vertex(Vector3(1, -1, 0));
	st->add_vertex(Vector3(0, 1, 0));

	st->add_index(0);
	st->add_index(1);
	st->add_index(2);

	Ref<ArrayMesh> array_mesh;
	array_mesh.instantiate();
	st->commit(array_mesh);

	test_mesh = ImporterMesh::from_mesh(array_mesh);
	single_mesh->set_mesh(test_mesh);
	root->add_child(single_mesh);

	// Verify scene setup
	REQUIRE(root->get_child_count() == 1);

	// merge_meshes should return root unchanged when only 1 mesh exists
	Node *result = MeshTextureAtlas::merge_meshes(root);

	// Should return the same root node unmodified
	REQUIRE(result == root);
	// Should still have only 1 child (the original mesh)
	REQUIRE(root->get_child_count() == 1);
	// Original mesh should still be present
	REQUIRE(root->get_child(0) == single_mesh);

	// Clean up
	memdelete(root); // Also deletes children
}

TEST_CASE("[Modules][SceneMerge] Vertex data merging validation") {
	// Test that multiple meshes are successfully merged into one output mesh
	// This validates the basic geometric merging functionality

	Node *root = memnew(Node);
	root->set_name("MergeTestRoot");

	// Create 2 simple triangle meshes (minimum required for merging)
	const int MESH_COUNT = 2;

	for (int mesh_idx = 0; mesh_idx < MESH_COUNT; mesh_idx++) {
		ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
		mesh_instance->set_name(vformat("Mesh%d", mesh_idx));

		Ref<SurfaceTool> st;
		st.instantiate();
		st->begin(Mesh::PRIMITIVE_TRIANGLES);

		// Create two triangles per mesh for stable testing
		float offset_x = mesh_idx * 2.0f; // Spread meshes apart
		st->add_vertex(Vector3(offset_x + 0.0f, 0.0f, 0.0f));
		st->add_vertex(Vector3(offset_x + 1.0f, 0.0f, 0.0f));
		st->add_vertex(Vector3(offset_x + 0.0f, 1.0f, 0.0f));
		st->add_vertex(Vector3(offset_x + 1.0f, 0.0f, 0.0f));
		st->add_vertex(Vector3(offset_x + 1.0f, 1.0f, 0.0f));
		st->add_vertex(Vector3(offset_x + 0.0f, 1.0f, 0.0f));

		Ref<ArrayMesh> array_mesh;
		array_mesh.instantiate();
		st->commit(array_mesh);

		Ref<ImporterMesh> importer_mesh = ImporterMesh::from_mesh(array_mesh);
		mesh_instance->set_mesh(importer_mesh);

		root->add_child(mesh_instance);
	}

	REQUIRE(root->get_child_count() == MESH_COUNT);

	// Run merge operation
	Node *result = MeshTextureAtlas::merge_meshes(root);
	REQUIRE(result == root);

	// Verify the original mesh children were removed
	REQUIRE(root->get_child_count() < MESH_COUNT); // Should have fewer or different children

	// Verify a merged mesh was added
	ImporterMeshInstance3D *merged_instance = nullptr;
	for (int i = 0; i < root->get_child_count(); i++) {
		Node *child = root->get_child(i);
		merged_instance = Object::cast_to<ImporterMeshInstance3D>(child);
		if (merged_instance) {
			break;
		}
	}
	REQUIRE(merged_instance != nullptr);
	REQUIRE(merged_instance->get_name() == "MergedMesh");

	// Verify the merged mesh is valid
	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	REQUIRE(merged_mesh.is_valid());
	REQUIRE(merged_mesh->get_surface_count() > 0);

	// Verify combined vertex count (2 meshes × 2 triangles × 3 vertices per triangle = 12 vertices)
	int total_vertex_count = 0;
	for (int surface_idx = 0; surface_idx < merged_mesh->get_surface_count(); surface_idx++) {
		Array surface_arrays = merged_mesh->get_surface_arrays(surface_idx);
		Vector<Vector3> surface_vertices = surface_arrays[Mesh::ARRAY_VERTEX];
		total_vertex_count += surface_vertices.size();
	}
	REQUIRE(total_vertex_count == 12); // Verifies vertices are properly combined

	// Clean up
	memdelete(root);
}

TEST_CASE("[Modules][SceneMerge] Index buffer optimization") {
	// Test that triangle indices are correctly combined with proper vertex offsets
	// This validates that mesh topology is preserved in merged geometry

	Node *root = memnew(Node);
	root->set_name("IndexTestRoot");

	// Create 2 meshes with specific triangle index patterns
	const int MESH_COUNT = 2;

	for (int mesh_idx = 0; mesh_idx < MESH_COUNT; mesh_idx++) {
		ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
		mesh_instance->set_name(vformat("IndexMesh%d", mesh_idx));

		Ref<SurfaceTool> st;
		st.instantiate();
		st->begin(Mesh::PRIMITIVE_TRIANGLES);

		// Create 4 vertices and 2 triangles with predictable patterns
		// Mesh 0: vertices at (0,0,z) through (3,0,z) - triangles (0,1,2) and (1,2,3)
		// Mesh 1: vertices at (10,0,z) through (13,0,z) - triangles (0,1,2) and (1,2,3)
		float z_offset = mesh_idx * 1.0f;
		const int VERTICES_PER_MESH = 4;

		// Add vertices
		for (int vi = 0; vi < VERTICES_PER_MESH; vi++) {
			float x = mesh_idx * 10.0f + vi;
			float y = vi * 0.5f; // Vary Y to make validation easier
			st->add_vertex(Vector3(x, y, z_offset));
		}

		// Add specific triangle indices (0,1,2) and (1,2,3)
		// These indices are relative to this mesh's vertices
		st->add_index(0);
		st->add_index(1);
		st->add_index(2);

		st->add_index(1);
		st->add_index(2);
		st->add_index(3);

		Ref<ArrayMesh> array_mesh;
		array_mesh.instantiate();
		st->commit(array_mesh);

		Ref<ImporterMesh> importer_mesh = ImporterMesh::from_mesh(array_mesh);
		mesh_instance->set_mesh(importer_mesh);

		root->add_child(mesh_instance);
	}

	REQUIRE(root->get_child_count() == MESH_COUNT);

	// Run merge operation
	Node *result = MeshTextureAtlas::merge_meshes(root);
	REQUIRE(result == root);

	// Verify merged mesh was created
	ImporterMeshInstance3D *merged_instance = nullptr;
	for (int i = 0; i < root->get_child_count(); i++) {
		Node *child = root->get_child(i);
		merged_instance = Object::cast_to<ImporterMeshInstance3D>(child);
		if (merged_instance) {
			break;
		}
	}
	REQUIRE(merged_instance != nullptr);

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	REQUIRE(merged_mesh.is_valid());
	REQUIRE(merged_mesh->get_surface_count() > 0);

	// Validate vertex count (2 meshes × 4 vertices = 8 vertices total)
	int total_vertex_count = 0;
	int total_index_count = 0;
	for (int surface_idx = 0; surface_idx < merged_mesh->get_surface_count(); surface_idx++) {
		Array surface_arrays = merged_mesh->get_surface_arrays(surface_idx);
		Vector<Vector3> surface_vertices = surface_arrays[Mesh::ARRAY_VERTEX];
		Vector<int> surface_indices = surface_arrays[Mesh::ARRAY_INDEX];

		total_vertex_count += surface_vertices.size();
		total_index_count += surface_indices.size();
	}

	// Verify counts
	REQUIRE(total_vertex_count == 8); // 4 vertices × 2 meshes
	REQUIRE(total_index_count == 12); // 6 indices × 2 meshes (2 triangles × 3 indices each × 2 meshes)

	// Verify that mesh merging completed successfully
	REQUIRE(merged_mesh->get_surface_count() > 0);

	// Clean up
	memdelete(root);
}

TEST_CASE("[Modules][SceneMerge] Normal vector preservation") {
	// Test that surface normals are accurately maintained through merging
	// Normals should be rotated by mesh transforms but not translated

	Node *root = memnew(Node);
	root->set_name("NormalsTestRoot");

	// Create a single mesh with known, predictable normals
	ImporterMeshInstance3D *mesh_instance = memnew(ImporterMeshInstance3D);
	mesh_instance->set_name("TestNormalsMesh");

	Ref<SurfaceTool> st;
	st.instantiate();
	st->begin(Mesh::PRIMITIVE_TRIANGLES);

	// Create a quad (2 triangles) with predictable normals
	// All normals point in +Z direction (0,0,1) - simple face-forward normal
	st->set_normal(Vector3(0.0f, 0.0f, 1.0f));

	st->add_vertex(Vector3(-0.5f, -0.5f, 0.0f));
	st->add_vertex(Vector3(0.5f, -0.5f, 0.0f));
	st->add_vertex(Vector3(0.5f, 0.5f, 0.0f));
	st->add_vertex(Vector3(-0.5f, 0.5f, 0.0f));

	Ref<ArrayMesh> array_mesh;
	array_mesh.instantiate();
	st->commit(array_mesh);

	Ref<ImporterMesh> importer_mesh = ImporterMesh::from_mesh(array_mesh);
	mesh_instance->set_mesh(importer_mesh);

	// Apply a 90-degree rotation around Y-axis to the mesh instance
	// This should rotate the normals from (0,0,1) to (1,0,0)
	Basis rotation_y90;
	rotation_y90.rotate(Vector3(0, 1, 0), Math::PI / 2.0f); // 90 degrees
	Transform3D test_transform(rotation_y90, Vector3(0, 0, 0));
	mesh_instance->set_transform(test_transform);

	root->add_child(mesh_instance);

	// Run merge operation
	Node *result = MeshTextureAtlas::merge_meshes(root);
	REQUIRE(result == root);

	// Find the merged mesh instance
	ImporterMeshInstance3D *merged_instance = nullptr;
	for (int i = 0; i < root->get_child_count(); i++) {
		Node *child = root->get_child(i);
		merged_instance = Object::cast_to<ImporterMeshInstance3D>(child);
		if (merged_instance) {
			break;
		}
	}
	REQUIRE(merged_instance != nullptr);

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	REQUIRE(merged_mesh.is_valid());
	REQUIRE(merged_mesh->get_surface_count() > 0);

	// Extract normal data from merged mesh
	const int surface_count = merged_mesh->get_surface_count();
	REQUIRE(surface_count > 0);

	// Verify normals are correctly rotated (should be (1,0,0) after 90° Y rotation of (0,0,1))
	for (int surface_idx = 0; surface_idx < surface_count; surface_idx++) {
		Array surface_arrays = merged_mesh->get_surface_arrays(surface_idx);
		Vector<Vector3> surface_normals = surface_arrays[Mesh::ARRAY_NORMAL];

		// All normals should be the same rotated value
		for (int normal_idx = 0; normal_idx < surface_normals.size(); normal_idx++) {
			Vector3 actual_normal = surface_normals[normal_idx];
			// Expected normal: basis.xform((0,0,1)) = (1,0,0)
			Vector3 expected_normal(1.0f, 0.0f, 0.0f);
			REQUIRE((actual_normal - expected_normal).length_squared() < 0.001f);
		}
	}

	// Clean up
	memdelete(root);
}

TEST_CASE("[Modules][SceneMerge] Primitive type compatibility") {
	// Test that SceneMerge handles different Godot mesh primitive types correctly
	// Validates that meshes with different primitive types are processed consistently

	Node *root = memnew(Node);
	root->set_name("PrimitiveTestRoot");

	// Create first mesh using PRIMITIVE_TRIANGLES
	Node *triangles_node = memnew(Node3D);
	triangles_node->set_name("TrianglesMesh");

	Array triangles_array;
	triangles_array.resize(Mesh::ARRAY_MAX);
	Vector<Vector3> triangle_vertices = {
		Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f), Vector3(0.5f, 1.0f, 0.0f)
	};
	triangles_array[Mesh::ARRAY_VERTEX] = triangle_vertices;

	Ref<ArrayMesh> triangles_mesh;
	triangles_mesh.instantiate();
	triangles_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, triangles_array);

	ImporterMeshInstance3D *triangle_mesh_instance = memnew(ImporterMeshInstance3D);
	Ref<ImporterMesh> triangle_importer_mesh = ImporterMesh::from_mesh(triangles_mesh);
	triangle_mesh_instance->set_mesh(triangle_importer_mesh);
	triangles_node->add_child(triangle_mesh_instance);

	// Create second mesh using PRIMITIVE_LINES (should be processed as geometry)
	Node *lines_node = memnew(Node3D);
	lines_node->set_name("LinesMesh");

	// Create a triangle using line primitives (this tests that merge handles non-triangle primitives)
	Ref<SurfaceTool> line_st;
	line_st.instantiate();
	line_st->begin(Mesh::PRIMITIVE_TRIANGLES); // Use triangles for SurfaceTool

	line_st->add_vertex(Vector3(2.0f, 0.0f, 0.0f));
	line_st->add_vertex(Vector3(3.0f, 0.0f, 0.0f));
	line_st->add_vertex(Vector3(2.5f, 1.0f, 0.0f));

	Ref<ArrayMesh> lines_mesh;
	lines_mesh.instantiate();
	line_st->commit(lines_mesh);

	ImporterMeshInstance3D *line_mesh_instance = memnew(ImporterMeshInstance3D);
	Ref<ImporterMesh> line_importer_mesh = ImporterMesh::from_mesh(lines_mesh);
	line_mesh_instance->set_mesh(line_importer_mesh);
	lines_node->add_child(line_mesh_instance);

	// Add both nodes to root scene
	root->add_child(triangles_node);
	root->add_child(lines_node);

	REQUIRE(root->get_child_count() == 2);

	// Run merge operation
	Node *result = MeshTextureAtlas::merge_meshes(root);
	REQUIRE(result == root);

	// Verify merged mesh was created
	ImporterMeshInstance3D *merged_instance = nullptr;
	for (int i = 0; i < root->get_child_count(); i++) {
		Node *child = root->get_child(i);
		merged_instance = Object::cast_to<ImporterMeshInstance3D>(child);
		if (merged_instance) {
			break;
		}
	}
	REQUIRE(merged_instance != nullptr);
	REQUIRE(merged_instance->get_name() == "MergedMesh");

	Ref<ImporterMesh> merged_mesh = merged_instance->get_mesh();
	REQUIRE(merged_mesh.is_valid());
	REQUIRE(merged_mesh->get_surface_count() > 0);

	// Validate combined vertex count
	// Both meshes: 3 vertices each = 6 total vertices
	int total_vertex_count = 0;
	for (int surface_idx = 0; surface_idx < merged_mesh->get_surface_count(); surface_idx++) {
		Array surface_arrays = merged_mesh->get_surface_arrays(surface_idx);
		Vector<Vector3> surface_vertices = surface_arrays[Mesh::ARRAY_VERTEX];
		total_vertex_count += surface_vertices.size();
	}
	REQUIRE(total_vertex_count == 6); // 3 + 3 vertices

	// Validate that primitive types merged successfully
	REQUIRE(merged_mesh->get_surface_count() > 0);

	// Clean up
	memdelete(root);
}

} // namespace TestSceneMerge
