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

#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/resources/surface_tool.h"

#include "modules/scene_merge/merge.h"

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

} // namespace TestSceneMerge
