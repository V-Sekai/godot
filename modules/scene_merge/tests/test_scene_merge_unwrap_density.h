/**************************************************************************/
/*  test_scene_merge_unwrap_density.h                                     */
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

#include "modules/scene_merge/scene_merge.h"
#include "scene/resources/3d/importer_mesh.h"

namespace TestSceneMerge {

TEST_CASE("[Modules][SceneMerge] SceneMerge unwrap_mesh with texel_density") {
	Ref<SceneMerge> scene_merge;
	scene_merge.instantiate();

	// Create a simple cube mesh manually to avoid RenderingServer dependency
	Ref<ImporterMesh> importer_mesh;
	importer_mesh.instantiate();

	// Define cube vertices
	Vector<Vector3> vertices;
	vertices.push_back(Vector3(-0.5, -0.5, -0.5)); // 0
	vertices.push_back(Vector3(0.5, -0.5, -0.5)); // 1
	vertices.push_back(Vector3(0.5, 0.5, -0.5)); // 2
	vertices.push_back(Vector3(-0.5, 0.5, -0.5)); // 3
	vertices.push_back(Vector3(-0.5, -0.5, 0.5)); // 4
	vertices.push_back(Vector3(0.5, -0.5, 0.5)); // 5
	vertices.push_back(Vector3(0.5, 0.5, 0.5)); // 6
	vertices.push_back(Vector3(-0.5, 0.5, 0.5)); // 7

	// Define cube indices (two triangles per face)
	Vector<int> indices;
	// Front face
	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);
	indices.push_back(0);
	indices.push_back(2);
	indices.push_back(3);
	// Back face
	indices.push_back(5);
	indices.push_back(4);
	indices.push_back(7);
	indices.push_back(5);
	indices.push_back(7);
	indices.push_back(6);
	// Left face
	indices.push_back(4);
	indices.push_back(0);
	indices.push_back(3);
	indices.push_back(4);
	indices.push_back(3);
	indices.push_back(7);
	// Right face
	indices.push_back(1);
	indices.push_back(5);
	indices.push_back(6);
	indices.push_back(1);
	indices.push_back(6);
	indices.push_back(2);
	// Top face
	indices.push_back(3);
	indices.push_back(2);
	indices.push_back(6);
	indices.push_back(3);
	indices.push_back(6);
	indices.push_back(7);
	// Bottom face
	indices.push_back(4);
	indices.push_back(5);
	indices.push_back(1);
	indices.push_back(4);
	indices.push_back(1);
	indices.push_back(0);

	// Define normals (one per vertex, pointing outwards)
	Vector<Vector3> normals;
	for (int i = 0; i < vertices.size(); i++) {
		normals.push_back(vertices[i].normalized());
	}

	// Create mesh arrays
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_INDEX] = indices;

	// Add surface to importer mesh
	importer_mesh->add_surface(Mesh::PRIMITIVE_TRIANGLES, arrays);

	// Test with custom texel density
	Ref<ImporterMesh> result = scene_merge->unwrap_mesh(importer_mesh, 64.0f, 0, 2);
	CHECK(result.is_valid());

	Array result_arrays = result->get_surface_arrays(0);
	Vector<Vector2> uvs = result_arrays[Mesh::ARRAY_TEX_UV];
	CHECK(uvs.size() > 0);

	// UVs should be scaled based on texel density
	// For a 1x1x1 cube with density 64, UVs should cover approximately 64x64 texels
	// But since resolution is estimated, we just check they're reasonable
	for (const Vector2 &uv : uvs) {
		CHECK(uv.x >= 0.0f);
		CHECK(uv.y >= 0.0f);
	}
}

} // namespace TestSceneMerge
