/**************************************************************************/
/*  test_fixtures.h                                                       */
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

#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"

namespace DDMTestFixtures {

// Simple cube mesh fixture (8 vertices, 12 triangles)
// Known topology for predictable testing
struct CubeMeshFixture {
	Vector<Vector3> vertices;
	Vector<Vector3> normals;
	Vector<int> indices;
	Vector<float> bone_weights;
	Vector<int> bone_indices;

	CubeMeshFixture() {
		// Unit cube vertices (0,0,0) to (1,1,1)
		vertices = {
			Vector3(0, 0, 0), // 0: Bottom-front-left
			Vector3(1, 0, 0), // 1: Bottom-front-right
			Vector3(1, 1, 0), // 2: Bottom-back-right
			Vector3(0, 1, 0), // 3: Bottom-back-left
			Vector3(0, 0, 1), // 4: Top-front-left
			Vector3(1, 0, 1), // 5: Top-front-right
			Vector3(1, 1, 1), // 6: Top-back-right
			Vector3(0, 1, 1) // 7: Top-back-left
		};

		// Simple normals (not normalized, just for testing)
		normals = {
			Vector3(-1, -1, -1), Vector3(1, -1, -1),
			Vector3(1, 1, -1), Vector3(-1, 1, -1),
			Vector3(-1, -1, 1), Vector3(1, -1, 1),
			Vector3(1, 1, 1), Vector3(-1, 1, 1)
		};

		// 12 triangles (2 per face)
		indices = {
			// Bottom face (z=0)
			0, 1, 2, 0, 2, 3,
			// Top face (z=1)
			4, 6, 5, 4, 7, 6,
			// Front face (y=0)
			0, 5, 1, 0, 4, 5,
			// Back face (y=1)
			3, 2, 6, 3, 6, 7,
			// Left face (x=0)
			0, 3, 7, 0, 7, 4,
			// Right face (x=1)
			1, 5, 6, 1, 6, 2
		};

		// Single bone with weight 1.0 for all vertices
		int vertex_count = vertices.size();
		bone_weights.resize(vertex_count * 4); // 4 weights per vertex
		bone_indices.resize(vertex_count * 4); // 4 bone indices per vertex

		for (int v = 0; v < vertex_count; v++) {
			bone_weights.set(v * 4 + 0, 1.0f); // Bone 0, weight 1.0
			bone_weights.set(v * 4 + 1, 0.0f);
			bone_weights.set(v * 4 + 2, 0.0f);
			bone_weights.set(v * 4 + 3, 0.0f);

			bone_indices.set(v * 4 + 0, 0); // Bone 0
			bone_indices.set(v * 4 + 1, 0);
			bone_indices.set(v * 4 + 2, 0);
			bone_indices.set(v * 4 + 3, 0);
		}
	}
};

// Expected adjacency for cube
// Each vertex connects to 3 neighbors (cube corners)
struct CubeAdjacencyFixture {
	LocalVector<LocalVector<int>> expected_adjacency;

	CubeAdjacencyFixture() {
		expected_adjacency.resize(8);

		// Vertex 0 (0,0,0) connects to: 1, 3, 4
		expected_adjacency[0].push_back(1);
		expected_adjacency[0].push_back(3);
		expected_adjacency[0].push_back(4);

		// Vertex 1 (1,0,0) connects to: 0, 2, 5
		expected_adjacency[1].push_back(0);
		expected_adjacency[1].push_back(2);
		expected_adjacency[1].push_back(5);

		// Vertex 2 (1,1,0) connects to: 1, 3, 6
		expected_adjacency[2].push_back(1);
		expected_adjacency[2].push_back(3);
		expected_adjacency[2].push_back(6);

		// Vertex 3 (0,1,0) connects to: 0, 2, 7
		expected_adjacency[3].push_back(0);
		expected_adjacency[3].push_back(2);
		expected_adjacency[3].push_back(7);

		// Vertex 4 (0,0,1) connects to: 0, 5, 7
		expected_adjacency[4].push_back(0);
		expected_adjacency[4].push_back(5);
		expected_adjacency[4].push_back(7);

		// Vertex 5 (1,0,1) connects to: 1, 4, 6
		expected_adjacency[5].push_back(1);
		expected_adjacency[5].push_back(4);
		expected_adjacency[5].push_back(6);

		// Vertex 6 (1,1,1) connects to: 2, 5, 7
		expected_adjacency[6].push_back(2);
		expected_adjacency[6].push_back(5);
		expected_adjacency[6].push_back(7);

		// Vertex 7 (0,1,1) connects to: 3, 4, 6
		expected_adjacency[7].push_back(3);
		expected_adjacency[7].push_back(4);
		expected_adjacency[7].push_back(6);
	}

	bool contains_neighbor(int vertex, int neighbor) const {
		for (unsigned int i = 0; i < expected_adjacency[vertex].size(); i++) {
			if (expected_adjacency[vertex][i] == neighbor) {
				return true;
			}
		}
		return false;
	}
};

// Transform fixtures for testing polar decomposition
struct TransformFixtures {
	Transform3D identity;
	Transform3D rotation_x_90;
	Transform3D rotation_y_45;
	Transform3D scale_uniform_2x;
	Transform3D scale_nonuniform;
	Transform3D combined_rotate_scale;

	TransformFixtures() {
		// Identity transform
		identity = Transform3D();

		// 90° rotation around X axis
		rotation_x_90 = Transform3D(Basis::from_euler(Vector3(Math::PI / 2.0, 0, 0)), Vector3(0, 0, 0));

		// 45° rotation around Y axis
		rotation_y_45 = Transform3D(Basis::from_euler(Vector3(0, Math::PI / 4.0, 0)), Vector3(0, 0, 0));

		// Uniform scale 2x
		scale_uniform_2x = Transform3D(Basis::from_scale(Vector3(2, 2, 2)), Vector3(0, 0, 0));

		// Non-uniform scale
		scale_nonuniform = Transform3D(Basis::from_scale(Vector3(1, 2, 3)), Vector3(0, 0, 0));

		// Combined: rotation + scale
		Basis rotation = Basis::from_euler(Vector3(Math::PI / 6.0, Math::PI / 4.0, 0));
		Basis scale = Basis::from_scale(Vector3(1.5, 1.5, 1.5));
		combined_rotate_scale = Transform3D(rotation * scale, Vector3(0, 0, 0));
	}
};

// Simple triangle mesh (3 vertices, 1 triangle)
// Minimal test case
struct TriangleMeshFixture {
	Vector<Vector3> vertices;
	Vector<int> indices;

	TriangleMeshFixture() {
		// Right triangle
		vertices = {
			Vector3(0, 0, 0),
			Vector3(1, 0, 0),
			Vector3(0, 1, 0)
		};

		indices = { 0, 1, 2 };
	}
};

// Degenerate mesh cases for edge case testing
struct DegenerateMeshFixtures {
	// Single vertex (isolated)
	Vector<Vector3> single_vertex = { Vector3(0, 0, 0) };
	Vector<int> no_indices;

	// Two vertices, no triangle
	Vector<Vector3> two_vertices = {
		Vector3(0, 0, 0),
		Vector3(1, 0, 0)
	};

	// Three collinear vertices (degenerate triangle)
	Vector<Vector3> collinear_vertices = {
		Vector3(0, 0, 0),
		Vector3(1, 0, 0),
		Vector3(2, 0, 0)
	};
	Vector<int> degenerate_triangle = { 0, 1, 2 };
};

} // namespace DDMTestFixtures
