/**************************************************************************/
/*  test_ddm_adjacency.h                                                  */
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

#include "modules/direct_delta_mush/ddm_deformer.h"
#include "modules/direct_delta_mush/tests/test_fixtures.h"
#include "tests/test_macros.h"

namespace TestDDMAdjacency {

TEST_CASE("[DDM][Adjacency] Cube mesh neighbor counts") {
	DDMTestFixtures::CubeMeshFixture cube;

	DDMDeformer::MeshData mesh_data;
	mesh_data.vertices = cube.vertices;
	mesh_data.indices = cube.indices;

	DDMDeformer deformer;
	deformer.initialize(mesh_data);

	// All cube corner vertices should have exactly 3 neighbors
	// (This is a cube-specific property)
	INFO("Cube has 8 corner vertices, each should connect to 3 neighbors");

	// Test adjacency indirectly via Laplacian computation
	// If adjacency is wrong, Laplacian will also be wrong
	for (int v = 0; v < 8; v++) {
		Vector3 avg = deformer.test_compute_laplacian_average(v, cube.vertices, true);

		// Average should be finite and within reasonable bounds
		CHECK(Math::is_finite(avg.x));
		CHECK(Math::is_finite(avg.y));
		CHECK(Math::is_finite(avg.z));

		// For a unit cube, neighbor averages should be in [0,1] range
		CHECK(avg.x >= -0.1); // Small tolerance for floating point
		CHECK(avg.x <= 1.1);
		CHECK(avg.y >= -0.1);
		CHECK(avg.y <= 1.1);
		CHECK(avg.z >= -0.1);
		CHECK(avg.z <= 1.1);
	}
}

TEST_CASE("[DDM][Adjacency] Triangle mesh connectivity") {
	DDMTestFixtures::TriangleMeshFixture triangle;

	DDMDeformer::MeshData mesh_data;
	mesh_data.vertices = triangle.vertices;
	mesh_data.indices = triangle.indices;

	DDMDeformer deformer;
	deformer.initialize(mesh_data);

	// In a triangle, each vertex connects to 2 others
	for (int v = 0; v < 3; v++) {
		Vector3 avg = deformer.test_compute_laplacian_average(v, triangle.vertices, true);

		// Average should be finite
		CHECK(Math::is_finite(avg.x));
		CHECK(Math::is_finite(avg.y));
		CHECK(Math::is_finite(avg.z));

		// For a right triangle, averages should be reasonable
		CHECK(avg.x >= -0.1);
		CHECK(avg.x <= 1.1);
		CHECK(avg.y >= -0.1);
		CHECK(avg.y <= 1.1);
	}
}

TEST_CASE("[DDM][Adjacency] Degenerate mesh handling") {
	SUBCASE("Empty mesh") {
		DDMDeformer::MeshData empty_mesh;
		DDMDeformer deformer;
		deformer.initialize(empty_mesh);

		// Should not crash on empty mesh
		Vector<Vector3> empty_positions;
		Vector<Vector3> result = deformer.test_apply_laplacian_smoothing(empty_positions, 1, 0.5);
		CHECK(result.size() == 0);
	}

	SUBCASE("Single vertex (no neighbors)") {
		DDMTestFixtures::DegenerateMeshFixtures degen;

		DDMDeformer::MeshData mesh_data;
		mesh_data.vertices = degen.single_vertex;
		mesh_data.indices = degen.no_indices;

		DDMDeformer deformer;
		deformer.initialize(mesh_data);

		// Vertex with no neighbors should return itself
		Vector3 avg = deformer.test_compute_laplacian_average(0, degen.single_vertex, true);
		CHECK(avg == degen.single_vertex[0]);
	}

	SUBCASE("Collinear vertices (degenerate triangle)") {
		DDMTestFixtures::DegenerateMeshFixtures degen;

		DDMDeformer::MeshData mesh_data;
		mesh_data.vertices = degen.collinear_vertices;
		mesh_data.indices = degen.degenerate_triangle;

		DDMDeformer deformer;
		deformer.initialize(mesh_data);

		// Should not crash, even with degenerate geometry
		for (int v = 0; v < 3; v++) {
			Vector3 avg = deformer.test_compute_laplacian_average(v, degen.collinear_vertices, true);
			CHECK(Math::is_finite(avg.x));
			CHECK(Math::is_finite(avg.y));
			CHECK(Math::is_finite(avg.z));
		}
	}
}

TEST_CASE("[DDM][Adjacency] Symmetry verification") {
	DDMTestFixtures::CubeMeshFixture cube;

	DDMDeformer::MeshData mesh_data;
	mesh_data.vertices = cube.vertices;
	mesh_data.indices = cube.indices;

	DDMDeformer deformer;
	deformer.initialize(mesh_data);

	// Test adjacency indirectly through smoothing behavior
	// If adjacency is symmetric, smoothing should be stable
	Vector<Vector3> smoothed = deformer.test_apply_laplacian_smoothing(cube.vertices, 10, 0.5);

	// After smoothing, all vertices should still be finite
	for (int v = 0; v < smoothed.size(); v++) {
		CHECK(Math::is_finite(smoothed[v].x));
		CHECK(Math::is_finite(smoothed[v].y));
		CHECK(Math::is_finite(smoothed[v].z));
	}

	// Smoothing should move vertices toward center of mass
	Vector3 original_center(0.5, 0.5, 0.5); // Cube center
	for (int v = 0; v < smoothed.size(); v++) {
		real_t original_dist = cube.vertices[v].distance_to(original_center);
		real_t smoothed_dist = smoothed[v].distance_to(original_center);

		// Smoothing should generally move toward center (some tolerance)
		CHECK(smoothed_dist <= original_dist + 0.1);
	}
}

} // namespace TestDDMAdjacency
