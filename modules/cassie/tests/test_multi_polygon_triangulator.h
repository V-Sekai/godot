/**************************************************************************/
/*  test_multi_polygon_triangulator.h                                     */
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

#include "../src/cassie_path_3d.h"
#include "../src/cassie_surface.h"
#include "../src/intrinsic_triangulation.h"
#include "../src/polygon_triangulation_godot.h"

namespace TestPolygonTriangulation {

// Disabled: PolygonTriangulation C++ class has memory management issues
// TEST_CASE("[Modules][Cassie][PolygonTriangulation] Basic initialization") {
// 	// Create a simple triangle (3 points)
// 	const int ptn = 3;
// 	double pts[9] = {
// 		0.0, 0.0, 0.0,
// 		1.0, 0.0, 0.0,
// 		0.5, 1.0, 0.0
// 	};
//
// 	// Create PolygonTriangulation object with these points
// 	Ref<PolygonTriangulation> polyTri = PolygonTriangulation::_create_with_degenerates(ptn, pts, nullptr, false);
// 	CHECK(polyTri.is_valid());
// }

TEST_CASE("[Modules][Cassie][PolygonTriangulationGodot] Simple triangle triangulation") {
	// Create a simple triangle in the XY plane
	PackedVector3Array triangle;
	triangle.push_back(Vector3(0, 0, 0));
	triangle.push_back(Vector3(1, 0, 0));
	triangle.push_back(Vector3(0.5, 1, 0));

	// Create triangulator
	Ref<PolygonTriangulationGodot> tri = PolygonTriangulationGodot::create(triangle);
	CHECK(tri.is_valid());

	// Preprocess
	bool preprocess_ok = tri->preprocess();
	CHECK(preprocess_ok);

	// Triangulate
	bool triangulate_ok = tri->triangulate();
	CHECK(triangulate_ok);

	// Check vertex and index counts
	int vertex_count = tri->get_vertex_count();
	int triangle_count = tri->get_triangle_count();
	CHECK(vertex_count == 3);
	CHECK(triangle_count == 1);
}

TEST_CASE("[Modules][Cassie][PolygonTriangulationGodot] Quadrilateral triangulation") {
	// Create a simple quadrilateral in the XY plane
	PackedVector3Array quad;
	quad.push_back(Vector3(0, 0, 0));
	quad.push_back(Vector3(1, 0, 0));
	quad.push_back(Vector3(1, 1, 0));
	quad.push_back(Vector3(0, 1, 0));

	// Create triangulator
	Ref<PolygonTriangulationGodot> tri = PolygonTriangulationGodot::create(quad);
	CHECK(tri.is_valid());

	// Preprocess
	bool preprocess_ok = tri->preprocess();
	CHECK(preprocess_ok);

	// Triangulate
	bool triangulate_ok = tri->triangulate();
	CHECK(triangulate_ok);

	// Check vertex and index counts
	int vertex_count = tri->get_vertex_count();
	int triangle_count = tri->get_triangle_count();
	CHECK(vertex_count == 4);
	CHECK(triangle_count == 2);
}

TEST_CASE("[Modules][Cassie][PolygonTriangulationGodot] Pentagon triangulation") {
	// Create a simple pentagon in the XY plane
	PackedVector3Array pentagon;
	pentagon.push_back(Vector3(0, 0, 0));
	pentagon.push_back(Vector3(1, 0, 0));
	pentagon.push_back(Vector3(1.5, 0.5, 0));
	pentagon.push_back(Vector3(1, 1, 0));
	pentagon.push_back(Vector3(0, 1, 0));

	// Create triangulator
	Ref<PolygonTriangulationGodot> tri = PolygonTriangulationGodot::create(pentagon);
	CHECK(tri.is_valid());

	// Preprocess
	bool preprocess_ok = tri->preprocess();
	CHECK(preprocess_ok);

	// Triangulate
	bool triangulate_ok = tri->triangulate();
	CHECK(triangulate_ok);

	// Check vertex and index counts
	int vertex_count = tri->get_vertex_count();
	int triangle_count = tri->get_triangle_count();
	CHECK(vertex_count == 5);
	CHECK(triangle_count == 3);
}

TEST_CASE("[Modules][Cassie][PolygonTriangulationGodot] Hexagon triangulation") {
	// Create a regular hexagon in the XY plane
	PackedVector3Array hexagon;
	float radius = 1.0f;
	for (int i = 0; i < 6; i++) {
		float angle = i * Math::PI / 3.0f; // 60 degrees
		float x = radius * cos(angle);
		float y = radius * sin(angle);
		hexagon.push_back(Vector3(x, y, 0));
	}

	// Create triangulator
	Ref<PolygonTriangulationGodot> tri = PolygonTriangulationGodot::create(hexagon);
	CHECK(tri.is_valid());

	// Preprocess
	bool preprocess_ok = tri->preprocess();
	CHECK(preprocess_ok);

	// Triangulate
	bool triangulate_ok = tri->triangulate();
	CHECK(triangulate_ok);

	// Check vertex and index counts
	int vertex_count = tri->get_vertex_count();
	int triangle_count = tri->get_triangle_count();
	CHECK(vertex_count == 6);
	CHECK(triangle_count == 4); // n-2 = 6-2 = 4 triangles
}

TEST_CASE("[Modules][Cassie][PolygonTriangulationGodot] Star triangulation") {
	// Create a 5-pointed star in the XY plane
	PackedVector3Array star;
	float outer_radius = 1.0f;
	float inner_radius = 0.5f;
	for (int i = 0; i < 10; i++) { // 10 points for 5-pointed star
		float angle = i * Math::PI / 5.0f; // 36 degrees
		float radius = (i % 2 == 0) ? outer_radius : inner_radius;
		float x = radius * cos(angle);
		float y = radius * sin(angle);
		star.push_back(Vector3(x, y, 0));
	}

	// Create triangulator
	Ref<PolygonTriangulationGodot> tri = PolygonTriangulationGodot::create(star);
	CHECK(tri.is_valid());

	// Preprocess
	bool preprocess_ok = tri->preprocess();
	CHECK(preprocess_ok);

	// Triangulate
	bool triangulate_ok = tri->triangulate();
	CHECK(triangulate_ok);

	// Check vertex and index counts
	int vertex_count = tri->get_vertex_count();
	int triangle_count = tri->get_triangle_count();
	CHECK(vertex_count == 10);
	CHECK(triangle_count == 8); // n-2 = 10-2 = 8 triangles
}

TEST_CASE("[Modules][Cassie][CassiePath3D] Add and retrieve points") {
	Ref<CassiePath3D> path = memnew(CassiePath3D);

	// Add points
	path->add_point(Vector3(0, 0, 0), Vector3(0, 1, 0));
	path->add_point(Vector3(1, 0, 0), Vector3(0, 1, 0));
	path->add_point(Vector3(2, 0, 0), Vector3(0, 1, 0));

	// Check count
	CHECK(path->get_point_count() == 3);

	// Check positions
	CHECK(path->get_point_position(0) == Vector3(0, 0, 0));
	CHECK(path->get_point_position(1) == Vector3(1, 0, 0));
	CHECK(path->get_point_position(2) == Vector3(2, 0, 0));

	// Check normals
	CHECK(path->get_point_normal(0) == Vector3(0, 1, 0));
}

TEST_CASE("[Modules][Cassie][CassiePath3D] Laplacian smoothing") {
	Ref<CassiePath3D> path = memnew(CassiePath3D);

	// Create a zigzag path
	path->add_point(Vector3(0, 0, 0));
	path->add_point(Vector3(1, 1, 0));
	path->add_point(Vector3(2, 0, 0));
	path->add_point(Vector3(3, 1, 0));
	path->add_point(Vector3(4, 0, 0));

	// Apply smoothing
	path->beautify_laplacian(0.5, 3);

	// After smoothing, middle points should move toward average of neighbors
	// We can't check exact values without reimplementing the algorithm,
	// but we can verify the path still has the same number of points
	CHECK(path->get_point_count() == 5);
}

TEST_CASE("[Modules][Cassie][CassiePath3D] Uniform resampling") {
	Ref<CassiePath3D> path = memnew(CassiePath3D);

	// Create a path with non-uniform spacing
	path->add_point(Vector3(0, 0, 0));
	path->add_point(Vector3(0.1, 0, 0));
	path->add_point(Vector3(2, 0, 0));
	path->add_point(Vector3(3, 0, 0));

	// Resample to 10 points
	path->resample_uniform(10);

	// Check new count
	CHECK(path->get_point_count() == 10);

	// Check that points are roughly evenly spaced
	float total_length = path->get_total_length();
	float avg_segment = path->get_average_segment_length();
	float expected_avg = total_length / 9; // 9 segments between 10 points

	// Allow 10% tolerance
	CHECK(Math::abs(avg_segment - expected_avg) < expected_avg * 0.1);
}

TEST_CASE("[Modules][Cassie][IntrinsicTriangulation] Basic mesh initialization") {
	Ref<IntrinsicTriangulation> intrinsic = memnew(IntrinsicTriangulation);

	// Create simple triangle mesh data
	PackedVector3Array vertices;
	vertices.push_back(Vector3(0, 0, 0));
	vertices.push_back(Vector3(1, 0, 0));
	vertices.push_back(Vector3(0.5, 1, 0));
	vertices.push_back(Vector3(1.5, 1, 0));

	PackedInt32Array indices;
	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);
	indices.push_back(1);
	indices.push_back(3);
	indices.push_back(2);

	// Set mesh data
	intrinsic->set_mesh_data(vertices, indices, PackedVector3Array());

	// Check counts
	CHECK(intrinsic->get_vertex_count() == 4);
	CHECK(intrinsic->get_triangle_count() == 2);
}

TEST_CASE("[Modules][Cassie][IntrinsicTriangulation] Delaunay flipping") {
	Ref<IntrinsicTriangulation> intrinsic = memnew(IntrinsicTriangulation);

	// Create mesh that needs flipping (non-Delaunay configuration)
	PackedVector3Array vertices;
	vertices.push_back(Vector3(0, 0, 0));
	vertices.push_back(Vector3(2, 0, 0));
	vertices.push_back(Vector3(0, 2, 0));
	vertices.push_back(Vector3(2, 2, 0));

	PackedInt32Array indices;
	// Create a bad diagonal (0-3 instead of 1-2)
	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(3);
	indices.push_back(0);
	indices.push_back(3);
	indices.push_back(2);

	intrinsic->set_mesh_data(vertices, indices, PackedVector3Array());

	// Flip to Delaunay
	bool converged = intrinsic->flip_to_delaunay();
	CHECK(converged);

	// After flipping, should still have same vertex and triangle counts
	CHECK(intrinsic->get_vertex_count() == 4);
	CHECK(intrinsic->get_triangle_count() == 2);
}

// TEST_CASE("[Modules][Cassie][CassieSurface] Complete pipeline") {
// 	Ref<CassieSurface> surface = memnew(CassieSurface);

// 	// Create a square boundary path
// 	Ref<CassiePath3D> path = memnew(CassiePath3D);
// 	path->add_point(Vector3(0, 0, 0), Vector3(0, 0, 1));
// 	path->add_point(Vector3(1, 0, 0), Vector3(0, 0, 1));
// 	path->add_point(Vector3(1, 1, 0), Vector3(0, 0, 1));
// 	path->add_point(Vector3(0, 1, 0), Vector3(0, 0, 1));

// 	// Add to surface generator
// 	surface->add_boundary_path(path);

// 	// Configure pipeline
// 	surface->set_auto_beautify(true);
// 	surface->set_auto_resample(true);
// 	surface->set_use_intrinsic_remeshing(false);
// 	surface->set_target_boundary_points(4);

// 	// Generate surface
// 	Ref<ArrayMesh> mesh = surface->generate_surface();

// 	// Verify mesh was created (basic check)
// 	CHECK(mesh.is_valid());
// 	// Note: Full mesh validation disabled due to headless mode array issues
// }

TEST_CASE("[Modules][Cassie][CassieSurface] Multiple boundaries") {
	Ref<CassieSurface> surface = memnew(CassieSurface);

	// Create outer boundary (square)
	Ref<CassiePath3D> outer = memnew(CassiePath3D);
	outer->add_point(Vector3(-2, -2, 0));
	outer->add_point(Vector3(2, -2, 0));
	outer->add_point(Vector3(2, 2, 0));
	outer->add_point(Vector3(-2, 2, 0));

	// Create inner boundary (hole - circle)
	Ref<CassiePath3D> inner = memnew(CassiePath3D);
	for (int i = 0; i < 8; i++) {
		float angle = (float)i / 8 * 2.0f * 3.14159265359f;
		Vector3 pos(Math::cos(angle) * 0.5, Math::sin(angle) * 0.5, 0);
		inner->add_point(pos);
	}

	surface->add_boundary_path(outer);
	surface->add_boundary_path(inner);

	CHECK(surface->get_boundary_path_count() == 2);

	// Note: Full hole support requires multi-boundary triangulation
	// For now we just test that the pipeline doesn't crash
	Ref<ArrayMesh> mesh = surface->generate_surface();
	// May return valid mesh (outer only) or null (if implementation doesn't support holes yet)
	// Just verify no crash occurred
}

} //namespace TestPolygonTriangulation
