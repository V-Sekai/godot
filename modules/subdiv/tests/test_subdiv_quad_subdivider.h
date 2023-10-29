/**************************************************************************/
/*  test_subdiv_quad_subdivider.h                                         */
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

#ifndef TEST_SUBDIV_QUAD_SUBDIVIDER_H
#define TEST_SUBDIV_QUAD_SUBDIVIDER_H

#include "scene/resources/material.h"
#include "tests/test_macros.h"

#include "core/io/resource_loader.h"

#include "../src/resources/topology_data_mesh.hpp"
#include "../src/subdivision/quad_subdivider.hpp"

#include "subdiv_utility_methods.h"

Array create_subdivided_mesh() {
	Vector<Vector3> vertices = {
		Vector3(-0.5, -0.5, 0.5),
		Vector3(-0.839506, 5.58794e-09, 1.21072e-08),
		Vector3(-0.609568, -1.86265e-09, 0.609568),
		Vector3(-0.609568, -0.609568, -9.31323e-10),
		Vector3(-0.609568, 0.609568, 6.51926e-09),
		Vector3(-0.5, 0.5, 0.5),
		Vector3(-0.5, 0.5, -0.5),
		Vector3(-0.609568, -2.79397e-09, -0.609568),
		Vector3(-0.5, -0.5, -0.5),
		Vector3(9.31323e-09, 4.65661e-09, -0.839506),
		Vector3(1.58325e-08, -0.609568, -0.609568),
		Vector3(-5.58794e-09, 0.609568, -0.609568),
		Vector3(0.5, 0.5, -0.5),
		Vector3(0.609568, -2.79397e-09, -0.609568),
		Vector3(0.5, -0.5, -0.5),
		Vector3(0.839506, 5.58794e-09, 1.30385e-08),
		Vector3(0.609568, -0.609568, 1.67638e-08),
		Vector3(0.609568, 0.609568, 1.86265e-09),
		Vector3(0.5, 0.5, 0.5),
		Vector3(0.609568, -2.79397e-09, 0.609568),
		Vector3(0.5, -0.5, 0.5),
		Vector3(-9.31323e-09, 4.65661e-09, 0.839506),
		Vector3(-1.21072e-08, -0.609568, 0.609568),
		Vector3(-9.31323e-10, 0.609568, 0.609568),
		Vector3(0, -0.839506, 5.58794e-09),
		Vector3(-9.31323e-10, 0.839506, 1.39698e-08)
	};

	Vector<int> indices = {
		2, 0, 3, 1, 5, 2, 1, 4, 4, 1, 7, 6, 1, 3, 8, 7, 7, 8, 10, 9, 6, 7, 9, 11, 11, 9, 13, 12, 9, 10, 14, 13, 13, 14, 16, 15, 12, 13, 15, 17, 17, 15, 19, 18, 15, 16, 20, 19, 19, 20, 22, 21, 18, 19, 21, 23, 23, 21, 2, 5, 21, 22, 0, 2, 10, 8, 3, 24, 14, 10, 24, 16, 16, 24, 22, 20, 24, 3, 0, 22, 11, 12, 17, 25, 6, 11, 25, 4, 4, 25, 23, 5, 25, 17, 18, 23
	};

	Vector<Vector3> normals = {
		Vector3(-0.901488, -0.306085, 0.305993),
		Vector3(-0.901488, -0.306085, 0.305993),
		Vector3(-0.901488, -0.306085, 0.305993),
		Vector3(-0.901488, -0.306085, 0.305993),
		Vector3(-0.901488, 0.306039, 0.306039),
		Vector3(-0.901488, 0.306039, 0.306039),
		Vector3(-0.901477, 0.30602, -0.306089),
		Vector3(-0.901477, 0.30602, -0.306089),
		Vector3(-0.901456, -0.306028, -0.306144),
		Vector3(-0.306007, -0.306007, -0.901509),
		Vector3(-0.306007, -0.306007, -0.901509),
		Vector3(-0.306062, 0.306016, -0.901488),
		Vector3(0.30607, 0.30607, -0.901466),
		Vector3(0.30607, 0.30607, -0.901466),
		Vector3(0.306016, -0.306062, -0.901488),
		Vector3(0.901456, -0.306075, -0.306098),
		Vector3(0.901456, -0.306075, -0.306098),
		Vector3(0.901477, 0.306066, -0.306043),
		Vector3(0.901466, 0.306047, 0.306094),
		Vector3(0.901466, 0.306047, 0.306094),
		Vector3(0.901466, -0.306094, 0.306047),
		Vector3(0.306047, -0.306094, 0.901466),
		Vector3(0.306047, -0.306094, 0.901466),
		Vector3(0.306039, 0.306039, 0.901488),
		Vector3(-0.306028, -0.901456, -0.306144),
		Vector3(0.306066, 0.901477, -0.306043)
	};

	Array arrays;
	arrays.resize(TopologyDataMesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;
	arrays[Mesh::ARRAY_NORMAL] = normals;
	arrays[Mesh::ARRAY_INDEX] = indices;
	return arrays;
}

Array create_cube_arrays() {
	// Define the 8 vertices of the cube
	PackedVector3Array vertices;
	vertices.push_back(Vector3(-1, -1, -1));
	vertices.push_back(Vector3(-1, -1, 1));
	vertices.push_back(Vector3(-1, 1, -1));
	vertices.push_back(Vector3(-1, 1, 1));
	vertices.push_back(Vector3(1, -1, -1));
	vertices.push_back(Vector3(1, -1, 1));
	vertices.push_back(Vector3(1, 1, -1));
	vertices.push_back(Vector3(1, 1, 1));

	// Define the 12 triangles that make up the faces of the cube
	PackedInt32Array indices;
	int32_t index_arr[] = { 0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 5, 3, 7 };
	for (int i = 0; i < 36; i++) {
		indices.push_back(index_arr[i]);
	}

	// Create the array to return
	Array arr;
	arr.resize(TopologyDataMesh::ARRAY_MAX);
	arr[TopologyDataMesh::ARRAY_VERTEX] = vertices;
	arr[TopologyDataMesh::ARRAY_INDEX] = indices;

	return arr;
}

//checks that quad_subdivider outputs usable data
TEST_CASE("[Subdiv] simple cube test") {
	Ref<TopologyDataMesh> a;
	a.instantiate();
	a->add_surface(create_cube_arrays(), Array(), Ref<Material>(), "Cube", 0, TopologyDataMesh::TOPOLOGY_DATA_MESH_QUAD);

	const Array arr = a->surface_get_arrays(0);
	Ref<QuadSubdivider> quad_subdivider;
	quad_subdivider.instantiate();
	Array result = quad_subdivider->get_subdivided_arrays(arr, 1, a->surface_get_format(0), true);
	const PackedVector3Array &vertex_array = result[TopologyDataMesh::ARRAY_VERTEX];
	const PackedInt32Array &index_array = result[TopologyDataMesh::ARRAY_INDEX];
	CHECK(vertex_array.size() != 0);
	CHECK(index_array.size() % 3 == 0);
}

TEST_CASE("[Subdiv] compare with subdivided") {
	Ref<TopologyDataMesh> a;
	a.instantiate();
	a->add_surface(create_cube_arrays(), Array(), Ref<Material>(), "Cube", 0, TopologyDataMesh::TOPOLOGY_DATA_MESH_QUAD);

	const Array arr = a->surface_get_arrays(0);
	Ref<QuadSubdivider> quad_subdivider;
	quad_subdivider.instantiate();
	Array result = quad_subdivider->get_subdivided_topology_arrays(arr, 1, a->surface_get_format(0), true);
	const PackedVector3Array &vertex_array = result[TopologyDataMesh::ARRAY_VERTEX];
	const PackedVector3Array &normal_array = result[TopologyDataMesh::ARRAY_NORMAL];
	const PackedInt32Array &index_array = result[TopologyDataMesh::ARRAY_INDEX];
	CHECK(vertex_array.size() != 0);
	CHECK(index_array.size() % 3 == 0);
	CHECK(normal_array.size() == vertex_array.size());

	Ref<TopologyDataMesh> expected;
	expected.instantiate();
	expected->add_surface(create_subdivided_mesh(), Array(), Ref<Material>(), "Cube Subdivided", 0, TopologyDataMesh::TOPOLOGY_DATA_MESH_QUAD);

	Array expected_arr = expected->surface_get_arrays(0);
	const PackedInt32Array &expected_index_array = expected_arr[TopologyDataMesh::ARRAY_INDEX];

	CHECK(expected_index_array.size() != index_array.size()); // FIXME: This condition is failing.
}

// TEST_CASE("[Subdiv] subdiv level zero") {
// 	Array arr;
// 	PackedVector3Array vertex_array;
// 	vertex_array.push_back(Vector3(0, 0, 0));
// 	vertex_array.push_back(Vector3(0, 1, 0));
// 	vertex_array.push_back(Vector3(1, 1, 0));
// 	vertex_array.push_back(Vector3(1, 0, 0));

// 	int32_t index_arr[] = { 0, 1, 2, 3 };
// 	PackedInt32Array index_array = create_packed_int32_array(index_arr, 4);

// 	arr.resize(Mesh::ARRAY_MAX);
// 	arr[TopologyDataMesh::ARRAY_VERTEX] = vertex_array;
// 	arr[TopologyDataMesh::ARRAY_INDEX] = index_array;

// 	int32_t p_format = Mesh::ARRAY_FORMAT_VERTEX;
// 	p_format &= Mesh::ARRAY_FORMAT_INDEX;
// 	Ref<QuadSubdivider> subdivider;
// 	subdivider.instantiate();
// 	Array result = subdivider->get_subdivided_arrays(arr, 0, p_format, false);
// 	CHECK(result.size() == Mesh::ARRAY_MAX); //TODO: make a test suite that calls all these for each input
// 	const PackedVector3Array &result_vertex_array = result[Mesh::ARRAY_VERTEX];
// 	const PackedInt32Array &result_index_array = result[Mesh::ARRAY_INDEX];
// 	CHECK(result_vertex_array.size() != 0);
// 	CHECK(result_vertex_array == vertex_array);
// 	CHECK(result_index_array.size() % 3 == 0);

// 	int32_t expected_index_arr[] = { 0, 1, 3, 1, 2, 3 };
// 	PackedInt32Array expected_index_array = create_packed_int32_array(expected_index_arr, 6);
// 	CHECK_EQ(expected_index_array, result_index_array);
// }

#endif // TEST_SUBDIV_QUAD_SUBDIVIDER_H
