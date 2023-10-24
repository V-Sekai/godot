/**************************************************************************/
/*  test_subdiv_triangle_subdivider.h                                     */
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

#ifndef TEST_SUBDIV_TRIANGLE_SUBDIVIDER_H
#define TEST_SUBDIV_TRIANGLE_SUBDIVIDER_H

#include "tests/test_macros.h"

#include "../src/resources/topology_data_mesh.hpp"
#include "../src/subdivision/triangle_subdivider.hpp"
#include "core/io/resource_loader.h"
#include "subdiv_utility_methods.h"

// //I don't think a lot of unit tests are needed for this. If it outputs non empty data subdivision very likely worked since opensubdiv let the data through
// TEST_CASE("[Subdiv] subdivide once") {
// 	Array arr;
// 	PackedVector3Array vertex_array;
// 	vertex_array.push_back(Vector3(0, 0, 0));
// 	vertex_array.push_back(Vector3(0, 1, 0));
// 	vertex_array.push_back(Vector3(1, 1, 0));

// 	int32_t index_arr[] = { 0, 1, 2 };
// 	PackedInt32Array index_array = create_packed_int32_array(index_arr, 3);

// 	arr.resize(Mesh::ARRAY_MAX);
// 	arr[TopologyDataMesh::ARRAY_VERTEX] = vertex_array;
// 	arr[TopologyDataMesh::ARRAY_INDEX] = index_array;

// 	int32_t p_format = Mesh::ARRAY_FORMAT_VERTEX;
// 	p_format &= Mesh::ARRAY_FORMAT_INDEX;
// 	Ref<TriangleSubdivider> subdivider;
// 	subdivider.instantiate();
// 	Array result = subdivider->get_subdivided_arrays(arr, 1, p_format, false);
// 	CHECK(result.size() == Mesh::ARRAY_MAX);
// 	const PackedVector3Array &result_vertex_array = result[Mesh::ARRAY_VERTEX];
// 	const PackedInt32Array &result_index_array = result[Mesh::ARRAY_INDEX];
// 	CHECK(result_vertex_array.size() != 0);
// 	CHECK(result_index_array.size() % 3 == 0);
// }

#endif
