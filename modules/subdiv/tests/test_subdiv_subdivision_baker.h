/**************************************************************************/
/*  test_subdiv_subdivision_baker.h                                       */
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

#include "../src/subdivision/subdivision_baker.hpp"
#include "core/io/resource_loader.h"
#include "tests/test_macros.h"

// TODO: 202308024 iFire
// //just checks for non empty usable data
// TEST_CASE("[Subdiv] Simple bake") {
// 	Ref<SubdivisionBaker> baker;
// 	baker.instantiate();
// 	Ref<TopologyDataMesh> source_mesh = core_bind::ResourceLoader::get_singleton()->load("res://test/skinning_test.tres");
// 	Array result_arrays = source_mesh->surface_get_arrays(0);
// 	CHECK(!result_arrays.is_empty());
// 	const PackedVector3Array &vertex_array = result_arrays[Mesh::ARRAY_VERTEX];
// 	int32_t vertex_amount = vertex_array.size();

// 	const PackedInt32Array &bones_array = result_arrays[Mesh::ARRAY_BONES];
// 	CHECK_EQ(bones_array.size(), vertex_amount * 4);
// 	const Vector<real_t> &weights_array = result_arrays[Mesh::ARRAY_WEIGHTS];
// 	CHECK_EQ(bones_array.size(), weights_array.size());
// }
