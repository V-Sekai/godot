/**************************************************************************/
/*  test_sliced_mesh.h                                                    */
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

#ifndef TEST_SLICED_MESH_H
#define TEST_SLICED_MESH_H

#include "tests/test_macros.h"

#include "../sliced_mesh.h"
#include "scene/resources/material.h"

namespace TestSlicedMesh {

TEST_SUITE("[SlicedMesh]") {
	TEST_CASE("[Modules][Slicer][SceneTree] Creates new meshes") {
		Intersector::SplitResult result;
		Vector<Intersector::SplitResult> results;
		result.material = Ref<StandardMaterial3D>();
		result.lower_faces.push_back(SlicerFace(Vector3(0, 0, 0), Vector3(0, 1, 0), Vector3(0, 1, 1)));
		result.lower_faces.push_back(SlicerFace(Vector3(0, 1, 1), Vector3(0, 0, 1), Vector3(0, 0, 0)));

		result.upper_faces.push_back(SlicerFace(Vector3(0, 1, 0), Vector3(0, 2, 0), Vector3(0, 2, 1)));
		result.upper_faces.push_back(SlicerFace(Vector3(0, 2, 1), Vector3(0, 1, 1), Vector3(0, 1, 0)));

		results.push_back(result);

		Vector<SlicerFace> cross_section_faces;
		Ref<StandardMaterial3D> cross_section_material;
		cross_section_material.instantiate();
		cross_section_faces.push_back(SlicerFace(Vector3(0, 1, 0), Vector3(1, 1, 0), Vector3(0, 1, 1)));

		SlicedMesh sliced(results, cross_section_faces, cross_section_material);
		REQUIRE_FALSE(sliced.lower_mesh.is_null());
		REQUIRE_FALSE(sliced.upper_mesh.is_null());

		REQUIRE(sliced.lower_mesh->get_surface_count() == 2);
		REQUIRE(sliced.upper_mesh->get_surface_count() == 2);

		REQUIRE(sliced.lower_mesh->surface_get_material(0) == result.material);
		REQUIRE(sliced.lower_mesh->surface_get_material(1) == cross_section_material);

		REQUIRE(sliced.upper_mesh->surface_get_material(0) == result.material);
		REQUIRE(sliced.upper_mesh->surface_get_material(1) == cross_section_material);
	}
}
} //namespace TestSlicedMesh

#endif // TEST_SLICED_MESH_H
