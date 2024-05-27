/**************************************************************************/
/*  test_slicer.h                                                         */
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

#ifndef TEST_SLICER_H
#define TEST_SLICER_H

#include "tests/test_macros.h"

#include "../slicer.h"
#include "scene/resources/3d/primitive_meshes.h"

namespace TestIntersector {

TEST_SUITE("[Slicer]") {
	Plane plane(Vector3(1, 0, 0), 0);

	TEST_CASE("[Modules][Slicer][SceneTree] Smoke test") {
		Ref<SphereMesh> sphere_mesh;
		sphere_mesh.instantiate();
		Slicer slicer;
		Ref<SlicedMesh> sliced_mesh = slicer.slice_by_plane(sphere_mesh, plane, NULL);
		REQUIRE_FALSE(sliced_mesh.is_null());
		REQUIRE_FALSE(sliced_mesh->upper_mesh.is_null());
		REQUIRE_FALSE(sliced_mesh->lower_mesh.is_null());
	}
}
} //namespace TestIntersector

#endif // TEST_SLICER_H
