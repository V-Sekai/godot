/**************************************************************************/
/*  test_csg_sculpted_complex_shapes.h                                    */
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

#include "modules/csg/csg_sculpted_primitive_base.h"
#include "modules/csg/csg_sculpted_ring.h"
#include "modules/csg/csg_sculpted_tube.h"

#include "tests/test_macros.h"

namespace TestCSG {

TEST_CASE("[SceneTree][CSG] CSGSculptedTube3D") {
	SUBCASE("[SceneTree][CSG] CSGSculptedTube3D: Basic shape generation") {
		CSGSculptedTube3D *tube = memnew(CSGSculptedTube3D);
		SceneTree::get_singleton()->get_root()->add_child(tube);

		tube->set_inner_radius(0.25);
		tube->set_outer_radius(0.5);
		tube->set_height(2.0);
		Vector<Vector3> faces = tube->get_brush_faces();

		CHECK_MESSAGE(faces.size() > 0, "Tube should generate faces");
		CHECK_MESSAGE(faces.size() % 3 == 0, "Faces should be triangles (multiple of 3)");

		SceneTree::get_singleton()->get_root()->remove_child(tube);
		memdelete(tube);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedTube3D: Property getters and setters") {
		CSGSculptedTube3D *tube = memnew(CSGSculptedTube3D);
		tube->set_inner_radius(0.3);
		tube->set_outer_radius(0.6);
		tube->set_height(3.0);
		CHECK(tube->get_inner_radius() == doctest::Approx(0.3));
		CHECK(tube->get_outer_radius() == doctest::Approx(0.6));
		CHECK(tube->get_height() == doctest::Approx(3.0));

		memdelete(tube);
	}
}

TEST_CASE("[SceneTree][CSG] CSGSculptedRing3D") {
	SUBCASE("[SceneTree][CSG] CSGSculptedRing3D: Basic shape generation") {
		CSGSculptedRing3D *ring = memnew(CSGSculptedRing3D);
		SceneTree::get_singleton()->get_root()->add_child(ring);

		ring->set_inner_radius(0.4);
		ring->set_outer_radius(0.5);
		ring->set_height(0.1);
		Vector<Vector3> faces = ring->get_brush_faces();

		CHECK_MESSAGE(faces.size() > 0, "Ring should generate faces");
		CHECK_MESSAGE(faces.size() % 3 == 0, "Faces should be triangles (multiple of 3)");

		SceneTree::get_singleton()->get_root()->remove_child(ring);
		memdelete(ring);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedRing3D: Property getters and setters") {
		CSGSculptedRing3D *ring = memnew(CSGSculptedRing3D);
		ring->set_inner_radius(0.3);
		ring->set_outer_radius(0.6);
		ring->set_height(0.2);
		CHECK(ring->get_inner_radius() == doctest::Approx(0.3));
		CHECK(ring->get_outer_radius() == doctest::Approx(0.6));
		CHECK(ring->get_height() == doctest::Approx(0.2));

		memdelete(ring);
	}
}

} // namespace TestCSG
