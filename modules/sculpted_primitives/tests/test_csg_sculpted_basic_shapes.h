/**************************************************************************/
/*  test_csg_sculpted_basic_shapes.h                                      */
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

#include "modules/sculpted_primitives/csg_sculpted_box.h"
#include "modules/sculpted_primitives/csg_sculpted_cylinder.h"
#include "modules/sculpted_primitives/csg_sculpted_primitive_base.h"
#include "modules/sculpted_primitives/csg_sculpted_prism.h"
#include "modules/sculpted_primitives/csg_sculpted_sphere.h"
#include "modules/sculpted_primitives/csg_sculpted_torus.h"

#include "tests/test_macros.h"

namespace TestCSG {

TEST_CASE("[CSG] CSGSculptedBox3D") {
	SUBCASE("Default initialization") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);

		// Check default size
		CHECK(box->get_size().is_equal_approx(Vector3(1.0, 1.0, 1.0)));

		// Check default profile curve (should be square for box)
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE);

		memdelete(box);
	}

	SUBCASE("Size property getters and setters") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);

		Vector3 test_size(3.0, 4.0, 5.0);
		box->set_size(test_size);
		CHECK(box->get_size().is_equal_approx(test_size));

		// Test zero size
		box->set_size(Vector3(0.0, 0.0, 0.0));
		CHECK(box->get_size().is_equal_approx(Vector3(0.0, 0.0, 0.0)));

		// Test negative size (should be allowed)
		box->set_size(Vector3(-1.0, -2.0, -3.0));
		CHECK(box->get_size().is_equal_approx(Vector3(-1.0, -2.0, -3.0)));

		memdelete(box);
	}

	SUBCASE("Profile curve validation") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);

		// Box should default to square profile
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE);

		// Test changing profile curve
		box->set_profile_curve(CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE);
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE);

		memdelete(box);
	}
}

TEST_CASE("[CSG] CSGSculptedCylinder3D") {
	SUBCASE("Default initialization") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);

		// Check default properties
		CHECK(cylinder->get_radius() == doctest::Approx(0.5));
		CHECK(cylinder->get_height() == doctest::Approx(1.0));

		// Check default profile curve (should be circle for cylinder)
		CHECK(cylinder->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE);

		memdelete(cylinder);
	}

	SUBCASE("Radius and height property getters and setters") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);

		cylinder->set_radius(2.5);
		cylinder->set_height(4.0);
		CHECK(cylinder->get_radius() == doctest::Approx(2.5));
		CHECK(cylinder->get_height() == doctest::Approx(4.0));

		// Test zero values
		cylinder->set_radius(0.0);
		cylinder->set_height(0.0);
		CHECK(cylinder->get_radius() == doctest::Approx(0.0));
		CHECK(cylinder->get_height() == doctest::Approx(0.0));

		// Test negative values (should be allowed)
		cylinder->set_radius(-1.0);
		cylinder->set_height(-2.0);
		CHECK(cylinder->get_radius() == doctest::Approx(-1.0));
		CHECK(cylinder->get_height() == doctest::Approx(-2.0));

		memdelete(cylinder);
	}
}

TEST_CASE("[CSG] CSGSculptedSphere3D") {
	SUBCASE("Default initialization") {
		CSGSculptedSphere3D *sphere = memnew(CSGSculptedSphere3D);

		// Check default radius
		CHECK(sphere->get_radius() == doctest::Approx(0.5));

		// Check default profile curve (should be circle for sphere)
		CHECK(sphere->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE);

		memdelete(sphere);
	}

	SUBCASE("Radius property getters and setters") {
		CSGSculptedSphere3D *sphere = memnew(CSGSculptedSphere3D);

		sphere->set_radius(3.5);
		CHECK(sphere->get_radius() == doctest::Approx(3.5));

		// Test zero radius
		sphere->set_radius(0.0);
		CHECK(sphere->get_radius() == doctest::Approx(0.0));

		// Test negative radius (should be allowed)
		sphere->set_radius(-1.0);
		CHECK(sphere->get_radius() == doctest::Approx(-1.0));

		memdelete(sphere);
	}
}

TEST_CASE("[CSG] CSGSculptedTorus3D") {
	SUBCASE("Default initialization") {
		CSGSculptedTorus3D *torus = memnew(CSGSculptedTorus3D);

		// Check default properties
		CHECK(torus->get_inner_radius() == doctest::Approx(0.25));
		CHECK(torus->get_outer_radius() == doctest::Approx(0.5));

		// Check default profile curve (should be circle for torus)
		CHECK(torus->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE);

		memdelete(torus);
	}

	SUBCASE("Inner and outer radius property getters and setters") {
		CSGSculptedTorus3D *torus = memnew(CSGSculptedTorus3D);

		torus->set_inner_radius(0.3);
		torus->set_outer_radius(0.8);
		CHECK(torus->get_inner_radius() == doctest::Approx(0.3));
		CHECK(torus->get_outer_radius() == doctest::Approx(0.8));

		// Test zero values
		torus->set_inner_radius(0.0);
		torus->set_outer_radius(0.0);
		CHECK(torus->get_inner_radius() == doctest::Approx(0.0));
		CHECK(torus->get_outer_radius() == doctest::Approx(0.0));

		// Test negative values (should be allowed)
		torus->set_inner_radius(-0.1);
		torus->set_outer_radius(-0.5);
		CHECK(torus->get_inner_radius() == doctest::Approx(-0.1));
		CHECK(torus->get_outer_radius() == doctest::Approx(-0.5));

		memdelete(torus);
	}

	SUBCASE("Radius relationship validation") {
		CSGSculptedTorus3D *torus = memnew(CSGSculptedTorus3D);

		// Inner radius should typically be less than outer radius, but not enforced
		torus->set_inner_radius(0.8);
		torus->set_outer_radius(0.3);
		CHECK(torus->get_inner_radius() == doctest::Approx(0.8));
		CHECK(torus->get_outer_radius() == doctest::Approx(0.3));

		memdelete(torus);
	}
}

TEST_CASE("[CSG] CSGSculptedPrism3D") {
	SUBCASE("Default initialization") {
		CSGSculptedPrism3D *prism = memnew(CSGSculptedPrism3D);

		// Check default size
		CHECK(prism->get_size().is_equal_approx(Vector3(1.0, 1.0, 1.0)));

		// Check default profile curve (should be triangle for prism)
		CHECK(prism->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_ISOTRI);

		memdelete(prism);
	}

	SUBCASE("Size property getters and setters") {
		CSGSculptedPrism3D *prism = memnew(CSGSculptedPrism3D);

		Vector3 test_size(2.0, 3.0, 4.0);
		prism->set_size(test_size);
		CHECK(prism->get_size().is_equal_approx(test_size));

		// Test zero size
		prism->set_size(Vector3(0.0, 0.0, 0.0));
		CHECK(prism->get_size().is_equal_approx(Vector3(0.0, 0.0, 0.0)));

		memdelete(prism);
	}
}

} // namespace TestCSG
