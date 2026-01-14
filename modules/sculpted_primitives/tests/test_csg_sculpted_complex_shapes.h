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

#include "modules/sculpted_primitives/csg_sculpted_primitive_base.h"
#include "modules/sculpted_primitives/csg_sculpted_ring.h"
#include "modules/sculpted_primitives/csg_sculpted_tube.h"

#include "tests/test_macros.h"

namespace TestCSG {

TEST_CASE("[CSG] CSGSculptedTube3D") {
	SUBCASE("Default initialization") {
		CSGSculptedTube3D *tube = memnew(CSGSculptedTube3D);

		// Check default properties
		CHECK(tube->get_inner_radius() == doctest::Approx(0.25));
		CHECK(tube->get_outer_radius() == doctest::Approx(0.5));
		CHECK(tube->get_height() == doctest::Approx(1.0));

		// Check default profile curve (should be circle for tube)
		CHECK(tube->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE);

		memdelete(tube);
	}

	SUBCASE("Inner and outer radius property getters and setters") {
		CSGSculptedTube3D *tube = memnew(CSGSculptedTube3D);

		tube->set_inner_radius(0.3);
		tube->set_outer_radius(0.8);
		tube->set_height(2.5);
		CHECK(tube->get_inner_radius() == doctest::Approx(0.3));
		CHECK(tube->get_outer_radius() == doctest::Approx(0.8));
		CHECK(tube->get_height() == doctest::Approx(2.5));

		// Test zero values
		tube->set_inner_radius(0.0);
		tube->set_outer_radius(0.0);
		tube->set_height(0.0);
		CHECK(tube->get_inner_radius() == doctest::Approx(0.0));
		CHECK(tube->get_outer_radius() == doctest::Approx(0.0));
		CHECK(tube->get_height() == doctest::Approx(0.0));

		// Test negative values (should be allowed)
		tube->set_inner_radius(-0.1);
		tube->set_outer_radius(-0.5);
		tube->set_height(-1.0);
		CHECK(tube->get_inner_radius() == doctest::Approx(-0.1));
		CHECK(tube->get_outer_radius() == doctest::Approx(-0.5));
		CHECK(tube->get_height() == doctest::Approx(-1.0));

		memdelete(tube);
	}

	SUBCASE("Radius relationship validation") {
		CSGSculptedTube3D *tube = memnew(CSGSculptedTube3D);

		// Inner radius should typically be less than outer radius, but not enforced
		tube->set_inner_radius(0.8);
		tube->set_outer_radius(0.3);
		CHECK(tube->get_inner_radius() == doctest::Approx(0.8));
		CHECK(tube->get_outer_radius() == doctest::Approx(0.3));

		// Test equal radii
		tube->set_inner_radius(0.5);
		tube->set_outer_radius(0.5);
		CHECK(tube->get_inner_radius() == doctest::Approx(0.5));
		CHECK(tube->get_outer_radius() == doctest::Approx(0.5));

		memdelete(tube);
	}
}

TEST_CASE("[CSG] CSGSculptedRing3D") {
	SUBCASE("Default initialization") {
		CSGSculptedRing3D *ring = memnew(CSGSculptedRing3D);

		// Check default properties
		CHECK(ring->get_inner_radius() == doctest::Approx(0.25));
		CHECK(ring->get_outer_radius() == doctest::Approx(0.5));
		CHECK(ring->get_height() == doctest::Approx(0.1));

		// Check default profile curve (should be circle for ring)
		CHECK(ring->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE);

		memdelete(ring);
	}

	SUBCASE("Ring property getters and setters") {
		CSGSculptedRing3D *ring = memnew(CSGSculptedRing3D);

		ring->set_inner_radius(0.4);
		ring->set_outer_radius(0.7);
		ring->set_height(0.2);
		CHECK(ring->get_inner_radius() == doctest::Approx(0.4));
		CHECK(ring->get_outer_radius() == doctest::Approx(0.7));
		CHECK(ring->get_height() == doctest::Approx(0.2));

		// Test very thin ring
		ring->set_height(0.01);
		CHECK(ring->get_height() == doctest::Approx(0.01));

		memdelete(ring);
	}

	SUBCASE("Ring geometry constraints") {
		CSGSculptedRing3D *ring = memnew(CSGSculptedRing3D);

		// Test when inner radius equals outer radius (degenerate case)
		ring->set_inner_radius(0.5);
		ring->set_outer_radius(0.5);
		CHECK(ring->get_inner_radius() == doctest::Approx(0.5));
		CHECK(ring->get_outer_radius() == doctest::Approx(0.5));

		// Test when inner radius is larger than outer radius
		ring->set_inner_radius(0.8);
		ring->set_outer_radius(0.3);
		CHECK(ring->get_inner_radius() == doctest::Approx(0.8));
		CHECK(ring->get_outer_radius() == doctest::Approx(0.3));

		memdelete(ring);
	}
}

} // namespace TestCSG
