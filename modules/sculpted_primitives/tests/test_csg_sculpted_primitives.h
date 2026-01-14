/**************************************************************************/
/*  test_csg_sculpted_primitives.h                                        */
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
#include "modules/sculpted_primitives/csg_sculpted_ring.h"
#include "modules/sculpted_primitives/csg_sculpted_sphere.h"
#include "modules/sculpted_primitives/csg_sculpted_texture.h"
#include "modules/sculpted_primitives/csg_sculpted_torus.h"
#include "modules/sculpted_primitives/csg_sculpted_tube.h"

#include "tests/test_macros.h"

namespace TestCSG {

// Unit tests for sculpted primitive helper functions
TEST_CASE("[CSG] Helper Functions: generate_profile_points") {
	SUBCASE("Circle profile - full circle") {
		Vector<Vector2> profile;
		Vector<Vector2> hollow_profile;
		generate_profile_points(CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE, 0.0, 1.0, 0.0,
				CSGSculptedPrimitive3D::HOLLOW_SAME, 8, profile, hollow_profile);

		// Should generate 9 points (8 + 1 closing point)
		CHECK(profile.size() == 9);
		// First and last points should be the same (closed loop)
		CHECK(profile[0].is_equal_approx(profile[8]));
		// No hollow profile when hollow = 0
		CHECK(hollow_profile.size() == 0);
	}

	SUBCASE("Circle profile - half circle") {
		Vector<Vector2> profile;
		Vector<Vector2> hollow_profile;
		generate_profile_points(CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE_HALF, 0.0, 1.0, 0.0,
				CSGSculptedPrimitive3D::HOLLOW_SAME, 8, profile, hollow_profile);

		// Should generate 5 points (4 + 1 closing point)
		CHECK(profile.size() == 5);
		CHECK(profile[0].is_equal_approx(profile[4]));
	}

	SUBCASE("Circle profile - with hollow") {
		Vector<Vector2> profile;
		Vector<Vector2> hollow_profile;
		generate_profile_points(CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE, 0.0, 1.0, 0.3,
				CSGSculptedPrimitive3D::HOLLOW_SAME, 8, profile, hollow_profile);

		// Both outer and inner profiles should be generated
		CHECK(profile.size() == 9);
		CHECK(hollow_profile.size() == 9);
		// Hollow profile should be smaller
		for (int i = 0; i < hollow_profile.size(); i++) {
			CHECK(hollow_profile[i].length() < profile[i].length());
		}
	}

	SUBCASE("Square profile") {
		Vector<Vector2> profile;
		Vector<Vector2> hollow_profile;
		generate_profile_points(CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE, 0.0, 1.0, 0.0,
				CSGSculptedPrimitive3D::HOLLOW_SAME, 8, profile, hollow_profile);

		// Should generate 5 points (4 corners + 1 closing point)
		CHECK(profile.size() == 5);
		CHECK(profile[0].is_equal_approx(profile[4]));
		// All points should be at distance 1 from center (normalized)
		for (const Vector2 &p : profile) {
			CHECK(p.length() == doctest::Approx(1.0));
		}
	}

	SUBCASE("Triangle profile") {
		Vector<Vector2> profile;
		Vector<Vector2> hollow_profile;
		generate_profile_points(CSGSculptedPrimitive3D::PROFILE_CURVE_ISOTRI, 0.0, 1.0, 0.0,
				CSGSculptedPrimitive3D::HOLLOW_SAME, 8, profile, hollow_profile);

		// Should generate 4 points (3 corners + 1 closing point)
		CHECK(profile.size() == 4);
		CHECK(profile[0].is_equal_approx(profile[3]));
	}
}

TEST_CASE("[CSG] Helper Functions: apply_path_transform") {
	SUBCASE("Linear path - no transformations") {
		Vector2 profile_point(1.0, 0.0);
		Vector3 result = apply_path_transform(profile_point, 0.5,
				CSGSculptedPrimitive3D::PATH_CURVE_LINE, 0.0, Vector2(0.0, 0.0),
				Vector2(0.0, 0.0), 0.0, 1.0, 0.0);

		// Should place point at (1, 0, 0) since path_pos=0.5 centers at z=0
		CHECK(result.x == doctest::Approx(1.0));
		CHECK(result.y == doctest::Approx(0.0));
		CHECK(result.z == doctest::Approx(0.0));
	}

	SUBCASE("Linear path - with taper") {
		Vector2 profile_point(1.0, 0.0);
		Vector3 result = apply_path_transform(profile_point, 1.0,
				CSGSculptedPrimitive3D::PATH_CURVE_LINE, 0.0, Vector2(0.5, 0.0),
				Vector2(0.0, 0.0), 0.0, 1.0, 0.0);

		// Taper.x=0.5 means scale by 0.5 at the end (path_pos=1.0)
		// taper_factor = 1.0 - (0.5 * (1.0-1.0) + 0.0 * 1.0) = 1.0 - 0.0 = 1.0
		// Wait, let me recalculate: taper_factor = 1.0 - (taper.x * (1.0 - normalized_path) + taper.y * normalized_path)
		// At path_pos=1.0, normalized_path=1.0: taper_factor = 1.0 - (0.5 * 0.0 + 0.0 * 1.0) = 1.0
		// This doesn't make sense. Let me check the code again...

		// Actually looking at the code: taper_factor = 1.0 - (p_taper.x * (1.0 - normalized_path) + p_taper.y * normalized_path)
		// For taper Vector2(0.5, 0.0) at path_pos=1.0: 1.0 - (0.5 * 0.0 + 0.0 * 1.0) = 1.0
		// Let me try at path_pos=0.0: 1.0 - (0.5 * 1.0 + 0.0 * 0.0) = 0.5
		// So taper.x affects the beginning, taper.y affects the end

		result = apply_path_transform(profile_point, 0.0,
				CSGSculptedPrimitive3D::PATH_CURVE_LINE, 0.0, Vector2(0.5, 0.0),
				Vector2(0.0, 0.0), 0.0, 1.0, 0.0);

		// At beginning (path_pos=0.0), taper_factor = 1.0 - 0.5 = 0.5
		CHECK(result.x == doctest::Approx(0.5));
	}

	SUBCASE("Circular path") {
		Vector2 profile_point(1.0, 0.0);
		Vector3 result = apply_path_transform(profile_point, 0.0,
				CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE, 0.0, Vector2(0.0, 0.0),
				Vector2(0.0, 0.0), 0.0, 1.0, 0.0);

		// At path_pos=0.0, should be at (radius, 0, 0) = (1, 0, 0)
		CHECK(result.x == doctest::Approx(1.0));
		CHECK(result.y == doctest::Approx(0.0));
		CHECK(result.z == doctest::Approx(0.0));
	}

	SUBCASE("Twist transformation") {
		Vector2 profile_point(1.0, 0.0);
		Vector3 result = apply_path_transform(profile_point, 0.5,
				CSGSculptedPrimitive3D::PATH_CURVE_LINE, 0.25, Vector2(0.0, 0.0),
				Vector2(0.0, 0.0), 0.0, 1.0, 0.0);

		// Twist of 0.25 at path_pos=0.5 means twist_angle = 0.25 * 0.5 * TAU = 0.125 * TAU
		// cos(0.125*TAU) ≈ cos(0.5*PI) ≈ 0, sin(0.125*TAU) ≈ sin(0.5*PI) ≈ 1
		// So twisted_profile = (1*cos - 0*sin, 1*sin + 0*cos) = (0, 1)
		// Result should be (0, 1, 0)
		CHECK(result.x == doctest::Approx(0.0).epsilon(0.01));
		CHECK(result.y == doctest::Approx(1.0).epsilon(0.01));
	}
}

TEST_CASE("[CSG] Helper Functions: Profile and path parameter validation") {
	SUBCASE("Profile parameter clamping") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);

		// Test profile_begin clamping
		box->set_profile_begin(-0.5);
		CHECK(box->get_profile_begin() == doctest::Approx(0.0));
		box->set_profile_begin(1.5);
		CHECK(box->get_profile_begin() == doctest::Approx(1.0));

		// Test profile_end clamping
		box->set_profile_end(-0.5);
		CHECK(box->get_profile_end() == doctest::Approx(0.0));
		box->set_profile_end(1.5);
		CHECK(box->get_profile_end() == doctest::Approx(1.0));

		memdelete(box);
	}

	SUBCASE("Path parameter clamping") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);

		// Test path_begin clamping
		box->set_path_begin(-0.5);
		CHECK(box->get_path_begin() == doctest::Approx(0.0));
		box->set_path_begin(1.5);
		CHECK(box->get_path_begin() == doctest::Approx(1.0));

		// Test path_end clamping
		box->set_path_end(-0.5);
		CHECK(box->get_path_end() == doctest::Approx(0.0));
		box->set_path_end(1.5);
		CHECK(box->get_path_end() == doctest::Approx(1.0));

		memdelete(box);
	}
}

} // namespace TestCSG
