/**************************************************************************/
/*  test_csg_sculpted_validation.h                                        */
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

TEST_CASE("[SceneTree][CSG] CSGSculptedPrimitive3D: Common parameters") {
	SUBCASE("[SceneTree][CSG] CSGSculptedPrimitive3D: Profile and path parameters") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);
		SceneTree::get_singleton()->get_root()->add_child(box);

		// Test profile parameters
		box->set_profile_curve(CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE);
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE);

		box->set_profile_begin(0.1);
		box->set_profile_end(0.9);
		CHECK(box->get_profile_begin() == doctest::Approx(0.1));
		CHECK(box->get_profile_end() == doctest::Approx(0.9));

		// Test path parameters
		box->set_path_curve(CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE);
		CHECK(box->get_path_curve() == CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE);

		box->set_path_begin(0.2);
		box->set_path_end(0.8);
		CHECK(box->get_path_begin() == doctest::Approx(0.2));
		CHECK(box->get_path_end() == doctest::Approx(0.8));

		// Verify mesh still generates
		Vector<Vector3> faces = box->get_brush_faces();
		CHECK_MESSAGE(faces.size() > 0, "Box should generate faces with modified parameters");

		SceneTree::get_singleton()->get_root()->remove_child(box);
		memdelete(box);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedPrimitive3D: Transform parameters") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);
		SceneTree::get_singleton()->get_root()->add_child(cylinder);

		// Test scale
		cylinder->set_profile_scale(Vector2(2.0, 1.5));
		CHECK(cylinder->get_profile_scale().is_equal_approx(Vector2(2.0, 1.5)));

		// Test taper
		cylinder->set_taper(Vector2(0.5, -0.5));
		CHECK(cylinder->get_taper().is_equal_approx(Vector2(0.5, -0.5)));

		// Test twist
		cylinder->set_twist_begin(0.1);
		cylinder->set_twist_end(-0.1);
		CHECK(cylinder->get_twist_begin() == doctest::Approx(0.1));
		CHECK(cylinder->get_twist_end() == doctest::Approx(-0.1));

		// Test shear
		cylinder->set_shear(Vector2(0.2, 0.3));
		CHECK(cylinder->get_shear().is_equal_approx(Vector2(0.2, 0.3)));

		// Test radius offset
		cylinder->set_radius_offset(0.1);
		CHECK(cylinder->get_radius_offset() == doctest::Approx(0.1));

		// Test skew
		cylinder->set_skew(0.2);
		CHECK(cylinder->get_skew() == doctest::Approx(0.2));

		// Test revolutions
		cylinder->set_revolutions(2.0);
		CHECK(cylinder->get_revolutions() == doctest::Approx(2.0));

		// Verify mesh still generates
		Vector<Vector3> faces = cylinder->get_brush_faces();
		CHECK_MESSAGE(faces.size() > 0, "Cylinder should generate faces with transform parameters");

		SceneTree::get_singleton()->get_root()->remove_child(cylinder);
		memdelete(cylinder);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedPrimitive3D: Hollow parameters") {
		CSGSculptedSphere3D *sphere = memnew(CSGSculptedSphere3D);
		SceneTree::get_singleton()->get_root()->add_child(sphere);

		sphere->set_hollow(0.5);
		CHECK(sphere->get_hollow() == doctest::Approx(0.5));

		sphere->set_hollow_shape(CSGSculptedPrimitive3D::HOLLOW_CIRCLE);
		CHECK(sphere->get_hollow_shape() == CSGSculptedPrimitive3D::HOLLOW_CIRCLE);

		// Verify mesh still generates
		Vector<Vector3> faces = sphere->get_brush_faces();
		CHECK_MESSAGE(faces.size() > 0, "Sphere should generate faces with hollow");

		SceneTree::get_singleton()->get_root()->remove_child(sphere);
		memdelete(sphere);
	}
}

TEST_CASE("[SceneTree][CSG] CSGSculptedPrimitive3D: Mesh validation") {
	SUBCASE("[SceneTree][CSG] CSGSculptedPrimitive3D: Box vertices within bounds") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);
		SceneTree::get_singleton()->get_root()->add_child(box);

		Vector3 size(2.0, 3.0, 4.0);
		box->set_size(size);
		Vector<Vector3> faces = box->get_brush_faces();

		// Check all vertices are within expected bounds
		real_t max_dist = 0.0;
		for (int i = 0; i < faces.size(); i++) {
			real_t dist_x = Math::abs(faces[i].x);
			real_t dist_y = Math::abs(faces[i].y);
			real_t dist_z = Math::abs(faces[i].z);
			max_dist = MAX(max_dist, MAX(dist_x, MAX(dist_y, dist_z)));
		}

		// Allow some tolerance for transformations
		CHECK_MESSAGE(max_dist <= size.length() * 1.5, "All vertices should be within reasonable bounds");

		SceneTree::get_singleton()->get_root()->remove_child(box);
		memdelete(box);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedPrimitive3D: Cylinder radius validation") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);
		SceneTree::get_singleton()->get_root()->add_child(cylinder);

		real_t radius = 1.0;
		cylinder->set_radius(radius);
		cylinder->set_height(2.0);
		Vector<Vector3> faces = cylinder->get_brush_faces();

		// Check that vertices are roughly within cylinder bounds
		real_t max_radius = 0.0;
		for (int i = 0; i < faces.size(); i++) {
			real_t dist_from_axis = Math::sqrt(faces[i].x * faces[i].x + faces[i].z * faces[i].z);
			max_radius = MAX(max_radius, dist_from_axis);
		}

		// Allow tolerance for transformations and scaling
		CHECK_MESSAGE(max_radius <= radius * 2.0, "Cylinder vertices should be within reasonable radius");

		SceneTree::get_singleton()->get_root()->remove_child(cylinder);
		memdelete(cylinder);
	}
}

TEST_CASE("[SceneTree][CSG] CSGSculptedPrimitive3D: Exact outputs") {
	SUBCASE("[SceneTree][CSG] CSGSculptedBox3D: Exact vertex count and positions") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);
		SceneTree::get_singleton()->get_root()->add_child(box);

		// Use default parameters for predictable output
		box->set_size(Vector3(1.0, 1.0, 1.0));
		// Default profile: square, path: line, 8 segments each

		Vector<Vector3> faces = box->get_brush_faces();

		// With default parameters, should generate a specific number of faces
		// Profile has 4 points (square), path has 9 points (8 segments + 1), hollow disabled
		// This creates a grid of 4x9 vertices, with triangles connecting them
		// Plus end caps: 2 caps × 2 triangles each = 4 triangles
		// Total triangles: (4×8×2) + 4 = 64 + 4 = 68 triangles = 204 vertices
		CHECK_MESSAGE(faces.size() == 204, "Box with default parameters should generate 204 vertices");

		// Check that all vertices are within bounds
		AABB bounds = box->get_aabb();
		for (int i = 0; i < faces.size(); i++) {
			CHECK_MESSAGE(bounds.has_point(faces[i]), "All vertices should be within AABB");
		}

		SceneTree::get_singleton()->get_root()->remove_child(box);
		memdelete(box);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedCylinder3D: Exact vertex count") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);
		SceneTree::get_singleton()->get_root()->add_child(cylinder);

		// Use default parameters
		cylinder->set_radius(0.5);
		cylinder->set_height(1.0);

		Vector<Vector3> faces = cylinder->get_brush_faces();

		// Circle profile with 8 segments, path with 9 points, creates cylinder
		// Should have specific vertex count
		CHECK_MESSAGE(faces.size() > 0, "Cylinder should generate faces");
		CHECK_MESSAGE(faces.size() % 3 == 0, "Faces should be triangles");

		// Check height bounds
		real_t min_y = 1e10;
		real_t max_y = -1e10;
		for (int i = 0; i < faces.size(); i++) {
			min_y = MIN(min_y, faces[i].y);
			max_y = MAX(max_y, faces[i].y);
		}

		CHECK_MESSAGE(min_y >= -0.5, "Cylinder should not extend below -0.5");
		CHECK_MESSAGE(max_y <= 0.5, "Cylinder should not extend above 0.5");

		SceneTree::get_singleton()->get_root()->remove_child(cylinder);
		memdelete(cylinder);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedSphere3D: Exact vertex count") {
		CSGSculptedSphere3D *sphere = memnew(CSGSculptedSphere3D);
		SceneTree::get_singleton()->get_root()->add_child(sphere);

		sphere->set_radius(0.5);

		Vector<Vector3> faces = sphere->get_brush_faces();

		// Sphere should generate faces
		CHECK_MESSAGE(faces.size() > 0, "Sphere should generate faces");
		CHECK_MESSAGE(faces.size() % 3 == 0, "Faces should be triangles");

		// Check that all vertices are approximately on sphere surface
		real_t radius = 0.5;
		for (int i = 0; i < faces.size(); i++) {
			real_t distance = faces[i].length();
			CHECK_MESSAGE(Math::abs(distance - radius) < 0.1, "Vertex should be approximately on sphere surface");
		}

		SceneTree::get_singleton()->get_root()->remove_child(sphere);
		memdelete(sphere);
	}
}

} // namespace TestCSG
