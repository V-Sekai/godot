/**************************************************************************/
/*  test_csg.h                                                            */
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

#include "../csg.h"
#include "../csg_shape.h"
#include "../csg_sculpted_primitive.h"

#include "tests/test_macros.h"

namespace TestCSG {

TEST_CASE("[SceneTree][CSG] CSGPolygon3D") {
	SUBCASE("[SceneTree][CSG] CSGPolygon3D: using accurate path tangent for polygon rotation") {
		const float polygon_radius = 10.0f;

		const Vector3 expected_min_bounds = Vector3(-polygon_radius, -polygon_radius, 0);
		const Vector3 expected_max_bounds = Vector3(100 + polygon_radius, polygon_radius, 100);
		const AABB expected_aabb = AABB(expected_min_bounds, expected_max_bounds - expected_min_bounds);

		Ref<Curve3D> curve;
		curve.instantiate();
		curve->add_point(
				// p_position
				Vector3(0, 0, 0),
				// p_in
				Vector3(),
				// p_out
				Vector3(0, 0, 60));
		curve->add_point(
				// p_position
				Vector3(100, 0, 100),
				// p_in
				Vector3(0, 0, -60),
				// p_out
				Vector3());

		Path3D *path = memnew(Path3D);
		path->set_curve(curve);

		CSGPolygon3D *csg_polygon_3d = memnew(CSGPolygon3D);
		SceneTree::get_singleton()->get_root()->add_child(csg_polygon_3d);

		csg_polygon_3d->add_child(path);
		csg_polygon_3d->set_path_node(csg_polygon_3d->get_path_to(path));
		csg_polygon_3d->set_mode(CSGPolygon3D::Mode::MODE_PATH);

		PackedVector2Array polygon;
		polygon.append(Vector2(-polygon_radius, 0));
		polygon.append(Vector2(0, polygon_radius));
		polygon.append(Vector2(polygon_radius, 0));
		polygon.append(Vector2(0, -polygon_radius));
		csg_polygon_3d->set_polygon(polygon);

		csg_polygon_3d->set_path_rotation(CSGPolygon3D::PathRotation::PATH_ROTATION_PATH);
		csg_polygon_3d->set_path_rotation_accurate(true);

		// Minimize the number of extrusions.
		// This decreases the number of samples taken from the curve.
		// Having fewer samples increases the inaccuracy of the line between samples as an approximation of the tangent of the curve.
		// With correct polygon orientation, the bounding box for the given curve should be independent of the number of extrusions.
		csg_polygon_3d->set_path_interval_type(CSGPolygon3D::PathIntervalType::PATH_INTERVAL_DISTANCE);
		csg_polygon_3d->set_path_interval(1000.0f);

		// Call get_brush_faces to force the bounding box to update.
		csg_polygon_3d->get_brush_faces();

		CHECK(csg_polygon_3d->get_aabb().is_equal_approx(expected_aabb));

		// Perform the bounding box check again with a greater number of extrusions.
		csg_polygon_3d->set_path_interval(1.0f);
		csg_polygon_3d->get_brush_faces();

		CHECK(csg_polygon_3d->get_aabb().is_equal_approx(expected_aabb));

		csg_polygon_3d->remove_child(path);
		SceneTree::get_singleton()->get_root()->remove_child(csg_polygon_3d);

		memdelete(csg_polygon_3d);
		memdelete(path);
	}

TEST_CASE("[SceneTree][CSG] CSGSculptedBox3D") {
		SUBCASE("[SceneTree][CSG] CSGSculptedBox3D: Basic shape generation") {
			CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);
			SceneTree::get_singleton()->get_root()->add_child(box);

			box->set_size(Vector3(2.0, 2.0, 2.0));
			Vector<Vector3> faces = box->get_brush_faces();

			CHECK_MESSAGE(faces.size() > 0, "Box should generate faces");
			CHECK_MESSAGE(faces.size() % 3 == 0, "Faces should be triangles (multiple of 3)");

			// Check bounding box
			AABB aabb = box->get_aabb();
			CHECK_MESSAGE(aabb.size.is_equal_approx(Vector3(2.0, 2.0, 2.0)), "AABB size should match box size");

			SceneTree::get_singleton()->get_root()->remove_child(box);
			memdelete(box);
		}

		SUBCASE("[SceneTree][CSG] CSGSculptedBox3D: Property getters and setters") {
			CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);
			box->set_size(Vector3(3.0, 4.0, 5.0));
			CHECK(box->get_size().is_equal_approx(Vector3(3.0, 4.0, 5.0)));

			memdelete(box);
		}
	}

	TEST_CASE("[SceneTree][CSG] CSGSculptedCylinder3D") {
		SUBCASE("[SceneTree][CSG] CSGSculptedCylinder3D: Basic shape generation") {
			CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);
			SceneTree::get_singleton()->get_root()->add_child(cylinder);

			cylinder->set_radius(1.0);
			cylinder->set_height(2.0);
			Vector<Vector3> faces = cylinder->get_brush_faces();

			CHECK_MESSAGE(faces.size() > 0, "Cylinder should generate faces");
			CHECK_MESSAGE(faces.size() % 3 == 0, "Faces should be triangles (multiple of 3)");

			// Check that vertices are within expected bounds
			AABB aabb = cylinder->get_aabb();
			CHECK_MESSAGE(aabb.size.y >= 2.0, "Cylinder height should be at least 2.0");
			CHECK_MESSAGE(aabb.size.x >= 2.0 && aabb.size.z >= 2.0, "Cylinder diameter should be at least 2.0");

			SceneTree::get_singleton()->get_root()->remove_child(cylinder);
			memdelete(cylinder);
		}

		SUBCASE("[SceneTree][CSG] CSGSculptedCylinder3D: Property getters and setters") {
			CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);
			cylinder->set_radius(1.5);
			cylinder->set_height(3.0);
			CHECK(cylinder->get_radius() == doctest::Approx(1.5));
			CHECK(cylinder->get_height() == doctest::Approx(3.0));

			memdelete(cylinder);
		}
	}

	TEST_CASE("[SceneTree][CSG] CSGSculptedSphere3D") {
		SUBCASE("[SceneTree][CSG] CSGSculptedSphere3D: Basic shape generation") {
			CSGSculptedSphere3D *sphere = memnew(CSGSculptedSphere3D);
			SceneTree::get_singleton()->get_root()->add_child(sphere);

			sphere->set_radius(1.0);
			Vector<Vector3> faces = sphere->get_brush_faces();

			CHECK_MESSAGE(faces.size() > 0, "Sphere should generate faces");
			CHECK_MESSAGE(faces.size() % 3 == 0, "Faces should be triangles (multiple of 3)");

			// Check bounding box
			AABB aabb = sphere->get_aabb();
			CHECK_MESSAGE(aabb.size.x >= 2.0 && aabb.size.y >= 2.0 && aabb.size.z >= 2.0, "Sphere diameter should be at least 2.0");

			SceneTree::get_singleton()->get_root()->remove_child(sphere);
			memdelete(sphere);
		}

		SUBCASE("[SceneTree][CSG] CSGSculptedSphere3D: Property getters and setters") {
			CSGSculptedSphere3D *sphere = memnew(CSGSculptedSphere3D);
			sphere->set_radius(2.5);
			CHECK(sphere->get_radius() == doctest::Approx(2.5));

			memdelete(sphere);
		}
	}

	TEST_CASE("[SceneTree][CSG] CSGSculptedTorus3D") {
		SUBCASE("[SceneTree][CSG] CSGSculptedTorus3D: Basic shape generation") {
			CSGSculptedTorus3D *torus = memnew(CSGSculptedTorus3D);
			SceneTree::get_singleton()->get_root()->add_child(torus);

			torus->set_inner_radius(0.25);
			torus->set_outer_radius(0.5);
			Vector<Vector3> faces = torus->get_brush_faces();

			CHECK_MESSAGE(faces.size() > 0, "Torus should generate faces");
			CHECK_MESSAGE(faces.size() % 3 == 0, "Faces should be triangles (multiple of 3)");

			SceneTree::get_singleton()->get_root()->remove_child(torus);
			memdelete(torus);
		}

		SUBCASE("[SceneTree][CSG] CSGSculptedTorus3D: Property getters and setters") {
			CSGSculptedTorus3D *torus = memnew(CSGSculptedTorus3D);
			torus->set_inner_radius(0.3);
			torus->set_outer_radius(0.6);
			CHECK(torus->get_inner_radius() == doctest::Approx(0.3));
			CHECK(torus->get_outer_radius() == doctest::Approx(0.6));

			memdelete(torus);
		}
	}

	TEST_CASE("[SceneTree][CSG] CSGSculptedPrism3D") {
		SUBCASE("[SceneTree][CSG] CSGSculptedPrism3D: Basic shape generation") {
			CSGSculptedPrism3D *prism = memnew(CSGSculptedPrism3D);
			SceneTree::get_singleton()->get_root()->add_child(prism);

			prism->set_size(Vector3(2.0, 2.0, 2.0));
			Vector<Vector3> faces = prism->get_brush_faces();

			CHECK_MESSAGE(faces.size() > 0, "Prism should generate faces");
			CHECK_MESSAGE(faces.size() % 3 == 0, "Faces should be triangles (multiple of 3)");

			SceneTree::get_singleton()->get_root()->remove_child(prism);
			memdelete(prism);
		}

		SUBCASE("[SceneTree][CSG] CSGSculptedPrism3D: Property getters and setters") {
			CSGSculptedPrism3D *prism = memnew(CSGSculptedPrism3D);
			prism->set_size(Vector3(3.0, 4.0, 5.0));
			CHECK(prism->get_size().is_equal_approx(Vector3(3.0, 4.0, 5.0)));

			memdelete(prism);
		}
	}

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
				max_dist = Math::max(max_dist, Math::max(dist_x, Math::max(dist_y, dist_z)));
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
				max_radius = Math::max(max_radius, dist_from_axis);
			}

			// Allow tolerance for transformations and scaling
			CHECK_MESSAGE(max_radius <= radius * 2.0, "Cylinder vertices should be within reasonable radius");

			SceneTree::get_singleton()->get_root()->remove_child(cylinder);
			memdelete(cylinder);
		}
	}

} // namespace TestCSG
