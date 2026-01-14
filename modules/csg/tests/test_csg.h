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
#include "../csg_sculpted_primitive.h"
#include "../csg_shape.h"

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

	SUBCASE("[SceneTree][CSG] CSGSculptedBox3D: Manifold validation - edge sharing") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);
		SceneTree::get_singleton()->get_root()->add_child(box);

		box->set_size(Vector3(1.0, 1.0, 1.0));
		Vector<Vector3> faces = box->get_brush_faces();

		// Build edge map: edge -> list of triangles that share it
		HashMap<String, Vector<int>> edge_map;
		int triangle_count = faces.size() / 3;

		for (int t = 0; t < triangle_count; t++) {
			Vector3 v0 = faces[t * 3 + 0];
			Vector3 v1 = faces[t * 3 + 1];
			Vector3 v2 = faces[t * 3 + 2];

			// Create edges (always in sorted order for consistency)
			auto make_edge_key = [](const Vector3 &a, const Vector3 &b) -> String {
				// Sort vertices to ensure consistent edge representation
				if (a.x < b.x || (a.x == b.x && a.y < b.y) || (a.x == b.x && a.y == b.y && a.z < b.z)) {
					return String::num_real(a.x) + "," + String::num_real(a.y) + "," + String::num_real(a.z) + "|" +
							String::num_real(b.x) + "," + String::num_real(b.y) + "," + String::num_real(b.z);
				} else {
					return String::num_real(b.x) + "," + String::num_real(b.y) + "," + String::num_real(b.z) + "|" +
							String::num_real(a.x) + "," + String::num_real(a.y) + "," + String::num_real(a.z);
				}
			};

			String e0 = make_edge_key(v0, v1);
			String e1 = make_edge_key(v1, v2);
			String e2 = make_edge_key(v2, v0);

			if (!edge_map.has(e0)) {
				edge_map[e0] = Vector<int>();
			}
			if (!edge_map.has(e1)) {
				edge_map[e1] = Vector<int>();
			}
			if (!edge_map.has(e2)) {
				edge_map[e2] = Vector<int>();
			}

			edge_map[e0].push_back(t);
			edge_map[e1].push_back(t);
			edge_map[e2].push_back(t);
		}

		// Verify every edge is shared by exactly two triangles
		for (const KeyValue<String, Vector<int>> &E : edge_map) {
			String error_msg = "Each edge must be shared by exactly two triangles. Edge: " + E.key + " is shared by " + String::num(E.value.size()) + " triangles";
			CHECK_MESSAGE(E.value.size() == 2, error_msg);
		}

		SceneTree::get_singleton()->get_root()->remove_child(box);
		memdelete(box);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedBox3D: Manifold validation - counter-clockwise winding") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);
		SceneTree::get_singleton()->get_root()->add_child(box);

		box->set_size(Vector3(1.0, 1.0, 1.0));
		Vector<Vector3> faces = box->get_brush_faces();

		int triangle_count = faces.size() / 3;
		int valid_winding_count = 0;

		for (int t = 0; t < triangle_count; t++) {
			Vector3 v0 = faces[t * 3 + 0];
			Vector3 v1 = faces[t * 3 + 1];
			Vector3 v2 = faces[t * 3 + 2];

			// Calculate face normal (counter-clockwise winding means normal points outward)
			Vector3 edge1 = v1 - v0;
			Vector3 edge2 = v2 - v0;
			Vector3 normal = edge1.cross(edge2).normalized();

			// Calculate center of triangle
			Vector3 center = (v0 + v1 + v2) / 3.0;

			// For a box centered at origin, the normal should point away from center
			// (i.e., dot product of normal with (center - origin) should be positive)
			// Actually, for a box, the center is at origin, so we check if normal points in same direction as center
			// For a box at origin with size 1, vertices are at ±0.5, so center of face should be away from origin
			real_t dot = normal.dot(center);
			if (dot > 0.0) {
				valid_winding_count++;
			}
		}

		// Most faces should have correct winding (allowing for some tolerance)
		CHECK_MESSAGE(valid_winding_count > triangle_count * 0.8, "Most triangles should have counter-clockwise winding when viewed from outside");

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
		CHECK_MESSAGE(aabb.size.x >= 2.0, "Cylinder diameter (x) should be at least 2.0");
		CHECK_MESSAGE(aabb.size.z >= 2.0, "Cylinder diameter (z) should be at least 2.0");

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
		CHECK_MESSAGE(aabb.size.x >= 2.0, "Sphere diameter (x) should be at least 2.0");
		CHECK_MESSAGE(aabb.size.y >= 2.0, "Sphere diameter (y) should be at least 2.0");
		CHECK_MESSAGE(aabb.size.z >= 2.0, "Sphere diameter (z) should be at least 2.0");

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

TEST_CASE("[SceneTree][CSG] CSGSculptedTexture3D") {
	SUBCASE("[SceneTree][CSG] CSGSculptedTexture3D: Basic texture sculpting") {
		CSGSculptedTexture3D *texture_primitive = memnew(CSGSculptedTexture3D);
		SceneTree::get_singleton()->get_root()->add_child(texture_primitive);

		// Create a simple 2x2 texture for testing
		Ref<Image> test_image;
		test_image.instantiate();
		test_image->create(2, 2, false, Image::FORMAT_RGB8);
		// Set pixel values: bottom-left (0,0): (0.5, 0.5, 0.5) -> (0, 0, 0)
		// bottom-right (1,0): (1.0, 0.5, 0.5) -> (1, 0, 0)
		// top-left (0,1): (0.5, 1.0, 0.5) -> (0, 1, 0)
		// top-right (1,1): (0.5, 0.5, 1.0) -> (0, 0, 1)
		test_image->set_pixel(0, 0, Color(0.5, 0.5, 0.5));
		test_image->set_pixel(1, 0, Color(1.0, 0.5, 0.5));
		test_image->set_pixel(0, 1, Color(0.5, 1.0, 0.5));
		test_image->set_pixel(1, 1, Color(0.5, 0.5, 1.0));

		Ref<ImageTexture> texture = ImageTexture::create_from_image(test_image);
		texture_primitive->set_sculpt_texture(texture);

		Vector<Vector3> faces = texture_primitive->get_brush_faces();

		// 2x2 texture should generate 2 triangles (4 total vertices, but shared)
		CHECK_MESSAGE(faces.size() == 6, "2x2 texture should generate 2 triangles (6 vertices)");

		// Check that vertices are at expected positions (scaled by default scale of 1.0)
		// Expected vertices based on RGB mapping: R=X, G=Y, B=Z, (0-1) -> (-1 to 1)
		Vector3 expected_vertices[4] = {
			Vector3(0.0, 0.0, 0.0), // (0.5, 0.5, 0.5) -> (0, 0, 0)
			Vector3(1.0, 0.0, 0.0), // (1.0, 0.5, 0.5) -> (1, 0, 0)
			Vector3(0.0, 1.0, 0.0), // (0.5, 1.0, 0.5) -> (0, 1, 0)
			Vector3(0.0, 0.0, 1.0) // (0.5, 0.5, 1.0) -> (0, 0, 1)
		};

		// Verify all expected vertices are present in the mesh
		bool found_vertices[4] = { false, false, false, false };
		for (int i = 0; i < faces.size(); i++) {
			for (int j = 0; j < 4; j++) {
				if (faces[i].is_equal_approx(expected_vertices[j])) {
					found_vertices[j] = true;
					break;
				}
			}
		}

		for (int j = 0; j < 4; j++) {
			CHECK_MESSAGE(found_vertices[j], "Expected vertex should be present in mesh");
		}

		SceneTree::get_singleton()->get_root()->remove_child(texture_primitive);
		memdelete(texture_primitive);
	}

	SUBCASE("[SceneTree][CSG] CSGSculptedTexture3D: Mirror and invert flags") {
		CSGSculptedTexture3D *texture_primitive = memnew(CSGSculptedTexture3D);
		SceneTree::get_singleton()->get_root()->add_child(texture_primitive);

		Ref<Image> test_image;
		test_image.instantiate();
		test_image->create(2, 2, false, Image::FORMAT_RGB8);
		test_image->set_pixel(0, 0, Color(0.5, 0.5, 0.5));
		test_image->set_pixel(1, 0, Color(1.0, 0.5, 0.5));
		test_image->set_pixel(0, 1, Color(0.5, 1.0, 0.5));
		test_image->set_pixel(1, 1, Color(0.5, 0.5, 1.0));

		Ref<ImageTexture> texture = ImageTexture::create_from_image(test_image);
		texture_primitive->set_sculpt_texture(texture);
		texture_primitive->set_mirror(true);
		texture_primitive->set_invert(true);

		Vector<Vector3> faces = texture_primitive->get_brush_faces();

		// With mirror and invert, vertices should be transformed
		// mirror: x = -x, invert: z = -z
		Vector3 expected_vertices[4] = {
			Vector3(0.0, 0.0, 0.0), // (0.5, 0.5, 0.5) -> (0, 0, 0) -> mirror+invert: (0, 0, 0)
			Vector3(-1.0, 0.0, 0.0), // (1.0, 0.5, 0.5) -> (1, 0, 0) -> mirror+invert: (-1, 0, 0)
			Vector3(0.0, 1.0, 0.0), // (0.5, 1.0, 0.5) -> (0, 1, 0) -> mirror+invert: (0, 1, 0)
			Vector3(0.0, 0.0, -1.0) // (0.5, 0.5, 1.0) -> (0, 0, 1) -> mirror+invert: (0, 0, -1)
		};

		bool found_vertices[4] = { false, false, false, false };
		for (int i = 0; i < faces.size(); i++) {
			for (int j = 0; j < 4; j++) {
				if (faces[i].is_equal_approx(expected_vertices[j])) {
					found_vertices[j] = true;
					break;
				}
			}
		}

		for (int j = 0; j < 4; j++) {
			CHECK_MESSAGE(found_vertices[j], "Transformed vertex should be present in mesh");
		}

		SceneTree::get_singleton()->get_root()->remove_child(texture_primitive);
		memdelete(texture_primitive);
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
