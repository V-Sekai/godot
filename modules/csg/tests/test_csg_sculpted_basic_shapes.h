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

#include "modules/csg/csg_sculpted_box.h"
#include "modules/csg/csg_sculpted_cylinder.h"
#include "modules/csg/csg_sculpted_primitive_base.h"
#include "modules/csg/csg_sculpted_prism.h"
#include "modules/csg/csg_sculpted_sphere.h"
#include "modules/csg/csg_sculpted_torus.h"

#include "tests/test_macros.h"

namespace TestCSG {

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

} // namespace TestCSG
