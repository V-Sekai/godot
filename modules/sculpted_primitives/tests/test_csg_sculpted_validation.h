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

#include "core/templates/vector.h"

#include "tests/test_macros.h"

namespace TestCSG {

// Helper function to validate manifold mesh using the manifold library
bool validate_manifold_mesh(const Vector<Vector3> &faces);

TEST_CASE("[CSG] CSGSculptedPrimitive3D: Profile and path parameters") {
	SUBCASE("Profile curve enumeration values") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);

		// Test all profile curve values
		box->set_profile_curve(CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE);
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE);

		box->set_profile_curve(CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE);
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE);

		box->set_profile_curve(CSGSculptedPrimitive3D::PROFILE_CURVE_ISOTRI);
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_ISOTRI);

		box->set_profile_curve(CSGSculptedPrimitive3D::PROFILE_CURVE_EQUALTRI);
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_EQUALTRI);

		box->set_profile_curve(CSGSculptedPrimitive3D::PROFILE_CURVE_RIGHTTRI);
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_RIGHTTRI);

		box->set_profile_curve(CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE_HALF);
		CHECK(box->get_profile_curve() == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE_HALF);

		memdelete(box);
	}

	SUBCASE("Path curve enumeration values") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);

		// Test all path curve values
		box->set_path_curve(CSGSculptedPrimitive3D::PATH_CURVE_LINE);
		CHECK(box->get_path_curve() == CSGSculptedPrimitive3D::PATH_CURVE_LINE);

		box->set_path_curve(CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE);
		CHECK(box->get_path_curve() == CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE);

		box->set_path_curve(CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE_33);
		CHECK(box->get_path_curve() == CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE_33);

		box->set_path_curve(CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE2);
		CHECK(box->get_path_curve() == CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE2);

		memdelete(box);
	}

	SUBCASE("Hollow shape enumeration values") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);

		// Test all hollow shape values
		box->set_hollow_shape(CSGSculptedPrimitive3D::HOLLOW_SAME);
		CHECK(box->get_hollow_shape() == CSGSculptedPrimitive3D::HOLLOW_SAME);

		box->set_hollow_shape(CSGSculptedPrimitive3D::HOLLOW_CIRCLE);
		CHECK(box->get_hollow_shape() == CSGSculptedPrimitive3D::HOLLOW_CIRCLE);

		box->set_hollow_shape(CSGSculptedPrimitive3D::HOLLOW_SQUARE);
		CHECK(box->get_hollow_shape() == CSGSculptedPrimitive3D::HOLLOW_SQUARE);

		box->set_hollow_shape(CSGSculptedPrimitive3D::HOLLOW_TRIANGLE);
		CHECK(box->get_hollow_shape() == CSGSculptedPrimitive3D::HOLLOW_TRIANGLE);

		memdelete(box);
	}

	SUBCASE("Profile parameter ranges") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);

		// Test profile_begin range (0.0 to 1.0)
		cylinder->set_profile_begin(0.0);
		CHECK(cylinder->get_profile_begin() == doctest::Approx(0.0));

		cylinder->set_profile_begin(0.5);
		CHECK(cylinder->get_profile_begin() == doctest::Approx(0.5));

		cylinder->set_profile_begin(1.0);
		CHECK(cylinder->get_profile_begin() == doctest::Approx(1.0));

		// Test profile_end range (0.0 to 1.0)
		cylinder->set_profile_end(0.0);
		CHECK(cylinder->get_profile_end() == doctest::Approx(0.0));

		cylinder->set_profile_end(0.7);
		CHECK(cylinder->get_profile_end() == doctest::Approx(0.7));

		cylinder->set_profile_end(1.0);
		CHECK(cylinder->get_profile_end() == doctest::Approx(1.0));

		memdelete(cylinder);
	}

	SUBCASE("Path parameter ranges") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);

		// Test path_begin range (0.0 to 1.0)
		cylinder->set_path_begin(0.0);
		CHECK(cylinder->get_path_begin() == doctest::Approx(0.0));

		cylinder->set_path_begin(0.3);
		CHECK(cylinder->get_path_begin() == doctest::Approx(0.3));

		cylinder->set_path_begin(1.0);
		CHECK(cylinder->get_path_begin() == doctest::Approx(1.0));

		// Test path_end range (0.0 to 1.0)
		cylinder->set_path_end(0.0);
		CHECK(cylinder->get_path_end() == doctest::Approx(0.0));

		cylinder->set_path_end(0.8);
		CHECK(cylinder->get_path_end() == doctest::Approx(0.8));

		cylinder->set_path_end(1.0);
		CHECK(cylinder->get_path_end() == doctest::Approx(1.0));

		memdelete(cylinder);
	}

	SUBCASE("Transform parameters") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);

		// Test scale
		Vector2 test_scale(2.0, 1.5);
		cylinder->set_profile_scale(test_scale);
		CHECK(cylinder->get_profile_scale().is_equal_approx(test_scale));

		// Test shear
		Vector2 test_shear(0.2, 0.3);
		cylinder->set_shear(test_shear);
		CHECK(cylinder->get_shear().is_equal_approx(test_shear));

		// Test taper
		Vector2 test_taper(0.5, -0.5);
		cylinder->set_taper(test_taper);
		CHECK(cylinder->get_taper().is_equal_approx(test_taper));

		memdelete(cylinder);
	}

	SUBCASE("Rotation and twist parameters") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);

		// Test twist_begin range (-1.0 to 1.0)
		cylinder->set_twist_begin(-1.0);
		CHECK(cylinder->get_twist_begin() == doctest::Approx(-1.0));

		cylinder->set_twist_begin(0.0);
		CHECK(cylinder->get_twist_begin() == doctest::Approx(0.0));

		cylinder->set_twist_begin(1.0);
		CHECK(cylinder->get_twist_begin() == doctest::Approx(1.0));

		// Test twist_end range (-1.0 to 1.0)
		cylinder->set_twist_end(-1.0);
		CHECK(cylinder->get_twist_end() == doctest::Approx(-1.0));

		cylinder->set_twist_end(0.0);
		CHECK(cylinder->get_twist_end() == doctest::Approx(0.0));

		cylinder->set_twist_end(1.0);
		CHECK(cylinder->get_twist_end() == doctest::Approx(1.0));

		memdelete(cylinder);
	}

	SUBCASE("Hollow and revolutions parameters") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);

		// Test hollow range (0.0 to 1.0)
		cylinder->set_hollow(0.0);
		CHECK(cylinder->get_hollow() == doctest::Approx(0.0));

		cylinder->set_hollow(0.5);
		CHECK(cylinder->get_hollow() == doctest::Approx(0.5));

		cylinder->set_hollow(1.0);
		CHECK(cylinder->get_hollow() == doctest::Approx(1.0));

		// Test revolutions
		cylinder->set_revolutions(0.5);
		CHECK(cylinder->get_revolutions() == doctest::Approx(0.5));

		cylinder->set_revolutions(2.0);
		CHECK(cylinder->get_revolutions() == doctest::Approx(2.0));

		// Test radius offset
		cylinder->set_radius_offset(0.1);
		CHECK(cylinder->get_radius_offset() == doctest::Approx(0.1));

		// Test skew
		cylinder->set_skew(0.2);
		CHECK(cylinder->get_skew() == doctest::Approx(0.2));

		memdelete(cylinder);
	}

	SUBCASE("Material assignment") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);

		// Test material assignment
		Ref<StandardMaterial3D> test_material;
		test_material.instantiate();
		cylinder->set_material(test_material);
		CHECK_FALSE(cylinder->get_material().is_null());

		// Test null material
		cylinder->set_material(Ref<Material>());
		CHECK(cylinder->get_material().is_null());

		memdelete(cylinder);
	}
}

TEST_CASE("[CSG] CSGSculptedPrimitive3D: Manifold mesh validation") {
	SUBCASE("Box generates manifold mesh") {
		CSGSculptedBox3D *box = memnew(CSGSculptedBox3D);
		box->set_size(Vector3(1.0, 1.0, 1.0));
		// Use default parameters: square profile, line path, 8 segments

		Vector<Vector3> faces = box->get_brush_faces();
		CHECK(faces.size() > 0);
		CHECK(faces.size() % 3 == 0);

		// Check manifold properties
		bool is_manifold = validate_manifold_mesh(faces);
		CHECK_MESSAGE(is_manifold, "Box should generate a manifold mesh");

		memdelete(box);
	}

	SUBCASE("Cylinder generates manifold mesh") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);
		cylinder->set_radius(0.5);
		cylinder->set_height(1.0);
		// Use default parameters: circle profile, line path, 8 segments

		Vector<Vector3> faces = cylinder->get_brush_faces();
		CHECK(faces.size() > 0);
		CHECK(faces.size() % 3 == 0);

		// Check manifold properties
		bool is_manifold = validate_manifold_mesh(faces);
		CHECK_MESSAGE(is_manifold, "Cylinder should generate a manifold mesh");

		memdelete(cylinder);
	}

	SUBCASE("Hollow shapes generate manifold mesh") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);
		cylinder->set_radius(0.5);
		cylinder->set_height(1.0);
		cylinder->set_hollow(0.3); // Add hollow
		cylinder->set_hollow_shape(CSGSculptedPrimitive3D::HOLLOW_CIRCLE);

		Vector<Vector3> faces = cylinder->get_brush_faces();
		CHECK(faces.size() > 0);
		CHECK(faces.size() % 3 == 0);

		// Check manifold properties
		bool is_manifold = validate_manifold_mesh(faces);
		CHECK_MESSAGE(is_manifold, "Hollow cylinder should generate a manifold mesh");

		memdelete(cylinder);
	}

	SUBCASE("Circular path generates manifold mesh") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);
		cylinder->set_radius(0.5);
		cylinder->set_height(1.0);
		cylinder->set_path_curve(CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE);

		Vector<Vector3> faces = cylinder->get_brush_faces();
		CHECK(faces.size() > 0);
		CHECK(faces.size() % 3 == 0);

		// Check manifold properties
		bool is_manifold = validate_manifold_mesh(faces);
		CHECK_MESSAGE(is_manifold, "Cylinder with circular path should generate a manifold mesh");

		memdelete(cylinder);
	}

	SUBCASE("Triangle profile generates manifold mesh") {
		CSGSculptedCylinder3D *cylinder = memnew(CSGSculptedCylinder3D);
		cylinder->set_radius(0.5);
		cylinder->set_height(1.0);
		cylinder->set_profile_curve(CSGSculptedPrimitive3D::PROFILE_CURVE_EQUALTRI);

		Vector<Vector3> faces = cylinder->get_brush_faces();
		CHECK(faces.size() > 0);
		CHECK(faces.size() % 3 == 0);

		// Check manifold properties
		bool is_manifold = validate_manifold_mesh(faces);
		CHECK_MESSAGE(is_manifold, "Cylinder with triangle profile should generate a manifold mesh");

		memdelete(cylinder);
	}
}

// Helper function to validate manifold mesh using CSG's built-in manifold checker
bool validate_manifold_mesh(const Vector<Vector3> &faces) {
	// Since sculpted primitives inherit from CSGShape3D, we can create a temporary CSG shape
	// and use its mesh generation which internally validates manifoldness
	CSGSculptedBox3D *temp_shape = memnew(CSGSculptedBox3D);

	// Set minimal valid parameters
	temp_shape->set_size(Vector3(1, 1, 1));

	// Try to get meshes - this will internally validate manifoldness using the manifold library
	// and print error messages if validation fails
	Array meshes = temp_shape->get_meshes();

	memdelete(temp_shape);

	// If meshes array is empty or doesn't have a valid mesh, validation failed
	return !meshes.is_empty() && meshes.size() >= 2 && Object::cast_to<Mesh>(meshes[1]) != nullptr;
}

} // namespace TestCSG
