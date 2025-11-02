/**************************************************************************/
/*  test_ik_kusudama_3d.h                                                 */
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

#include "tests/test_macros.h"

#include "core/math/ik_kusudama_3d.h"

namespace TestIKKusudama3D {

TEST_CASE("[IKKusudama3D][IKLimitCone3D] Test POD struct properties") {
	IKLimitCone3D cone;
	cone.set_control_point(Vector3(0, 1, 0));
	cone.set_radius(Math::PI / 4.0);

	CHECK_MESSAGE(cone.get_control_point() == Vector3(0, 1, 0), "Control point should be set correctly");
	CHECK_MESSAGE(Math::is_equal_approx(cone.get_radius(), Math::PI / 4.0), "Radius should be set correctly");
}

TEST_CASE("[IKKusudama3D][IKLimitCone3D] Test cone constraint solving") {
	IKLimitCone3D cone;
	cone.set_control_point(Vector3(0, 1, 0));
	cone.set_radius(Math::PI / 6.0); // 30 degrees

	// Test point within cone
	Vector3 test_point = Vector3(0, 0.866, 0.5).normalized(); // ~30 degrees from Y axis
	Vector<double> in_bounds;
	in_bounds.resize(1);
	in_bounds.write[0] = -1.0;

	Vector3 result = cone.closest_to_cone(test_point, &in_bounds);
	CHECK_MESSAGE(in_bounds[0] >= 0.0, "Point should be within cone bounds");

	// Test point outside cone
	Vector3 outside_point = Vector3(0, 0.5, 0.866).normalized(); // ~60 degrees from Y axis
	in_bounds.write[0] = -1.0;
	result = cone.closest_to_cone(outside_point, &in_bounds);
	CHECK_MESSAGE(in_bounds[0] < 0.0, "Point should be outside cone bounds");
}

TEST_CASE("[IKKusudama3D][IKLimitCone3D] Test tangent circle updates") {
	IKLimitCone3D cone1;
	cone1.set_control_point(Vector3(0, 1, 0));
	cone1.set_radius(Math::PI / 6.0);

	IKLimitCone3D cone2;
	cone2.set_control_point(Vector3(0.707, 0.707, 0).normalized());
	cone2.set_radius(Math::PI / 8.0);

	cone1.update_tangent_handles(&cone2);

	// Check that tangent circles were computed
	CHECK_MESSAGE(cone1.tangent_circle_center_next_1.length() > 0, "Tangent circle center 1 should be computed");
	CHECK_MESSAGE(cone1.tangent_circle_center_next_2.length() > 0, "Tangent circle center 2 should be computed");
	CHECK_MESSAGE(cone1.tangent_circle_radius_next > 0, "Tangent circle radius should be computed");
}

TEST_CASE("[IKKusudama3D][IKLimitCone3D] Test multi-cone path finding") {
	IKLimitCone3D cone1;
	cone1.set_control_point(Vector3(0, 1, 0));
	cone1.set_radius(Math::PI / 6.0);

	IKLimitCone3D cone2;
	cone2.set_control_point(Vector3(0.707, 0.707, 0).normalized());
	cone2.set_radius(Math::PI / 8.0);

	cone1.update_tangent_handles(&cone2);

	// Test path finding between cones
	Vector3 test_point = Vector3(0.5, 0.5, 0.707).normalized();
	Vector3 path_point = cone1.get_closest_path_point(&cone2, test_point);

	CHECK_MESSAGE(path_point.is_finite(), "Path point should be finite");
	CHECK_MESSAGE(!Math::is_nan(path_point.x), "Path point X should not be NaN");
	CHECK_MESSAGE(!Math::is_nan(path_point.y), "Path point Y should not be NaN");
	CHECK_MESSAGE(!Math::is_nan(path_point.z), "Path point Z should not be NaN");
}

TEST_CASE("[IKKusudama3D] Test Resource class creation") {
	Ref<IKKusudama3D> kusudama = memnew(IKKusudama3D);

	CHECK_MESSAGE(kusudama.is_valid(), "IKKusudama3D should be created successfully");
	CHECK_MESSAGE(!kusudama->is_enabled(), "New kusudama should not be enabled by default");
}

TEST_CASE("[IKKusudama3D] Test open cone management") {
	Ref<IKKusudama3D> kusudama = memnew(IKKusudama3D);

	IKLimitCone3D cone1;
	cone1.set_control_point(Vector3(0, 1, 0));
	cone1.set_radius(Math::PI / 4.0);

	IKLimitCone3D cone2;
	cone2.set_control_point(Vector3(0.707, 0.707, 0).normalized());
	cone2.set_radius(Math::PI / 6.0);

	kusudama->add_open_cone(cone1);
	kusudama->add_open_cone(cone2);

	TypedArray<IKLimitCone3D> cones = kusudama->get_open_cones();
	CHECK_MESSAGE(cones.size() == 2, "Should have 2 open cones");

	// Test removal
	kusudama->remove_open_cone(cone1);
	cones = kusudama->get_open_cones();
	CHECK_MESSAGE(cones.size() == 1, "Should have 1 open cone after removal");

	// Test clearing
	kusudama->clear_open_cones();
	cones = kusudama->get_open_cones();
	CHECK_MESSAGE(cones.size() == 0, "Should have no open cones after clearing");
}

TEST_CASE("[IKKusudama3D] Test axial limits") {
	Ref<IKKusudama3D> kusudama = memnew(IKKusudama3D);

	kusudama->set_axial_limits(Math::PI / 4.0, Math::PI / 2.0);

	CHECK_MESSAGE(kusudama->is_axially_constrained(), "Should be axially constrained after setting limits");
	CHECK_MESSAGE(Math::is_equal_approx(kusudama->get_min_axial_angle(), Math::PI / 4.0), "Min axial angle should be set correctly");
	CHECK_MESSAGE(Math::is_equal_approx(kusudama->get_range_angle(), Math::PI / 2.0), "Range angle should be set correctly");
}

TEST_CASE("[IKKusudama3D] Test constraint solving") {
	Ref<IKKusudama3D> kusudama = memnew(IKKusudama3D);

	// Add a single cone
	IKLimitCone3D cone;
	cone.set_control_point(Vector3(0, 1, 0));
	cone.set_radius(Math::PI / 6.0);
	kusudama->add_open_cone(cone);

	kusudama->enable_orientational_limits();

	// Test constraint solving
	Vector3 input_direction = Vector3(0, 0.5, 0.866).normalized(); // 60 degrees from Y axis
	Vector3 constrained = kusudama->_solve(input_direction);

	CHECK_MESSAGE(constrained.is_finite(), "Constrained direction should be finite");
	CHECK_MESSAGE(constrained.length() > 0.99, "Constrained direction should be normalized");
}

TEST_CASE("[IKKusudama3D] Test multi-cone constraint solving") {
	Ref<IKKusudama3D> kusudama = memnew(IKKusudama3D);

	// Add two cones
	IKLimitCone3D cone1;
	cone1.set_control_point(Vector3(0, 1, 0));
	cone1.set_radius(Math::PI / 6.0);

	IKLimitCone3D cone2;
	cone2.set_control_point(Vector3(0.707, 0.707, 0).normalized());
	cone2.set_radius(Math::PI / 8.0);

	kusudama->add_open_cone(cone1);
	kusudama->add_open_cone(cone2);
	kusudama->enable_orientational_limits();

	// Update tangent handles
	kusudama->update_tangent_radii();

	// Test constraint solving with multi-cone setup
	Vector3 input_direction = Vector3(0.5, 0.5, 0.707).normalized();
	Vector3 constrained = kusudama->_solve(input_direction);

	CHECK_MESSAGE(constrained.is_finite(), "Constrained direction should be finite");
	CHECK_MESSAGE(constrained.length() > 0.99, "Constrained direction should be normalized");
}

} // namespace TestIKKusudama3D
