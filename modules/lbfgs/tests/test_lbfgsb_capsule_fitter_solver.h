/**************************************************************************/
/*  test_lbfgsb_capsule_fitter_solver->h                                   */
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

#include "../lbfgsb_capsule_fitter_solver.h"
#include "core/math/basis.h"
#include "core/math/vector3.h"
#include "core/variant/dictionary.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/mesh.h"

#include "tests/test_macros.h"
#include <cfloat>

namespace TestLBFGSBCapsuleFitterSolver {

// Helper function for approximate Vector3 comparison
static bool vector3_is_equal_approx(const Vector3 &v1, const Vector3 &v2, double epsilon = CMP_EPSILON) {
	return Math::is_equal_approx(real_t(v1.x), real_t(v2.x), real_t(epsilon)) &&
		   Math::is_equal_approx(real_t(v1.y), real_t(v2.y), real_t(epsilon)) &&
		   Math::is_equal_approx(real_t(v1.z), real_t(v2.z), real_t(epsilon));
}

static Ref<ArrayMesh> create_cylinder_points_mesh(float r, float h, int num_points_circle, int num_height_steps) {
	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	PackedVector3Array vertices;

	if (num_height_steps <= 0) {
		num_height_steps = 1;
	}
	if (num_points_circle <= 0) {
		num_points_circle = 1;
	}
	for (int i = 0; i < num_height_steps; ++i) {
		float current_h = -h / 2.0f;
		if (num_height_steps > 1) {
			current_h += (h * i / (float)(num_height_steps - 1));
		} else if (num_height_steps == 1) {
			current_h = 0; // Single slice at mid-height if h is non-zero, or at 0.
		}

		for (int j = 0; j < num_points_circle; ++j) {
			float angle = 0.0f;
			if (num_points_circle > 1) {
				angle = (2.0f * Math::PI * j) / (float)num_points_circle;
			}
			vertices.push_back(Vector3(r * cos(angle), current_h, r * sin(angle)));
		}
	}

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;

	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_POINTS, arrays);
	return mesh;
}

static Ref<ArrayMesh> create_tilted_cylinder_points_mesh(float r, float h, int num_points_circle, int num_height_steps, const Basis &orientation) {
	Ref<ArrayMesh> mesh = memnew(ArrayMesh);
	PackedVector3Array vertices;

	if (num_height_steps <= 0) {
		num_height_steps = 1;
	}
	if (num_points_circle <= 0) {
		num_points_circle = 1;
	}

	for (int i = 0; i < num_height_steps; ++i) {
		float current_h_local = -h / 2.0f;
		if (num_height_steps > 1) {
			current_h_local += (h * i / (float)(num_height_steps - 1));
		} else if (num_height_steps == 1) {
			current_h_local = 0;
		}

		for (int j = 0; j < num_points_circle; ++j) {
			float angle = 0.0f;
			if (num_points_circle > 1) {
				angle = (2.0f * Math::PI * j) / (float)num_points_circle;
			}
			Vector3 local_p(r * cos(angle), current_h_local, r * sin(angle));
			vertices.push_back(orientation.xform(local_p));
		}
	}

	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;

	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_POINTS, arrays);
	return mesh;
}

TEST_CASE("[SceneTree][LBFGSBCapsuleFitterSolver] Optimize Radius for Fixed Axis") {
	LBFGSBCapsuleFitterSolver *solver = memnew(LBFGSBCapsuleFitterSolver);

	Ref<ArrayMesh> mesh = create_cylinder_points_mesh(1.0f, 2.0f, 16, 5);
	solver->set_source_mesh(mesh);
	solver->set_surface_index(0);

	solver->set_axis_point_a(Vector3(0, -1.0, 0));
	solver->set_axis_point_b(Vector3(0, 1.0, 0));
	solver->set_radius(0.1);

	solver->set_max_iterations(200);
	solver->set_epsilon(1e-4);

	Dictionary result = solver->optimize_radius_for_fixed_axis();

	INFO("Radius optimization result: ", result);
	CHECK_MESSAGE(!result.is_empty(), "Optimization result should not be empty.");
	REQUIRE_MESSAGE(result.has("final_fx"), "Result should contain 'final_fx'.");
	REQUIRE_MESSAGE(result.has("optimized_radius"), "Result should contain 'optimized_radius'.");

	double optimized_radius = result["optimized_radius"];
	MESSAGE("Optimized radius: ", optimized_radius);
	CHECK_MESSAGE(optimized_radius == doctest::Approx(1.0).epsilon(0.05), "Optimized radius should be close to 1.0.");

	double final_f = result["final_fx"];
	CHECK_MESSAGE(final_f < 1e-2, "Final objective function value should be small for a good fit.");
	memdelete(solver);
}

TEST_CASE("[SceneTree][LBFGSBCapsuleFitterSolver] Optimize Orientation for Fixed Size") {
	LBFGSBCapsuleFitterSolver *solver = memnew(LBFGSBCapsuleFitterSolver);

	Basis tilt_basis = Basis(Vector3(0, 0, 1), Math::deg_to_rad(30.0f));
	Ref<ArrayMesh> mesh = create_tilted_cylinder_points_mesh(0.5f, 2.0f, 16, 5, tilt_basis);
	solver->set_source_mesh(mesh);
	solver->set_surface_index(0);

	// Set initial capsule pose slightly off from the mesh's actual pose
	// Mesh is tilted around Z by 30 deg, centered at origin. Radius 0.5, Height 2.0.
	// Initial capsule: radius 0.5, height 2.0, axis along Y, centered at (0.1, -0.1, 0.1) to test translation.
	solver->set_radius(0.5);
	solver->set_height(2.0); // This will set _orientation_opt_initial_half_axis correctly if called before optimize_orientation
	solver->set_axis_point_a(Vector3(0.1, -1.0 - 0.1, 0.1)); // Offset center
	solver->set_axis_point_b(Vector3(0.1, 1.0 - 0.1, 0.1));

	solver->set_huber_delta(0.05); // Adjust Huber delta for potentially smaller errors
	solver->set_max_iterations(300);
	solver->set_epsilon(DBL_EPSILON); // Tightened solver epsilon for better positional accuracy

	Dictionary result = solver->optimize_orientation_for_fixed_size();

	INFO("Orientation optimization result: ", result);
	CHECK_MESSAGE(!result.is_empty(), "Optimization result should not be empty.");
	REQUIRE_MESSAGE(result.has("final_fx"), "Result should contain 'final_fx'.");
	REQUIRE_MESSAGE(result.has("optimized_axis_a"), "Result should contain 'optimized_axis_a'.");
	REQUIRE_MESSAGE(result.has("optimized_axis_b"), "Result should contain 'optimized_axis_b'.");

	Vector3 optimized_a = result["optimized_axis_a"];
	Vector3 optimized_b = result["optimized_axis_b"];

	MESSAGE("Optimized Axis A: ", optimized_a, " Optimized Axis B: ", optimized_b);

	CHECK_MESSAGE((optimized_b - optimized_a).length() == doctest::Approx(2.0).epsilon(0.01), "Optimized axis length should remain 2.0.");

	Vector3 optimized_direction = (optimized_b - optimized_a).normalized();
	Vector3 expected_direction = tilt_basis.xform(Vector3(0, 1, 0)).normalized();
	Vector3 optimized_center = (optimized_a + optimized_b) / 2.0;
	Vector3 expected_center = Vector3(0, 0, 0); // Mesh is centered at origin

	MESSAGE("Optimized Direction: ", optimized_direction, " Expected Direction (approx): ", expected_direction);
	CHECK_MESSAGE(vector3_is_equal_approx(optimized_direction, expected_direction, 1e-5), "Optimized axis direction should be very close to the tilted mesh's direction.");

	MESSAGE("Optimized Center: ", optimized_center, " Expected Center: ", expected_center);
	CHECK_MESSAGE(vector3_is_equal_approx(optimized_center, expected_center, 3e-3), "Optimized capsule center should be very close to the mesh's center.");

	double final_f = result["final_fx"];
	// The final_fx is now a sum of Huber losses, which can be small but not necessarily < 1e-2 for a good fit.
	// We rely more on geometric checks (direction, center, length).
	// Check if it's a valid, non-negative number, and perhaps within a broader range if needed.
	CHECK_MESSAGE(final_f >= 0, "Final objective function value should be non-negative.");
	// Example: Check if it's less than a loose upper bound, e.g. number of points * huber_delta^2
	// For 16*5 = 80 points, 80 * 0.05^2 = 80 * 0.0025 = 0.2
	CHECK_MESSAGE(final_f < 0.5, "Final objective function value should be reasonably small.");

	// Check number of iterations if available
	if (result.has("iterations")) {
		int iterations = result["iterations"];
		MESSAGE("Solver iterations: ", iterations);
		CHECK_MESSAGE(iterations > 0, "Solver should take some iterations.");
		CHECK_MESSAGE(iterations < solver->get_max_iterations(), "Solver should converge before max_iterations.");
	}

	memdelete(solver);
}
} // namespace TestLBFGSBCapsuleFitterSolver
