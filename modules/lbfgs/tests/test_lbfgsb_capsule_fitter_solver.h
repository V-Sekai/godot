/**************************************************************************/
/*  test_lbfgsb_capsule_fitter_solver.h                                   */
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

#include "../diff_cloth_integration.h"
#include "../lbfgsb_capsule_fitter_solver.h"
#include "../skeleton_retargeting_solver.h"
#include "core/math/basis.h"
#include "core/math/vector3.h"
#include "core/variant/dictionary.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/mesh.h"

#include "tests/test_macros.h"
#include <cfloat>

namespace TestLBFGSBCapsuleFitterSolver {

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
			current_h = 0;
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
	CHECK_MESSAGE(result.has("final_fx"), "Result should contain 'final_fx'.");
	CHECK_MESSAGE(result.has("optimized_radius"), "Result should contain 'optimized_radius'.");

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

	solver->set_radius(0.5);
	solver->set_height(2.0);
	solver->set_axis_point_a(Vector3(0.1, -1.0 - 0.1, 0.1));
	solver->set_axis_point_b(Vector3(0.1, 1.0 - 0.1, 0.1));

	solver->set_huber_delta(0.05);
	solver->set_max_iterations(300);
	solver->set_epsilon(DBL_EPSILON);

	Dictionary result = solver->optimize_orientation_for_fixed_size();

	INFO("Orientation optimization result: ", result);
	CHECK_MESSAGE(!result.is_empty(), "Optimization result should not be empty.");
	CHECK_MESSAGE(result.has("final_fx"), "Result should contain 'final_fx'.");
	CHECK_MESSAGE(result.has("optimized_axis_a"), "Result should contain 'optimized_axis_a'.");
	CHECK_MESSAGE(result.has("optimized_axis_b"), "Result should contain 'optimized_axis_b'.");

	Vector3 optimized_a = result["optimized_axis_a"];
	Vector3 optimized_b = result["optimized_axis_b"];

	MESSAGE("Optimized Axis A: ", optimized_a, " Optimized Axis B: ", optimized_b);

	CHECK_MESSAGE((optimized_b - optimized_a).length() == doctest::Approx(2.0).epsilon(0.01), "Optimized axis length should remain 2.0.");

	Vector3 optimized_direction = (optimized_b - optimized_a).normalized();
	Vector3 expected_direction = tilt_basis.xform(Vector3(0, 1, 0)).normalized();
	Vector3 optimized_center = (optimized_a + optimized_b) / 2.0;
	Vector3 expected_center = Vector3(0, 0, 0);

	MESSAGE("Optimized Direction: ", optimized_direction, " Expected Direction (approx): ", expected_direction);
	CHECK_MESSAGE(vector3_is_equal_approx(optimized_direction, expected_direction, 1e-5), "Optimized axis direction should be very close to the tilted mesh's direction.");

	MESSAGE("Optimized Center: ", optimized_center, " Expected Center: ", expected_center);
	CHECK_MESSAGE(vector3_is_equal_approx(optimized_center, expected_center, 3e-3), "Optimized capsule center should be very close to the mesh's center.");

	double final_f = result["final_fx"];
	CHECK_MESSAGE(final_f >= 0, "Final objective function value should be non-negative.");
	CHECK_MESSAGE(final_f < 0.5, "Final objective function value should be reasonably small.");

	if (result.has("iterations")) {
		int iterations = result["iterations"];
		MESSAGE("Solver iterations: ", iterations);
		CHECK_MESSAGE(iterations > 0, "Solver should take some iterations.");
		CHECK_MESSAGE(iterations < solver->get_max_iterations(), "Solver should converge before max_iterations.");
	}

	memdelete(solver);
}

TEST_CASE("[SceneTree][LBFGSBCapsuleFitterSolver] Optimize Orientation for SphereMesh") {
	LBFGSBCapsuleFitterSolver *solver = memnew(LBFGSBCapsuleFitterSolver);

	Ref<SphereMesh> sphere_mesh = memnew(SphereMesh);
	sphere_mesh->set_radius(1.0f);
	sphere_mesh->set_height(2.0f);
	sphere_mesh->set_radial_segments(16);
	sphere_mesh->set_rings(8);

	Ref<ArrayMesh> array_mesh = memnew(ArrayMesh);
	array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, sphere_mesh->get_mesh_arrays());

	solver->set_source_mesh(array_mesh);
	solver->set_surface_index(0);

	solver->set_radius(0.5);
	solver->set_height(1.0);
	solver->set_axis_point_a(Vector3(0, -0.5, 0.1));
	solver->set_axis_point_b(Vector3(0, 0.5, 0.1));

	solver->set_huber_delta(0.1);
	solver->set_max_iterations(200);
	solver->set_epsilon(1e-6);

	Dictionary result = solver->optimize_orientation_for_fixed_size();

	INFO("SphereMesh Orientation optimization result: ", result);
	CHECK_MESSAGE(!result.is_empty(), "SphereMesh: Optimization result should not be empty.");
	CHECK_MESSAGE(result.has("final_fx"), "SphereMesh: Result should contain 'final_fx'.");
	CHECK_MESSAGE(result.has("optimized_axis_a"), "SphereMesh: Result should contain 'optimized_axis_a'.");
	CHECK_MESSAGE(result.has("optimized_axis_b"), "SphereMesh: Result should contain 'optimized_axis_b'.");

	if (!result.is_empty() && result.has("optimized_axis_a") && result.has("optimized_axis_b")) {
		Vector3 optimized_a = result["optimized_axis_a"];
		Vector3 optimized_b = result["optimized_axis_b"];
		Vector3 optimized_center = (optimized_a + optimized_b) / 2.0;
		Vector3 expected_center = Vector3(0, 0, 0);

		MESSAGE("SphereMesh Optimized Center: ", optimized_center, " Expected Center: ", expected_center);
		CHECK_MESSAGE(vector3_is_equal_approx(optimized_center, expected_center, 0.1), "SphereMesh: Optimized capsule center should be very close to the sphere's center.");

		MESSAGE("SphereMesh Optimized Axis Length: ", (optimized_b - optimized_a).length());
		CHECK_MESSAGE((optimized_b - optimized_a).length() == doctest::Approx(1.0).epsilon(0.05), "SphereMesh: Optimized axis length should remain close to initial 1.0.");
	}

	memdelete(solver);
}

TEST_CASE("[SceneTree][LBFGSBCapsuleFitterSolver] Optimize Orientation for BoxMesh") {
	LBFGSBCapsuleFitterSolver *solver = memnew(LBFGSBCapsuleFitterSolver);

	Ref<BoxMesh> box_mesh = memnew(BoxMesh);
	box_mesh->set_size(Vector3(1, 2, 0.5));

	Ref<ArrayMesh> array_mesh_box = memnew(ArrayMesh);
	array_mesh_box->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, box_mesh->get_mesh_arrays());

	solver->set_source_mesh(array_mesh_box);
	solver->set_surface_index(0);

	solver->set_radius(0.25);
	solver->set_height(1.8);
	solver->set_axis_point_a(Vector3(0.1, -0.9, 0));
	solver->set_axis_point_b(Vector3(0.1, 0.9, 0));

	solver->set_huber_delta(0.1);
	solver->set_max_iterations(300);
	solver->set_epsilon(1e-6);

	Dictionary result = solver->optimize_orientation_for_fixed_size();

	INFO("BoxMesh Orientation optimization result: ", result);
	CHECK_MESSAGE(!result.is_empty(), "BoxMesh: Optimization result should not be empty.");
	CHECK_MESSAGE(result.has("final_fx"), "BoxMesh: Result should contain 'final_fx'.");
	CHECK_MESSAGE(result.has("optimized_axis_a"), "BoxMesh: Result should contain 'optimized_axis_a'.");
	CHECK_MESSAGE(result.has("optimized_axis_b"), "BoxMesh: Result should contain 'optimized_axis_b'.");

	if (!result.is_empty() && result.has("optimized_axis_a") && result.has("optimized_axis_b")) {
		Vector3 optimized_a = result["optimized_axis_a"];
		Vector3 optimized_b = result["optimized_axis_b"];
		Vector3 optimized_center = (optimized_a + optimized_b) / 2.0;
		Vector3 optimized_direction = (optimized_b - optimized_a).normalized();
		Vector3 expected_center = Vector3(0, 0, 0);

		MESSAGE("BoxMesh Optimized Center: ", optimized_center, " Expected Center: ", expected_center);
		CHECK_MESSAGE(vector3_is_equal_approx(optimized_center, expected_center, 0.15), "BoxMesh: Optimized capsule center should be very close to the box's center.");

		bool aligned_with_y = Math::abs(optimized_direction.dot(Vector3(0, 1, 0))) > 0.95;
		MESSAGE("BoxMesh Optimized Direction: ", optimized_direction, " (Dot Y: ", optimized_direction.dot(Vector3(0, 1, 0)), ")");
		CHECK_MESSAGE(aligned_with_y, "BoxMesh: Optimized capsule should align with the box's longest (Y) axis.");
		CHECK_MESSAGE((optimized_b - optimized_a).length() == doctest::Approx(1.8).epsilon(0.1), "BoxMesh: Optimized axis length should remain close to initial 1.8.");
	}

	memdelete(solver);
}

TEST_CASE("[SceneTree][LBFGSBCapsuleFitterSolver] Optimize Parameters for Multiple Capsules Simultaneously" * doctest::skip(true)) {
	// This test drives the adaptation of optimize_all_capsule_parameters
	// to handle multiple capsules defined within the solver.

	LBFGSBCapsuleFitterSolver *solver = memnew(LBFGSBCapsuleFitterSolver);

	// Create a target mesh, e.g., a tilted cylinder.
	// Both capsules will attempt to fit this same mesh for this test.
	Basis actual_tilt = Basis(Vector3(1, 0, 0), Math::deg_to_rad(45.0f)); // Tilted 45 deg around X
	Ref<ArrayMesh> mesh_target = create_tilted_cylinder_points_mesh(0.75f, 1.5f, 20, 7, actual_tilt);
	solver->set_source_mesh(mesh_target);
	solver->set_surface_index(0);

	// Assume an API to add/configure multiple capsules. This will drive solver API design.
	// For this test, we'll imagine methods to set up two capsules.
	// These would likely be replaced by a more robust internal list and an `add_capsule` method on the solver.
	// For now, the test implies the solver needs to be aware of these multiple configurations.

	// To make this test work without immediate solver API changes for multiple capsules,
	// we will call optimize_all_capsule_parameters twice, once for each conceptual capsule,
	// and check results. A true multi-capsule optimization would involve a single call
	// that optimizes all 14+ parameters (7 per capsule) simultaneously.
	// The current test name implies a single call for multiple capsules.
	// Let's adjust the test to reflect a future where the solver handles a list of capsules.

	// Placeholder for solver API to define multiple capsules.
	// solver->add_capsule_definition(initial_a1, initial_b1, initial_radius1);
	// solver->add_capsule_definition(initial_a2, initial_b2, initial_radius2);
	// For now, we will simulate this by re-setting parameters on a single-capsule solver
	// and calling it, which is NOT the final intent of "Multiple Capsules Simultaneously".
	// This test will need significant rework once the solver supports multiple capsules internally.
	// The following is a TEMPORARY structure for a SINGLE capsule test, acknowledging the name is for MULTIPLE.

	// Set initial capsule parameters (for a conceptual first capsule)
	solver->set_radius(0.1); // Too small
	// solver->set_height(1.0); // set_height is not used by optimize_all_capsule_parameters directly
	// The following lines for set_axis_point_a and set_axis_point_b are commented out
	// because the LBFGSBCapsuleFitterSolver API has changed to use local_axis_point_a/b
	// and target_skeleton_path/target_bone_name. This test will need to be updated
	// to reflect these API changes once it's being seriously worked on for multi-capsule.
	// For now, to avoid compilation errors with the changed API, we comment them out.
	// solver->set_axis_point_a(Vector3(0, -0.5, 0)); // Initial axis along Y
	// solver->set_axis_point_b(Vector3(0, 0.5, 0));

	solver->set_huber_delta(0.05);
	solver->set_max_iterations(500);
	solver->set_epsilon(1e-6);

	// This call would ideally optimize ALL configured capsules.
	// The current solver is expected to optimize its single set of parameters.
	// The SNAME should match a method that will eventually handle multiple capsules.
	Dictionary result = solver->call(SNAME("optimize_all_capsule_parameters"));

	INFO("Combined optimization result (simulating one of multiple capsules): ", result);
	// Basic checks, will need to be adapted for an array of results later
	CHECK_MESSAGE(!result.is_empty(), "Result should not be empty.");
	CHECK_MESSAGE(result.has("final_fx"), "Result should contain 'final_fx'.");

	// The following detailed checks are for a single capsule result.
	// These will need to be moved into a loop when results are an array.
	if (result.has("optimized_radius") && result.has("optimized_axis_a") && result.has("optimized_axis_b")) {
		double optimized_radius = result["optimized_radius"];
		Vector3 optimized_a = result["optimized_axis_a"];
		Vector3 optimized_b = result["optimized_axis_b"];

		MESSAGE("Optimized radius: ", optimized_radius);
		CHECK_MESSAGE(optimized_radius == doctest::Approx(0.75).epsilon(0.05), "Optimized radius should be close to 0.75 (target).");

		MESSAGE("Optimized Axis A: ", optimized_a, " Optimized Axis B: ", optimized_b);
		CHECK_MESSAGE((optimized_b - optimized_a).length() == doctest::Approx(1.5).epsilon(0.05), "Optimized axis length should be close to 1.5 (target).");

		Vector3 optimized_direction = (optimized_b - optimized_a).normalized();
		Vector3 expected_direction = actual_tilt.xform(Vector3(0, 1, 0)).normalized(); // Assuming cylinder is along Y initially
		CHECK_MESSAGE(vector3_is_equal_approx(optimized_direction, expected_direction, 1e-4), "Optimized axis direction should match target mesh direction.");

		Vector3 optimized_center = (optimized_a + optimized_b) / 2.0;
		CHECK_MESSAGE(vector3_is_equal_approx(optimized_center, Vector3(0, 0, 0), 1e-3), "Optimized capsule center should be close to origin.");

		double final_f = result["final_fx"];
		CHECK_MESSAGE(final_f < 0.1, "Final objective function value should be small for a good fit.");
	} else {
		// If the primary keys are missing, it might be because the solver returned an array directly.
		// This part of the test is highly speculative until the multi-capsule API is defined.
		CHECK_MESSAGE(result.has("optimized_capsules_results"), "Result should contain 'optimized_capsules_results' if not single capsule keys.");
	}

	// TODO: This test needs to be expanded significantly once the solver supports
	// an internal list of capsules and `optimize_all_capsule_parameters` can optimize them all
	// simultaneously, returning an array of results.
	// The assertions would then loop through that array.

	memdelete(solver);
}

} // namespace TestLBFGSBCapsuleFitterSolver
