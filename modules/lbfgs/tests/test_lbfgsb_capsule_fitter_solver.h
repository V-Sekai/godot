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

#include "../lbfgsb_capsule_fitter_solver.h"

#include "core/io/json.h" // For JSON::stringify
#include "core/math/math_funcs.h" // For Math::TAU
#include "core/math/vector3.h"
#include "core/variant/dictionary.h"
#include "scene/resources/mesh.h" // Should include SphereMesh and ImporterMesh

#include "tests/test_macros.h"

namespace TestLBFGSBCapsuleFitterSolver {

static Ref<ArrayMesh> create_sphere_test_mesh(const Vector3 &p_center, real_t p_radius, int p_quality = 16) {
	PackedVector3Array vertices;
	PackedVector3Array normals;
	PackedInt32Array indices;

	int segments = MAX(3, p_quality); // Radial segments
	int rings = MAX(2, p_quality / 2); // Height segments

	for (int i = 0; i <= rings; ++i) {
		real_t v_angle = Math::PI * (real_t)i / rings; // Angle from Y+ axis
		real_t y = p_radius * Math::cos(v_angle);
		real_t current_ring_radius = p_radius * Math::sin(v_angle);

		for (int j = 0; j <= segments; ++j) {
			real_t u_angle = Math::TAU * (real_t)j / segments; // Angle around Y axis
			real_t x = current_ring_radius * Math::cos(u_angle);
			real_t z = current_ring_radius * Math::sin(u_angle);
			Vector3 point_local = Vector3(x, y, z);
			vertices.push_back(p_center + point_local);
			normals.push_back(point_local.normalized());
		}
	}

	for (int i = 0; i < rings; ++i) {
		for (int j = 0; j < segments; ++j) {
			int first = (i * (segments + 1)) + j;
			int second = first + segments + 1;

			indices.push_back(first);
			indices.push_back(second);
			indices.push_back(first + 1);

			indices.push_back(second);
			indices.push_back(second + 1);
			indices.push_back(first + 1);
		}
	}

	Ref<ArrayMesh> array_mesh;
	array_mesh.instantiate();
	Array mesh_arrays;
	mesh_arrays.resize(Mesh::ARRAY_MAX);
	mesh_arrays[Mesh::ARRAY_VERTEX] = vertices;
	mesh_arrays[Mesh::ARRAY_NORMAL] = normals;
	mesh_arrays[Mesh::ARRAY_INDEX] = indices;

	uint32_t fmt = 0;
	if (!vertices.is_empty()) {
		fmt |= Mesh::ARRAY_FORMAT_VERTEX;
	}
	if (!normals.is_empty()) {
		fmt |= Mesh::ARRAY_FORMAT_NORMAL;
	}
	if (!indices.is_empty()) {
		fmt |= Mesh::ARRAY_FORMAT_INDEX;
	}

	array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, mesh_arrays, Array(), Dictionary(), fmt);
	return array_mesh;
}

// Helper function to create a triangulated cylinder mesh.
// Returns Ref<ArrayMesh>.
static Ref<ArrayMesh> create_cylinder_points_mesh(float p_radius, float p_height, int p_radial_segments = 16, int p_height_segments = 1, bool p_caps = true) {
	PackedVector3Array vertices;
	PackedVector3Array normals;
	PackedInt32Array indices;

	p_radial_segments = MAX(3, p_radial_segments);
	p_height_segments = MAX(1, p_height_segments);

	float half_height = p_height / 2.0f;

	// Cylinder Wall
	for (int i = 0; i <= p_height_segments; i++) {
		float y = -half_height + (p_height * i / p_height_segments);
		for (int j = 0; j <= p_radial_segments; j++) {
			float angle = Math::TAU * j / p_radial_segments;
			float x = p_radius * Math::cos(angle);
			float z = p_radius * Math::sin(angle);
			vertices.push_back(Vector3(x, y, z));
			normals.push_back(Vector3(x, 0, z).normalized()); // Normal points outwards from cylinder axis
		}
	}

	for (int i = 0; i < p_height_segments; i++) {
		for (int j = 0; j < p_radial_segments; j++) {
			int first = (i * (p_radial_segments + 1)) + j;
			int second = first + p_radial_segments + 1;

			indices.push_back(first);
			indices.push_back(second + 1);
			indices.push_back(first + 1);

			indices.push_back(first);
			indices.push_back(second);
			indices.push_back(second + 1);
		}
	}

	if (p_caps) {
		// Top Cap
		int top_center_idx = vertices.size();
		vertices.push_back(Vector3(0, half_height, 0));
		normals.push_back(Vector3(0, 1, 0));
		int top_ring_start_idx = (p_height_segments * (p_radial_segments + 1));
		for (int j = 0; j < p_radial_segments; j++) {
			indices.push_back(top_center_idx);
			indices.push_back(top_ring_start_idx + j);
			indices.push_back(top_ring_start_idx + j + 1);
		}

		// Bottom Cap
		int bottom_center_idx = vertices.size();
		vertices.push_back(Vector3(0, -half_height, 0));
		normals.push_back(Vector3(0, -1, 0));
		int bottom_ring_start_idx = 0;
		for (int j = 0; j < p_radial_segments; j++) {
			indices.push_back(bottom_center_idx);
			indices.push_back(bottom_ring_start_idx + j + 1);
			indices.push_back(bottom_ring_start_idx + j);
		}
	}

	Ref<ArrayMesh> array_mesh;
	array_mesh.instantiate();
	Array mesh_arrays;
	mesh_arrays.resize(Mesh::ARRAY_MAX);
	mesh_arrays[Mesh::ARRAY_VERTEX] = vertices;
	mesh_arrays[Mesh::ARRAY_NORMAL] = normals;
	mesh_arrays[Mesh::ARRAY_INDEX] = indices;

	uint32_t fmt = 0;
	if (!vertices.is_empty()) {
		fmt |= Mesh::ARRAY_FORMAT_VERTEX;
	}
	if (!normals.is_empty()) {
		fmt |= Mesh::ARRAY_FORMAT_NORMAL;
	}
	if (!indices.is_empty()) {
		fmt |= Mesh::ARRAY_FORMAT_INDEX;
	}

	array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, mesh_arrays, Array(), Dictionary(), fmt);
	return array_mesh;
}

TEST_CASE("[SceneTree][LBFGSBCapsuleFitterSolverBase] Basic Setters and Getters") {
	Ref<ArrayMesh> mesh = create_sphere_test_mesh(Vector3(0, 0, 0), 1.0, 8);
	LBFGSBCapsuleRadiusSolver *solver = memnew(LBFGSBCapsuleRadiusSolver);

	SUBCASE("Source Mesh and Basic Parameters") {
		solver->set_source_mesh(mesh);
		CHECK(solver->get_source_mesh() == mesh);
		solver->set_surface_index(0);
		CHECK(solver->get_surface_index() == 0);
		solver->set_huber_delta(0.15);
		CHECK(solver->get_huber_delta() == doctest::Approx(0.15));
		solver->set_orientation_distance_threshold(0.2);
		CHECK(solver->get_orientation_distance_threshold() == doctest::Approx(0.2));
		solver->set_orientation_angle_threshold_rad(0.5);
		CHECK(solver->get_orientation_angle_threshold_rad() == doctest::Approx(0.5));
		solver->set_orientation_penalty_factor(1.5);
		CHECK(solver->get_orientation_penalty_factor() == doctest::Approx(1.5));
	}

	SUBCASE("Capsule Instance Management") {
		CHECK(solver->get_num_capsule_instances() == 0);
		solver->add_capsule_instance(Vector3(0, 0, -0.5), Vector3(0, 0, 0.5), 0.2);
		CHECK(solver->get_num_capsule_instances() == 1);
		Dictionary capsule_data = solver->get_capsule_instance_data(0);
		CHECK(Vector3(capsule_data["initial_axis_a"]) == Vector3(0, 0, -0.5));
		CHECK(Vector3(capsule_data["initial_axis_b"]) == Vector3(0, 0, 0.5));
		CHECK(double(capsule_data["initial_radius"]) == doctest::Approx(0.2));
		solver->clear_capsule_instances();
		CHECK(solver->get_num_capsule_instances() == 0);
	}

	memdelete(solver);
}

TEST_CASE("[SceneTree][LBFGSBCapsuleRadiusSolver] Optimize Radius for a Sphere") {
	LBFGSBCapsuleRadiusSolver *radius_solver = memnew(LBFGSBCapsuleRadiusSolver);

	Ref<ArrayMesh> sphere_mesh = create_sphere_test_mesh(Vector3(1, 1, 1), 2.0, 16); // Sphere centered at (1,1,1) with radius 2.0
	radius_solver->set_source_mesh(sphere_mesh);
	radius_solver->set_surface_index(0);
	radius_solver->add_capsule_instance(Vector3(1, 1, 1), Vector3(1, 1, 1), 0.5);

	Dictionary result = radius_solver->optimize_radius(0);

	String msg_radius_result = String("Radius optimization result (Sphere): ") + JSON::stringify(result, "\t");
	WARN_MESSAGE(true, msg_radius_result.utf8().get_data());

	String msg_check_error_radius = String("Optimization should not produce an error. Error: ") + (result.has("error") ? String(result["error"]) : String("N/A"));
	CHECK_MESSAGE(!result.has("error"), msg_check_error_radius.utf8().get_data());
	if (!result.has("error")) {
		String msg_check_has_radius = String("Result should contain optimized_radius");
		CHECK_MESSAGE(result.has("optimized_radius"), msg_check_has_radius.utf8().get_data());
		if (result.has("optimized_radius")) {
			double optimized_r = result["optimized_radius"];
			String msg_opt_radius_val = String("Optimized radius (Sphere): ") + String::num(optimized_r);
			WARN_MESSAGE(true, msg_opt_radius_val.utf8().get_data());
			String msg_before_check_sphere_radius = String("CHECK: optimized_r (") + String::num(optimized_r) + String(") vs Approx(2.0)");
			WARN_MESSAGE(true, msg_before_check_sphere_radius.utf8().get_data());
			CHECK(optimized_r == doctest::Approx(2.0).epsilon(0.15)); // Allow some tolerance
		}
	}

	memdelete(radius_solver);
}

TEST_CASE("[SceneTree][LBFGSBCapsuleAxisSolver] Optimize Axes for a Cylinder") {
	LBFGSBCapsuleAxisSolver *axis_solver = memnew(LBFGSBCapsuleAxisSolver);

	float cylinder_radius = 0.5f;
	float cylinder_height = 3.0f;
	Ref<ArrayMesh> cylinder_mesh = create_cylinder_points_mesh(cylinder_radius, cylinder_height, 16, 4);
	axis_solver->set_source_mesh(cylinder_mesh);
	axis_solver->set_surface_index(0);

	Vector3 initial_a = Vector3(0.1, -cylinder_height / 2.0 + 0.2, 0.1);
	Vector3 initial_b = Vector3(-0.1, cylinder_height / 2.0 - 0.2, -0.1);
	double initial_radius = cylinder_radius;
	axis_solver->add_capsule_instance(initial_a, initial_b, initial_radius);

	Dictionary result = axis_solver->optimize_axes(0);

	String msg_axes_result = String("Axes optimization result (Cylinder): ") + JSON::stringify(result, "\t");
	WARN_MESSAGE(true, msg_axes_result.utf8().get_data());

	String msg_check_error_axes = String("Optimization should not produce an error. Error: ") + (result.has("error") ? String(result["error"]) : String("N/A"));
	CHECK_MESSAGE(!result.has("error"), msg_check_error_axes.utf8().get_data());
	if (!result.has("error")) {
		String msg_check_has_axes = String("Result should contain optimized_axis_a and optimized_axis_b");
		bool has_axes = result.has("optimized_axis_a") && result.has("optimized_axis_b");
		CHECK_MESSAGE(has_axes, msg_check_has_axes.utf8().get_data());
		if (result.has("optimized_axis_a") && result.has("optimized_axis_b")) {
			Vector3 opt_a = Vector3(result["optimized_axis_a"]);
			Vector3 opt_b = Vector3(result["optimized_axis_b"]);
			String msg_opt_axis_a = String("Optimized axis_a (Cylinder): ") + opt_a.operator String();
			WARN_MESSAGE(true, msg_opt_axis_a.utf8().get_data());
			String msg_opt_axis_b = String("Optimized axis_b (Cylinder): ") + opt_b.operator String();
			WARN_MESSAGE(true, msg_opt_axis_b.utf8().get_data());

			String msg_before_check_opt_ax = String("CHECK: opt_a.x (") + String::num(opt_a.x) + String(") vs Approx(0.0)");
			WARN_MESSAGE(true, msg_before_check_opt_ax.utf8().get_data());
			CHECK(opt_a.x == doctest::Approx(0.0).epsilon(0.15));
			String msg_before_check_opt_az = String("CHECK: opt_a.z (") + String::num(opt_a.z) + String(") vs Approx(0.0)");
			WARN_MESSAGE(true, msg_before_check_opt_az.utf8().get_data());
			CHECK(opt_a.z == doctest::Approx(0.0).epsilon(0.15));
			bool a_is_bottom = std::abs(opt_a.y - (-cylinder_height / 2.0)) < std::abs(opt_a.y - (cylinder_height / 2.0));

			if (a_is_bottom) {
				String msg_before_check_opt_ay_bottom = String("CHECK: opt_a.y (") + String::num(opt_a.y) + String(") vs Approx(") + String::num(-cylinder_height / 2.0) + String(")");
				WARN_MESSAGE(true, msg_before_check_opt_ay_bottom.utf8().get_data());
				CHECK(opt_a.y == doctest::Approx(-cylinder_height / 2.0).epsilon(0.25));
				String msg_before_check_opt_by_top = String("CHECK: opt_b.y (") + String::num(opt_b.y) + String(") vs Approx(") + String::num(cylinder_height / 2.0) + String(")");
				WARN_MESSAGE(true, msg_before_check_opt_by_top.utf8().get_data());
				CHECK(opt_b.y == doctest::Approx(cylinder_height / 2.0).epsilon(0.25));
			} else {
				String msg_before_check_opt_ay_top = String("CHECK: opt_a.y (") + String::num(opt_a.y) + String(") vs Approx(") + String::num(cylinder_height / 2.0) + String(")");
				WARN_MESSAGE(true, msg_before_check_opt_ay_top.utf8().get_data());
				CHECK(opt_a.y == doctest::Approx(cylinder_height / 2.0).epsilon(0.25));
				String msg_before_check_opt_by_bottom = String("CHECK: opt_b.y (") + String::num(opt_b.y) + String(") vs Approx(") + String::num(-cylinder_height / 2.0) + String(")");
				WARN_MESSAGE(true, msg_before_check_opt_by_bottom.utf8().get_data());
				CHECK(opt_b.y == doctest::Approx(-cylinder_height / 2.0).epsilon(0.25));
			}
			String msg_before_check_opt_bx = String("CHECK: opt_b.x (") + String::num(opt_b.x) + String(") vs Approx(0.0)");
			WARN_MESSAGE(true, msg_before_check_opt_bx.utf8().get_data());
			CHECK(opt_b.x == doctest::Approx(0.0).epsilon(0.15));
			String msg_before_check_opt_bz = String("CHECK: opt_b.z (") + String::num(opt_b.z) + String(") vs Approx(0.0)");
			WARN_MESSAGE(true, msg_before_check_opt_bz.utf8().get_data());
			CHECK(opt_b.z == doctest::Approx(0.0).epsilon(0.15));
			
			double current_optimized_radius = result.has("optimized_radius") ? double(result["optimized_radius"]) : -1.0; // Default if not found
			String msg_before_check_cyl_radius = String("CHECK: optimized_radius (") + String::num(current_optimized_radius) + String(") vs Approx(initial_radius:") + String::num(initial_radius) + String(")");
			WARN_MESSAGE(true, msg_before_check_cyl_radius.utf8().get_data());
			CHECK(current_optimized_radius == doctest::Approx(initial_radius));
		}
	}

	memdelete(axis_solver);
}

} // namespace TestLBFGSBCapsuleFitterSolver
