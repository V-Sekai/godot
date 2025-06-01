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

	array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, mesh_arrays);
	return array_mesh;
}

TEST_CASE("[SceneTree][SceneTree][LBFGSBCapsuleFitterSolver] Instantiation and Basic Setup") {
	LBFGSBCapsuleFitterSolver *solver = memnew(LBFGSBCapsuleFitterSolver);
	Ref<ArrayMesh> mesh = create_sphere_test_mesh(Vector3(0, 0, 0), 1.0);
	solver->set_source_mesh(mesh);
	REQUIRE(solver->get_source_mesh() == mesh);
	solver->set_surface_index(0);
	REQUIRE(solver->get_surface_index() == 0);
	solver->set_huber_delta(0.1);
	REQUIRE(solver->get_huber_delta() == doctest::Approx(0.1));
}

TEST_CASE("[SceneTree][LBFGSBCapsuleFitterSolver] Optimize All Capsule Parameters - Zero Capsules") {
	LBFGSBCapsuleFitterSolver *solver = memnew(LBFGSBCapsuleFitterSolver);
	Ref<ArrayMesh> mesh = create_sphere_test_mesh(Vector3(1, 1, 1), 0.5);
	solver->set_source_mesh(mesh);
	solver->set_surface_index(0);

	solver->clear_capsule_instances();
	REQUIRE(solver->get_num_capsule_instances() == 0);

	Dictionary result = solver->optimize_all_capsule_parameters();

	CHECK_FALSE(result.has("error"));
	REQUIRE(result.has("message"));
	CHECK(String(result["message"]).contains("No capsule instances defined"));
	REQUIRE(result.has("optimized_capsules_results"));
	CHECK(Array(result["optimized_capsules_results"]).is_empty());
	REQUIRE(result.has("final_fx"));
	CHECK(double(result["final_fx"]) == doctest::Approx(0.0));
	REQUIRE(result.has("iterations"));
	CHECK(int(result["iterations"]) == 0);
}

TEST_CASE("[SceneTree][LBFGSBCapsuleFitterSolver] Optimize All Capsule Parameters - Single Capsule Fit to Sphere") {
	LBFGSBCapsuleFitterSolver *solver = memnew(LBFGSBCapsuleFitterSolver);
	Vector3 mesh_center = Vector3(2, 3, 4);
	double mesh_radius = 0.75;
	Ref<ArrayMesh> mesh = create_sphere_test_mesh(mesh_center, mesh_radius);
	solver->set_source_mesh(mesh);
	solver->set_surface_index(0);
	solver->clear_capsule_instances();
	solver->add_capsule_instance(mesh_center + Vector3(0.1, -0.1, 0.1), mesh_center + Vector3(0.2, -0.2, 0.2), mesh_radius * 0.5);
	REQUIRE(solver->get_num_capsule_instances() == 1);
	Dictionary result = solver->optimize_all_capsule_parameters();
	INFO("Solver result (single capsule): ", JSON::stringify(result));
	CHECK_FALSE(result.has("error"));
	REQUIRE(result.has("optimized_capsules_results"));
	Array capsule_results = result.get("optimized_capsules_results", Array());
	CHECK(capsule_results.size() == 1);
	if (capsule_results.size() == 1) {
		Dictionary capsule_data = capsule_results[0]; // Access is now conditional
		REQUIRE(capsule_data.has("optimized_radius"));
		REQUIRE(capsule_data.has("optimized_axis_a"));
		REQUIRE(capsule_data.has("optimized_axis_b"));
		double optimized_radius = capsule_data["optimized_radius"];
		Vector3 optimized_axis_a = capsule_data["optimized_axis_a"];
		Vector3 optimized_axis_b = capsule_data["optimized_axis_b"];
		CHECK(optimized_radius == doctest::Approx(mesh_radius).epsilon(0.15)); // Allow some tolerance
		Vector3 optimized_center = (optimized_axis_a + optimized_axis_b) / 2.0;
		CHECK(optimized_center.distance_to(mesh_center) == doctest::Approx(0.0).epsilon(0.15));
		double optimized_height = optimized_axis_a.distance_to(optimized_axis_b);
		CHECK(optimized_height == doctest::Approx(0.0).epsilon(0.2)); // For a sphere, height should be minimal
	} else {
		FAIL_CHECK("Optimized capsule results size was expected to be 1, but got " << capsule_results.size() << ".");
	}
}

TEST_CASE("[SceneTree][LBFGSBCapsuleFitterSolver] Optimize Parameters for Multiple Capsules Simultaneously") {
	LBFGSBCapsuleFitterSolver *solver = memnew(LBFGSBCapsuleFitterSolver);

	Ref<ArrayMesh> mesh = create_sphere_test_mesh(Vector3(0, 0, 0), 1.0); // Simple sphere target
	solver->set_source_mesh(mesh);
	solver->set_surface_index(0);

	solver->clear_capsule_instances();
	solver->add_capsule_instance(Vector3(-0.5, 0.1, 0), Vector3(0.5, -0.1, 0), 0.2); // Capsule 1 - initial guess
	solver->add_capsule_instance(Vector3(0.1, -0.5, 0), Vector3(-0.1, 0.5, 0), 0.25); // Capsule 2 - another initial guess
	REQUIRE(solver->get_num_capsule_instances() == 2);

	Dictionary result = solver->optimize_all_capsule_parameters();
	INFO("Solver result (multi-capsule): ", JSON::stringify(result));

	CHECK_FALSE(result.has("error"));
	REQUIRE(result.has("optimized_capsules_results"));
	Array capsule_results_array = result["optimized_capsules_results"];
	REQUIRE(capsule_results_array.size() == 2);

	for (int i = 0; i < capsule_results_array.size(); ++i) {
		Dictionary capsule_data = capsule_results_array[i];
		REQUIRE(capsule_data.has("optimized_radius"));
		REQUIRE(capsule_data.has("optimized_axis_a"));
		REQUIRE(capsule_data.has("optimized_axis_b"));

		double opt_radius = capsule_data["optimized_radius"];
		Vector3 opt_axis_a = capsule_data["optimized_axis_a"];
		Vector3 opt_axis_b = capsule_data["optimized_axis_b"];

		CHECK_FALSE(Math::is_nan(opt_radius));
		CHECK_FALSE(!opt_axis_a.is_finite());
		CHECK_FALSE(!opt_axis_b.is_finite());
		CHECK(opt_radius > 0.0);
	}

	Dictionary cap0_internal = solver->get_capsule_instance_data(0);
	REQUIRE(cap0_internal.has("optimized_radius"));
	CHECK(double(cap0_internal["optimized_radius"]) > 0.0);

	Dictionary cap1_internal = solver->get_capsule_instance_data(1);
	REQUIRE(cap1_internal.has("optimized_radius"));
	CHECK(double(cap1_internal["optimized_radius"]) > 0.0);
}

} // namespace TestLBFGSBCapsuleFitterSolver
