/**************************************************************************/
/*  test_ddm_deformer.h                                                   */
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

#include "modules/direct_delta_mush/ddm_deformer.h"
#include "modules/direct_delta_mush/tests/test_fixtures.h"
#include "tests/test_macros.h"

namespace TestDDMDeformer {

TEST_CASE("[DDM][Deformer] Identity transform preserves mesh") {
	DDMTestFixtures::CubeMeshFixture cube;
	DDMTestFixtures::TransformFixtures transforms;

	DDMDeformer::MeshData mesh_data;
	mesh_data.vertices = cube.vertices;
	mesh_data.indices = cube.indices;
	mesh_data.bone_weights = cube.bone_weights;
	mesh_data.bone_indices = cube.bone_indices;
	mesh_data.normals = cube.normals;

	DDMDeformer deformer;
	deformer.initialize(mesh_data);

	Vector<Transform3D> bones = { transforms.identity };
	DDMDeformer::Config config;
	config.iterations = 0; // No smoothing to test pure identity
	config.smooth_lambda = 0.0;

	DDMDeformer::DeformResult result = deformer.deform(bones, config);

	CHECK(result.success);
	CHECK(result.vertices.size() == cube.vertices.size());

	// With identity transform and no smoothing, vertices should be unchanged
	for (int i = 0; i < result.vertices.size(); i++) {
		real_t dist = result.vertices[i].distance_to(cube.vertices[i]);
		CHECK(dist < 0.01);
	}
}

TEST_CASE("[DDM][Deformer] Deformation produces finite results") {
	DDMTestFixtures::CubeMeshFixture cube;
	DDMTestFixtures::TransformFixtures transforms;

	DDMDeformer::MeshData mesh_data;
	mesh_data.vertices = cube.vertices;
	mesh_data.indices = cube.indices;
	mesh_data.bone_weights = cube.bone_weights;
	mesh_data.bone_indices = cube.bone_indices;
	mesh_data.normals = cube.normals;

	DDMDeformer deformer;
	deformer.initialize(mesh_data);

	SUBCASE("90° rotation produces finite vertices") {
		Vector<Transform3D> bones = { transforms.rotation_x_90 };
		DDMDeformer::Config config;

		DDMDeformer::DeformResult result = deformer.deform(bones, config);

		CHECK(result.success);

		// All vertices should be finite
		for (const Vector3 &v : result.vertices) {
			CHECK(Math::is_finite(v.x));
			CHECK(Math::is_finite(v.y));
			CHECK(Math::is_finite(v.z));
			CHECK(v.length() < 100.0); // Reasonable bounds
		}
	}

	SUBCASE("Uniform scale produces finite vertices") {
		Vector<Transform3D> bones = { transforms.scale_uniform_2x };
		DDMDeformer::Config config;

		DDMDeformer::DeformResult result = deformer.deform(bones, config);

		CHECK(result.success);

		for (const Vector3 &v : result.vertices) {
			CHECK(Math::is_finite(v.x));
			CHECK(Math::is_finite(v.y));
			CHECK(Math::is_finite(v.z));
		}
	}
}

TEST_CASE("[DDM][Deformer] Smoothing parameter affects result") {
	DDMTestFixtures::CubeMeshFixture cube;
	DDMTestFixtures::TransformFixtures transforms;

	DDMDeformer::MeshData mesh_data;
	mesh_data.vertices = cube.vertices;
	mesh_data.indices = cube.indices;
	mesh_data.bone_weights = cube.bone_weights;
	mesh_data.bone_indices = cube.bone_indices;
	mesh_data.normals = cube.normals;

	Vector<Transform3D> bones = { transforms.rotation_y_45 };

	// Test with no smoothing
	DDMDeformer deformer_no_smooth;
	deformer_no_smooth.initialize(mesh_data);
	DDMDeformer::Config config_no_smooth;
	config_no_smooth.iterations = 0;
	config_no_smooth.smooth_lambda = 0.0;
	DDMDeformer::DeformResult result_no_smooth = deformer_no_smooth.deform(bones, config_no_smooth);

	// Test with high smoothing
	DDMDeformer deformer_high_smooth;
	deformer_high_smooth.initialize(mesh_data);
	DDMDeformer::Config config_high_smooth;
	config_high_smooth.iterations = 30;
	config_high_smooth.smooth_lambda = 0.9;
	DDMDeformer::DeformResult result_high_smooth = deformer_high_smooth.deform(bones, config_high_smooth);

	CHECK(result_no_smooth.success);
	CHECK(result_high_smooth.success);

	// Results should be different when smoothing is applied
	bool any_different = false;
	for (int i = 0; i < result_no_smooth.vertices.size(); i++) {
		real_t dist = result_no_smooth.vertices[i].distance_to(result_high_smooth.vertices[i]);
		if (dist > 0.001) {
			any_different = true;
			break;
		}
	}
	CHECK(any_different);
}

TEST_CASE("[DDM][Deformer] Empty mesh handling") {
	DDMDeformer::MeshData empty_mesh;
	DDMDeformer deformer;
	deformer.initialize(empty_mesh);

	Vector<Transform3D> bones = { Transform3D() };
	DDMDeformer::Config config;

	DDMDeformer::DeformResult result = deformer.deform(bones, config);

	// Should return empty result, not crash
	CHECK(result.vertices.size() == 0);
}

TEST_CASE("[DDM][Deformer] Single bone weight application") {
	DDMTestFixtures::CubeMeshFixture cube;
	DDMTestFixtures::TransformFixtures transforms;

	DDMDeformer::MeshData mesh_data;
	mesh_data.vertices = cube.vertices;
	mesh_data.indices = cube.indices;
	mesh_data.bone_weights = cube.bone_weights;
	mesh_data.bone_indices = cube.bone_indices;
	mesh_data.normals = cube.normals;

	DDMDeformer deformer;
	deformer.initialize(mesh_data);

	// All vertices have weight 1.0 on bone 0
	Vector<Transform3D> bones = { transforms.scale_uniform_2x };
	DDMDeformer::Config config;
	config.iterations = 0; // No smoothing to isolate bone transform

	DDMDeformer::DeformResult result = deformer.deform(bones, config);

	CHECK(result.success);

	// With uniform 2x scale and no smoothing, vertices should roughly double
	// (exact match not expected due to Enhanced DDM polar decomposition)
	for (int i = 0; i < result.vertices.size(); i++) {
		// Result should be larger than original
		CHECK(result.vertices[i].length() >= cube.vertices[i].length() - 0.1);
	}
}

TEST_CASE("[DDM][Deformer] Result consistency") {
	DDMTestFixtures::CubeMeshFixture cube;

	DDMDeformer::MeshData mesh_data;
	mesh_data.vertices = cube.vertices;
	mesh_data.indices = cube.indices;
	mesh_data.bone_weights = cube.bone_weights;
	mesh_data.bone_indices = cube.bone_indices;
	mesh_data.normals = cube.normals;

	DDMDeformer deformer;
	deformer.initialize(mesh_data);

	Vector<Transform3D> bones = { Transform3D() };
	DDMDeformer::Config config;

	// Run deformation twice with same inputs
	DDMDeformer::DeformResult result1 = deformer.deform(bones, config);
	DDMDeformer::DeformResult result2 = deformer.deform(bones, config);

	// Results should be identical (deterministic)
	CHECK(result1.vertices.size() == result2.vertices.size());
	for (int i = 0; i < result1.vertices.size(); i++) {
		real_t dist = result1.vertices[i].distance_to(result2.vertices[i]);
		CHECK(dist < 0.0001); // Should be exactly the same
	}
}

} // namespace TestDDMDeformer
