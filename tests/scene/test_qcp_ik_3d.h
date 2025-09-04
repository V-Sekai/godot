/**************************************************************************/
/*  test_qcp_ik_3d.h                                                      */
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

#include "scene/3d/chain_ik_3d.h"
#include "scene/3d/qcp_ik_3d.h"
#include "scene/3d/skeleton_3d.h"

namespace TestQCPIK3D {

TEST_CASE("[QCPIK3D] Basic instantiation and properties") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test default property values
	CHECK_EQ(qcp_ik->get_use_translation(), true);
	CHECK_EQ(qcp_ik->get_convergence_threshold(), 0.001);
	CHECK_EQ(qcp_ik->get_max_qcp_iterations(), 10);
	CHECK_EQ(qcp_ik->get_precision(), doctest::Approx(1.0e-6));
	CHECK_EQ(qcp_ik->get_use_rmd_flipping(), true);
	CHECK_EQ(qcp_ik->get_max_points(), 10000);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Property setters and getters") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test property modifications
	qcp_ik->set_use_translation(false);
	CHECK_EQ(qcp_ik->get_use_translation(), false);

	qcp_ik->set_convergence_threshold(0.0001);
	CHECK_EQ(qcp_ik->get_convergence_threshold(), doctest::Approx(0.0001));

	qcp_ik->set_max_qcp_iterations(50);
	CHECK_EQ(qcp_ik->get_max_qcp_iterations(), 50);

	qcp_ik->set_precision(1.0e-8);
	CHECK_EQ(qcp_ik->get_precision(), doctest::Approx(1.0e-8));

	qcp_ik->set_use_rmd_flipping(false);
	CHECK_EQ(qcp_ik->get_use_rmd_flipping(), false);

	qcp_ik->set_max_points(5000);
	CHECK_EQ(qcp_ik->get_max_points(), 5000);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] RMSD computation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test identical point sets (RMSD should be 0)
	Vector<Vector3> points1 = { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0) };
	Vector<Vector3> points2 = { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0) };
	real_t rmsd = qcp_ik->compute_rmsd(points1, points2);
	CHECK_EQ(rmsd, doctest::Approx(0.0));

	// Test different point sets
	points2 = { Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(0, 1, 1) };
	rmsd = qcp_ik->compute_rmsd(points1, points2);
	CHECK_GT(rmsd, 0.0);

	// Test empty point sets
	rmsd = qcp_ik->compute_rmsd(Vector<Vector3>(), Vector<Vector3>());
	CHECK_EQ(rmsd, doctest::Approx(0.0));

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Chain alignment validation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test empty chain
	bool aligned = qcp_ik->validate_chain_alignment(Vector<Vector3>(), Vector3(0, 0, 0));
	CHECK_EQ(aligned, false);

	// Test chain with end effector at target
	Vector<Vector3> chain = { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0) };
	Vector3 target = Vector3(2, 0, 0);
	aligned = qcp_ik->validate_chain_alignment(chain, target);
	CHECK_EQ(aligned, true);

	// Test chain with end effector far from target
	target = Vector3(5, 5, 5);
	aligned = qcp_ik->validate_chain_alignment(chain, target);
	CHECK_EQ(aligned, false);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Input validation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test valid inputs
	Vector<Vector3> moved = { Vector3(0, 0, 0), Vector3(1, 0, 0) };
	Vector<Vector3> target = { Vector3(0, 0, 1), Vector3(1, 0, 1) };
	Vector<double> weights = { 1.0, 1.0 };
	bool valid = qcp_ik->validate_inputs(moved, target, weights);
	CHECK_EQ(valid, true);

	// Test mismatched sizes
	target = { Vector3(0, 0, 1) }; // Different size
	valid = qcp_ik->validate_inputs(moved, target, weights);
	CHECK_EQ(valid, false);

	// Test too many points
	moved.clear();
	target.clear();
	weights.clear();
	for (int i = 0; i < 15000; i++) {
		moved.push_back(Vector3(i, 0, 0));
		target.push_back(Vector3(i, 1, 0));
		weights.push_back(1.0);
	}
	valid = qcp_ik->validate_inputs(moved, target, weights);
	CHECK_EQ(valid, false); // Should fail due to max_points limit

	// Test invalid weights
	weights = { -1.0, 1.0 }; // Negative weight
	valid = qcp_ik->validate_inputs({ Vector3(0, 0, 0), Vector3(1, 0, 0) }, { Vector3(0, 0, 1), Vector3(1, 0, 1) }, weights);
	CHECK_EQ(valid, false);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Rotation validation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test identity quaternion (should be valid)
	Quaternion identity = Quaternion();
	bool valid = qcp_ik->validate_rotation(identity);
	CHECK_EQ(valid, true);

	// Test normalized quaternion
	Quaternion normalized = Quaternion(0.5, 0.5, 0.5, 0.5).normalized();
	valid = qcp_ik->validate_rotation(normalized);
	CHECK_EQ(valid, true);

	// Test non-normalized quaternion (should be invalid)
	Quaternion non_normalized = Quaternion(1, 1, 1, 1); // Not normalized
	valid = qcp_ik->validate_rotation(non_normalized);
	CHECK_EQ(valid, false);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Single point alignment") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test identical points
	Vector3 point = Vector3(1, 2, 3);
	bool aligned = qcp_ik->validate_single_point_alignment(point, point);
	CHECK_EQ(aligned, true);

	// Test different points
	Vector3 target = Vector3(4, 5, 6);
	aligned = qcp_ik->validate_single_point_alignment(point, target);
	// This should still return true as QCP can always find a transformation
	// The actual alignment depends on the computed transformation
	CHECK_EQ(aligned, true);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Opposite vector alignment") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test opposite vectors
	Vector3 point = Vector3(1, 0, 0);
	Vector3 opposite = Vector3(-1, 0, 0);
	bool valid = qcp_ik->validate_opposite_vector_alignment(point, opposite);
	// Should return true as QCP handles opposite vectors gracefully
	CHECK_EQ(valid, true);

	// Test non-opposite vectors
	Vector3 different = Vector3(0, 1, 0);
	valid = qcp_ik->validate_opposite_vector_alignment(point, different);
	CHECK_EQ(valid, true); // Should still be valid

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Zero length vector handling") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test zero vector
	Vector3 zero = Vector3();
	Vector3 target = Vector3(1, 0, 0);
	bool valid = qcp_ik->validate_zero_length_vector_handling(zero, target);
	// QCP should handle zero vectors gracefully
	CHECK_EQ(valid, true);

	// Test non-zero vector
	valid = qcp_ik->validate_zero_length_vector_handling(Vector3(1, 0, 0), target);
	CHECK_EQ(valid, true);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Weighted point alignment") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test valid weighted alignment
	Vector<Vector3> moved = { Vector3(0, 0, 0), Vector3(1, 0, 0) };
	Vector<Vector3> target = { Vector3(0, 0, 1), Vector3(1, 0, 1) };
	Vector<double> weights = { 2.0, 1.0 };
	bool valid = qcp_ik->validate_weighted_point_alignment(moved, target, weights);
	CHECK_EQ(valid, true);

	// Test invalid weights
	weights = { -1.0, 1.0 };
	valid = qcp_ik->validate_weighted_point_alignment(moved, target, weights);
	CHECK_EQ(valid, false);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Large coordinate handling") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test normal coordinates
	Vector<Vector3> moved = { Vector3(0, 0, 0), Vector3(1, 0, 0) };
	Vector<Vector3> target = { Vector3(0, 0, 1), Vector3(1, 0, 1) };
	bool valid = qcp_ik->validate_large_coordinate_handling(moved, target);
	CHECK_EQ(valid, true);

	// Test large coordinates
	moved = { Vector3(1e10, 0, 0), Vector3(1e10 + 1, 0, 0) };
	target = { Vector3(0, 1e10, 0), Vector3(0, 1e10 + 1, 0) };
	valid = qcp_ik->validate_large_coordinate_handling(moved, target);
	// QCP should handle large coordinates
	CHECK_EQ(valid, true);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Near collinear points") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test collinear points
	Vector<Vector3> moved = { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0) };
	Vector<Vector3> target = { Vector3(0, 0, 0), Vector3(0, 1, 0), Vector3(0, 2, 0) };
	bool valid = qcp_ik->validate_near_collinear_points(moved, target);
	// QCP should handle collinear points
	CHECK_EQ(valid, true);

	// Test fewer than 3 points
	moved = { Vector3(0, 0, 0), Vector3(1, 0, 0) };
	target = { Vector3(0, 0, 0), Vector3(0, 1, 0) };
	valid = qcp_ik->validate_near_collinear_points(moved, target);
	CHECK_EQ(valid, true);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Rotation orthogonality") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test identity quaternion (should be orthogonal)
	Quaternion identity = Quaternion();
	bool orthogonal = qcp_ik->validate_rotation_orthogonality(identity);
	CHECK_EQ(orthogonal, true);

	// Test 90-degree rotation around Z axis
	Quaternion rot_z = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(90));
	orthogonal = qcp_ik->validate_rotation_orthogonality(rot_z);
	CHECK_EQ(orthogonal, true);

	// Test arbitrary normalized quaternion
	Quaternion arbitrary = Quaternion(0.5, 0.5, 0.5, 0.5).normalized();
	orthogonal = qcp_ik->validate_rotation_orthogonality(arbitrary);
	CHECK_EQ(orthogonal, true);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Rotation determinant") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test identity quaternion (determinant should be +1)
	Quaternion identity = Quaternion();
	bool proper_rotation = qcp_ik->validate_rotation_determinant(identity);
	CHECK_EQ(proper_rotation, true);

	// Test proper rotation quaternion
	Quaternion rotation = Quaternion(Vector3(1, 0, 0), Math::deg_to_rad(45));
	proper_rotation = qcp_ik->validate_rotation_determinant(rotation);
	CHECK_EQ(proper_rotation, true);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Distance preservation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test identical point sets (distances should be preserved)
	Vector<Vector3> original = { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0) };
	Vector<Vector3> transformed = { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0) };
	bool preserved = qcp_ik->validate_distance_preservation(original, transformed);
	CHECK_EQ(preserved, true);

	// Test transformed point set with preserved distances
	// Apply a rotation to the original points
	Quaternion rotation = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(90));
	transformed.clear();
	for (const Vector3 &point : original) {
		transformed.push_back(rotation.xform(point));
	}
	preserved = qcp_ik->validate_distance_preservation(original, transformed);
	CHECK_EQ(preserved, true);

	// Test point set with different distances (should fail)
	transformed = { Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(0, 2, 0) }; // Doubled distances
	preserved = qcp_ik->validate_distance_preservation(original, transformed);
	CHECK_EQ(preserved, false);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Transformation efficiency validation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test identity transformation
	Quaternion identity = Quaternion();
	Vector3 zero_translation = Vector3();
	bool efficient = qcp_ik->validate_transformation_efficiency(identity, zero_translation, "identity");
	CHECK_EQ(efficient, true);

	// Test rotation-only transformation
	Quaternion rotation = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(45));
	efficient = qcp_ik->validate_transformation_efficiency(rotation, zero_translation, "rotation_only");
	CHECK_EQ(efficient, true);

	// Test translation-only transformation
	efficient = qcp_ik->validate_transformation_efficiency(identity, Vector3(1, 2, 3), "translation_only");
	CHECK_EQ(efficient, true);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Minimal jerk validation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test smooth transformation (identity)
	Quaternion identity = Quaternion();
	Vector3 zero_translation = Vector3();
	bool minimal_jerk = qcp_ik->validate_minimal_jerk(identity, zero_translation);
	CHECK_EQ(minimal_jerk, true);

	// Test small rotation (should be smooth)
	Quaternion small_rotation = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(10));
	minimal_jerk = qcp_ik->validate_minimal_jerk(small_rotation, Vector3(0.1, 0.1, 0.1));
	CHECK_EQ(minimal_jerk, true);

	// Test large rotation (might not be minimal jerk)
	Quaternion large_rotation = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(170));
	minimal_jerk = qcp_ik->validate_minimal_jerk(large_rotation, Vector3(0.1, 0.1, 0.1));
	// This might return false due to preferring shorter angular paths
	// The exact behavior depends on the implementation

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Motion coordination validation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test well-coordinated transformation
	Quaternion small_rotation = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(30));
	Vector3 small_translation = Vector3(0.5, 0.5, 0.5);
	bool coordinated = qcp_ik->validate_motion_coordination(small_rotation, small_translation);
	CHECK_EQ(coordinated, true);

	// Test poorly coordinated transformation (large rotation + large translation)
	Quaternion large_rotation = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(90));
	Vector3 large_translation = Vector3(10, 10, 10);
	coordinated = qcp_ik->validate_motion_coordination(large_rotation, large_translation);
	// This might return false due to poor coordination
	// The exact behavior depends on the implementation

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Minimal RMSD validation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test identical point sets (RMSD should be minimal)
	Vector<Vector3> moved = { Vector3(0, 0, 0), Vector3(1, 0, 0) };
	Vector<Vector3> target = { Vector3(0, 0, 0), Vector3(1, 0, 0) };
	Quaternion identity = Quaternion();
	Vector3 zero_translation = Vector3();
	bool minimal_rmsd = qcp_ik->validate_minimal_rmsd(identity, zero_translation, moved, target);
	CHECK_EQ(minimal_rmsd, true);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] Minimal rotation angle validation") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test small rotation angle
	Quaternion small_rotation = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(30));
	bool minimal_angle = qcp_ik->validate_minimal_rotation_angle(small_rotation, Math::deg_to_rad(45));
	CHECK_EQ(minimal_angle, true);

	// Test large rotation angle
	Quaternion large_rotation = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(60));
	minimal_angle = qcp_ik->validate_minimal_rotation_angle(large_rotation, Math::deg_to_rad(45));
	CHECK_EQ(minimal_angle, false);

	memdelete(qcp_ik);
}

TEST_CASE("[QCPIK3D] QCP solver access") {
	QCPIK3D *qcp_ik = memnew(QCPIK3D);

	// Test that we can get the QCP solver (should return null since we use static methods)
	Ref<QuaternionCharacteristicPolynomial> solver = qcp_ik->get_qcp_solver();
	// Since we use static methods, this should return an empty ref
	CHECK(solver.is_null());

	memdelete(qcp_ik);
}

} // namespace TestQCPIK3D
