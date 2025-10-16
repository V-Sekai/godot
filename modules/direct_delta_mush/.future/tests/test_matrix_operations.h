/**************************************************************************/
/*  test_matrix_operations.h                                              */
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

#include "../ddm_math.h"

namespace DDMTests {

TEST_SUITE("[Modules][DirectDeltaMush][MatrixOperations]") {
	TEST_CASE("[MatrixOperations] QR decomposition - identity matrix") {
		using namespace DDMMath;

		Matrix3 A; // Identity by default
		Matrix3 Q, R;

		qr_decomposition_3x3(A, Q, R);

		CHECK_MESSAGE(Math::is_equal_approx(Q.determinant(), 1.0f), "Q should be orthogonal with det=1");
		CHECK_MESSAGE(Math::is_equal_approx(R.m[0][0], 1.0f), "R[0][0] should be 1");
	}

	TEST_CASE("[MatrixOperations] QR decomposition - simple rotation") {
		using namespace DDMMath;

		// 45 degree rotation around Z-axis
		float angle = Math_PI / 4.0f;
		float c = std::cos(angle);
		float s = std::sin(angle);

		Matrix3 A;
		A.m[0][0] = c;
		A.m[0][1] = -s;
		A.m[0][2] = 0;
		A.m[1][0] = s;
		A.m[1][1] = c;
		A.m[1][2] = 0;
		A.m[2][0] = 0;
		A.m[2][1] = 0;
		A.m[2][2] = 1;

		Matrix3 Q, R;
		qr_decomposition_3x3(A, Q, R);

		// Q should equal A (since A is already orthogonal)
		// R should be close to identity
		CHECK_MESSAGE(Math::is_equal_approx(Q.determinant(), 1.0f), "Q should have determinant 1");
		CHECK_MESSAGE(Math::is_equal_approx(R.m[0][0], 1.0f, 0.01f), "R should be close to identity");
	}

	TEST_CASE("[MatrixOperations] Matrix decomposition - scale only") {
		using namespace DDMMath;

		Matrix3 M;
		M.m[0][0] = 2.0f;
		M.m[0][1] = 0.0f;
		M.m[0][2] = 0.0f;
		M.m[1][0] = 0.0f;
		M.m[1][1] = 3.0f;
		M.m[1][2] = 0.0f;
		M.m[2][0] = 0.0f;
		M.m[2][1] = 0.0f;
		M.m[2][2] = 4.0f;

		Matrix3 M_rigid, M_scale;
		matrix_decomposition(M, M_rigid, M_scale);

		CHECK_MESSAGE(Math::is_equal_approx(M_rigid.determinant(), 1.0f), "Rigid part should be orthogonal");
		CHECK_MESSAGE(Math::is_equal_approx(M_scale.m[0][0], 2.0f), "Scale X should be 2");
		CHECK_MESSAGE(Math::is_equal_approx(M_scale.m[1][1], 3.0f), "Scale Y should be 3");
		CHECK_MESSAGE(Math::is_equal_approx(M_scale.m[2][2], 4.0f), "Scale Z should be 4");
	}

	TEST_CASE("[MatrixOperations] Matrix decomposition - rotation + scale") {
		using namespace DDMMath;

		// Create rotation matrix
		float angle = Math_PI / 6.0f; // 30 degrees
		float c = std::cos(angle);
		float s = std::sin(angle);

		Matrix3 R;
		R.m[0][0] = c;
		R.m[0][1] = -s;
		R.m[0][2] = 0;
		R.m[1][0] = s;
		R.m[1][1] = c;
		R.m[1][2] = 0;
		R.m[2][0] = 0;
		R.m[2][1] = 0;
		R.m[2][2] = 1;

		// Add scale
		Matrix3 S;
		S.m[0][0] = 2.0f;
		S.m[1][1] = 2.0f;
		S.m[2][2] = 2.0f;

		// M = R * S
		Matrix3 M;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				M.m[i][j] = 0.0f;
				for (int k = 0; k < 3; k++) {
					M.m[i][j] += R.m[i][k] * S.m[k][j];
				}
			}
		}

		Matrix3 M_rigid, M_scale;
		matrix_decomposition(M, M_rigid, M_scale);

		CHECK_MESSAGE(Math::is_equal_approx(M_rigid.determinant(), 1.0f, 0.01f),
				"Rigid part should be orthogonal");
		CHECK_MESSAGE(validate_matrix_decomposition(M, M_rigid, M_scale, 0.01f),
				"Decomposition should reconstruct original");
	}

	TEST_CASE("[MatrixOperations] Polar decomposition - identity") {
		using namespace DDMMath;

		Matrix3 M; // Identity
		Matrix3 R = polar_decomposition(M);

		CHECK_MESSAGE(Math::is_equal_approx(R.determinant(), 1.0f), "R should have determinant 1");
		CHECK_MESSAGE(validate_polar_decomposition(R, 0.01f), "R should be orthogonal");
	}

	TEST_CASE("[MatrixOperations] Polar decomposition - rotation") {
		using namespace DDMMath;

		// Pure rotation matrix
		float angle = Math_PI / 3.0f; // 60 degrees
		float c = std::cos(angle);
		float s = std::sin(angle);

		Matrix3 M;
		M.m[0][0] = c;
		M.m[0][1] = -s;
		M.m[0][2] = 0;
		M.m[1][0] = s;
		M.m[1][1] = c;
		M.m[1][2] = 0;
		M.m[2][0] = 0;
		M.m[2][1] = 0;
		M.m[2][2] = 1;

		Matrix3 R = polar_decomposition(M);

		// R should equal M (since M is already a rotation)
		CHECK_MESSAGE(Math::is_equal_approx(R.determinant(), 1.0f, 0.01f), "R determinant should be 1");
		CHECK_MESSAGE(validate_polar_decomposition(R, 0.01f), "R should be orthogonal");
	}

	TEST_CASE("[MatrixOperations] Polar decomposition - rotation + scale") {
		using namespace DDMMath;

		// Rotation + uniform scale
		float angle = Math_PI / 4.0f;
		float c = std::cos(angle);
		float s = std::sin(angle);
		float scale = 2.0f;

		Matrix3 M;
		M.m[0][0] = c * scale;
		M.m[0][1] = -s * scale;
		M.m[0][2] = 0;
		M.m[1][0] = s * scale;
		M.m[1][1] = c * scale;
		M.m[1][2] = 0;
		M.m[2][0] = 0;
		M.m[2][1] = 0;
		M.m[2][2] = scale;

		Matrix3 R = polar_decomposition(M);

		CHECK_MESSAGE(Math::is_equal_approx(R.determinant(), 1.0f, 0.01f), "R determinant should be 1");
		CHECK_MESSAGE(validate_polar_decomposition(R, 0.01f), "R should be orthogonal");

		// R should extract the rotation part
		CHECK_MESSAGE(Math::is_equal_approx(R.m[0][0], c, 0.01f), "Rotation should be preserved");
	}

	TEST_CASE("[MatrixOperations] Reflection correction") {
		using namespace DDMMath;

		// Matrix with negative determinant (reflection)
		Matrix3 M;
		M.m[0][0] = -1;
		M.m[0][1] = 0;
		M.m[0][2] = 0;
		M.m[1][0] = 0;
		M.m[1][1] = 1;
		M.m[1][2] = 0;
		M.m[2][0] = 0;
		M.m[2][1] = 0;
		M.m[2][2] = 1;

		correct_reflection(M);

		CHECK_MESSAGE(M.determinant() > 0.0f, "Determinant should be positive after correction");
	}

	TEST_CASE("[MatrixOperations] Reorthogonalization") {
		using namespace DDMMath;

		// Slightly non-orthogonal matrix
		Matrix3 M;
		M.m[0][0] = 1.0f;
		M.m[0][1] = 0.01f;
		M.m[0][2] = 0.0f;
		M.m[1][0] = 0.01f;
		M.m[1][1] = 1.0f;
		M.m[1][2] = 0.0f;
		M.m[2][0] = 0.0f;
		M.m[2][1] = 0.0f;
		M.m[2][2] = 1.0f;

		reorthogonalize(M);

		// Check orthogonality: dot products of columns should be ~0
		Vector3 c0 = M.col(0);
		Vector3 c1 = M.col(1);
		Vector3 c2 = M.col(2);

		CHECK_MESSAGE(Math::is_zero_approx(c0.dot(c1)), "Columns 0 and 1 should be orthogonal");
		CHECK_MESSAGE(Math::is_zero_approx(c0.dot(c2)), "Columns 0 and 2 should be orthogonal");
		CHECK_MESSAGE(Math::is_zero_approx(c1.dot(c2)), "Columns 1 and 2 should be orthogonal");
	}

	TEST_CASE("[MatrixOperations] Non-rigid displacement") {
		using namespace DDMMath;

		Vector3 rest_vertex(1.0f, 0.0f, 0.0f);

		// Scale matrix (2x in all directions)
		Matrix3 M_scale;
		M_scale.m[0][0] = 2.0f;
		M_scale.m[1][1] = 2.0f;
		M_scale.m[2][2] = 2.0f;

		Vector3 displacement = compute_displacement(rest_vertex, M_scale);

		// Displacement should be: (2.0 * vertex) - vertex = vertex
		CHECK_MESSAGE(Math::is_equal_approx(displacement.x, 1.0f), "Displacement X should be 1");
		CHECK_MESSAGE(Math::is_equal_approx(displacement.y, 0.0f), "Displacement Y should be 0");
		CHECK_MESSAGE(Math::is_equal_approx(displacement.z, 0.0f), "Displacement Z should be 0");
	}

	TEST_CASE("[MatrixOperations] Cotangent weight computation") {
		using namespace DDMMath;

		// 90 degree angles
		float angle_90 = Math_PI / 2.0f;
		float weight = compute_cotangent_weight_cpu(angle_90, angle_90);

		// cot(90°) = 0, so weight should be near 0
		CHECK_MESSAGE(Math::is_zero_approx(weight), "Cotangent of 90° should be ~0");
	}

	TEST_CASE("[MatrixOperations] Angle computation") {
		using namespace DDMMath;

		Vector3 v0(1, 0, 0);
		Vector3 v1(0, 0, 0); // Vertex at origin
		Vector3 v2(0, 1, 0);

		float angle = compute_angle(v0, v1, v2);

		// Should be 90 degrees
		CHECK_MESSAGE(Math::is_equal_approx(angle, Math_PI / 2.0f, 0.01f),
				"Angle should be 90 degrees");
	}

	TEST_CASE("[MatrixOperations] Validation - decomposition correctness") {
		using namespace DDMMath;

		Matrix3 M;
		M.m[0][0] = 2.0f;
		M.m[0][1] = 0.0f;
		M.m[0][2] = 0.0f;
		M.m[1][0] = 0.0f;
		M.m[1][1] = 3.0f;
		M.m[1][2] = 0.0f;
		M.m[2][0] = 0.0f;
		M.m[2][1] = 0.0f;
		M.m[2][2] = 4.0f;

		Matrix3 Q, R;
		qr_decomposition_3x3(M, Q, R);

		bool valid = validate_matrix_decomposition(M, Q, R, 0.01f);
		CHECK_MESSAGE(valid, "QR decomposition should be valid");
	}

	TEST_CASE("[MatrixOperations] Validation - polar decomposition correctness") {
		using namespace DDMMath;

		Matrix3 M;
		M.m[0][0] = 1.0f;
		M.m[0][1] = 0.0f;
		M.m[0][2] = 0.0f;
		M.m[1][0] = 0.0f;
		M.m[1][1] = 1.0f;
		M.m[1][2] = 0.0f;
		M.m[2][0] = 0.0f;
		M.m[2][1] = 0.0f;
		M.m[2][2] = 1.0f;

		Matrix3 R = polar_decomposition(M);

		bool valid = validate_polar_decomposition(R, 0.01f);
		CHECK_MESSAGE(valid, "Polar decomposition should produce valid rotation");
	}
}

} // namespace DDMTests
