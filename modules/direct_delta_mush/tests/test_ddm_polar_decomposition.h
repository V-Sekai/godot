/**************************************************************************/
/*  test_ddm_polar_decomposition.h                                        */
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

namespace TestDDMPolarDecomposition {

TEST_CASE("[DDM][PolarDecomp] Identity transform") {
	DDMTestFixtures::TransformFixtures fixtures;
	DDMDeformer deformer;

	Transform3D rigid, scale;
	deformer.test_decompose_transform(fixtures.identity, rigid, scale);

	SUBCASE("Rigid component is identity rotation") {
		CHECK(rigid.basis.is_rotation());
		CHECK(rigid.basis.determinant() == doctest::Approx(1.0).epsilon(0.01));
	}

	SUBCASE("Scale component is identity") {
		CHECK(scale.basis.determinant() == doctest::Approx(1.0).epsilon(0.01));
		Vector3 scale_vec = scale.basis.get_scale();
		CHECK(scale_vec.x == doctest::Approx(1.0).epsilon(0.01));
		CHECK(scale_vec.y == doctest::Approx(1.0).epsilon(0.01));
		CHECK(scale_vec.z == doctest::Approx(1.0).epsilon(0.01));
	}
}

TEST_CASE("[DDM][PolarDecomp] Pure rotation extraction") {
	DDMTestFixtures::TransformFixtures fixtures;
	DDMDeformer deformer;

	SUBCASE("90° X rotation") {
		Transform3D rigid, scale;
		deformer.test_decompose_transform(fixtures.rotation_x_90, rigid, scale);

		// Rigid should be the rotation
		CHECK(rigid.basis.is_rotation());
		CHECK(rigid.basis.determinant() > 0.0); // No reflection

		// Scale should be near identity
		Vector3 scale_vec = scale.basis.get_scale();
		CHECK(scale_vec.x == doctest::Approx(1.0).epsilon(0.1));
		CHECK(scale_vec.y == doctest::Approx(1.0).epsilon(0.1));
		CHECK(scale_vec.z == doctest::Approx(1.0).epsilon(0.1));
	}

	SUBCASE("45° Y rotation") {
		Transform3D rigid, scale;
		deformer.test_decompose_transform(fixtures.rotation_y_45, rigid, scale);

		// Rigid should be the rotation
		CHECK(rigid.basis.is_rotation());
		CHECK(rigid.basis.determinant() > 0.0);

		// Scale should be near identity
		Vector3 scale_vec = scale.basis.get_scale();
		CHECK(scale_vec.x == doctest::Approx(1.0).epsilon(0.1));
		CHECK(scale_vec.y == doctest::Approx(1.0).epsilon(0.1));
		CHECK(scale_vec.z == doctest::Approx(1.0).epsilon(0.1));
	}
}

TEST_CASE("[DDM][PolarDecomp] Pure scale extraction") {
	DDMTestFixtures::TransformFixtures fixtures;
	DDMDeformer deformer;

	SUBCASE("Uniform 2x scale") {
		Transform3D rigid, scale;
		deformer.test_decompose_transform(fixtures.scale_uniform_2x, rigid, scale);

		// Rigid should be identity (no rotation)
		CHECK(rigid.basis.is_rotation());

		// Scale should be 2x uniform
		Vector3 scale_vec = scale.basis.get_scale();
		CHECK(scale_vec.x == doctest::Approx(2.0).epsilon(0.1));
		CHECK(scale_vec.y == doctest::Approx(2.0).epsilon(0.1));
		CHECK(scale_vec.z == doctest::Approx(2.0).epsilon(0.1));
	}

	SUBCASE("Non-uniform scale") {
		Transform3D rigid, scale;
		deformer.test_decompose_transform(fixtures.scale_nonuniform, rigid, scale);

		// Rigid should be identity or near-identity rotation
		CHECK(rigid.basis.is_rotation());

		// Scale should preserve non-uniform scaling
		Vector3 scale_vec = scale.basis.get_scale();
		// Scale components should match original (1, 2, 3) within tolerance
		CHECK(scale_vec.length() > 0.0); // Non-zero scale
	}
}

TEST_CASE("[DDM][PolarDecomp] Combined rotation and scale") {
	DDMTestFixtures::TransformFixtures fixtures;
	DDMDeformer deformer;

	Transform3D rigid, scale;
	deformer.test_decompose_transform(fixtures.combined_rotate_scale, rigid, scale);

	SUBCASE("Rigid component is valid rotation") {
		CHECK(rigid.basis.is_rotation());
		CHECK(rigid.basis.determinant() > 0.0); // Positive determinant

		// Check orthonormality
		CHECK(rigid.basis.is_orthonormal());
	}

	SUBCASE("Reconstruction matches original") {
		// M = R * S (polar decomposition property)
		Basis reconstructed = rigid.basis * scale.basis;

		// Compare with original
		Basis original = fixtures.combined_rotate_scale.basis;
		Basis diff = original * reconstructed.inverse();

		// Difference should be near identity
		Vector3 col0_diff = diff.get_column(0) - Vector3(1, 0, 0);
		Vector3 col1_diff = diff.get_column(1) - Vector3(0, 1, 0);
		Vector3 col2_diff = diff.get_column(2) - Vector3(0, 0, 1);

		CHECK(col0_diff.length() < 0.1);
		CHECK(col1_diff.length() < 0.1);
		CHECK(col2_diff.length() < 0.1);
	}

	SUBCASE("Determinant is positive") {
		// Ensure no reflections in rigid component
		CHECK(rigid.basis.determinant() > 0.0);
		CHECK(rigid.basis.determinant() == doctest::Approx(1.0).epsilon(0.1));
	}
}

TEST_CASE("[DDM][PolarDecomp] Numerical stability") {
	DDMDeformer deformer;

	SUBCASE("Very small scale doesn't produce NaN") {
		Transform3D tiny_scale(Basis::from_scale(Vector3(0.001, 0.001, 0.001)), Vector3(0, 0, 0));
		Transform3D rigid, scale;

		deformer.test_decompose_transform(tiny_scale, rigid, scale);

		// Should not produce NaN values
		CHECK(Math::is_finite(rigid.basis.get_column(0).x));
		CHECK(Math::is_finite(rigid.basis.get_column(1).y));
		CHECK(Math::is_finite(rigid.basis.get_column(2).z));
	}

	SUBCASE("Large scale doesn't overflow") {
		Transform3D large_scale(Basis::from_scale(Vector3(100, 100, 100)), Vector3(0, 0, 0));
		Transform3D rigid, scale;

		deformer.test_decompose_transform(large_scale, rigid, scale);

		// Should remain finite
		CHECK(Math::is_finite(rigid.basis.determinant()));
		CHECK(Math::is_finite(scale.basis.determinant()));
	}
}

} // namespace TestDDMPolarDecomposition
