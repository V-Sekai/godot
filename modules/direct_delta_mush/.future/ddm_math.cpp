/**************************************************************************/
/*  ddm_math.cpp                                                          */
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

// Enhanced Direct Delta Mush - CPU Math Validation Library
// Implements double precision emulation and matrix operations

#include "ddm_math.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace DDMMath {

// ============================================================================
// Double Precision Emulation - Dekker Multiplication
// ============================================================================

DoublePrecision double_mul(DoublePrecision a, DoublePrecision b) {
	// Dekker's algorithm for double precision multiplication
	// Result: (a_hi + a_lo) * (b_hi + b_lo)

	// Veltkamp splitting constant for float (2^13 + 1)
	const float C = 8193.0f;

	// Split a
	float a_h = C * a.hi;
	a_h = (a.hi - (a_h - a.hi)) + (a_h - a.hi);
	float a_l = a.hi - a_h;

	// Split b
	float b_h = C * b.hi;
	b_h = (b.hi - (b_h - b.hi)) + (b_h - b.hi);
	float b_l = b.hi - b_h;

	// Compute product components
	float p_hh = a_h * b_h;
	float p_hl = a_h * b_l + a_l * b_h;
	float p_ll = a_l * b_l;

	// Renormalize
	float r_h = p_hh + p_hl;
	float r_l = (p_hh - r_h + p_hl) + p_ll + a.hi * b.lo + a.lo * b.hi;

	return DoublePrecision(r_h, r_l);
}

// ============================================================================
// Double Precision Emulation - Shewchuk Summation
// ============================================================================

DoublePrecision double_add(DoublePrecision a, DoublePrecision b) {
	// Shewchuk's error-free addition
	// Produces (s_h, s_l) such that s_h + s_l = a + b (exactly)

	// High-order addition
	float s = a.hi + b.hi;

	// Compute roundoff error
	float e = 0.0f;
	if (std::abs(a.hi) >= std::abs(b.hi)) {
		e = (a.hi - s) + b.hi;
	} else {
		e = (b.hi - s) + a.hi;
	}

	// Add low-order parts
	e = e + a.lo + b.lo;

	// Renormalize
	float r_h = s + e;
	float r_l = (s - r_h) + e;

	return DoublePrecision(r_h, r_l);
}

DoublePrecision double_sub(DoublePrecision a, DoublePrecision b) {
	return double_add(a, DoublePrecision(-b.hi, -b.lo));
}

// ============================================================================
// Double Precision Emulation - Division
// ============================================================================

DoublePrecision double_div(DoublePrecision a, DoublePrecision b) {
	// Approximate division using Newton-Raphson refinement

	if (std::abs(b.hi) < 1e-10f) {
		return DoublePrecision(std::numeric_limits<float>::infinity(), 0.0f);
	}

	// Initial approximation
	float q_h = a.hi / b.hi;

	// Newton-Raphson: q_new = q + (a - q*b) / b
	// Refined as: q_new = q + (a - q*b) / b.hi (using high-order division)
	DoublePrecision prod = double_mul(DoublePrecision(q_h), b);
	DoublePrecision remainder = double_sub(a, prod);
	float q_l = remainder.hi / b.hi;

	return DoublePrecision(q_h, q_l);
}

// ============================================================================
// Double Precision Emulation - Square Root
// ============================================================================

DoublePrecision double_sqrt(DoublePrecision x) {
	if (x.hi <= 0.0f) {
		return DoublePrecision(0.0f, 0.0f);
	}

	// Initial approximation using single-precision sqrt
	float q = std::sqrt(x.hi);
	DoublePrecision q_dbl(q, 0.0f);

	// Newton-Raphson refinement: q_new = q + (x - q²) / (2*q)
	DoublePrecision q_squared = double_mul(q_dbl, q_dbl);
	DoublePrecision difference = double_sub(x, q_squared);
	DoublePrecision two_q(2.0f * q, 0.0f);
	DoublePrecision correction = double_div(difference, two_q);

	return double_add(q_dbl, correction);
}

// ============================================================================
// Double Precision Emulation - Reciprocal
// ============================================================================

DoublePrecision double_recip(DoublePrecision x) {
	if (std::abs(x.hi) < 1e-10f) {
		return DoublePrecision(std::numeric_limits<float>::infinity(), 0.0f);
	}

	// Initial approximation
	float r = 1.0f / x.hi;
	DoublePrecision r_dbl(r, 0.0f);

	// Newton-Raphson: r_new = r(2 - x*r)
	DoublePrecision x_r = double_mul(x, r_dbl);
	DoublePrecision two(2.0f, 0.0f);
	DoublePrecision two_minus_xr = double_sub(two, x_r);

	return double_mul(r_dbl, two_minus_xr);
}

// ============================================================================
// Double Precision Emulation - Trigonometric
// ============================================================================

DoublePrecision double_cot(DoublePrecision angle) {
	// cot(x) = cos(x) / sin(x)
	// Use single-precision sin/cos (acceptable error)

	float cos_val = std::cos(angle.hi);
	float sin_val = std::sin(angle.hi);

	if (std::abs(sin_val) < 1e-10f) {
		return DoublePrecision(1e10f, 0.0f); // Very large value
	}

	DoublePrecision cos_dbl(cos_val, 0.0f);
	DoublePrecision sin_dbl(sin_val, 0.0f);

	return double_div(cos_dbl, sin_dbl);
}

DoublePrecision double_cotangent_weight(float angle1, float angle2) {
	// Weight = (cot(angle1) + cot(angle2)) / 2
	DoublePrecision cot1 = double_cot(DoublePrecision(angle1));
	DoublePrecision cot2 = double_cot(DoublePrecision(angle2));

	DoublePrecision sum = double_add(cot1, cot2);
	DoublePrecision two(2.0f, 0.0f);

	return double_div(sum, two);
}

// ============================================================================
// DoublePrecision Member Functions
// ============================================================================

DoublePrecision DoublePrecision::operator+(const DoublePrecision &b) const {
	return double_add(*this, b);
}

DoublePrecision DoublePrecision::operator-(const DoublePrecision &b) const {
	return double_sub(*this, b);
}

DoublePrecision DoublePrecision::operator*(const DoublePrecision &b) const {
	return double_mul(*this, b);
}

DoublePrecision DoublePrecision::operator/(const DoublePrecision &b) const {
	return double_div(*this, b);
}

DoublePrecision DoublePrecision::abs() const {
	return DoublePrecision(std::abs(hi), (hi < 0.0f) ? -lo : lo);
}

DoublePrecision DoublePrecision::sqrt() const {
	return double_sqrt(*this);
}

DoublePrecision DoublePrecision::recip() const {
	return double_recip(*this);
}

// ============================================================================
// Matrix Decomposition - QR via Gram-Schmidt
// ============================================================================

void qr_decomposition_3x3(
		const Matrix3 &A,
		Matrix3 &Q,
		Matrix3 &R) {
	// Modified Gram-Schmidt orthogonalization

	// Initialize R to zero
	R = Matrix3();
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			R.m[i][j] = 0.0f;
		}
	}

	// Process each column
	Vector3 col0 = A.col(0);
	Vector3 col1 = A.col(1);
	Vector3 col2 = A.col(2);

	// First column
	R.m[0][0] = col0.length();
	if (R.m[0][0] > 1e-10f) {
		col0 = col0 * (1.0f / R.m[0][0]);
	}

	// Second column
	R.m[0][1] = col0.dot(col1);
	col1 = col1 - (col0 * R.m[0][1]);
	R.m[1][1] = col1.length();
	if (R.m[1][1] > 1e-10f) {
		col1 = col1 * (1.0f / R.m[1][1]);
	}

	// Third column
	R.m[0][2] = col0.dot(col2);
	R.m[1][2] = col1.dot(col2);
	col2 = col2 - (col0 * R.m[0][2]) - (col1 * R.m[1][2]);
	R.m[2][2] = col2.length();
	if (R.m[2][2] > 1e-10f) {
		col2 = col2 * (1.0f / R.m[2][2]);
	}

	// Store orthonormal vectors
	Q.set_col(0, col0);
	Q.set_col(1, col1);
	Q.set_col(2, col2);

	// Reorthogonalize for numerical stability
	Vector3 q1 = Q.col(1);
	q1 = q1 - (Q.col(0) * Q.col(0).dot(q1));
	q1 = q1.normalized();
	Q.set_col(1, q1);

	Vector3 q2 = Q.col(2);
	q2 = q2 - (Q.col(0) * Q.col(0).dot(q2)) - (Q.col(1) * Q.col(1).dot(q2));
	q2 = q2.normalized();
	Q.set_col(2, q2);
}

void matrix_decomposition(
		const Matrix3 &M,
		Matrix3 &M_rigid,
		Matrix3 &M_scale_shear) {
	// Decompose M = Q*R where Q is rigid and R is scale/shear
	Matrix3 Q, R;
	qr_decomposition_3x3(M, Q, R);

	// Correct for reflections (ensure determinant = +1)
	if (Q.determinant() < 0.0f) {
		Q.set_col(2, Q.col(2) * -1.0f);
	}

	M_rigid = Q;
	M_scale_shear = R;
}

// ============================================================================
// Polar Decomposition - Eigenvalue Based
// ============================================================================

void eigen_decomposition_3x3(
		const Matrix3 &A,
		Vector3 &eigenvalues,
		Matrix3 &eigenvectors) {
	// Power iteration for dominant eigenvalue
	Vector3 v(1.0f, 0.5f, 0.25f); // Initial guess

	for (int iter = 0; iter < 10; iter++) {
		// Multiply A by v
		Vector3 Av(0.0f, 0.0f, 0.0f);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				// Access A as column-major
				float val = A.m[i][j];
				if (i == 0) {
					Av.x += val * v.x;
				} else if (i == 1) {
					Av.y += val * v.x;
				} else {
					Av.z += val * v.x;
				}

				if (i == 0) {
					Av.x += val * v.y;
				} else if (i == 1) {
					Av.y += val * v.y;
				} else {
					Av.z += val * v.y;
				}

				if (i == 0) {
					Av.x += val * v.z;
				} else if (i == 1) {
					Av.y += val * v.z;
				} else {
					Av.z += val * v.z;
				}
			}
		}

		float norm = Av.length();
		if (norm > 1e-10f) {
			v = Av * (1.0f / norm);
		}
	}

	// First eigenvalue (Rayleigh quotient)
	Vector3 Av(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			float val = A.m[i][j];
			if (i == 0) {
				Av.x += val * v.x + val * v.y + val * v.z;
			} else if (i == 1) {
				Av.y += val * v.x + val * v.y + val * v.z;
			} else {
				Av.z += val * v.x + val * v.y + val * v.z;
			}
		}
	}
	float eval1 = v.dot(Av);
	eigenvalues.x = eval1;
	eigenvectors.set_col(0, v);

	// Simplified: set remaining eigenvalues/vectors
	// In practice, would continue with deflation for 2nd and 3rd eigenvalues
	eigenvalues.y = eval1 * 0.5f; // Placeholder
	eigenvalues.z = eval1 * 0.25f; // Placeholder

	eigenvectors.set_col(1, Vector3(-v.y, v.x, 0.0f).normalized());
	eigenvectors.set_col(2, (v.cross(eigenvectors.col(1))).normalized());
}

Matrix3 matrix_inv_sqrt(const Matrix3 &A) {
	// Compute A^(-1/2) using eigendecomposition
	Vector3 eigenvalues;
	Matrix3 eigenvectors;
	eigen_decomposition_3x3(A, eigenvalues, eigenvectors);

	// Clamp eigenvalues to prevent division by zero
	eigenvalues.x = std::max(eigenvalues.x, 1e-10f);
	eigenvalues.y = std::max(eigenvalues.y, 1e-10f);
	eigenvalues.z = std::max(eigenvalues.z, 1e-10f);

	// Compute 1/sqrt(eigenvalues)
	eigenvalues.x = 1.0f / std::sqrt(eigenvalues.x);
	eigenvalues.y = 1.0f / std::sqrt(eigenvalues.y);
	eigenvalues.z = 1.0f / std::sqrt(eigenvalues.z);

	// Reconstruct: V * D^(-1/2) * V^T
	// Simplified implementation
	Matrix3 D;
	D.m[0][0] = eigenvalues.x;
	D.m[1][1] = eigenvalues.y;
	D.m[2][2] = eigenvalues.z;

	// V * D
	Matrix3 VD;
	for (int i = 0; i < 3; i++) {
		VD.set_col(i, eigenvectors.col(i) * eigenvalues[i]);
	}

	// (V * D) * V^T
	Matrix3 VT = eigenvectors.transpose();
	Matrix3 result;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			result.m[i][j] = 0.0f;
			for (int k = 0; k < 3; k++) {
				result.m[i][j] += VD.m[i][k] * VT.m[k][j];
			}
		}
	}

	return result;
}

Matrix3 polar_decomposition(const Matrix3 &M) {
	// Compute R = M * (M^T * M)^(-1/2)

	Matrix3 Mt = M.transpose();
	Matrix3 MtM;

	// Compute M^T * M
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			MtM.m[i][j] = 0.0f;
			for (int k = 0; k < 3; k++) {
				MtM.m[i][j] += Mt.m[i][k] * M.m[k][j];
			}
		}
	}

	// Compute (M^T * M)^(-1/2)
	Matrix3 inv_sqrt_MtM = matrix_inv_sqrt(MtM);

	// Compute R = M * (M^T * M)^(-1/2)
	Matrix3 R;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			R.m[i][j] = 0.0f;
			for (int k = 0; k < 3; k++) {
				R.m[i][j] += M.m[i][k] * inv_sqrt_MtM.m[k][j];
			}
		}
	}

	return R;
}

void correct_reflection(Matrix3 &R) {
	if (R.determinant() < 0.0f) {
		R.set_col(2, R.col(2) * -1.0f);
	}
}

void reorthogonalize(Matrix3 &R) {
	Vector3 c0 = R.col(0).normalized();
	Vector3 c1 = R.col(1) - (c0 * c0.dot(R.col(1)));
	c1 = c1.normalized();
	Vector3 c2 = R.col(2) - (c0 * c0.dot(R.col(2))) - (c1 * c1.dot(R.col(2)));
	c2 = c2.normalized();

	R.set_col(0, c0);
	R.set_col(1, c1);
	R.set_col(2, c2);
}

// ============================================================================
// Non-Rigid Displacement
// ============================================================================

Vector3 compute_displacement(
		const Vector3 &rest_vertex,
		const Matrix3 &M_scale_shear) {
	// d = M_sj * u - u
	Vector3 transformed(0.0f, 0.0f, 0.0f);

	// Matrix-vector multiplication
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == 0) {
				transformed.x += M_scale_shear.m[i][j] * rest_vertex.x;
			} else if (i == 1) {
				transformed.y += M_scale_shear.m[i][j] * rest_vertex.y;
			} else {
				transformed.z += M_scale_shear.m[i][j] * rest_vertex.z;
			}
		}
	}

	return transformed - rest_vertex;
}

// ============================================================================
// Laplacian Computation
// ============================================================================

float compute_angle(
		const Vector3 &v0,
		const Vector3 &v1,
		const Vector3 &v2) {
	// Angle at v1 between edges (v1->v0) and (v1->v2)
	Vector3 e0 = (v0 - v1).normalized();
	Vector3 e1 = (v2 - v1).normalized();

	float cos_angle = std::clamp(e0.dot(e1), -1.0f, 1.0f);
	return std::acos(cos_angle);
}

float compute_cotangent_weight_cpu(float angle1_rad, float angle2_rad) {
	// Clamp angles to valid range
	angle1_rad = std::clamp(angle1_rad, 1e-6f, 3.14159f - 1e-6f);
	angle2_rad = std::clamp(angle2_rad, 1e-6f, 3.14159f - 1e-6f);

	// Compute cot(angle) = cos(angle) / sin(angle)
	float cot1 = std::cos(angle1_rad) / std::sin(angle1_rad);
	float cot2 = std::cos(angle2_rad) / std::sin(angle2_rad);

	// Weight = (cot(α) + cot(β)) / 2
	float weight = (cot1 + cot2) * 0.5f;

	// Clamp to prevent extreme values
	weight = std::clamp(weight, -1e5f, 1e5f);

	return weight;
}

// ============================================================================
// Enhanced DDM Deformation
// ============================================================================

Vector3 enhanced_deformation(
		const Vector3 &smoothed_vertex,
		const std::vector<Matrix3> &rigid_transforms,
		const std::vector<Vector3> &displacements,
		const std::vector<float> &omega_weights,
		const std::vector<float> &bone_weights,
		const std::vector<uint32_t> &bone_indices,
		uint32_t max_bones) {
	Vector3 result(0.0f, 0.0f, 0.0f);

	for (uint32_t b = 0; b < max_bones && b < bone_weights.size(); b++) {
		float weight = bone_weights[b];

		if (weight < 1e-10f) {
			continue;
		}

		uint32_t bone_idx = std::min(bone_indices[b], (uint32_t)rigid_transforms.size() - 1);
		const Matrix3 &M_rigid = rigid_transforms[bone_idx];

		Vector3 displacement = displacements[b];
		float omega = omega_weights[b];

		// Transform: M_rj * (displacement + u)
		Vector3 displaced = smoothed_vertex + displacement;
		Vector3 transformed(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == 0) {
					transformed.x += M_rigid.m[i][j] * displaced.x;
				} else if (i == 1) {
					transformed.y += M_rigid.m[i][j] * displaced.y;
				} else {
					transformed.z += M_rigid.m[i][j] * displaced.z;
				}
			}
		}

		result = result + (transformed * weight * omega);
	}

	return result;
}

Vector3 standard_deformation(
		const Vector3 &smoothed_vertex,
		const std::vector<Matrix3> &bone_transforms,
		const std::vector<float> &bone_weights,
		const std::vector<uint32_t> &bone_indices,
		uint32_t max_bones) {
	Vector3 result(0.0f, 0.0f, 0.0f);

	for (uint32_t b = 0; b < max_bones && b < bone_weights.size(); b++) {
		float weight = bone_weights[b];

		if (weight < 1e-10f) {
			continue;
		}

		uint32_t bone_idx = std::min(bone_indices[b], (uint32_t)bone_transforms.size() - 1);
		const Matrix3 &M = bone_transforms[bone_idx];

		Vector3 transformed(0.0f, 0.0f, 0.0f);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == 0) {
					transformed.x += M.m[i][j] * smoothed_vertex.x;
				} else if (i == 1) {
					transformed.y += M.m[i][j] * smoothed_vertex.y;
				} else {
					transformed.z += M.m[i][j] * smoothed_vertex.z;
				}
			}
		}

		result = result + (transformed * weight);
	}

	return result;
}

// ============================================================================
// Validation Utilities
// ============================================================================

bool validate_matrix_decomposition(
		const Matrix3 &original,
		const Matrix3 &rigid,
		const Matrix3 &scale_shear,
		float epsilon) {
	// Check if rigid * scale_shear ≈ original
	Matrix3 reconstructed;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			reconstructed.m[i][j] = 0.0f;
			for (int k = 0; k < 3; k++) {
				reconstructed.m[i][j] += rigid.m[i][k] * scale_shear.m[k][j];
			}
		}
	}

	// Check error
	float max_error = 0.0f;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			float err = std::abs(reconstructed.m[i][j] - original.m[i][j]);
			max_error = std::max(max_error, err);
		}
	}

	return max_error < epsilon;
}

bool validate_polar_decomposition(
		const Matrix3 &R,
		float epsilon) {
	// Check determinant = +1
	float det = R.determinant();
	if (std::abs(det - 1.0f) > epsilon) {
		return false;
	}

	// Check orthogonality: R^T * R = I
	Matrix3 Rt = R.transpose();
	Matrix3 RtR;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			RtR.m[i][j] = 0.0f;
			for (int k = 0; k < 3; k++) {
				RtR.m[i][j] += Rt.m[i][k] * R.m[k][j];
			}
		}
	}

	// Check if identity
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			float expected = (i == j) ? 1.0f : 0.0f;
			if (std::abs(RtR.m[i][j] - expected) > epsilon) {
				return false;
			}
		}
	}

	return true;
}

} // namespace DDMMath
