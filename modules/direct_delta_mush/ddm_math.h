// Enhanced Direct Delta Mush - CPU Math Validation Library
// C++ implementations of GPU shader algorithms for validation and fallback
// Provides CPU-side double precision and complex matrix operations

#pragma once

#ifndef DDM_MATH_H
#define DDM_MATH_H

#include <cmath>
#include <stdint.h>
#include <vector>

namespace DDMMath {

// ============================================================================
// Vector and Matrix Types (using float arrays for portability)
// ============================================================================

struct Vector3 {
    float x, y, z;
    
    Vector3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Vector3 operator+(const Vector3& v) const {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }
    
    Vector3 operator-(const Vector3& v) const {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }
    
    Vector3 operator*(float s) const {
        return Vector3(x * s, y * s, z * s);
    }
    
    float dot(const Vector3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    
    Vector3 cross(const Vector3& v) const {
        return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    Vector3 normalized() const {
        float len = length();
        if (len < 1e-10f) return Vector3(0, 0, 0);
        return (*this) * (1.0f / len);
    }
};

struct Matrix3 {
    float m[3][3];
    
    Matrix3() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    
    Vector3 col(int i) const {
        return Vector3(m[0][i], m[1][i], m[2][i]);
    }
    
    void set_col(int i, const Vector3& v) {
        m[0][i] = v.x;
        m[1][i] = v.y;
        m[2][i] = v.z;
    }
    
    float determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
    
    Matrix3 transpose() const {
        Matrix3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.m[i][j] = m[j][i];
            }
        }
        return result;
    }
};

// ============================================================================
// Double Precision Emulation (CPU)
// ============================================================================

struct DoublePrecision {
    float hi;  // High precision component
    float lo;  // Low precision component

    DoublePrecision() : hi(0.0f), lo(0.0f) {}
    DoublePrecision(float h, float l) : hi(h), lo(l) {}
    DoublePrecision(float x) : hi(x), lo(0.0f) {}

    // Convert to single precision
    float to_float() const { return hi + lo; }

    // Convert from single precision
    static DoublePrecision from_float(float x) {
        return DoublePrecision(x, 0.0f);
    }

    // Arithmetic operators
    DoublePrecision operator+(const DoublePrecision& b) const;
    DoublePrecision operator-(const DoublePrecision& b) const;
    DoublePrecision operator*(const DoublePrecision& b) const;
    DoublePrecision operator/(const DoublePrecision& b) const;

    // Utility functions
    DoublePrecision abs() const;
    DoublePrecision sqrt() const;
    DoublePrecision recip() const;  // 1/x
};

// Double precision arithmetic operations
DoublePrecision double_add(DoublePrecision a, DoublePrecision b);
DoublePrecision double_sub(DoublePrecision a, DoublePrecision b);
DoublePrecision double_mul(DoublePrecision a, DoublePrecision b);
DoublePrecision double_div(DoublePrecision a, DoublePrecision b);
DoublePrecision double_sqrt(DoublePrecision x);
DoublePrecision double_recip(DoublePrecision x);

// Trigonometric operations with double precision
DoublePrecision double_cot(DoublePrecision angle_rad);
DoublePrecision double_cotangent_weight(float angle1, float angle2);

// ============================================================================
// Matrix Decomposition (CPU)
// ============================================================================

// QR decomposition using Gram-Schmidt orthogonalization
void qr_decomposition_3x3(
    const Matrix3& A,
    Matrix3& Q,
    Matrix3& R
);

// Matrix decomposition: M = Q*R
void matrix_decomposition(
    const Matrix3& M,
    Matrix3& M_rigid,
    Matrix3& M_scale_shear
);

// ============================================================================
// Polar Decomposition (CPU)
// ============================================================================

// Eigenvalue decomposition for 3x3 symmetric positive-definite matrix
void eigen_decomposition_3x3(
    const Matrix3& A,
    Vector3& eigenvalues,
    Matrix3& eigenvectors
);

// Matrix square root inverse: A^(-1/2)
Matrix3 matrix_inv_sqrt(const Matrix3& A);

// Polar decomposition: R = M * (M^T * M)^(-1/2)
Matrix3 polar_decomposition(const Matrix3& M);

// Reflection correction: ensure determinant = +1
void correct_reflection(Matrix3& R);

// Reorthogonalization using Gram-Schmidt
void reorthogonalize(Matrix3& R);

// ============================================================================
// Non-Rigid Displacement (CPU)
// ============================================================================

// Compute non-rigid displacement
Vector3 compute_displacement(
    const Vector3& rest_vertex,
    const Matrix3& M_scale_shear
);

// ============================================================================
// Laplacian Computation (CPU)
// ============================================================================

// Compute cotangent weight
float compute_cotangent_weight_cpu(float angle1_rad, float angle2_rad);

// Compute angle at vertex
float compute_angle(
    const Vector3& v0,
    const Vector3& v1,
    const Vector3& v2
);

// ============================================================================
// Enhanced DDM Deformation (CPU)
// ============================================================================

// Apply Enhanced DDM deformation
Vector3 enhanced_deformation(
    const Vector3& smoothed_vertex,
    const std::vector<Matrix3>& rigid_transforms,
    const std::vector<Vector3>& displacements,
    const std::vector<float>& omega_weights,
    const std::vector<float>& bone_weights,
    const std::vector<uint32_t>& bone_indices,
    uint32_t max_bones = 4
);

// Standard DDM fallback
Vector3 standard_deformation(
    const Vector3& smoothed_vertex,
    const std::vector<Matrix3>& bone_transforms,
    const std::vector<float>& bone_weights,
    const std::vector<uint32_t>& bone_indices,
    uint32_t max_bones = 4
);

// ============================================================================
// Validation Utilities
// ============================================================================

// Validate matrix decomposition correctness
bool validate_matrix_decomposition(
    const Matrix3& original,
    const Matrix3& rigid,
    const Matrix3& scale_shear,
    float epsilon = 1e-5f
);

// Validate polar decomposition
bool validate_polar_decomposition(
    const Matrix3& R,
    float epsilon = 1e-4f
);

// ============================================================================
// Performance Profiling
// ============================================================================

struct PerformanceMetrics {
    double double_precision_time_ms;
    double matrix_decomposition_time_ms;
    double polar_decomposition_time_ms;
    double displacement_time_ms;
    double laplacian_time_ms;
    double deformation_time_ms;
    double total_time_ms;

    PerformanceMetrics() : 
        double_precision_time_ms(0.0),
        matrix_decomposition_time_ms(0.0),
        polar_decomposition_time_ms(0.0),
        displacement_time_ms(0.0),
        laplacian_time_ms(0.0),
        deformation_time_ms(0.0),
        total_time_ms(0.0) {}
};

}  // namespace DDMMath

#endif  // DDM_MATH_H
