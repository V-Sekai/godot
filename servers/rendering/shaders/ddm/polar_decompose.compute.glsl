// Enhanced Direct Delta Mush - Polar Decomposition Shader
// Computes polar decomposition R_i = M_i(M_i^T M_i)^(-1/2)
// Extracts rotation component from arbitrary 3x3 matrix using eigenvalue-based approach
// Handles reflection correction to ensure rotation-only output

#version 460

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input: Scale/shear matrices (M_sj from matrix decomposition)
layout(std430, binding = 0) readonly buffer ScaleShearMatrices {
    mat4 scale_shear[];
};

// Output: Rotation matrices (for detail preservation)
layout(std430, binding = 1) writeonly buffer RotationMatrices {
    mat4 rotation_matrices[];
};

// Uniform: Number of matrices to process
layout(std140, binding = 0) uniform Config {
    uint matrix_count;
    uint pad0, pad1, pad2;
};

// ============================================================================
// 3x3 Matrix Utilities
// ============================================================================

// Extract 3x3 linear part from 4x4 matrix
mat3 extract_linear_part(mat4 m) {
    return mat3(
        m[0].xyz,
        m[1].xyz,
        m[2].xyz
    );
}

// Compute matrix transpose
mat3 transpose_3x3(mat3 m) {
    return mat3(
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2]
    );
}

// Multiply two 3x3 matrices
mat3 mul_3x3(mat3 a, mat3 b) {
    return mat3(
        vec3(dot(a[0], vec3(b[0][0], b[1][0], b[2][0])),
             dot(a[0], vec3(b[0][1], b[1][1], b[2][1])),
             dot(a[0], vec3(b[0][2], b[1][2], b[2][2]))),
        vec3(dot(a[1], vec3(b[0][0], b[1][0], b[2][0])),
             dot(a[1], vec3(b[0][1], b[1][1], b[2][1])),
             dot(a[1], vec3(b[0][2], b[1][2], b[2][2]))),
        vec3(dot(a[2], vec3(b[0][0], b[1][0], b[2][0])),
             dot(a[2], vec3(b[0][1], b[1][1], b[2][1])),
             dot(a[2], vec3(b[0][2], b[1][2], b[2][2])))
    );
}

// ============================================================================
// Eigenvalue/Eigenvector Computation (3x3 Symmetric)
// ============================================================================
// For symmetric matrix, compute eigenvalues and eigenvectors
// Using power iteration and deflation method

// Power iteration to find dominant eigenvector
void power_iteration(mat3 A, out vec3 eigvec, out float eigval) {
    vec3 v = vec3(1.0, 0.5, 0.25);  // Initial guess
    
    for (int iter = 0; iter < 10; iter++) {
        vec3 Av = mul_3x3(A, mat3(v, vec3(0.0), vec3(0.0)))[0];
        float norm = length(Av);
        
        if (norm > 1e-10) {
            v = Av / norm;
        }
    }
    
    // Compute Rayleigh quotient for eigenvalue
    vec3 Av = mul_3x3(A, mat3(v, vec3(0.0), vec3(0.0)))[0];
    eigval = dot(v, Av);
    eigvec = v;
}

// Compute eigenvalues and eigenvectors of symmetric 3x3 matrix
void eigen_decomposition_3x3(mat3 A, 
                              out vec3 eigenvalues,
                              out mat3 eigenvectors) {
    // A must be symmetric: A = A^T
    
    vec3 eig1, eig2, eig3;
    float eval1, eval2, eval3;
    
    // Find first eigenvector (dominant)
    power_iteration(A, eig1, eval1);
    
    // Deflate matrix: A2 = A - eval1 * eig1 * eig1^T
    mat3 A2 = A - eval1 * outerProduct(eig1, eig1);
    power_iteration(A2, eig2, eval2);
    
    // Orthogonalize second eigenvector
    eig2 = eig2 - dot(eig1, eig2) * eig1;
    eig2 = normalize(eig2);
    
    // Third eigenvector is cross product
    eig3 = normalize(cross(eig1, eig2));
    
    // Store eigenvalues and eigenvectors
    eigenvalues = vec3(eval1, eval2, eval3);
    eigenvectors = mat3(eig1, eig2, eig3);
}

// ============================================================================
// Matrix Square Root via Eigendecomposition
// ============================================================================
// Compute A^(-1/2) where A is symmetric positive definite
// Using: A = V * diag(λ) * V^T
//        A^(-1/2) = V * diag(1/√λ) * V^T

mat3 matrix_inv_sqrt(mat3 A) {
    // Compute eigendecomposition
    vec3 eigenvalues;
    mat3 eigenvectors;
    eigen_decomposition_3x3(A, eigenvalues, eigenvectors);
    
    // Compute diagonal matrix with 1/√λ
    vec3 inv_sqrt_eigs = vec3(
        1.0 / sqrt(max(eigenvalues[0], 1e-10)),
        1.0 / sqrt(max(eigenvalues[1], 1e-10)),
        1.0 / sqrt(max(eigenvalues[2], 1e-10))
    );
    
    // Construct D = diag(inv_sqrt_eigs)
    mat3 D = mat3(
        vec3(inv_sqrt_eigs[0], 0.0, 0.0),
        vec3(0.0, inv_sqrt_eigs[1], 0.0),
        vec3(0.0, 0.0, inv_sqrt_eigs[2])
    );
    
    // Compute V * D * V^T
    mat3 VD = mul_3x3(eigenvectors, D);
    mat3 result = mul_3x3(VD, transpose_3x3(eigenvectors));
    
    return result;
}

// ============================================================================
// Polar Decomposition
// ============================================================================
// Compute R = M * (M^T * M)^(-1/2)
// R is orthogonal (rotation-only)

mat3 polar_decomposition(mat3 M) {
    // Compute M^T
    mat3 Mt = transpose_3x3(M);
    
    // Compute A = M^T * M (symmetric, positive definite)
    mat3 A = mul_3x3(Mt, M);
    
    // Compute A^(-1/2)
    mat3 A_inv_sqrt = matrix_inv_sqrt(A);
    
    // Compute R = M * A^(-1/2)
    mat3 R = mul_3x3(M, A_inv_sqrt);
    
    return R;
}

// ============================================================================
// Reflection Correction
// ============================================================================
// Ensure matrix represents rotation (det = +1) not reflection (det = -1)

void correct_reflection(inout mat3 R) {
    float det = determinant(R);
    
    // If determinant is negative, we have a reflection
    // Negate one column to convert to rotation
    if (det < 0.0) {
        R[2] = -R[2];  // Flip third column
    }
}

// ============================================================================
// Reorthogonalization for Numerical Stability
// ============================================================================
// Improve orthogonality using Gram-Schmidt if needed

void reorthogonalize(inout mat3 R) {
    // Gram-Schmidt reorthogonalization
    R[0] = normalize(R[0]);
    R[1] = normalize(R[1] - dot(R[0], R[1]) * R[0]);
    R[2] = normalize(R[2] - dot(R[0], R[2]) * R[0] - dot(R[1], R[2]) * R[1]);
}

// ============================================================================
// Main Computation
// ============================================================================

void main() {
    uint matrix_idx = gl_GlobalInvocationID.x;
    
    // Bounds check
    if (matrix_idx >= matrix_count) {
        return;
    }
    
    // Load scale/shear matrix
    mat4 M_sj = scale_shear[matrix_idx];
    
    // Extract 3x3 linear part
    mat3 M = extract_linear_part(M_sj);
    
    // Compute polar decomposition
    mat3 R = polar_decomposition(M);
    
    // Correct for reflections (ensure det = +1)
    correct_reflection(R);
    
    // Reorthogonalize for numerical stability
    reorthogonalize(R);
    
    // Construct output 4x4 matrix (rotation only, no translation)
    mat4 result = mat4(
        vec4(R[0], 0.0),
        vec4(R[1], 0.0),
        vec4(R[2], 0.0),
        vec4(0.0, 0.0, 0.0, 1.0)
    );
    
    // Store result
    rotation_matrices[matrix_idx] = result;
}
