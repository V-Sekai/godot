// Enhanced Direct Delta Mush - Matrix Decomposition Shader
// Decomposes 4x4 bone transformation matrices into rigid (M_rj) and scale/shear (M_sj) components
// Uses QR decomposition: M = Q*R where Q is orthogonal (rigid) and R is upper triangular (scale/shear)

#version 460

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input: Bone transformation matrices (4x4)
layout(std430, binding = 0) readonly buffer BoneTransforms {
    mat4 transforms[];
};

// Output: Rigid components (rotation + translation)
layout(std430, binding = 1) writeonly buffer RigidComponents {
    mat4 rigid_matrices[];
};

// Output: Scale/shear components
layout(std430, binding = 2) writeonly buffer ScaleShearComponents {
    mat4 scale_shear_matrices[];
};

// Uniform: Number of bones to process
layout(std140, binding = 0) uniform Config {
    uint bone_count;
    uint pad0, pad1, pad2;
};

// ============================================================================
// Matrix Utilities
// ============================================================================

// Extract 3x3 linear part from 4x4 matrix
mat3 extract_linear_part(mat4 m) {
    return mat3(
        m[0].xyz,
        m[1].xyz,
        m[2].xyz
    );
}

// Construct 4x4 from 3x3 linear part and translation
mat4 construct_with_translation(mat3 linear, vec3 translation) {
    return mat4(
        vec4(linear[0], translation.x),
        vec4(linear[1], translation.y),
        vec4(linear[2], translation.z),
        vec4(0.0, 0.0, 0.0, 1.0)
    );
}

// ============================================================================
// Gram-Schmidt Orthogonalization
// ============================================================================
// Compute QR decomposition using modified Gram-Schmidt
// Input: 3x3 matrix
// Output: Q (orthogonal), R (upper triangular)

void qr_decomposition(mat3 A, out mat3 Q, out mat3 R) {
    // Initialize R to zero
    R = mat3(0.0);
    
    // Process each column
    vec3 col0 = A[0];
    vec3 col1 = A[1];
    vec3 col2 = A[2];
    
    // First column
    R[0][0] = length(col0);
    if (R[0][0] > 1e-10) {
        col0 = col0 / R[0][0];
    }
    
    // Second column
    R[0][1] = dot(col0, col1);
    col1 = col1 - R[0][1] * col0;
    R[1][1] = length(col1);
    if (R[1][1] > 1e-10) {
        col1 = col1 / R[1][1];
    }
    
    // Third column
    R[0][2] = dot(col0, col2);
    R[1][2] = dot(col1, col2);
    col2 = col2 - R[0][2] * col0 - R[1][2] * col1;
    R[2][2] = length(col2);
    if (R[2][2] > 1e-10) {
        col2 = col2 / R[2][2];
    }
    
    // Store orthonormal vectors in Q
    Q[0] = col0;
    Q[1] = col1;
    Q[2] = col2;
    
    // Ensure Q is actually orthogonal (numerical stability)
    // Reorthogonalize if needed
    Q[1] = normalize(Q[1] - dot(Q[0], Q[1]) * Q[0]);
    Q[2] = normalize(Q[2] - dot(Q[0], Q[2]) * Q[0] - dot(Q[1], Q[2]) * Q[1]);
}

// ============================================================================
// Reflection Correction
// ============================================================================
// Ensure Q represents rotation (det = +1) not reflection (det = -1)

void correct_reflection(inout mat3 Q) {
    float det = determinant(Q);
    
    // If determinant is negative, we have a reflection
    // Flip one column to convert to rotation
    if (det < 0.0) {
        Q[2] = -Q[2];  // Flip third column
    }
}

// ============================================================================
// Main Computation
// ============================================================================

void main() {
    uint bone_idx = gl_GlobalInvocationID.x;
    
    // Bounds check
    if (bone_idx >= bone_count) {
        return;
    }
    
    // Load bone transformation matrix
    mat4 M = transforms[bone_idx];
    
    // Extract 3x3 linear transformation (excluding translation)
    mat3 M_linear = extract_linear_part(M);
    vec3 translation = M[3].xyz;  // Translation column
    
    // Compute QR decomposition
    // M_linear = Q * R
    // where Q is orthogonal (rigid rotation) and R is upper triangular (scale/shear)
    mat3 Q;  // Orthogonal component (rotation)
    mat3 R;  // Upper triangular component (scale/shear)
    qr_decomposition(M_linear, Q, R);
    
    // Correct for reflections
    // Ensure Q is a pure rotation (determinant = +1)
    correct_reflection(Q);
    
    // Construct output matrices
    // M_rj = [Q | translation; 0 0 0 | 1] (rigid: rotation + translation)
    mat4 rigid = construct_with_translation(Q, translation);
    rigid[3] = vec4(0.0, 0.0, 0.0, 1.0);
    
    // M_sj = [R | 0; 0 0 0 | 1] (scale/shear: no translation)
    mat4 scale_shear = construct_with_translation(R, vec3(0.0));
    scale_shear[3] = vec4(0.0, 0.0, 0.0, 1.0);
    
    // Store results
    rigid_matrices[bone_idx] = rigid;
    scale_shear_matrices[bone_idx] = scale_shear;
}
