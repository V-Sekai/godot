#[compute]
#version 450

#include "double_precision.glsl"

layout(local_size_x = 64) in;

// Input: Transformation matrices M (4x4)
layout(set = 0, binding = 0, std430) readonly buffer InputTransforms {
	mat4 input_transforms[];
};

// Output: Rotation matrices R (4x4)
layout(set = 0, binding = 1, std430) writeonly buffer RotationMatrices {
	mat4 rotation_matrices[];
};

// Push constants
layout(push_constant) uniform Params {
	uint matrix_count;
	uint max_iterations; // For eigenvalue computation
	uint padding1;
	uint padding2;
}
params;

// Compute M^T * M (symmetric positive definite)
mat3 compute_mtm(mat3 M) {
	mat3 Mt = transpose(M);
	return Mt * M;
}

// Power iteration for dominant eigenvalue
float power_iteration(mat3 A, out vec3 eigenvector) {
	// Initial guess
	vec3 v = normalize(vec3(1.0, 0.5, 0.25));

	uint iterations = min(params.max_iterations, 20u);
	for (uint i = 0u; i < iterations; i++) {
		vec3 Av = A * v;
		float norm = length(Av);

		if (norm < 1e-10) {
			eigenvector = vec3(1, 0, 0);
			return 0.0;
		}

		v = Av / norm;
	}

	eigenvector = v;

	// Rayleigh quotient for eigenvalue
	vec3 Av = A * v;
	return dot(v, Av);
}

// Simplified eigen decomposition for 3x3 symmetric matrix
void eigen_decomposition_symmetric(mat3 A, out vec3 eigenvalues, out mat3 eigenvectors) {
	// First eigenvalue/vector via power iteration
	vec3 v1;
	eigenvalues.x = power_iteration(A, v1);
	eigenvectors[0] = v1;

	// For simplicity, use approximate eigenvalues for remaining
	// In production, would use full deflation method
	float trace = A[0][0] + A[1][1] + A[2][2];
	eigenvalues.y = (trace - eigenvalues.x) * 0.5;
	eigenvalues.z = (trace - eigenvalues.x) * 0.5;

	// Orthogonal basis
	vec3 v2 = abs(v1.x) < 0.9 ? vec3(1, 0, 0) : vec3(0, 1, 0);
	v2 = normalize(v2 - v1 * dot(v1, v2));
	eigenvectors[1] = v2;

	vec3 v3 = cross(v1, v2);
	eigenvectors[2] = normalize(v3);
}

// Compute matrix inverse square root: A^(-1/2)
mat3 matrix_inv_sqrt(mat3 A) {
	vec3 eigenvalues;
	mat3 V; // Eigenvectors

	eigen_decomposition_symmetric(A, eigenvalues, V);

	// Clamp eigenvalues to prevent division by zero
	eigenvalues = max(eigenvalues, vec3(1e-10));

	// Compute 1/sqrt(eigenvalues)
	vec3 inv_sqrt_eig = 1.0 / sqrt(eigenvalues);

	// Diagonal matrix D^(-1/2)
	mat3 D_inv_sqrt = mat3(
			inv_sqrt_eig.x, 0.0, 0.0,
			0.0, inv_sqrt_eig.y, 0.0,
			0.0, 0.0, inv_sqrt_eig.z);

	// Reconstruct: V * D^(-1/2) * V^T
	mat3 VD = V * D_inv_sqrt;
	mat3 Vt = transpose(V);
	return VD * Vt;
}

// Polar decomposition: R = M * (M^T * M)^(-1/2)
mat3 polar_decomposition(mat3 M) {
	// Compute M^T * M
	mat3 MtM = compute_mtm(M);

	// Compute (M^T * M)^(-1/2)
	mat3 inv_sqrt_MtM = matrix_inv_sqrt(MtM);

	// R = M * (M^T * M)^(-1/2)
	mat3 R = M * inv_sqrt_MtM;

	return R;
}

// Reorthogonalize using Gram-Schmidt
mat3 reorthogonalize(mat3 R) {
	vec3 c0 = normalize(R[0]);

	vec3 c1 = R[1] - c0 * dot(c0, R[1]);
	c1 = normalize(c1);

	vec3 c2 = R[2] - c0 * dot(c0, R[2]) - c1 * dot(c1, R[2]);
	c2 = normalize(c2);

	return mat3(c0, c1, c2);
}

// Correct reflection (ensure determinant = +1)
mat3 correct_reflection(mat3 R) {
	float det = determinant(R);
	if (det < 0.0) {
		// Flip third column
		R[2] = -R[2];
	}
	return R;
}

// Extract 3x3 from 4x4
mat3 extract_3x3(mat4 m) {
	return mat3(m[0].xyz, m[1].xyz, m[2].xyz);
}

// Create 4x4 from 3x3
mat4 create_4x4(mat3 m, vec3 translation) {
	return mat4(
			vec4(m[0], 0.0),
			vec4(m[1], 0.0),
			vec4(m[2], 0.0),
			vec4(translation, 1.0));
}

void main() {
	uint matrix_idx = gl_GlobalInvocationID.x;

	if (matrix_idx >= params.matrix_count) {
		return;
	}

	// Get input transform
	mat4 M_4x4 = input_transforms[matrix_idx];
	vec3 translation = M_4x4[3].xyz;

	// Extract 3x3 portion
	mat3 M = extract_3x3(M_4x4);

	// Check for degenerate matrix
	float scale_check = abs(M[0][0]) + abs(M[1][1]) + abs(M[2][2]);
	if (scale_check < 1e-6) {
		rotation_matrices[matrix_idx] = mat4(1.0);
		return;
	}

	// Perform polar decomposition
	mat3 R = polar_decomposition(M);

	// Reorthogonalize for numerical stability
	R = reorthogonalize(R);

	// Correct for reflections
	R = correct_reflection(R);

	// Store result
	rotation_matrices[matrix_idx] = create_4x4(R, translation);
}
