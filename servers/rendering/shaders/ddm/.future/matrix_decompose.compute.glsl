#[compute]
#version 450

#include "double_precision.glsl"

layout(local_size_x = 64) in;

// Input: Bone transformation matrices (4x4)
layout(set = 0, binding = 0, std430) readonly buffer BoneTransforms {
	mat4 bone_transforms[];
};

// Output: Rigid components (M_rj) - rotation only
layout(set = 0, binding = 1, std430) writeonly buffer RigidComponents {
	mat4 rigid_components[];
};

// Output: Scale/shear components (M_sj)
layout(set = 0, binding = 2, std430) writeonly buffer ScaleShearComponents {
	mat4 scale_shear_components[];
};

// Push constants
layout(push_constant) uniform Params {
	uint bone_count;
	uint padding1;
	uint padding2;
	uint padding3;
}
params;

// Extract 3x3 rotation/scale from 4x4 transform
mat3 extract_3x3(mat4 m) {
	return mat3(
			m[0].xyz,
			m[1].xyz,
			m[2].xyz);
}

// Create 4x4 from 3x3 (preserving translation)
mat4 create_4x4(mat3 m, vec3 translation) {
	return mat4(
			vec4(m[0], 0.0),
			vec4(m[1], 0.0),
			vec4(m[2], 0.0),
			vec4(translation, 1.0));
}

// QR Decomposition using Modified Gram-Schmidt
void qr_decomposition(mat3 A, out mat3 Q, out mat3 R) {
	// Initialize R to zero
	R = mat3(0.0);

	// Extract columns
	vec3 a0 = A[0];
	vec3 a1 = A[1];
	vec3 a2 = A[2];

	// First column
	R[0][0] = length(a0);
	vec3 q0 = (R[0][0] > 1e-10) ? a0 / R[0][0] : vec3(1, 0, 0);

	// Second column
	R[0][1] = dot(q0, a1);
	vec3 a1_orth = a1 - q0 * R[0][1];
	R[1][1] = length(a1_orth);
	vec3 q1 = (R[1][1] > 1e-10) ? a1_orth / R[1][1] : vec3(0, 1, 0);

	// Third column
	R[0][2] = dot(q0, a2);
	R[1][2] = dot(q1, a2);
	vec3 a2_orth = a2 - q0 * R[0][2] - q1 * R[1][2];
	R[2][2] = length(a2_orth);
	vec3 q2 = (R[2][2] > 1e-10) ? a2_orth / R[2][2] : vec3(0, 0, 1);

	// Reorthogonalize for numerical stability
	q1 = q1 - q0 * dot(q0, q1);
	q1 = normalize(q1);

	q2 = q2 - q0 * dot(q0, q2) - q1 * dot(q1, q2);
	q2 = normalize(q2);

	// Construct Q
	Q = mat3(q0, q1, q2);
}

// Correct for reflections (ensure determinant = +1)
void correct_reflection(inout mat3 Q, inout mat3 R) {
	float det = determinant(Q);
	if (det < 0.0) {
		// Flip third column
		Q[2] = -Q[2];
		R[2][0] = -R[2][0];
		R[2][1] = -R[2][1];
		R[2][2] = -R[2][2];
	}
}

void main() {
	uint bone_idx = gl_GlobalInvocationID.x;

	if (bone_idx >= params.bone_count) {
		return;
	}

	// Get bone transform
	mat4 M = bone_transforms[bone_idx];
	vec3 translation = M[3].xyz;

	// Extract 3x3 rotation/scale portion
	mat3 M_3x3 = extract_3x3(M);

	// Handle degenerate/zero matrices
	float scale_check = abs(M_3x3[0][0]) + abs(M_3x3[1][1]) + abs(M_3x3[2][2]);
	if (scale_check < 1e-6) {
		// Use identity for degenerate matrices
		rigid_components[bone_idx] = mat4(1.0);
		scale_shear_components[bone_idx] = mat4(1.0);
		return;
	}

	// QR decomposition: M = Q * R
	// Q is orthogonal (rigid rotation)
	// R is upper triangular (scale + shear)
	mat3 Q, R;
	qr_decomposition(M_3x3, Q, R);

	// Correct for reflections
	correct_reflection(Q, R);

	// Store results as 4x4 matrices
	rigid_components[bone_idx] = create_4x4(Q, translation);
	scale_shear_components[bone_idx] = create_4x4(R, vec3(0.0));
}
