// Enhanced Direct Delta Mush - Non-Rigid Displacement Shader
// Computes per-vertex, per-bone displacement from scale/shear transformations
// Displacement: d_ij = M_sj × u_i - u_i
// Handles multiple bone influences (up to 4 bones per vertex via LBS)

#version 460

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input: Rest pose vertices
layout(std430, binding = 0) readonly buffer RestVertices {
	vec4 vertices[]; // xyz = position, w = unused
};

// Input: Scale/shear transformation matrices (M_sj)
layout(std430, binding = 1) readonly buffer ScaleShearMatrices {
	mat4 scale_shear[];
};

// Input: Bone weights and indices per vertex
layout(std430, binding = 2) readonly buffer BoneWeights {
	vec4 weights[]; // Weight values (4 bones max)
};

layout(std430, binding = 3) readonly buffer BoneIndices {
	uvec4 indices[]; // Bone indices (4 bones max)
};

// Output: Displacement matrices (4x4) per vertex per bone
// Layout: [displacement_0, displacement_1, displacement_2, displacement_3] per vertex
layout(std430, binding = 4) writeonly buffer DisplacementMatrices {
	mat4 displacements[]; // Linear layout: vertex_idx * 4 + bone_idx
};

// Uniform: Parameters
layout(std140, binding = 0) uniform Config {
	uint vertex_count;
	uint max_bones; // Max bones per vertex (typically 4)
	uint total_bones; // Total number of bones
	uint pad0;
};

// ============================================================================
// Matrix Utilities
// ============================================================================

// Extract 3x3 linear part from 4x4 matrix
mat3 extract_linear_part(mat4 m) {
	return mat3(
			m[0].xyz,
			m[1].xyz,
			m[2].xyz);
}

// Transform 3D point by 3x3 matrix (linear transformation only)
vec3 transform_linear(mat3 M, vec3 p) {
	return M * p;
}

// ============================================================================
// Displacement Computation
// ============================================================================
// Compute displacement for one bone influence: d = M_sj * u - u

vec4 compute_single_displacement(vec3 rest_vertex, mat4 M_sj) {
	// Extract linear part of scale/shear matrix
	mat3 M_linear = extract_linear_part(M_sj);

	// Apply scale/shear transformation: p' = M_sj * u
	vec3 transformed = transform_linear(M_linear, rest_vertex);

	// Compute displacement: d = p' - u = (M_sj - I) * u
	vec3 displacement = transformed - rest_vertex;

	return vec4(displacement, 0.0);
}

// ============================================================================
// Displacement Matrix Construction
// ============================================================================
// Create 4x4 displacement matrix from displacement vector
// Used to accumulate displacements for multiple bones

mat4 make_displacement_matrix(vec4 disp) {
	// Displacement matrix format:
	// [M_sj - I, 0]
	// [0,        1]
	// where M_sj - I is the 3x3 linear displacement part

	return mat4(
			vec4(disp.x, 0.0, 0.0, 0.0),
			vec4(0.0, disp.y, 0.0, 0.0),
			vec4(0.0, 0.0, disp.z, 0.0),
			vec4(0.0, 0.0, 0.0, 0.0));
}

// ============================================================================
// Edge Case Handling
// ============================================================================

// Check if vertex has any bone influence
bool has_bone_influence(vec4 weights) {
	return (weights.x > 1e-10 || weights.y > 1e-10 ||
			weights.z > 1e-10 || weights.w > 1e-10);
}

// Safely get bone index with bounds checking
uint safe_bone_index(uint idx) {
	return min(idx, total_bones - 1u);
}

// ============================================================================
// Main Computation
// ============================================================================

void main() {
	uint vertex_idx = gl_GlobalInvocationID.x;

	// Bounds check
	if (vertex_idx >= vertex_count) {
		return;
	}

	// Load vertex data
	vec3 rest_pos = vertices[vertex_idx].xyz;
	vec4 bone_weights = weights[vertex_idx];
	uvec4 bone_indices = indices[vertex_idx];

	// Handle edge case: vertex with no bone influences
	if (!has_bone_influence(bone_weights)) {
		// Store zero displacements for all bones
		for (uint b = 0; b < max_bones; b++) {
			uint output_idx = vertex_idx * max_bones + b;
			displacements[output_idx] = mat4(0.0);
		}
		return;
	}

	// Normalize weights (account for floating point precision)
	float weight_sum = bone_weights.x + bone_weights.y +
			bone_weights.z + bone_weights.w;

	if (weight_sum > 1e-10) {
		bone_weights = bone_weights / weight_sum;
	}

	// Compute displacement for each bone influence
	for (uint b = 0; b < max_bones; b++) {
		float bone_weight = bone_weights[b];

		// Skip zero-weight bones
		if (bone_weight < 1e-10) {
			uint output_idx = vertex_idx * max_bones + b;
			displacements[output_idx] = mat4(0.0);
			continue;
		}

		// Get bone index with bounds checking
		uint bone_idx = safe_bone_index(bone_indices[b]);

		// Load scale/shear matrix for this bone
		mat4 M_sj = scale_shear[bone_idx];

		// Compute displacement: d_ij = M_sj * u_i - u_i
		vec4 disp_vec = compute_single_displacement(rest_pos, M_sj);

		// Scale by bone weight
		disp_vec = disp_vec * bone_weight;

		// Create displacement matrix
		mat4 disp_matrix = make_displacement_matrix(disp_vec);

		// Store result
		uint output_idx = vertex_idx * max_bones + b;
		displacements[output_idx] = disp_matrix;
	}
}
