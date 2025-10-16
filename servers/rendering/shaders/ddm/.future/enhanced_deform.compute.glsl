// Enhanced Direct Delta Mush - Enhanced Deformation Shader
// Applies Enhanced DDM equation with non-rigid joint transformations
// Equation: v'_i = Σ_j M_rj × D_ij × Ω_ij × p_i
// Where M_rj = rigid (rotation), D_ij = non-rigid displacement, Ω_ij = weights

#version 460

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input: Smoothed vertex positions (from Laplacian solve)
layout(std430, binding = 0) readonly buffer SmoothedPositions {
	vec4 smoothed[];
};

// Input: Rigid bone transformations (rotation + translation)
layout(std430, binding = 1) readonly buffer RigidTransforms {
	mat4 rigid[];
};

// Input: Non-rigid displacement matrices (per vertex per bone)
layout(std430, binding = 2) readonly buffer DisplacementMatrices {
	mat4 displacements[];
};

// Input: Omega weights (influence of each displacement on final deformation)
layout(std430, binding = 3) readonly buffer OmegaWeights {
	vec4 omega[]; // 4 bones per vertex max
};

// Input: Bone weights and indices
layout(std430, binding = 4) readonly buffer BoneWeights {
	vec4 weights[];
};

layout(std430, binding = 5) readonly buffer BoneIndices {
	uvec4 indices[];
};

// Input: Rest pose vertices (for reference)
layout(std430, binding = 6) readonly buffer RestVertices {
	vec4 rest_positions[];
};

// Output: Deformed vertex positions
layout(std430, binding = 7) writeonly buffer DeformedPositions {
	vec4 deformed[];
};

// Uniform: Parameters
layout(std140, binding = 0) uniform Config {
	uint vertex_count;
	uint max_bones;
	uint total_bones;
	float blend_factor; // Blend between standard DDM and Enhanced DDM (0.0 = pure enhanced)
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

// Transform point by 3x3 matrix only (linear transformation)
vec3 transform_linear(mat3 M, vec3 p) {
	return M * p;
}

// Transform point by 4x4 matrix (affine transformation)
vec3 transform_affine(mat4 M, vec3 p) {
	vec4 p_homo = vec4(p, 1.0);
	vec4 result = M * p_homo;
	return result.xyz / max(result.w, 1e-6);
}

// ============================================================================
// Enhanced DDM Deformation
// ============================================================================

// Apply Enhanced DDM: v'_i = Σ_j M_rj × (D_ij + I) × Ω_ij × p_i
vec3 compute_enhanced_deformation(
		vec3 smoothed_pos,
		uint vertex_idx,
		vec4 bone_weights,
		uvec4 bone_indices) {
	vec3 result = vec3(0.0);

	// Accumulate contributions from each bone
	for (uint b = 0; b < max_bones; b++) {
		float weight = bone_weights[b];

		// Skip zero-weight bones
		if (weight < 1e-10) {
			continue;
		}

		// Get bone index with bounds checking
		uint bone_idx = min(bone_indices[b], total_bones - 1u);

		// Load rigid transformation M_rj
		mat4 M_rigid = rigid[bone_idx];

		// Load non-rigid displacement matrix D_ij
		uint disp_idx = vertex_idx * max_bones + b;
		mat4 D = displacements[disp_idx];

		// Load omega weight Ω_ij
		float omega_weight = omega[disp_idx].x;

		// Compute displacement matrix component: (D_ij + I)
		// D_ij is stored as M_sj - I, so (D_ij + I) = M_sj
		mat4 M_sj = D + mat4(1.0); // Add identity to get M_sj

		// Apply transformations: M_rj × M_sj × p
		mat4 combined = M_rigid * M_sj;
		vec3 transformed = transform_affine(combined, smoothed_pos);

		// Weight by bone weight and omega
		result += transformed * weight * omega_weight;
	}

	return result;
}

// ============================================================================
// Standard DDM Deformation (Fallback)
// ============================================================================

// Standard DDM: v'_i = Σ_j M_j × p_i
vec3 compute_standard_deformation(
		vec3 smoothed_pos,
		uint vertex_idx,
		vec4 bone_weights,
		uvec4 bone_indices) {
	vec3 result = vec3(0.0);

	// Accumulate bone transformations (Linear Blend Skinning style)
	for (uint b = 0; b < max_bones; b++) {
		float weight = bone_weights[b];

		// Skip zero-weight bones
		if (weight < 1e-10) {
			continue;
		}

		// Get bone index with bounds checking
		uint bone_idx = min(bone_indices[b], total_bones - 1u);

		// Load rigid transformation
		mat4 M_rigid = rigid[bone_idx];

		// Apply transformation
		vec3 transformed = transform_affine(M_rigid, smoothed_pos);

		// Weight contribution
		result += transformed * weight;
	}

	return result;
}

// ============================================================================
// Detail Preservation (Optional)
// ============================================================================

// Compute detail delta (difference between rest and smoothed)
// For detail-preserving deformation, add back some of the lost detail
vec3 preserve_detail(
		vec3 rest_pos,
		vec3 smoothed_pos,
		vec3 deformed_pos) {
	// Detail = smoothed - rest (what was lost during smoothing)
	vec3 detail = smoothed_pos - rest_pos;

	// Add back a fraction of the detail
	// This helps prevent over-smoothing
	const float detail_blend = 0.5;

	return deformed_pos + detail * detail_blend;
}

// ============================================================================
// Edge Case Handling
// ============================================================================

// Check if vertex has meaningful bone influences
bool has_significant_influence(vec4 weights) {
	return (weights.x > 1e-10 || weights.y > 1e-10 ||
			weights.z > 1e-10 || weights.w > 1e-10);
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

	// Load input data
	vec3 smoothed_pos = smoothed[vertex_idx].xyz;
	vec3 rest_pos = rest_positions[vertex_idx].xyz;
	vec4 bone_weights = weights[vertex_idx];
	uvec4 bone_indices = indices[vertex_idx];

	// Handle edge case: vertex with no bone influences
	if (!has_significant_influence(bone_weights)) {
		// Output rest position unchanged
		deformed[vertex_idx] = vec4(smoothed_pos, 1.0);
		return;
	}

	// Normalize weights to ensure they sum to 1.0
	float weight_sum = bone_weights.x + bone_weights.y +
			bone_weights.z + bone_weights.w;

	if (weight_sum > 1e-10) {
		bone_weights = bone_weights / weight_sum;
	} else {
		// Fallback: uniform weights if all zero
		bone_weights = vec4(0.25, 0.25, 0.25, 0.25);
	}

	// Compute deformation
	vec3 enhanced_deform = compute_enhanced_deformation(
			smoothed_pos,
			vertex_idx,
			bone_weights,
			bone_indices);

	// Optional: Blend with standard DDM if needed
	vec3 standard_deform = compute_standard_deformation(
			smoothed_pos,
			vertex_idx,
			bone_weights,
			bone_indices);

	// Blend between enhanced and standard (blend_factor: 0.0 = pure enhanced)
	vec3 final_deform = mix(enhanced_deform, standard_deform, blend_factor);

	// Optional: Preserve detail from smoothing
	// Uncomment to enable detail preservation
	// final_deform = preserve_detail(rest_pos, smoothed_pos, final_deform);

	// Handle NaN/Inf
	if (any(isnan(final_deform)) || any(isinf(final_deform))) {
		final_deform = smoothed_pos; // Fallback to smoothed position
	}

	// Output deformed position
	deformed[vertex_idx] = vec4(final_deform, 1.0);
}
