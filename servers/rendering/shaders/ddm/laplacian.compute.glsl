// Enhanced Direct Delta Mush - Laplacian Shader with Emulated Double Precision
// Computes cotangent-weighted Laplacian for mesh smoothing
// Uses emulated double precision to prevent numerical issues and vertex degeneration
// Laplacian: L_ij = (cot(α) + cot(β)) / 2 for each edge

#version 460

#include "double_precision.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input: Vertex positions
layout(std430, binding = 0) readonly buffer VertexPositions {
	vec4 vertices[];
};

// Input: Triangle indices (3 indices per triangle, 3 components each)
layout(std430, binding = 1) readonly buffer TriangleIndices {
	uvec3 triangles[];
};

// Input: Edge metadata (stores angle pairs for cotangent computation)
layout(std430, binding = 2) readonly buffer EdgeMetadata {
	vec4 edge_angles[]; // packed: alpha_rad.x, alpha_rad.y, beta_rad.x, beta_rad.y
};

// Output: Laplacian weights (one per edge, per vertex pair)
layout(std430, binding = 3) writeonly buffer LaplacianWeights {
	float weights[]; // One weight per unique edge
};

// Output: Laplacian matrix structure (for validation)
// Stores: vertex_degree[i] = sum of absolute weights for vertex i
layout(std430, binding = 4) writeonly buffer VertexDegrees {
	float degrees[];
};

// Uniform: Parameters
layout(std140, binding = 0) uniform Config {
	uint vertex_count;
	uint edge_count;
	uint triangle_count;
	uint pad0;
};

// ============================================================================
// Triangle Geometry Utilities
// ============================================================================

// Compute angle at vertex between two edge vectors
float compute_angle(vec3 v0, vec3 v1, vec3 v2) {
	// Angle at v1 between edges (v1->v0) and (v1->v2)
	vec3 e0 = normalize(v0 - v1);
	vec3 e1 = normalize(v2 - v1);

	float cos_angle = clamp(dot(e0, e1), -1.0, 1.0);
	return acos(cos_angle);
}

// Compute cotangent of angle
// Using emulated double precision for accuracy
float compute_cotangent_accurate(float angle) {
	// Use emulated double precision for angle computation
	double_t angle_dbl = float_to_double(angle);
	double_t cot_dbl = double_cot(angle_dbl);

	// Convert back to float
	return double_to_float(cot_dbl);
}

// ============================================================================
// Laplacian Weight Computation
// ============================================================================

// Compute cotangent weight for a single edge
// Input: Two vertices forming an edge, and the opposite vertices in the two adjacent triangles
float compute_cotangent_weight(vec3 v_i, vec3 v_j, vec3 v_opposite1, vec3 v_opposite2) {
	// Compute angles opposite the edge (v_i, v_j)

	// Angle at v_opposite1 (in first adjacent triangle)
	float alpha = compute_angle(v_i, v_opposite1, v_j);

	// Angle at v_opposite2 (in second adjacent triangle)
	float beta = compute_angle(v_i, v_opposite2, v_j);

	// Avoid degenerate angles
	alpha = clamp(alpha, 1e-6, 3.14159 - 1e-6);
	beta = clamp(beta, 1e-6, 3.14159 - 1e-6);

	// Compute cotangent sum with double precision
	float cot_alpha = compute_cotangent_accurate(alpha);
	float cot_beta = compute_cotangent_accurate(beta);

	// Weight = (cot(α) + cot(β)) / 2
	float weight = (cot_alpha + cot_beta) * 0.5;

	// Clamp to reasonable range to prevent extreme values
	weight = clamp(weight, -1e6, 1e6);

	return weight;
}

// ============================================================================
// Laplacian Assembly
// ============================================================================

// Accumulate Laplacian contributions for a vertex
// L_i = sum_j w_ij * (v_j - v_i)
// Returns the sum of absolute weights (vertex degree for validation)

float compute_vertex_laplacian_degree(uint vertex_idx) {
	float degree = 0.0;

	// Iterate through all edges connected to this vertex
	// (Edge enumeration would be done in a separate preprocessing pass)
	// For now, we compute degree as we process triangles

	return degree;
}

// ============================================================================
// Edge-Based Processing (Alternative: Triangle-Based)
// ============================================================================

// Process triangle to extract edges and compute Laplacian weights
void process_triangle(uint tri_idx) {
	// Load triangle vertices
	uvec3 triangle = triangles[tri_idx];
	vec3 v0 = vertices[triangle.x].xyz;
	vec3 v1 = vertices[triangle.y].xyz;
	vec3 v2 = vertices[triangle.z].xyz;

	// For each edge of the triangle, compute cotangent weight
	// Edge (0,1), opposite vertex 2
	float angle_at_v2 = compute_angle(v0, v2, v1);
	angle_at_v2 = clamp(angle_at_v2, 1e-6, 3.14159 - 1e-6);
	float cot_at_v2 = compute_cotangent_accurate(angle_at_v2);

	// Edge (1,2), opposite vertex 0
	float angle_at_v0 = compute_angle(v1, v0, v2);
	angle_at_v0 = clamp(angle_at_v0, 1e-6, 3.14159 - 1e-6);
	float cot_at_v0 = compute_cotangent_accurate(angle_at_v0);

	// Edge (2,0), opposite vertex 1
	float angle_at_v1 = compute_angle(v2, v1, v0);
	angle_at_v1 = clamp(angle_at_v1, 1e-6, 3.14159 - 1e-6);
	float cot_at_v1 = compute_cotangent_accurate(angle_at_v1);

	// Note: These weights would be accumulated into edge storage
	// and later retrieved during deformation computation
}

// ============================================================================
// Main Computation
// ============================================================================

void main() {
	uint edge_idx = gl_GlobalInvocationID.x;

	// Bounds check
	if (edge_idx >= edge_count) {
		return;
	}

	// Load edge angle metadata
	vec4 angles = edge_angles[edge_idx];

	// Unpack angles (stored as components for double precision computation)
	float alpha_rad = sqrt(angles.x * angles.x + angles.y * angles.y);
	float beta_rad = sqrt(angles.z * angles.z + angles.w * angles.w);

	// Ensure valid angle range
	alpha_rad = clamp(alpha_rad, 1e-6, 3.14159 - 1e-6);
	beta_rad = clamp(beta_rad, 1e-6, 3.14159 - 1e-6);

	// Compute cotangent weight using emulated double precision
	// Weight = (cot(α) + cot(β)) / 2
	float cot_alpha = compute_cotangent_accurate(alpha_rad);
	float cot_beta = compute_cotangent_accurate(beta_rad);

	float weight = (cot_alpha + cot_beta) * 0.5;

	// Clamp to reasonable range to prevent inf/nan propagation
	weight = clamp(weight, -1e5, 1e5);

	// Handle special cases
	if (isnan(weight) || isinf(weight)) {
		weight = 0.0;
	}

	// Store Laplacian weight
	weights[edge_idx] = weight;

	// Update vertex degree (validation metric)
	// This would require atomic operations or synchronization
	// For now, we store the raw weight
	degrees[edge_idx] = abs(weight);
}
