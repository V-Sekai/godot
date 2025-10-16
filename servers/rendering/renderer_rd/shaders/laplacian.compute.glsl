#[compute]

#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Input buffer
layout(set = 0, binding = 0) readonly buffer Adjacency {
	int adjacency[];
};

// Output buffer - Laplacian matrix as [index, weight] pairs
layout(set = 0, binding = 1) writeonly buffer Laplacian {
	float laplacian[];
};

// Uniforms
layout(set = 0, binding = 2) uniform Params {
	uint vertex_count;
	uint max_neighbors;
	uint entries_per_vertex; // max_neighbors * 2 (index + weight)
};

void main() {
	uint vertex_id = gl_GlobalInvocationID.x;

	if (vertex_id >= vertex_count) {
		return;
	}

	uint adjacency_start = vertex_id * max_neighbors;
	uint laplacian_start = vertex_id * entries_per_vertex;

	// Calculate vertex degree (number of neighbors)
	uint degree = 0;
	for (uint i = 0; i < max_neighbors; i++) {
		if (adjacency[adjacency_start + i] >= 0) {
			degree++;
		} else {
			break;
		}
	}

	if (degree == 0) {
		// Isolated vertex - set diagonal to 1
		uint diagonal_idx = laplacian_start + (max_neighbors - 1) * 2;
		laplacian[diagonal_idx] = float(vertex_id); // self index
		laplacian[diagonal_idx + 1] = 1.0; // weight = 1.0
		return;
	}

	// Build Laplacian entries for neighbors
	for (uint i = 0; i < max_neighbors; i++) {
		int neighbor_idx = adjacency[adjacency_start + i];
		if (neighbor_idx < 0) {
			break;
		}

		uint entry_idx = laplacian_start + i * 2;
		laplacian[entry_idx] = float(neighbor_idx); // neighbor index
		laplacian[entry_idx + 1] = -1.0 / float(degree); // weight = -1/degree
	}

	// Set diagonal element (normalized Laplacian)
	uint diagonal_idx = laplacian_start + (max_neighbors - 1) * 2;
	laplacian[diagonal_idx] = float(vertex_id); // self index
	laplacian[diagonal_idx + 1] = 1.0; // weight = 1.0
}
