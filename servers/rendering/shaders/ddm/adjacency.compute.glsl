#[compute]
#version 450

layout(local_size_x = 256) in;

// Input buffers
layout(set = 0, binding = 0) readonly buffer VertexBuffer {
	vec4 vertices[];
};

layout(set = 0, binding = 1) readonly buffer IndexBuffer {
	uint indices[];
};

// Output buffer (adjacency matrix in CSR format)
layout(set = 0, binding = 2) writeonly buffer AdjacencyBuffer {
	int adjacency[];
};

// Params
layout(push_constant) uniform Params {
	uint vertex_count;
	uint index_count;
	uint max_neighbors;
}
params;

shared int shared_adjacency[256 * 32]; // Local cache for thread group

void main() {
	uint gid = gl_GlobalInvocationID.x;
	uint lid = gl_LocalInvocationID.x;

	// Initialize local adjacency for this thread's vertices
	uint vertex_idx = gid;
	if (vertex_idx >= params.vertex_count) {
		return;
	}

	// Initialize with -1 (empty)
	for (uint i = 0; i < params.max_neighbors; i++) {
		shared_adjacency[lid * params.max_neighbors + i] = -1;
	}

	// Process triangles that reference this vertex
	uint neighbor_count = 0;
	for (uint tri = 0; tri < params.index_count / 3; tri++) {
		uint i0 = indices[tri * 3];
		uint i1 = indices[tri * 3 + 1];
		uint i2 = indices[tri * 3 + 2];

		if (i0 == vertex_idx || i1 == vertex_idx || i2 == vertex_idx) {
			// Add neighboring vertices
			if (i0 != vertex_idx && neighbor_count < params.max_neighbors) {
				shared_adjacency[lid * params.max_neighbors + neighbor_count] = int(i0);
				neighbor_count++;
			}
			if (i1 != vertex_idx && neighbor_count < params.max_neighbors) {
				shared_adjacency[lid * params.max_neighbors + neighbor_count] = int(i1);
				neighbor_count++;
			}
			if (i2 != vertex_idx && neighbor_count < params.max_neighbors) {
				shared_adjacency[lid * params.max_neighbors + neighbor_count] = int(i2);
				neighbor_count++;
			}
		}
	}

	barrier();

	// Write back to global memory
	for (uint i = 0; i < params.max_neighbors; i++) {
		adjacency[vertex_idx * params.max_neighbors + i] = shared_adjacency[lid * params.max_neighbors + i];
	}
}
