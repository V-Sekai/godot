#[compute]

#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Input buffers
layout(set = 0, binding = 0) readonly buffer Vertices {
    vec3 vertices[];
};

layout(set = 0, binding = 1) readonly buffer Indices {
    uint indices[];
};

// Output buffer
layout(set = 0, binding = 2) writeonly buffer Adjacency {
    int adjacency[];
};

// Uniforms
layout(set = 0, binding = 3) uniform Params {
    uint vertex_count;
    uint index_count;
    uint max_neighbors;
    float tolerance;
};

// Shared memory for vertex positions
shared vec3 shared_vertices[64];

void main() {
    uint vertex_id = gl_GlobalInvocationID.x;

    if (vertex_id >= vertex_count) {
        return;
    }

    // Load vertex position
    vec3 vertex_pos = vertices[vertex_id];

    // Initialize adjacency list for this vertex
    uint adjacency_start = vertex_id * max_neighbors;

    // Clear adjacency list
    for (uint i = 0; i < max_neighbors; i++) {
        adjacency[adjacency_start + i] = -1;
    }

    // Find adjacent vertices by checking triangles
    uint neighbor_count = 0;

    for (uint tri = 0; tri < index_count && neighbor_count < max_neighbors; tri += 3) {
        uint i0 = indices[tri];
        uint i1 = indices[tri + 1];
        uint i2 = indices[tri + 2];

        // Check if this vertex is part of this triangle
        if (i0 == vertex_id || i1 == vertex_id || i2 == vertex_id) {
            // Add the other two vertices as neighbors
            if (i0 == vertex_id) {
                add_neighbor(i1);
                add_neighbor(i2);
            } else if (i1 == vertex_id) {
                add_neighbor(i0);
                add_neighbor(i2);
            } else { // i2 == vertex_id
                add_neighbor(i0);
                add_neighbor(i1);
            }
        }
    }
}

void add_neighbor(uint neighbor_id) {
    if (neighbor_id == gl_GlobalInvocationID.x) {
        return; // Don't add self
    }

    uint adjacency_start = gl_GlobalInvocationID.x * max_neighbors;

    // Check if already added
    for (uint i = 0; i < max_neighbors; i++) {
        if (adjacency[adjacency_start + i] == int(neighbor_id)) {
            return; // Already exists
        }
        if (adjacency[adjacency_start + i] == -1) {
            // Add new neighbor
            adjacency[adjacency_start + i] = int(neighbor_id);
            return;
        }
    }
}
