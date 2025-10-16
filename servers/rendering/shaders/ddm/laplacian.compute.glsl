#[compute]
#version 450

layout(local_size_x = 256) in;

// Input buffer (adjacency matrix)
layout(set = 0, binding = 0) readonly buffer AdjacencyBuffer {
    int adjacency[];
};

// Output buffer (Laplacian matrix stored as [index, weight] pairs)
layout(set = 0, binding = 1) writeonly buffer LaplacianBuffer {
    float laplacian[];
};

// Params
layout(push_constant) uniform Params {
    uint vertex_count;
    uint max_neighbors;
} params;

void main() {
    uint vertex_idx = gl_GlobalInvocationID.x;
    
    if (vertex_idx >= params.vertex_count) {
        return;
    }
    
    // Calculate vertex degree (number of neighbors)
    int degree = 0;
    for (uint j = 0; j < params.max_neighbors; j++) {
        int neighbor = adjacency[vertex_idx * params.max_neighbors + j];
        if (neighbor < 0) {
            break;
        }
        degree++;
    }
    
    if (degree == 0) {
        return; // Isolated vertex
    }
    
    // Build Laplacian row for this vertex
    // For normalized Laplacian: L[i,j] = -1/deg(i) for neighbors, L[i,i] = 1
    uint laplacian_row_start = vertex_idx * params.max_neighbors * 2;
    
    for (uint j = 0; j < params.max_neighbors; j++) {
        int neighbor_idx = adjacency[vertex_idx * params.max_neighbors + j];
        if (neighbor_idx < 0) {
            break;
        }
        
        // Store neighbor index and weight
        uint entry_idx = laplacian_row_start + j * 2;
        laplacian[entry_idx] = float(neighbor_idx);
        laplacian[entry_idx + 1] = -1.0 / float(degree);
    }
    
    // Set diagonal element (L[i,i] = 1 for normalized Laplacian)
    uint diagonal_idx = laplacian_row_start + (params.max_neighbors - 1) * 2;
    laplacian[diagonal_idx] = float(vertex_idx);
    laplacian[diagonal_idx + 1] = 1.0;
}
