#[compute]
#version 450

layout(local_size_x = 256) in;

// Input buffers
layout(set = 0, binding = 0) readonly buffer VertexBuffer {
    vec4 vertices[];
};

layout(set = 0, binding = 1) readonly buffer WeightsBuffer {
    vec4 weights[];
};

layout(set = 0, binding = 2) readonly buffer LaplacianBuffer {
    float laplacian[];
};

// Output buffer (omega matrices: 4x4 per vertex per bone)
layout(set = 0, binding = 3) writeonly buffer OmegaBuffer {
    float omegas[];
};

// Params
layout(push_constant) uniform Params {
    uint vertex_count;
    uint bone_count;
    uint max_neighbors;
    uint iterations;
    float lambda;
} params;

void main() {
    uint vertex_idx = gl_GlobalInvocationID.x;
    
    if (vertex_idx >= params.vertex_count) {
        return;
    }
    
    // For now, initialize omega matrices as identity matrices scaled by vertex weights
    // In a full implementation, this would solve the Laplacian system using AMGCL
    
    for (uint bone = 0; bone < params.bone_count; bone++) {
        // Get bone weight for this vertex
        vec4 bone_weights = weights[vertex_idx];
        float omega_value = bone_weights[bone % 4];
        
        // Create 4x4 omega matrix (identity scaled by weight for now)
        uint matrix_idx = (vertex_idx * params.bone_count + bone) * 16;
        
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                uint idx = matrix_idx + i * 4 + j;
                if (i == j) {
                    omegas[idx] = omega_value;
                } else {
                    omegas[idx] = 0.0;
                }
            }
        }
    }
}
