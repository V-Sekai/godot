#[compute]

#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Input buffers
layout(set = 0, binding = 0) readonly buffer Vertices {
    vec3 vertices[];
};

layout(set = 0, binding = 1) readonly buffer Weights {
    vec4 weights[];  // bone weights (xyzw)
    ivec4 bones[];   // bone indices (xyzw)
};

layout(set = 0, binding = 2) readonly buffer Laplacian {
    float laplacian[]; // [index, weight] pairs
};

// Output buffer
layout(set = 0, binding = 3) writeonly buffer Omegas {
    float omegas[]; // 4x4 matrices as 16 floats each
};

// Uniforms
layout(set = 0, binding = 4) uniform Params {
    uint vertex_count;
    uint bone_count;
    uint max_neighbors;
    uint iterations;
    float lambda;
};

void main() {
    uint vertex_id = gl_GlobalInvocationID.x;

    if (vertex_id >= vertex_count) {
        return;
    }

    vec3 vertex_pos = vertices[vertex_id];
    vec4 bone_weights = weights[vertex_id];
    ivec4 bone_indices = bones[vertex_id];

    // Initialize omega matrices for this vertex
    // For each bone that influences this vertex, create initial omega matrix
    for (uint bi = 0; bi < 4; bi++) { // Max 4 bones per vertex
        int bone_idx = bone_indices[bi];
        float weight = bone_weights[bi];

        if (bone_idx < 0 || weight <= 0.0) {
            continue;
        }

        // Create initial omega matrix (position-weighted identity)
        uint omega_idx = (vertex_id * 32 + bone_idx) * 16; // 32 bones max, 16 floats per matrix

        // Identity matrix with position
        omegas[omega_idx + 0] = weight;  // m00
        omegas[omega_idx + 1] = 0.0;     // m01
        omegas[omega_idx + 2] = 0.0;     // m02
        omegas[omega_idx + 3] = vertex_pos.x * weight; // m03

        omegas[omega_idx + 4] = 0.0;     // m10
        omegas[omega_idx + 5] = weight;  // m11
        omegas[omega_idx + 6] = 0.0;     // m12
        omegas[omega_idx + 7] = vertex_pos.y * weight; // m13

        omegas[omega_idx + 8] = 0.0;     // m20
        omegas[omega_idx + 9] = 0.0;     // m21
        omegas[omega_idx + 10] = weight; // m22
        omegas[omega_idx + 11] = vertex_pos.z * weight; // m23

        omegas[omega_idx + 12] = 0.0;    // m30
        omegas[omega_idx + 13] = 0.0;    // m31
        omegas[omega_idx + 14] = 0.0;    // m32
        omegas[omega_idx + 15] = weight; // m33
    }

    // TODO: Implement iterative Laplacian smoothing
    // This requires multiple passes and careful synchronization
    // For now, we use the initial omega matrices
}
