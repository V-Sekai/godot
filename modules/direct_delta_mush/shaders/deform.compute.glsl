#[compute]

#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Input buffers
layout(set = 0, binding = 0) readonly buffer Vertices {
    vec3 vertices[];
};

layout(set = 0, binding = 1) readonly buffer Normals {
    vec3 normals[];
};

layout(set = 0, binding = 2) readonly buffer Bones {
    mat4 bones[]; // bone transformation matrices
};

layout(set = 0, binding = 3) readonly buffer Omegas {
    float omegas[]; // omega matrices (4x4 as 16 floats each)
};

// Output buffers
layout(set = 0, binding = 4) writeonly buffer OutputVertices {
    vec3 output_vertices[];
};

layout(set = 0, binding = 5) writeonly buffer OutputNormals {
    vec3 output_normals[];
};

// Uniforms
layout(set = 0, binding = 6) uniform Params {
    uint vertex_count;
    uint bone_count;
};

mat4 extract_omega_matrix(uint vertex_id, uint bone_id) {
    uint omega_idx = (vertex_id * 32 + bone_id) * 16; // 32 bones max, 16 floats per matrix

    // Extract 4x4 matrix from flat array
    mat4 omega = mat4(
        omegas[omega_idx + 0], omegas[omega_idx + 1], omegas[omega_idx + 2], omegas[omega_idx + 3],
        omegas[omega_idx + 4], omegas[omega_idx + 5], omegas[omega_idx + 6], omegas[omega_idx + 7],
        omegas[omega_idx + 8], omegas[omega_idx + 9], omegas[omega_idx + 10], omegas[omega_idx + 11],
        omegas[omega_idx + 12], omegas[omega_idx + 13], omegas[omega_idx + 14], omegas[omega_idx + 15]
    );

    return omega;
}

void main() {
    uint vertex_id = gl_GlobalInvocationID.x;

    if (vertex_id >= vertex_count) {
        return;
    }

    vec3 original_vertex = vertices[vertex_id];
    vec3 original_normal = normals[vertex_id];

    // Accumulate transformation from all bones
    mat4 accumulated_transform = mat4(0.0);

    // For each bone that could influence this vertex
    for (uint bi = 0; bi < bone_count; bi++) {
        mat4 bone_transform = bones[bi];
        mat4 omega_matrix = extract_omega_matrix(vertex_id, bi);

        // Check if omega matrix is valid (non-zero)
        bool has_influence = false;
        for (uint i = 0; i < 16; i++) {
            if (abs(omega_matrix[i/4][i%4]) > 1e-6) {
                has_influence = true;
                break;
            }
        }

        if (has_influence) {
            // Apply bone transformation to omega matrix
            mat4 bone_omega = bone_transform * omega_matrix;
            accumulated_transform += bone_omega;
        }
    }

    // Apply accumulated transformation to vertex and normal
    vec4 transformed_vertex = accumulated_transform * vec4(original_vertex, 1.0);
    vec4 transformed_normal = accumulated_transform * vec4(original_normal, 0.0);

    // Store results
    output_vertices[vertex_id] = transformed_vertex.xyz;
    output_normals[vertex_id] = normalize(transformed_normal.xyz);
}
