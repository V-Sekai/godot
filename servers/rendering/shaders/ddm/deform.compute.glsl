#[compute]
#version 450

layout(local_size_x = 256) in;

// Input buffers
layout(set = 0, binding = 0) readonly buffer InputVertices {
    vec4 input_vertices[];
};

layout(set = 0, binding = 1) readonly buffer InputNormals {
    vec4 input_normals[];
};

layout(set = 0, binding = 2) readonly buffer BonesBuffer {
    mat4 bones[];
};

layout(set = 0, binding = 3) readonly buffer OmegaBuffer {
    float omegas[];
};

layout(set = 0, binding = 4) readonly buffer WeightsBuffer {
    vec4 weights[];
};

// Output buffers
layout(set = 0, binding = 5) writeonly buffer OutputVertices {
    vec4 output_vertices[];
};

layout(set = 0, binding = 6) writeonly buffer OutputNormals {
    vec4 output_normals[];
};

// Params
layout(push_constant) uniform Params {
    uint vertex_count;
    uint bone_count;
} params;

void main() {
    uint vertex_idx = gl_GlobalInvocationID.x;
    
    if (vertex_idx >= params.vertex_count) {
        return;
    }
    
    // Get input vertex and normal
    vec3 vertex = input_vertices[vertex_idx].xyz;
    vec3 normal = input_normals[vertex_idx].xyz;
    vec4 weights = weights[vertex_idx];
    
    // Get omega matrices and bone transforms
    vec3 deformed_vertex = vec3(0.0);
    vec3 deformed_normal = vec3(0.0);
    
    // Apply weighted blending using omega matrices
    for (uint bone = 0; bone < params.bone_count; bone++) {
        float weight = weights[bone % 4];
        if (weight > 0.0) {
            // Get omega matrix for this bone
            uint matrix_idx = (vertex_idx * params.bone_count + bone) * 16;
            mat4 omega_matrix = mat4(
                omegas[matrix_idx +  0], omegas[matrix_idx +  1], omegas[matrix_idx +  2], omegas[matrix_idx +  3],
                omegas[matrix_idx +  4], omegas[matrix_idx +  5], omegas[matrix_idx +  6], omegas[matrix_idx +  7],
                omegas[matrix_idx +  8], omegas[matrix_idx +  9], omegas[matrix_idx + 10], omegas[matrix_idx + 11],
                omegas[matrix_idx + 12], omegas[matrix_idx + 13], omegas[matrix_idx + 14], omegas[matrix_idx + 15]
            );
            
            // Apply bone transform with omega matrix blending
            mat4 bone_transform = bones[bone];
            vec3 transformed = (omega_matrix * bone_transform * vec4(vertex, 1.0)).xyz;
            vec3 transformed_normal = (omega_matrix * bone_transform * vec4(normal, 0.0)).xyz;
            
            deformed_vertex += transformed * weight;
            deformed_normal += transformed_normal * weight;
        }
    }
    
    // Write output
    output_vertices[vertex_idx] = vec4(deformed_vertex, 1.0);
    output_normals[vertex_idx] = vec4(normalize(deformed_normal), 0.0);
}
