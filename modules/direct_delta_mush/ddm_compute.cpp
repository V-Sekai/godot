/* ddm_compute.cpp */

#include "ddm_compute.h"

void DDMCompute::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize", "rendering_device"), &DDMCompute::initialize);
    ClassDB::bind_method(D_METHOD("cleanup"), &DDMCompute::cleanup);
}

bool DDMCompute::initialize(RenderingDevice *p_rd) {
    rd = p_rd;
    if (!rd) {
        return false;
    }

    // Load and compile compute shaders
    adjacency_shader = load_shader_from_file("shaders/adjacency.compute.glsl");
    laplacian_shader = load_shader_from_file("shaders/laplacian.compute.glsl");
    omega_shader = load_shader_from_file("shaders/omega_precompute.compute.glsl");
    deform_shader = load_shader_from_file("shaders/deform.compute.glsl");

    if (adjacency_shader.is_null() || laplacian_shader.is_null() ||
        omega_shader.is_null() || deform_shader.is_null()) {
        ERR_PRINT("Failed to load Direct Delta Mush compute shaders");
        return false;
    }

    // Create compute pipelines
    Vector<StringName> adjacency_uniforms = {"Vertices", "Indices", "Adjacency", "Params"};
    Vector<StringName> laplacian_uniforms = {"Adjacency", "Laplacian", "Params"};
    Vector<StringName> omega_uniforms = {"Vertices", "Weights", "Laplacian", "Omegas", "Params"};
    Vector<StringName> deform_uniforms = {"Vertices", "Normals", "Bones", "Omegas", "OutputVertices", "OutputNormals", "Params"};

    if (!create_pipeline(adjacency_pipeline, adjacency_shader, adjacency_uniforms) ||
        !create_pipeline(laplacian_pipeline, laplacian_shader, laplacian_uniforms) ||
        !create_pipeline(omega_pipeline, omega_shader, omega_uniforms) ||
        !create_pipeline(deform_pipeline, deform_shader, deform_uniforms)) {
        ERR_PRINT("Failed to create Direct Delta Mush compute pipelines");
        return false;
    }

    return true;
}

bool DDMCompute::compute_adjacency(const RID &vertex_buffer, const RID &index_buffer,
                                  RID &output_buffer, int vertex_count) {
    // TODO: Implement adjacency matrix computation on GPU
    return true;
}

bool DDMCompute::compute_laplacian(const RID &adjacency_buffer, RID &output_buffer, int vertex_count) {
    // TODO: Implement Laplacian matrix computation on GPU
    return true;
}

bool DDMCompute::compute_omega_matrices(const RID &laplacian_buffer, const RID &vertex_buffer,
                                       const RID &weights_buffer, RID &output_buffer,
                                       int vertex_count, int bone_count, int iterations, float lambda) {
    // TODO: Implement Omega matrix precomputation on GPU
    return true;
}

bool DDMCompute::deform_mesh(const RID &omega_buffer, const RID &bones_buffer,
                            const RID &input_vertices, const RID &input_normals,
                            RID &output_vertices, RID &output_normals, int vertex_count) {
    // TODO: Implement runtime mesh deformation on GPU
    return true;
}

void DDMCompute::cleanup() {
    // TODO: Clean up GPU resources
    if (rd) {
        // rd->free_rid(adjacency_pipeline);
        // rd->free_rid(laplacian_pipeline);
        // rd->free_rid(omega_pipeline);
        // rd->free_rid(deform_pipeline);
        // rd->free_rid(adjacency_shader);
        // rd->free_rid(laplacian_shader);
        // rd->free_rid(omega_shader);
        // rd->free_rid(deform_shader);
    }
}

RID DDMCompute::load_shader_from_file(const String &shader_path) {
    if (!rd) {
        return RID();
    }

    // Load shader file
    Ref<FileAccess> file = FileAccess::open(shader_path, FileAccess::READ);
    if (file.is_null()) {
        ERR_PRINT("Failed to open shader file: " + shader_path);
        return RID();
    }

    String shader_code = file->get_as_text();

    // Create shader from source
    return load_shader(shader_code);
}

RID DDMCompute::load_shader(const String &shader_code) {
    if (!rd) {
        return RID();
    }

    // Compile GLSL compute shader
    Vector<uint8_t> spirv;
    String error;
    bool compile_ok = rd->shader_compile_spirv_from_source(
        RD::ShaderStage::SHADER_STAGE_COMPUTE,
        shader_code,
        RD::ShaderLanguage::SHADER_LANGUAGE_GLSL,
        spirv,
        error
    );

    if (!compile_ok) {
        ERR_PRINT("Failed to compile compute shader: " + error);
        return RID();
    }

    // Create shader RID
    return rd->shader_create_from_spirv(spirv);
}

bool DDMCompute::create_pipeline(RID &pipeline, RID shader, const Vector<StringName> &uniform_names) {
    if (!rd || shader.is_null()) {
        return false;
    }

    // Create compute pipeline
    pipeline = rd->compute_pipeline_create(shader);
    return !pipeline.is_null();
}

bool DDMCompute::compute_adjacency(const RID &vertex_buffer, const RID &index_buffer,
                                  RID &output_buffer, int vertex_count) {
    if (!rd || adjacency_pipeline.is_null()) {
        return false;
    }

    // Set up compute list
    uint32_t compute_list = rd->compute_list_begin();

    // Bind pipeline
    rd->compute_list_bind_compute_pipeline(compute_list, adjacency_pipeline);

    // Bind buffers
    rd->compute_list_bind_uniform_buffer(compute_list, vertex_buffer, 0, 0);
    rd->compute_list_bind_uniform_buffer(compute_list, index_buffer, 0, 1);
    rd->compute_list_bind_uniform_buffer(compute_list, output_buffer, 0, 2);

    // Set uniforms
    struct Params {
        uint32_t vertex_count;
        uint32_t index_count;
        uint32_t max_neighbors;
        float tolerance;
    } params = { (uint32_t)vertex_count, 0, 32, 1e-4f };

    rd->compute_list_set_push_constant(compute_list, &params, sizeof(Params));

    // Dispatch compute
    uint32_t workgroup_count = (vertex_count + 63) / 64; // 64 threads per workgroup
    rd->compute_list_dispatch(compute_list, workgroup_count, 1, 1);

    // End compute list
    rd->compute_list_end();

    return true;
}

bool DDMCompute::compute_laplacian(const RID &adjacency_buffer, RID &output_buffer, int vertex_count) {
    if (!rd || laplacian_pipeline.is_null()) {
        return false;
    }

    uint32_t compute_list = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(compute_list, laplacian_pipeline);

    rd->compute_list_bind_uniform_buffer(compute_list, adjacency_buffer, 0, 0);
    rd->compute_list_bind_uniform_buffer(compute_list, output_buffer, 0, 1);

    struct Params {
        uint32_t vertex_count;
        uint32_t max_neighbors;
        uint32_t entries_per_vertex;
    } params = { (uint32_t)vertex_count, 32, 64 }; // 32 neighbors * 2 (index+weight)

    rd->compute_list_set_push_constant(compute_list, &params, sizeof(Params));

    uint32_t workgroup_count = (vertex_count + 63) / 64;
    rd->compute_list_dispatch(compute_list, workgroup_count, 1, 1);

    rd->compute_list_end();

    return true;
}

bool DDMCompute::compute_omega_matrices(const RID &laplacian_buffer, const RID &vertex_buffer,
                                       const RID &weights_buffer, RID &output_buffer,
                                       int vertex_count, int bone_count, int iterations, float lambda) {
    if (!rd || omega_pipeline.is_null()) {
        return false;
    }

    uint32_t compute_list = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(compute_list, omega_pipeline);

    rd->compute_list_bind_uniform_buffer(compute_list, vertex_buffer, 0, 0);
    rd->compute_list_bind_uniform_buffer(compute_list, weights_buffer, 0, 1);
    rd->compute_list_bind_uniform_buffer(compute_list, laplacian_buffer, 0, 2);
    rd->compute_list_bind_uniform_buffer(compute_list, output_buffer, 0, 3);

    struct Params {
        uint32_t vertex_count;
        uint32_t bone_count;
        uint32_t max_neighbors;
        uint32_t iterations;
        float lambda;
    } params = { (uint32_t)vertex_count, (uint32_t)bone_count, 32, (uint32_t)iterations, lambda };

    rd->compute_list_set_push_constant(compute_list, &params, sizeof(Params));

    uint32_t workgroup_count = (vertex_count + 63) / 64;
    rd->compute_list_dispatch(compute_list, workgroup_count, 1, 1);

    rd->compute_list_end();

    return true;
}

bool DDMCompute::deform_mesh(const RID &omega_buffer, const RID &bones_buffer,
                            const RID &input_vertices, const RID &input_normals,
                            RID &output_vertices, RID &output_normals, int vertex_count) {
    if (!rd || deform_pipeline.is_null()) {
        return false;
    }

    uint32_t compute_list = rd->compute_list_begin();
    rd->compute_list_bind_compute_pipeline(compute_list, deform_pipeline);

    rd->compute_list_bind_uniform_buffer(compute_list, input_vertices, 0, 0);
    rd->compute_list_bind_uniform_buffer(compute_list, input_normals, 0, 1);
    rd->compute_list_bind_uniform_buffer(compute_list, bones_buffer, 0, 2);
    rd->compute_list_bind_uniform_buffer(compute_list, omega_buffer, 0, 3);
    rd->compute_list_bind_uniform_buffer(compute_list, output_vertices, 0, 4);
    rd->compute_list_bind_uniform_buffer(compute_list, output_normals, 0, 5);

    struct Params {
        uint32_t vertex_count;
        uint32_t bone_count;
    } params = { (uint32_t)vertex_count, 32 }; // Assume max 32 bones

    rd->compute_list_set_push_constant(compute_list, &params, sizeof(Params));

    uint32_t workgroup_count = (vertex_count + 63) / 64;
    rd->compute_list_dispatch(compute_list, workgroup_count, 1, 1);

    rd->compute_list_end();

    return true;
}
