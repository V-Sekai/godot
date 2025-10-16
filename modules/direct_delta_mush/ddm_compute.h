/* ddm_compute.h */

#ifndef DDM_COMPUTE_H
#define DDM_COMPUTE_H

#include "core/object/ref_counted.h"
#include "servers/rendering/rendering_device.h"

class DDMCompute : public RefCounted {
    GDCLASS(DDMCompute, RefCounted);

private:
    RenderingDevice *rd = nullptr;

    // Compute pipelines
    RID adjacency_pipeline;
    RID laplacian_pipeline;
    RID omega_pipeline;
    RID deform_pipeline;

    // Shader RIDs
    RID adjacency_shader;
    RID laplacian_shader;
    RID omega_shader;
    RID deform_shader;

public:
    bool initialize(RenderingDevice *p_rd);

    // Compute operations
    bool compute_adjacency(const RID &vertex_buffer, const RID &index_buffer,
                          RID &output_buffer, int vertex_count);
    bool compute_laplacian(const RID &adjacency_buffer, RID &output_buffer, int vertex_count);
    bool compute_omega_matrices(const RID &laplacian_buffer, const RID &vertex_buffer,
                               const RID &weights_buffer, RID &output_buffer,
                               int vertex_count, int bone_count, int iterations, float lambda);
    bool deform_mesh(const RID &omega_buffer, const RID &bones_buffer,
                    const RID &input_vertices, const RID &input_normals,
                    RID &output_vertices, RID &output_normals, int vertex_count);

    void cleanup();

protected:
    static void _bind_methods();

private:
    RID load_shader(const String &shader_code);
    bool create_pipeline(RID &pipeline, RID shader, const Vector<StringName> &uniform_names);
};

#endif // DDM_COMPUTE_H
