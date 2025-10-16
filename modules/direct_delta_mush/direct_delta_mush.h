/* direct_delta_mush.h */

#ifndef DIRECT_DELTA_MUSH_H
#define DIRECT_DELTA_MUSH_H

#include "scene/3d/mesh_instance_3d.h"
#include "servers/rendering/rendering_device.h"

class DirectDeltaMush : public MeshInstance3D {
    GDCLASS(DirectDeltaMush, MeshInstance3D);

private:
    // Direct Delta Mush parameters
    int iterations = 30;
    float smooth_lambda = 0.9f;
    float adjacency_tolerance = 1e-4f;
    bool use_compute = true;

    // Precomputed data
    RID omega_buffer;
    RID adjacency_buffer;
    RID laplacian_buffer;

    // Runtime state
    Ref<Mesh> deformed_mesh;
    RenderingDevice *rd = nullptr;

    // Internal methods
    void precompute_data();
    void update_deformation();
    void build_adjacency_matrix();
    void compute_laplacian_matrix();
    void precompute_omega_matrices();

protected:
    static void _bind_methods();
    void _notification(int p_what);

public:
    DirectDeltaMush();
    ~DirectDeltaMush();

    // Property setters/getters
    void set_iterations(int p_iterations);
    int get_iterations() const;

    void set_smooth_lambda(float p_lambda);
    float get_smooth_lambda() const;

    void set_adjacency_tolerance(float p_tolerance);
    float get_adjacency_tolerance() const;

    void set_use_compute(bool p_use_compute);
    bool get_use_compute() const;

    // Public methods
    void precompute();
};

#endif // DIRECT_DELTA_MUSH_H
