/* ddm_precomputer.h */

#ifndef DDM_PRECOMPUTER_H
#define DDM_PRECOMPUTER_H

#include "core/object/ref_counted.h"
#include "scene/resources/mesh.h"

class DDMPrecomputer : public RefCounted {
    GDCLASS(DDMPrecomputer, RefCounted);

private:
    // Precomputation data structures
    Vector<int> adjacency_matrix;
    Vector<float> laplacian_matrix;
    Vector<float> omega_matrices;

public:
    // Main precomputation methods
    bool precompute(const Ref<Mesh>& mesh, int iterations, float lambda, float tolerance);

    // Individual computation steps
    bool build_adjacency_matrix(const Ref<Mesh>& mesh, float tolerance);
    bool compute_laplacian_matrix();
    bool precompute_omega_matrices(const Ref<Mesh>& mesh, int iterations, float lambda);

    // Data access
    const Vector<int>& get_adjacency_matrix() const { return adjacency_matrix; }
    const Vector<float>& get_laplacian_matrix() const { return laplacian_matrix; }
    const Vector<float>& get_omega_matrices() const { return omega_matrices; }

protected:
    static void _bind_methods();
};

#endif // DDM_PRECOMPUTER_H
