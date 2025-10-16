/* ddm_mesh.h */

#ifndef DDM_MESH_H
#define DDM_MESH_H

#include "scene/resources/mesh.h"
#include "core/templates/hash_map.h"

// DDMMesh is a resource that contains precomputed Direct Delta Mush data
class DDMMesh : public Resource {
    GDCLASS(DDMMesh, Resource);

private:
    RID source_mesh; // Original mesh with bone weights
    RID ddm_mesh;    // Processed mesh for rendering

    // Precomputed data
    Vector<int> adjacency_matrix;
    Vector<float> laplacian_matrix;
    Vector<float> omega_matrices;

    int vertex_count = 0;
    int bone_count = 0;

protected:
    static void _bind_methods();

public:
    DDMMesh();
    ~DDMMesh();

    RID get_rid() const;
    void set_rid(RID p_rid);

    // Precomputation
    void set_precomputed_data(const Vector<int>& p_adjacency,
                             const Vector<float>& p_laplacian,
                             const Vector<float>& p_omega);
    void set_mesh_data(RID p_source_mesh, int p_vertex_count, int p_bone_count);

    // Data access
    const Vector<int>& get_adjacency_matrix() const { return adjacency_matrix; }
    const Vector<float>& get_laplacian_matrix() const { return laplacian_matrix; }
    const Vector<float>& get_omega_matrices() const { return omega_matrices; }

    int get_vertex_count() const { return vertex_count; }
    int get_bone_count() const { return bone_count; }

    // Runtime deformation
    void update_deformation(const Vector<Transform3D>& bone_transforms);
};

#endif // DDM_MESH_H
