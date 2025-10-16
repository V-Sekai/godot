/* ddm_mesh.cpp */

#include "ddm_mesh.h"

#include "core/config/engine.h"
#include "ddm_deformer.h"

void DDMMesh::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_precomputed_data", "adjacency", "laplacian", "omega"), &DDMMesh::set_precomputed_data);
    ClassDB::bind_method(D_METHOD("set_mesh_data", "source_mesh", "vertex_count", "bone_count"), &DDMMesh::set_mesh_data);
    ClassDB::bind_method(D_METHOD("update_deformation", "bone_transforms"), &DDMMesh::update_deformation);
}

DDMMesh::DDMMesh() {
    // Initialize with default values
}

DDMMesh::~DDMMesh() {
    // Cleanup will be handled by Resource base class
}

RID DDMMesh::get_rid() const {
    return ddm_mesh;
}

void DDMMesh::set_rid(RID p_rid) {
    ddm_mesh = p_rid;
}

void DDMMesh::set_precomputed_data(const Vector<int>& p_adjacency,
                                  const Vector<float>& p_laplacian,
                                  const Vector<float>& p_omega) {
    adjacency_matrix = p_adjacency;
    laplacian_matrix = p_laplacian;
    omega_matrices = p_omega;
}

void DDMMesh::set_mesh_data(RID p_source_mesh, int p_vertex_count, int p_bone_count) {
    source_mesh = p_source_mesh;
    vertex_count = p_vertex_count;
    bone_count = p_bone_count;
}

void DDMMesh::update_deformation(const Vector<Transform3D>& bone_transforms) {
    // TODO: Implement runtime deformation using precomputed data
    // This will use the DDMDeformer class to apply Direct Delta Mush algorithm

    if (omega_matrices.is_empty() || bone_transforms.is_empty()) {
        return;
    }

    // For now, this is a placeholder - actual implementation will use DDMDeformer
    // DDMDeformer deformer;
    // deformer.deform(bone_transforms, omega_matrices, vertex_count);
}
