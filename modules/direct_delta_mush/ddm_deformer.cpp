/* ddm_deformer.cpp */

#include "ddm_deformer.h"

void DDMDeformer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("deform", "bone_transforms", "omega_matrices", "vertex_count"), &DDMDeformer::deform);
    ClassDB::bind_method(D_METHOD("set_original_mesh", "vertices", "normals"), &DDMDeformer::set_original_mesh);
}

bool DDMDeformer::deform(const Vector<Transform3D>& bone_transforms,
                        const Vector<float>& omega_matrices,
                        int vertex_count) {
    // Ported from Unity DDMSkinnedMeshGPUVar0.UpdateMeshOnCPU
    // Direct Delta Mush deformation with SVD

    if (original_vertices.is_empty() || bone_transforms.is_empty()) {
        return false;
    }

    // Ensure output arrays are sized correctly
    deformed_vertices.resize(original_vertices.size());
    deformed_normals.resize(original_normals.size());

    // Process each vertex
    for (int vi = 0; vi < vertex_count; vi++) {
        // Accumulate transformation from all influencing bones
        Transform3D accumulated_transform;

        // For each bone that influences this vertex
        for (int bi = 0; bi < bone_transforms.size(); bi++) {
            // Get omega matrix for this vertex-bone pair
            int omega_idx = (vi * 32 + bi) * 16; // 32 bones max, 16 floats per matrix

            if (omega_idx + 15 >= omega_matrices.size()) {
                continue; // No more omega matrices for this vertex
            }

            // Extract 4x4 omega matrix
            Transform3D omega_transform;
            omega_transform.basis.rows[0] = Vector3(omega_matrices[omega_idx + 0],
                                                   omega_matrices[omega_idx + 1],
                                                   omega_matrices[omega_idx + 2]);
            omega_transform.origin.x = omega_matrices[omega_idx + 3];

            omega_transform.basis.rows[1] = Vector3(omega_matrices[omega_idx + 4],
                                                   omega_matrices[omega_idx + 5],
                                                   omega_matrices[omega_idx + 6]);
            omega_transform.origin.y = omega_matrices[omega_idx + 7];

            omega_transform.basis.rows[2] = Vector3(omega_matrices[omega_idx + 8],
                                                   omega_matrices[omega_idx + 9],
                                                   omega_matrices[omega_idx + 10]);
            omega_transform.origin.z = omega_matrices[omega_idx + 11];

            // Apply bone transformation: bone_transform * omega_transform
            Transform3D bone_omega = bone_transforms[bi] * omega_transform;

            // Accumulate (this is simplified - should be weighted)
            accumulated_transform = accumulated_transform * bone_omega;
        }

        // Apply SVD-based deformation (simplified for now)
        // TODO: Implement proper SVD decomposition
        // For now, just apply the accumulated transformation

        Vector3 original_pos = original_vertices[vi];
        Vector3 original_normal = original_normals[vi];

        // Transform position and normal
        deformed_vertices.set(vi, accumulated_transform.xform(original_pos));
        deformed_normals.set(vi, accumulated_transform.basis.xform(original_normal).normalized());
    }

    return true;
}

void DDMDeformer::set_original_mesh(const Vector<Vector3>& vertices, const Vector<Vector3>& normals) {
    original_vertices = vertices;
    original_normals = normals;
    deformed_vertices.resize(vertices.size());
    deformed_normals.resize(normals.size());
}
