/* ddm_deformer.h */

#ifndef DDM_DEFORMER_H
#define DDM_DEFORMER_H

#include "core/object/ref_counted.h"

class DDMDeformer : public RefCounted {
    GDCLASS(DDMDeformer, RefCounted);

private:
    // Deformation data
    Vector<Vector3> original_vertices;
    Vector<Vector3> original_normals;
    Vector<Vector3> deformed_vertices;
    Vector<Vector3> deformed_normals;

public:
    // Runtime deformation
    bool deform(const Vector<Transform3D>& bone_transforms,
                const Vector<float>& omega_matrices,
                int vertex_count);

    // Data access
    void set_original_mesh(const Vector<Vector3>& vertices, const Vector<Vector3>& normals);
    const Vector<Vector3>& get_deformed_vertices() const { return deformed_vertices; }
    const Vector<Vector3>& get_deformed_normals() const { return deformed_normals; }

protected:
    static void _bind_methods();
};

#endif // DDM_DEFORMER_H
