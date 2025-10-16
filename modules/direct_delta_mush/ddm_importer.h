/* ddm_importer.h */

#ifndef DDM_IMPORTER_H
#define DDM_IMPORTER_H

#include "core/object/ref_counted.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/mesh.h"

#include "ddm_mesh.h"

class DDMImporter : public RefCounted {
    GDCLASS(DDMImporter, RefCounted);

public:
    enum ImportMode {
        IMPORT_TIME_PRECOMPUTE = 0,  // Precompute everything at import time
        RUNTIME_PRECOMPUTE,          // Precompute adjacency/laplacian at import, omega at runtime
        FULL_RUNTIME,                // Everything at runtime
    };

private:
    struct MeshSurfaceData {
        PackedVector3Array vertex_array;
        PackedVector3Array normal_array;
        PackedInt32Array index_array;
        PackedInt32Array bones_array;
        Vector<float> weights_array;

        MeshSurfaceData(const Array& p_mesh_arrays);
        MeshSurfaceData() {};
    };

    // Precomputation data structures
    Vector<int> adjacency_matrix;
    Vector<float> laplacian_matrix;

protected:
    static void _bind_methods();

public:
    DDMImporter();
    ~DDMImporter();

    // Main import function
    Ref<DDMMesh> import_mesh(const Ref<Mesh>& mesh, ImportMode import_mode);

    // Individual processing steps
    bool extract_mesh_data(const Ref<Mesh>& mesh, MeshSurfaceData& surface_data);
    bool build_adjacency_matrix(const MeshSurfaceData& surface_data, float tolerance);
    bool compute_laplacian_matrix();
    Ref<DDMMesh> create_ddm_mesh(const Ref<Mesh>& source_mesh, const MeshSurfaceData& surface_data);

    // Utility functions
    static MeshInstance3D* replace_mesh_instance_with_ddm(MeshInstance3D* mesh_instance, ImportMode import_mode);

private:
    // Helper methods
    bool validate_mesh_data(const Ref<Mesh>& mesh) const;
    int find_vertex_adjacency(const MeshSurfaceData& surface_data, int vertex_index, float tolerance);
};

VARIANT_ENUM_CAST(DDMImporter::ImportMode);

#endif // DDM_IMPORTER_H
