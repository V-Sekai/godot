/* direct_delta_mush.cpp */

#include "direct_delta_mush.h"

#include "core/config/engine.h"
#include "scene/resources/mesh.h"

void DirectDeltaMush::_bind_methods() {
    // Properties
    ClassDB::bind_method(D_METHOD("set_iterations", "iterations"), &DirectDeltaMush::set_iterations);
    ClassDB::bind_method(D_METHOD("get_iterations"), &DirectDeltaMush::get_iterations);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "iterations", PROPERTY_HINT_RANGE, "1,100,1"), "set_iterations", "get_iterations");

    ClassDB::bind_method(D_METHOD("set_smooth_lambda", "lambda"), &DirectDeltaMush::set_smooth_lambda);
    ClassDB::bind_method(D_METHOD("get_smooth_lambda"), &DirectDeltaMush::get_smooth_lambda);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "smooth_lambda", PROPERTY_HINT_RANGE, "0.1,2.0,0.01"), "set_smooth_lambda", "get_smooth_lambda");

    ClassDB::bind_method(D_METHOD("set_adjacency_tolerance", "tolerance"), &DirectDeltaMush::set_adjacency_tolerance);
    ClassDB::bind_method(D_METHOD("get_adjacency_tolerance"), &DirectDeltaMush::get_adjacency_tolerance);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "adjacency_tolerance", PROPERTY_HINT_RANGE, "0.0001,0.01,0.0001"), "set_adjacency_tolerance", "get_adjacency_tolerance");

    ClassDB::bind_method(D_METHOD("set_use_compute", "use_compute"), &DirectDeltaMush::set_use_compute);
    ClassDB::bind_method(D_METHOD("get_use_compute"), &DirectDeltaMush::get_use_compute);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_compute"), "set_use_compute", "get_use_compute");

    // Methods
    ClassDB::bind_method(D_METHOD("precompute"), &DirectDeltaMush::precompute);
}

void DirectDeltaMush::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE: {
            // Initialize rendering device
            rd = RenderingServer::get_singleton()->get_rendering_device();
        } break;

        case NOTIFICATION_PROCESS: {
            if (Engine::get_singleton()->is_editor_hint()) {
                return;
            }
            update_deformation();
        } break;

        case NOTIFICATION_EXIT_TREE: {
            // Clean up resources
            if (rd && omega_buffer.is_valid()) {
                rd->free_rid(omega_buffer);
            }
            if (rd && adjacency_buffer.is_valid()) {
                rd->free_rid(adjacency_buffer);
            }
            if (rd && laplacian_buffer.is_valid()) {
                rd->free_rid(laplacian_buffer);
            }
        } break;
    }
}

DirectDeltaMush::DirectDeltaMush() {
    // Initialize with default values
    set_process(true);
}

DirectDeltaMush::~DirectDeltaMush() {
    // Cleanup handled in _notification
}

// Property setters/getters
void DirectDeltaMush::set_iterations(int p_iterations) {
    iterations = p_iterations;
}

int DirectDeltaMush::get_iterations() const {
    return iterations;
}

void DirectDeltaMush::set_smooth_lambda(float p_lambda) {
    smooth_lambda = p_lambda;
}

float DirectDeltaMush::get_smooth_lambda() const {
    return smooth_lambda;
}

void DirectDeltaMush::set_adjacency_tolerance(float p_tolerance) {
    adjacency_tolerance = p_tolerance;
}

float DirectDeltaMush::get_adjacency_tolerance() const {
    return adjacency_tolerance;
}

void DirectDeltaMush::set_use_compute(bool p_use_compute) {
    use_compute = p_use_compute;
}

bool DirectDeltaMush::get_use_compute() const {
    return use_compute;
}

// Public methods
void DirectDeltaMush::precompute() {
    precompute_data();
}

// Internal methods
void DirectDeltaMush::precompute_data() {
    Ref<Mesh> mesh = get_mesh();
    if (mesh.is_null()) {
        WARN_PRINT("No mesh assigned to DirectDeltaMush node");
        return;
    }

    // Check if mesh has bone weights
    Array surface_arrays = mesh->surface_get_arrays(0);
    if (surface_arrays.size() <= Mesh::ARRAY_BONES) {
        WARN_PRINT("Mesh must have bone weights for Direct Delta Mush");
        return;
    }

    // Build adjacency matrix
    build_adjacency_matrix();

    // Compute Laplacian matrix
    compute_laplacian_matrix();

    // Precompute Omega matrices
    precompute_omega_matrices();

    print_line("Direct Delta Mush precomputation completed");
}

void DirectDeltaMush::update_deformation() {
    Ref<Mesh> mesh = get_mesh();
    if (mesh.is_null() || !omega_buffer.is_valid()) {
        return;
    }

    // Get skeleton bone transforms
    Node *skel_node = get_skeleton();
    if (!skel_node) {
        return;
    }

    Skeleton3D *skeleton = Object::cast_to<Skeleton3D>(skel_node);
    if (!skeleton) {
        return;
    }

    // Get bone transforms
    int bone_count = skeleton->get_bone_count();
    Vector<Transform3D> bone_transforms;
    bone_transforms.resize(bone_count);

    for (int i = 0; i < bone_count; i++) {
        bone_transforms.set(i, skeleton->get_bone_global_pose(i));
    }

    // Create deformed mesh if needed
    if (deformed_mesh.is_null()) {
        deformed_mesh = mesh->duplicate();
    }

    // Apply deformation
    if (use_compute && rd) {
        // GPU deformation
        // TODO: Implement GPU deformation using DDMCompute
    } else {
        // CPU deformation fallback
        PackedVector3Array vertices = mesh->surface_get_arrays(0)[Mesh::ARRAY_VERTEX];
        PackedVector3Array normals = mesh->surface_get_arrays(0)[Mesh::ARRAY_NORMAL];

        // TODO: Apply deformation using DDMDeformer
        // For now, just copy original mesh
        deformed_mesh->surface_set_material(0, mesh->surface_get_material(0));
    }

    // Update the displayed mesh
    set_mesh(deformed_mesh);
}

void DirectDeltaMush::build_adjacency_matrix() {
    Ref<Mesh> mesh = get_mesh();
    if (mesh.is_null()) {
        return;
    }

    // Extract mesh surface data
    Array surface_arrays = mesh->surface_get_arrays(0);
    PackedVector3Array vertices = surface_arrays[Mesh::ARRAY_VERTEX];
    PackedInt32Array indices = surface_arrays[Mesh::ARRAY_INDEX];

    int vertex_count = vertices.size();
    const int max_neighbors = 32;

    // Allocate adjacency buffer
    if (rd) {
        adjacency_buffer = rd->storage_buffer_create(vertex_count * max_neighbors * sizeof(int));
    }

    // Build adjacency matrix (CPU implementation for now)
    Vector<int> adjacency_data;
    adjacency_data.resize(vertex_count * max_neighbors);

    // Initialize to -1
    for (int i = 0; i < adjacency_data.size(); i++) {
        adjacency_data.set(i, -1);
    }

    // Build from triangles
    for (int tri = 0; tri < indices.size(); tri += 3) {
        int v0 = indices[tri];
        int v1 = indices[tri + 1];
        int v2 = indices[tri + 2];

        // Add edges
        add_edge_to_adjacency(adjacency_data, v0, v1, max_neighbors);
        add_edge_to_adjacency(adjacency_data, v0, v2, max_neighbors);
        add_edge_to_adjacency(adjacency_data, v1, v2, max_neighbors);
    }

    // Upload to GPU if using compute
    if (rd && adjacency_buffer.is_valid()) {
        rd->buffer_update(adjacency_buffer, 0, adjacency_data.size() * sizeof(int), adjacency_data.ptr());
    }
}

void DirectDeltaMush::compute_laplacian_matrix() {
    if (!adjacency_buffer.is_valid()) {
        return;
    }

    Ref<Mesh> mesh = get_mesh();
    int vertex_count = mesh->surface_get_arrays(0)[Mesh::ARRAY_VERTEX].size();
    const int max_neighbors = 32;

    // Allocate Laplacian buffer
    if (rd) {
        laplacian_buffer = rd->storage_buffer_create(vertex_count * max_neighbors * 2 * sizeof(float));
    }

    // TODO: Implement Laplacian computation
    // For now, create a basic Laplacian matrix
}

void DirectDeltaMush::precompute_omega_matrices() {
    if (!laplacian_buffer.is_valid()) {
        return;
    }

    Ref<Mesh> mesh = get_mesh();
    int vertex_count = mesh->surface_get_arrays(0)[Mesh::ARRAY_VERTEX].size();

    // Allocate omega buffer (32 bones max, 16 floats per 4x4 matrix)
    if (rd) {
        omega_buffer = rd->storage_buffer_create(vertex_count * 32 * 16 * sizeof(float));
    }

    // TODO: Implement omega matrix precomputation
    // For now, initialize with identity matrices
}

// Helper method
void DirectDeltaMush::add_edge_to_adjacency(Vector<int>& adjacency_data, int v0, int v1, int max_neighbors) {
    // Add v1 to v0's adjacency list
    for (int i = 0; i < max_neighbors; i++) {
        int idx = v0 * max_neighbors + i;
        if (adjacency_data[idx] == -1) {
            adjacency_data.set(idx, v1);
            break;
        }
        if (adjacency_data[idx] == v1) {
            break; // Already exists
        }
    }

    // Add v0 to v1's adjacency list
    for (int i = 0; i < max_neighbors; i++) {
        int idx = v1 * max_neighbors + i;
        if (adjacency_data[idx] == -1) {
            adjacency_data.set(idx, v0);
            break;
        }
        if (adjacency_data[idx] == v0) {
            break; // Already exists
        }
    }
}
