#ifndef UFBX_EXPORT_H_INCLUDED
#define UFBX_EXPORT_H_INCLUDED

#include "ufbx.h"

#ifdef __cplusplus
extern "C" {
#endif

// -- Export API

// Forward declarations for export structures
typedef struct ufbx_export_scene ufbx_export_scene;
typedef struct ufbx_export_opts ufbx_export_opts;

// Export result structure
typedef struct {
    void *data;
    size_t size;
    bool success;
    ufbx_error error;
} ufbx_export_result;

// Export scene metadata
typedef struct {
    const char *creator;
    uint32_t version;
    bool is_big_endian;
} ufbx_export_metadata;

// Export scene settings
typedef struct {
    ufbx_coordinate_axes axes;
    ufbx_real unit_meters;
    ufbx_real frames_per_second;
} ufbx_export_settings;

// Export scene builder - used to construct scenes for export
struct ufbx_export_scene {
    // Scene metadata
    ufbx_export_metadata metadata;
    
    // Scene settings
    ufbx_export_settings settings;
    
    // Root node of the scene
    ufbx_node *root_node;
    
    // Lists of elements in the scene
    ufbx_node_list nodes;
    ufbx_mesh_list meshes;
    ufbx_material_list materials;
    ufbx_texture_list textures;
    ufbx_light_list lights;
    ufbx_camera_list cameras;
    ufbx_anim_stack_list anim_stacks;
    
    // Internal data for export
    void *_internal_data;
};

// Options for FBX export
struct ufbx_export_opts {
    uint32_t _begin_zero;
    
    // Allocator options
    ufbx_allocator_opts temp_allocator;
    ufbx_allocator_opts result_allocator;
    
    // FBX version to export (e.g., 7400 for FBX 7.4)
    uint32_t fbx_version;
    
    // Export as ASCII instead of binary
    bool ascii_format;
    
    // Coordinate system settings
    ufbx_coordinate_axes axes;
    ufbx_real unit_meters;
    
    // Export options
    bool embed_textures;
    bool export_animations;
    bool export_materials;
    
    uint32_t _end_zero;
};

// Scene construction functions

// Create an empty export scene
ufbx_abi ufbx_export_scene *ufbx_create_scene(const ufbx_export_opts *opts);

// Add a node to the scene
ufbx_abi ufbx_node *ufbx_add_node(ufbx_export_scene *scene, const char *name, ufbx_node *parent);

// Add a mesh to the scene
ufbx_abi ufbx_mesh *ufbx_add_mesh(ufbx_export_scene *scene, const char *name);

// Add a material to the scene
ufbx_abi ufbx_material *ufbx_add_material(ufbx_export_scene *scene, const char *name);

// Add a texture to the scene
ufbx_abi ufbx_texture *ufbx_add_texture(ufbx_export_scene *scene, const char *name, const char *filename);

// Add an animation stack to the scene
ufbx_abi ufbx_anim_stack *ufbx_add_animation(ufbx_export_scene *scene, const char *name);

// Mesh construction helpers

// Set mesh vertex data
ufbx_abi bool ufbx_set_mesh_vertices(ufbx_mesh *mesh, const ufbx_vec3 *positions, size_t num_vertices);

// Set mesh face data
ufbx_abi bool ufbx_set_mesh_indices(ufbx_mesh *mesh, const uint32_t *indices, size_t num_indices);

// Set mesh normals
ufbx_abi bool ufbx_set_mesh_normals(ufbx_mesh *mesh, const ufbx_vec3 *normals, size_t num_normals);

// Set mesh UV coordinates
ufbx_abi bool ufbx_set_mesh_uvs(ufbx_mesh *mesh, const ufbx_vec2 *uvs, size_t num_uvs);

// Material construction helpers

// Set material diffuse color
ufbx_abi bool ufbx_set_material_diffuse_color(ufbx_material *material, ufbx_vec3 color);

// Set material diffuse texture
ufbx_abi bool ufbx_set_material_diffuse_texture(ufbx_material *material, ufbx_texture *texture);

// Set material PBR properties
ufbx_abi bool ufbx_set_material_pbr(ufbx_material *material, ufbx_vec3 base_color, ufbx_real metalness, ufbx_real roughness);

// Node construction helpers

// Set node transform
ufbx_abi void ufbx_set_node_transform(ufbx_node *node, const ufbx_transform *transform);

// Attach mesh to node
ufbx_abi bool ufbx_attach_mesh_to_node(ufbx_node *node, ufbx_mesh *mesh);

// Attach material to mesh
ufbx_abi bool ufbx_attach_material_to_mesh(ufbx_mesh *mesh, ufbx_material *material);

// Export functions

// Export scene to FBX binary format
ufbx_abi ufbx_error ufbx_export_to_file(const ufbx_export_scene *scene, const char *filename, const ufbx_export_opts *opts);

// Export scene to memory buffer
ufbx_abi ufbx_export_result ufbx_export_to_memory(const ufbx_export_scene *scene, const ufbx_export_opts *opts);

// Free export result
ufbx_abi void ufbx_free_export_result(ufbx_export_result *result);

// Cleanup

// Free an export scene
ufbx_abi void ufbx_free_export_scene(ufbx_export_scene *scene);

#ifdef __cplusplus
}
#endif

#endif // UFBX_EXPORT_H_INCLUDED
