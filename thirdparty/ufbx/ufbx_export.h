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

// Internal implementation structure (exposed for writer access)
typedef struct {
    ufbx_export_scene scene;
    ufbx_allocator allocator;
    
    // Dynamic arrays for scene elements
    ufbx_node **nodes;
    size_t num_nodes;
    size_t nodes_cap;
    
    ufbx_mesh **meshes;
    size_t num_meshes;
    size_t meshes_cap;
    
    ufbx_material **materials;
    size_t num_materials;
    size_t materials_cap;
    
    ufbx_anim_stack **anim_stacks;
    size_t num_anim_stacks;
    size_t anim_stacks_cap;
    
    // Error handling
    ufbx_error error;
    bool has_error;
} ufbxi_export_scene;

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
ufbx_abi ufbx_texture *ufbx_add_texture(ufbx_export_scene *scene, const char *name);

// Add an animation stack to the scene
ufbx_abi ufbx_anim_stack *ufbx_add_animation(ufbx_export_scene *scene, const char *name);

// Mesh construction helpers

// Set mesh vertex data
ufbx_abi bool ufbx_set_mesh_vertices(ufbx_mesh *mesh, const ufbx_vec3 *positions, size_t num_vertices, ufbx_error *error);

// Set mesh face data with variable topology (triangles, quads, n-gons)
ufbx_abi bool ufbx_set_mesh_faces(ufbx_mesh *mesh, const uint32_t *indices, size_t num_indices, 
                                  const ufbx_face *face_definitions, size_t num_faces, ufbx_error *error);

// Set mesh face data (uniform topology - backward compatibility)
ufbx_abi bool ufbx_set_mesh_indices(ufbx_mesh *mesh, const uint32_t *indices, size_t num_indices, size_t vertices_per_face, ufbx_error *error);

// Set mesh normals
ufbx_abi bool ufbx_set_mesh_normals(ufbx_mesh *mesh, const ufbx_vec3 *normals, size_t num_normals, ufbx_error *error);

// Set mesh UV coordinates
ufbx_abi bool ufbx_set_mesh_uvs(ufbx_mesh *mesh, const ufbx_vec2 *uvs, size_t num_uvs, ufbx_error *error);

// Material construction helpers

// Set material property (generic)
ufbx_abi bool ufbx_set_material_property(ufbx_material *material, const char *property_name, int type, const void *value);

// Set material albedo color
ufbx_abi bool ufbx_set_material_albedo(ufbx_material *material, ufbx_real r, ufbx_real g, ufbx_real b, ufbx_real a, ufbx_error *error);

// Set material metallic and roughness
ufbx_abi bool ufbx_set_material_metallic_roughness(ufbx_material *material, ufbx_real metallic, ufbx_real roughness, ufbx_error *error);

// Set material emission
ufbx_abi bool ufbx_set_material_emission(ufbx_material *material, ufbx_real r, ufbx_real g, ufbx_real b, ufbx_error *error);

// Set texture data
ufbx_abi bool ufbx_set_texture_data(ufbx_texture *texture, const void *data, size_t size, const char *format, ufbx_error *error);

// Attach texture to material
ufbx_abi bool ufbx_attach_texture_to_material(ufbx_material *material, ufbx_texture *texture, const char *property_name, ufbx_error *error);

// Node construction helpers

// Set node transform
ufbx_abi void ufbx_set_node_transform(ufbx_node *node, const ufbx_transform *transform, ufbx_error *error);

// Attach mesh to node
ufbx_abi bool ufbx_attach_mesh_to_node(ufbx_node *node, ufbx_mesh *mesh, ufbx_error *error);

// Attach material to mesh
ufbx_abi bool ufbx_attach_material_to_mesh(ufbx_mesh *mesh, ufbx_material *material, int surface_index, ufbx_error *error);

// Skinning construction helpers

// Add a skin deformer to the scene
ufbx_abi ufbx_skin_deformer *ufbx_add_skin_deformer(ufbx_export_scene *scene, const char *name);

// Add a skin cluster to a skin deformer
ufbx_abi ufbx_skin_cluster *ufbx_add_skin_cluster(ufbx_export_scene *scene, ufbx_skin_deformer *skin, ufbx_node *bone_node, const char *name);

// Set skin weight data
ufbx_abi bool ufbx_set_skin_weights(ufbx_skin_deformer *skin, const ufbx_skin_weight *weights, size_t num_weights, ufbx_error *error);

// Set skin vertex data
ufbx_abi bool ufbx_set_skin_vertices(ufbx_skin_deformer *skin, const ufbx_skin_vertex *vertices, size_t num_vertices, ufbx_error *error);

// Attach skin deformer to mesh
ufbx_abi bool ufbx_attach_skin_to_mesh(ufbx_mesh *mesh, ufbx_skin_deformer *skin, ufbx_error *error);

// Morph target construction helpers

// Add a blend deformer to the scene
ufbx_abi ufbx_blend_deformer *ufbx_add_blend_deformer(ufbx_export_scene *scene, const char *name);

// Add a blend channel to a blend deformer
ufbx_abi ufbx_blend_channel *ufbx_add_blend_channel(ufbx_export_scene *scene, ufbx_blend_deformer *deformer, const char *name);

// Add a blend shape to the scene
ufbx_abi ufbx_blend_shape *ufbx_add_blend_shape(ufbx_export_scene *scene, const char *name);

// Set blend shape offset data
ufbx_abi bool ufbx_set_blend_shape_offsets(ufbx_blend_shape *shape, const ufbx_vec3 *position_offsets, const ufbx_vec3 *normal_offsets, size_t num_offsets, ufbx_error *error);

// Attach blend deformer to mesh
ufbx_abi bool ufbx_attach_blend_to_mesh(ufbx_mesh *mesh, ufbx_blend_deformer *blend, ufbx_error *error);

// Export functions

// Get export buffer size
ufbx_abi size_t ufbx_get_export_size(const ufbx_export_scene *scene, const ufbx_export_opts *opts, ufbx_error *error);

// Export scene to memory buffer
ufbx_abi size_t ufbx_export_to_memory(const ufbx_export_scene *scene, void *buffer, size_t buffer_size, const ufbx_export_opts *opts, ufbx_error *error);

// Export scene to FBX file
ufbx_abi bool ufbx_export_to_file(const ufbx_export_scene *scene, const char *filename, const ufbx_export_opts *opts, ufbx_error *error);

// Free export result
ufbx_abi void ufbx_free_export_result(ufbx_export_result *result);

// Cleanup

// Free an export scene
ufbx_abi void ufbx_free_export_scene(ufbx_export_scene *scene);

#ifdef __cplusplus
}
#endif

#endif // UFBX_EXPORT_H_INCLUDED
