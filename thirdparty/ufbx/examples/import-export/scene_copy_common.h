#ifndef SCENE_COPY_COMMON_H
#define SCENE_COPY_COMMON_H

#include "scene_utils.h"

// Helper structures to track mappings during copy
typedef struct {
    ufbx_node *src_node;
    ufbx_node *export_node;
} node_mapping;

typedef struct {
    ufbx_material *src_material;
    ufbx_material *export_material;
} material_mapping;

typedef struct {
    ufbx_mesh *src_mesh;
    ufbx_mesh *export_mesh;
} mesh_mapping;

typedef struct {
    ufbx_anim_stack *src_stack;
    ufbx_anim_stack *export_stack;
} anim_stack_mapping;

typedef struct {
    ufbx_anim_layer *src_layer;
    ufbx_anim_layer *export_layer;
} anim_layer_mapping;

typedef struct {
    ufbx_skin_deformer *src_skin;
    ufbx_skin_deformer *export_skin;
} skin_mapping;

typedef struct {
    ufbx_blend_deformer *src_blend;
    ufbx_blend_deformer *export_blend;
} blend_mapping;

// Function declarations for node operations
bool copy_nodes(ufbx_scene *source_scene, ufbx_export_scene *export_scene, 
                node_mapping **node_mappings, size_t *num_mappings);
bool setup_node_hierarchy(node_mapping *node_mappings, size_t num_mappings);
bool attach_elements_to_nodes(ufbx_scene *source_scene, node_mapping *node_mappings, size_t num_mappings,
                              mesh_mapping *mesh_mappings);

// Function declarations for material operations
bool copy_materials(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                    material_mapping **material_mappings);

// Function declarations for mesh operations
bool copy_meshes(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                 mesh_mapping **mesh_mappings);

// Function declarations for animation operations
bool copy_animations(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                     node_mapping *node_mappings, size_t num_mappings,
                     anim_stack_mapping **stack_mappings, anim_layer_mapping **layer_mappings);

// Function declarations for deformer operations
bool copy_skin_deformers(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                         node_mapping *node_mappings, size_t num_mappings,
                         mesh_mapping *mesh_mappings, skin_mapping **skin_mappings);
bool copy_blend_deformers(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                          mesh_mapping *mesh_mappings, blend_mapping **blend_mappings);

// Function declarations for additional element types
bool copy_lights_and_cameras(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                             node_mapping *node_mappings, size_t num_mappings);
bool copy_constraints(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                      node_mapping *node_mappings, size_t num_mappings);
bool copy_user_properties(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                          node_mapping *node_mappings, size_t num_mappings);

#endif // SCENE_COPY_COMMON_H
