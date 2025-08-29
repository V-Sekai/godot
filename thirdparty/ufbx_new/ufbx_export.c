#include "ufbx_export.h"
#include "ufbx.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Internal structures for building export scenes
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
} ufbx_export_scene_imp;

// Helper function to grow dynamic arrays
static bool ufbx_export_grow_array(void **ptr, size_t *cap, size_t elem_size, size_t min_cap) {
    if (*cap >= min_cap) return true;
    
    size_t new_cap = *cap ? *cap * 2 : 16;
    if (new_cap < min_cap) new_cap = min_cap;
    
    void *new_ptr = realloc(*ptr, new_cap * elem_size);
    if (!new_ptr) return false;
    
    *ptr = new_ptr;
    *cap = new_cap;
    return true;
}

// Create a new export scene
ufbx_export_scene *ufbx_create_scene(const ufbx_export_opts *opts, ufbx_error *error) {
    ufbx_export_scene_imp *scene_imp = (ufbx_export_scene_imp*)calloc(1, sizeof(ufbx_export_scene_imp));
    if (!scene_imp) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            error->description = "Failed to allocate export scene";
        }
        return NULL;
    }
    
    // Initialize allocator
    if (opts && opts->temp_allocator.alloc_fn) {
        scene_imp->allocator = opts->temp_allocator;
    } else {
        scene_imp->allocator.alloc_fn = malloc;
        scene_imp->allocator.realloc_fn = realloc;
        scene_imp->allocator.free_fn = free;
    }
    
    // Initialize scene metadata
    scene_imp->scene.metadata.creator = "ufbx_export";
    scene_imp->scene.metadata.version = 7400; // FBX 2014/2015 format
    scene_imp->scene.metadata.is_big_endian = false;
    
    // Set coordinate system (Godot uses Y-up, right-handed)
    scene_imp->scene.settings.axes.up = UFBX_COORDINATE_AXIS_POSITIVE_Y;
    scene_imp->scene.settings.axes.front = UFBX_COORDINATE_AXIS_NEGATIVE_Z;
    scene_imp->scene.settings.axes.right = UFBX_COORDINATE_AXIS_POSITIVE_X;
    scene_imp->scene.settings.unit_meters = 1.0;
    scene_imp->scene.settings.frames_per_second = 30.0;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return &scene_imp->scene;
}

// Free an export scene
void ufbx_free_export_scene(ufbx_export_scene *scene) {
    if (!scene) return;
    
    ufbx_export_scene_imp *scene_imp = (ufbx_export_scene_imp*)scene;
    
    // Free all allocated nodes
    for (size_t i = 0; i < scene_imp->num_nodes; i++) {
        if (scene_imp->nodes[i]) {
            free(scene_imp->nodes[i]);
        }
    }
    free(scene_imp->nodes);
    
    // Free all allocated meshes
    for (size_t i = 0; i < scene_imp->num_meshes; i++) {
        if (scene_imp->meshes[i]) {
            ufbx_mesh *mesh = scene_imp->meshes[i];
            free(mesh->vertices.data);
            free(mesh->vertex_indices.data);
            free(mesh->faces.data);
            free(mesh);
        }
    }
    free(scene_imp->meshes);
    
    // Free all allocated materials
    for (size_t i = 0; i < scene_imp->num_materials; i++) {
        if (scene_imp->materials[i]) {
            free(scene_imp->materials[i]);
        }
    }
    free(scene_imp->materials);
    
    // Free animation stacks
    for (size_t i = 0; i < scene_imp->num_anim_stacks; i++) {
        if (scene_imp->anim_stacks[i]) {
            free(scene_imp->anim_stacks[i]);
        }
    }
    free(scene_imp->anim_stacks);
    
    free(scene_imp);
}

// Add a node to the scene
ufbx_node *ufbx_add_node(ufbx_export_scene *scene, ufbx_node *parent, const char *name, ufbx_error *error) {
    if (!scene || !name) {
        if (error) {
            error->type = UFBX_ERROR_BAD_ARGUMENT;
            error->description = "Invalid scene or name";
        }
        return NULL;
    }
    
    ufbx_export_scene_imp *scene_imp = (ufbx_export_scene_imp*)scene;
    
    // Grow nodes array if needed
    if (!ufbx_export_grow_array((void**)&scene_imp->nodes, &scene_imp->nodes_cap, 
                                sizeof(ufbx_node*), scene_imp->num_nodes + 1)) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            error->description = "Failed to grow nodes array";
        }
        return NULL;
    }
    
    // Allocate new node
    ufbx_node *node = (ufbx_node*)calloc(1, sizeof(ufbx_node));
    if (!node) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            error->description = "Failed to allocate node";
        }
        return NULL;
    }
    
    // Initialize node
    node->element.name.data = strdup(name);
    node->element.name.length = strlen(name);
    node->element.element_id = scene_imp->num_nodes + 1;
    node->element.type = UFBX_ELEMENT_NODE;
    
    // Set parent relationship
    node->parent = parent;
    if (parent) {
        // TODO: Add to parent's children list
    }
    
    // Initialize transform to identity
    node->local_transform.translation = (ufbx_vec3){ 0, 0, 0 };
    node->local_transform.rotation = (ufbx_quat){ 0, 0, 0, 1 };
    node->local_transform.scale = (ufbx_vec3){ 1, 1, 1 };
    
    // Add to scene
    scene_imp->nodes[scene_imp->num_nodes] = node;
    scene_imp->num_nodes++;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return node;
}

// Add a mesh to the scene
ufbx_mesh *ufbx_add_mesh(ufbx_export_scene *scene, const char *name, ufbx_error *error) {
    if (!scene || !name) {
        if (error) {
            error->type = UFBX_ERROR_BAD_ARGUMENT;
            error->description = "Invalid scene or name";
        }
        return NULL;
    }
    
    ufbx_export_scene_imp *scene_imp = (ufbx_export_scene_imp*)scene;
    
    // Grow meshes array if needed
    if (!ufbx_export_grow_array((void**)&scene_imp->meshes, &scene_imp->meshes_cap,
                                sizeof(ufbx_mesh*), scene_imp->num_meshes + 1)) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            error->description = "Failed to grow meshes array";
        }
        return NULL;
    }
    
    // Allocate new mesh
    ufbx_mesh *mesh = (ufbx_mesh*)calloc(1, sizeof(ufbx_mesh));
    if (!mesh) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            error->description = "Failed to allocate mesh";
        }
        return NULL;
    }
    
    // Initialize mesh
    mesh->element.name.data = strdup(name);
    mesh->element.name.length = strlen(name);
    mesh->element.element_id = scene_imp->num_meshes + 1000; // Offset to avoid ID conflicts
    mesh->element.type = UFBX_ELEMENT_MESH;
    
    // Add to scene
    scene_imp->meshes[scene_imp->num_meshes] = mesh;
    scene_imp->num_meshes++;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return mesh;
}

// Add a material to the scene
ufbx_material *ufbx_add_material(ufbx_export_scene *scene, const char *name, ufbx_error *error) {
    if (!scene || !name) {
        if (error) {
            error->type = UFBX_ERROR_BAD_ARGUMENT;
            error->description = "Invalid scene or name";
        }
        return NULL;
    }
    
    ufbx_export_scene_imp *scene_imp = (ufbx_export_scene_imp*)scene;
    
    // Grow materials array if needed
    if (!ufbx_export_grow_array((void**)&scene_imp->materials, &scene_imp->materials_cap,
                                sizeof(ufbx_material*), scene_imp->num_materials + 1)) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            error->description = "Failed to grow materials array";
        }
        return NULL;
    }
    
    // Allocate new material
    ufbx_material *material = (ufbx_material*)calloc(1, sizeof(ufbx_material));
    if (!material) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            error->description = "Failed to allocate material";
        }
        return NULL;
    }
    
    // Initialize material
    material->element.name.data = strdup(name);
    material->element.name.length = strlen(name);
    material->element.element_id = scene_imp->num_materials + 2000; // Offset to avoid ID conflicts
    material->element.type = UFBX_ELEMENT_MATERIAL;
    
    // Set default material properties
    material->pbr.base_factor = (ufbx_vec3){ 0.8f, 0.8f, 0.8f };
    material->pbr.roughness.value_real = 0.5;
    material->pbr.metallic.value_real = 0.0;
    
    // Add to scene
    scene_imp->materials[scene_imp->num_materials] = material;
    scene_imp->num_materials++;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return material;
}

// Set mesh vertex data
bool ufbx_set_mesh_vertices(ufbx_mesh *mesh, const ufbx_vec3 *vertices, size_t num_vertices) {
    if (!mesh || !vertices || num_vertices == 0) return false;
    
    // Allocate vertex data
    ufbx_vec3 *vertex_data = (ufbx_vec3*)malloc(num_vertices * sizeof(ufbx_vec3));
    if (!vertex_data) return false;
    
    // Copy vertex data
    memcpy(vertex_data, vertices, num_vertices * sizeof(ufbx_vec3));
    
    // Set mesh vertex data
    mesh->vertices.data = vertex_data;
    mesh->vertices.count = num_vertices;
    mesh->num_vertices = num_vertices;
    
    return true;
}

// Set mesh face indices
bool ufbx_set_mesh_indices(ufbx_mesh *mesh, const uint32_t *indices, size_t num_indices) {
    if (!mesh || !indices || num_indices == 0) return false;
    
    // Allocate index data
    uint32_t *index_data = (uint32_t*)malloc(num_indices * sizeof(uint32_t));
    if (!index_data) return false;
    
    // Copy index data
    memcpy(index_data, indices, num_indices * sizeof(uint32_t));
    
    // Set mesh index data
    mesh->vertex_indices.data = index_data;
    mesh->vertex_indices.count = num_indices;
    
    // Calculate number of faces (assuming triangles)
    size_t num_faces = num_indices / 3;
    ufbx_face *faces = (ufbx_face*)malloc(num_faces * sizeof(ufbx_face));
    if (!faces) {
        free(index_data);
        return false;
    }
    
    // Initialize faces
    for (size_t i = 0; i < num_faces; i++) {
        faces[i].index_begin = i * 3;
        faces[i].num_indices = 3;
    }
    
    mesh->faces.data = faces;
    mesh->faces.count = num_faces;
    mesh->num_faces = num_faces;
    mesh->num_triangles = num_faces;
    
    return true;
}

// Set mesh normals
bool ufbx_set_mesh_normals(ufbx_mesh *mesh, const ufbx_vec3 *normals, size_t num_normals) {
    if (!mesh || !normals || num_normals == 0) return false;
    
    // Allocate normal data
    ufbx_vec3 *normal_data = (ufbx_vec3*)malloc(num_normals * sizeof(ufbx_vec3));
    if (!normal_data) return false;
    
    // Copy normal data
    memcpy(normal_data, normals, num_normals * sizeof(ufbx_vec3));
    
    // Set mesh normal data
    mesh->vertex_normal.data = normal_data;
    mesh->vertex_normal.count = num_normals;
    mesh->vertex_normal.exists = true;
    
    return true;
}

// Set mesh UV coordinates
bool ufbx_set_mesh_uvs(ufbx_mesh *mesh, const ufbx_vec2 *uvs, size_t num_uvs) {
    if (!mesh || !uvs || num_uvs == 0) return false;
    
    // Allocate UV data
    ufbx_vec2 *uv_data = (ufbx_vec2*)malloc(num_uvs * sizeof(ufbx_vec2));
    if (!uv_data) return false;
    
    // Copy UV data
    memcpy(uv_data, uvs, num_uvs * sizeof(ufbx_vec2));
    
    // Set mesh UV data
    mesh->vertex_uv.data = uv_data;
    mesh->vertex_uv.count = num_uvs;
    mesh->vertex_uv.exists = true;
    
    return true;
}

// Attach mesh to node
bool ufbx_attach_mesh_to_node(ufbx_node *node, ufbx_mesh *mesh) {
    if (!node || !mesh) return false;
    
    node->mesh = mesh;
    node->attrib = &mesh->element;
    node->attrib_type = UFBX_ELEMENT_MESH;
    
    return true;
}

// Attach material to mesh
bool ufbx_attach_material_to_mesh(ufbx_mesh *mesh, ufbx_material *material) {
    if (!mesh || !material) return false;
    
    // For now, just set the first material
    // TODO: Support multiple materials per mesh
    mesh->materials.data = &material;
    mesh->materials.count = 1;
    
    return true;
}

// Set node transform
void ufbx_set_node_transform(ufbx_node *node, const ufbx_transform *transform) {
    if (!node || !transform) return;
    
    node->local_transform = *transform;
    
    // TODO: Update world transform based on parent hierarchy
    node->node_to_parent = *transform;
}

// Set material property
bool ufbx_set_material_property(ufbx_material *material, const char *property_name, 
                                ufbx_material_property_type type, const void *value) {
    if (!material || !property_name || !value) return false;
    
    // Handle common material properties
    if (strcmp(property_name, "DiffuseColor") == 0 && type == UFBX_MATERIAL_PROPERTY_VEC3) {
        material->pbr.base_factor = *(const ufbx_vec3*)value;
        return true;
    } else if (strcmp(property_name, "Roughness") == 0 && type == UFBX_MATERIAL_PROPERTY_REAL) {
        material->pbr.roughness.value_real = *(const ufbx_real*)value;
        return true;
    } else if (strcmp(property_name, "Metallic") == 0 && type == UFBX_MATERIAL_PROPERTY_REAL) {
        material->pbr.metallic.value_real = *(const ufbx_real*)value;
        return true;
    }
    
    // TODO: Support more material properties and custom properties
    return false;
}

// Forward declarations for writer functions
ufbx_error ufbx_export_to_file_impl(const ufbx_export_scene *scene, const char *filename, 
                                    const ufbx_export_opts *opts);
ufbx_export_result ufbx_export_to_memory_impl(const ufbx_export_scene *scene, 
                                              const ufbx_export_opts *opts);

// Validation function
static ufbx_error ufbx_validate_export_scene(const ufbx_export_scene *scene) {
    ufbx_error error = { UFBX_ERROR_NONE };
    
    if (!scene) {
        error.type = UFBX_ERROR_BAD_ARGUMENT;
        error.description = "Scene is NULL";
        return error;
    }
    
    // Validate scene metadata
    if (!scene->metadata.creator) {
        error.type = UFBX_ERROR_BAD_ARGUMENT;
        error.description = "Scene metadata creator is NULL";
        return error;
    }
    
    // Validate coordinate system
    if (scene->settings.axes.up == UFBX_COORDINATE_AXIS_UNKNOWN ||
        scene->settings.axes.front == UFBX_COORDINATE_AXIS_UNKNOWN ||
        scene->settings.axes.right == UFBX_COORDINATE_AXIS_UNKNOWN) {
        error.type = UFBX_ERROR_BAD_ARGUMENT;
        error.description = "Invalid coordinate system in scene settings";
        return error;
    }
    
    // Validate unit scale
    if (scene->settings.unit_meters <= 0.0) {
        error.type = UFBX_ERROR_BAD_ARGUMENT;
        error.description = "Invalid unit scale in scene settings";
        return error;
    }
    
    return error;
}

// Export scene to file
ufbx_error ufbx_export_to_file(const ufbx_export_scene *scene, const char *filename, 
                               const ufbx_export_opts *opts) {
    // Validate inputs
    if (!scene || !filename) {
        ufbx_error error = { UFBX_ERROR_BAD_ARGUMENT };
        error.description = "Invalid scene or filename";
        return error;
    }
    
    // Validate scene structure
    ufbx_error validation_error = ufbx_validate_export_scene(scene);
    if (validation_error.type != UFBX_ERROR_NONE) {
        return validation_error;
    }
    
    // Call the actual implementation
    return ufbx_export_to_file_impl(scene, filename, opts);
}

// Export scene to memory
ufbx_export_result ufbx_export_to_memory(const ufbx_export_scene *scene, 
                                         const ufbx_export_opts *opts) {
    ufbx_export_result result = { 0 };
    
    if (!scene) {
        result.error.type = UFBX_ERROR_BAD_ARGUMENT;
        result.error.description = "Scene is NULL";
        return result;
    }
    
    // Validate scene structure
    ufbx_error validation_error = ufbx_validate_export_scene(scene);
    if (validation_error.type != UFBX_ERROR_NONE) {
        result.error = validation_error;
        return result;
    }
    
    // Call the actual implementation
    return ufbx_export_to_memory_impl(scene, opts);
}
