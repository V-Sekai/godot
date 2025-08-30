#include "ufbx_export.h"
#include "ufbx.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

static bool ufbx_export_grow_array(void **ptr, size_t *cap, size_t elem_size, size_t min_cap) {
    if (*cap >= min_cap) {
        return true;
    }
    
    size_t new_cap = *cap ? *cap * 2 : 16;
    if (new_cap < min_cap) {
        new_cap = min_cap;
    }
    
    void *new_ptr = realloc(*ptr, new_cap * elem_size);
    if (!new_ptr) {
        return false;
    }
    
    *ptr = new_ptr;
    *cap = new_cap;
    return true;
}

ufbx_export_scene *ufbx_create_scene(const ufbx_export_opts *opts) {
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)calloc(1, sizeof(ufbxi_export_scene));
    if (!scene_imp) {
        return NULL;
    }
    
    if (opts && opts->temp_allocator.allocator.alloc_fn) {
        scene_imp->allocator = opts->temp_allocator.allocator;
    } else {
        scene_imp->allocator.alloc_fn = (ufbx_alloc_fn*)malloc;
        scene_imp->allocator.realloc_fn = (ufbx_realloc_fn*)realloc;
        scene_imp->allocator.free_fn = (ufbx_free_fn*)free;
        scene_imp->allocator.user = NULL;
    }
    
    scene_imp->scene.metadata.creator = "ufbx_export";
    scene_imp->scene.metadata.version = 7400; // FBX 2014/2015 format
    scene_imp->scene.metadata.is_big_endian = false;
    
    // Set coordinate system (Godot uses Y-up, right-handed)
    scene_imp->scene.settings.axes.up = UFBX_COORDINATE_AXIS_POSITIVE_Y;
    scene_imp->scene.settings.axes.front = UFBX_COORDINATE_AXIS_NEGATIVE_Z;
    scene_imp->scene.settings.axes.right = UFBX_COORDINATE_AXIS_POSITIVE_X;
    scene_imp->scene.settings.unit_meters = 1.0;
    scene_imp->scene.settings.frames_per_second = 30.0;
    
    scene_imp->scene.nodes.data = NULL;
    scene_imp->scene.nodes.count = 0;
    scene_imp->scene.meshes.data = NULL;
    scene_imp->scene.meshes.count = 0;
    scene_imp->scene.materials.data = NULL;
    scene_imp->scene.materials.count = 0;
    scene_imp->scene.anim_stacks.data = NULL;
    scene_imp->scene.anim_stacks.count = 0;
    
    return &scene_imp->scene;
}

void ufbx_free_export_scene(ufbx_export_scene *scene) {
    if (!scene) {
        return;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    for (size_t i = 0; i < scene_imp->num_nodes; i++) {
        if (scene_imp->nodes[i]) {
            free(scene_imp->nodes[i]);
        }
    }
    free(scene_imp->nodes);
    
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
    
    for (size_t i = 0; i < scene_imp->num_materials; i++) {
        if (scene_imp->materials[i]) {
            free(scene_imp->materials[i]);
        }
    }
    free(scene_imp->materials);
    
    for (size_t i = 0; i < scene_imp->num_anim_stacks; i++) {
        if (scene_imp->anim_stacks[i]) {
            free(scene_imp->anim_stacks[i]);
        }
    }
    free(scene_imp->anim_stacks);
    
    free(scene_imp);
}

ufbx_node *ufbx_add_node(ufbx_export_scene *scene, const char *name, ufbx_node *parent) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    if (!ufbx_export_grow_array((void**)&scene_imp->nodes, &scene_imp->nodes_cap, 
                                sizeof(ufbx_node*), scene_imp->num_nodes + 1)) {
        return NULL;
    }
    
    ufbx_node *node = (ufbx_node*)calloc(1, sizeof(ufbx_node));
    if (!node) {
        return NULL;
    }
    
    node->element.name.data = strdup(name);
    node->element.name.length = strlen(name);
    node->element.element_id = scene_imp->num_nodes + 1;
    node->element.type = UFBX_ELEMENT_NODE;
    
    node->parent = parent;
    if (parent) {
        // TODO: Add to parent's children list
    }
    
    node->local_transform.translation = (ufbx_vec3){ 0, 0, 0 };
    node->local_transform.rotation = (ufbx_quat){ 0, 0, 0, 1 };
    node->local_transform.scale = (ufbx_vec3){ 1, 1, 1 };
    
    scene_imp->nodes[scene_imp->num_nodes] = node;
    scene_imp->num_nodes++;
    
    scene_imp->scene.nodes.data = (ufbx_node**)scene_imp->nodes;
    scene_imp->scene.nodes.count = scene_imp->num_nodes;
    
    return node;
}

// Add a mesh to the scene
ufbx_mesh *ufbx_add_mesh(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    // Grow meshes array if needed
    if (!ufbx_export_grow_array((void**)&scene_imp->meshes, &scene_imp->meshes_cap,
                                sizeof(ufbx_mesh*), scene_imp->num_meshes + 1)) {
        return NULL;
    }
    
    // Allocate new mesh
    ufbx_mesh *mesh = (ufbx_mesh*)calloc(1, sizeof(ufbx_mesh));
    if (!mesh) {
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
    
    // Update scene list pointer and count
    scene_imp->scene.meshes.data = (ufbx_mesh**)scene_imp->meshes;
    scene_imp->scene.meshes.count = scene_imp->num_meshes;
    
    return mesh;
}

// Add a material to the scene
ufbx_material *ufbx_add_material(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    // Grow materials array if needed
    if (!ufbx_export_grow_array((void**)&scene_imp->materials, &scene_imp->materials_cap,
                                sizeof(ufbx_material*), scene_imp->num_materials + 1)) {
        return NULL;
    }
    
    // Allocate new material
    ufbx_material *material = (ufbx_material*)calloc(1, sizeof(ufbx_material));
    if (!material) {
        return NULL;
    }
    
    // Initialize material
    material->element.name.data = strdup(name);
    material->element.name.length = strlen(name);
    material->element.element_id = scene_imp->num_materials + 2000; // Offset to avoid ID conflicts
    material->element.type = UFBX_ELEMENT_MATERIAL;
    
    // Set default material properties
    material->pbr.base_color.value_vec3 = (ufbx_vec3){ 0.8f, 0.8f, 0.8f };
    material->pbr.base_color.has_value = true;
    material->pbr.roughness.value_real = 0.5;
    material->pbr.roughness.has_value = true;
    material->pbr.metalness.value_real = 0.0;
    material->pbr.metalness.has_value = true;
    
    // Add to scene
    scene_imp->materials[scene_imp->num_materials] = material;
    scene_imp->num_materials++;
    
    // Update scene list pointer and count
    scene_imp->scene.materials.data = (ufbx_material**)scene_imp->materials;
    scene_imp->scene.materials.count = scene_imp->num_materials;
    
    return material;
}

// Set mesh vertex data
bool ufbx_set_mesh_vertices(ufbx_mesh *mesh, const ufbx_vec3 *vertices, size_t num_vertices, ufbx_error *error) {
    if (!mesh || !vertices || num_vertices == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, vertices, or vertex count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate vertex data
    ufbx_vec3 *vertex_data = (ufbx_vec3*)malloc(num_vertices * sizeof(ufbx_vec3));
    if (!vertex_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate vertex data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy vertex data
    memcpy(vertex_data, vertices, num_vertices * sizeof(ufbx_vec3));
    
    // Set mesh vertex data
    mesh->vertices.data = vertex_data;
    mesh->vertices.count = num_vertices;
    mesh->num_vertices = num_vertices;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set mesh face indices with variable topology (triangles, quads, n-gons)
bool ufbx_set_mesh_faces(ufbx_mesh *mesh, const uint32_t *indices, size_t num_indices, 
                         const ufbx_face *face_definitions, size_t num_faces, ufbx_error *error) {
    if (!mesh || !indices || !face_definitions || num_indices == 0 || num_faces == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, indices, face definitions, or counts");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Validate face definitions
    size_t total_expected_indices = 0;
    size_t triangle_count = 0;
    for (size_t i = 0; i < num_faces; i++) {
        if (face_definitions[i].num_indices < 3) {
            if (error) {
                error->type = UFBX_ERROR_UNKNOWN;
                snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Face %zu has invalid vertex count: %u (minimum 3)", i, face_definitions[i].num_indices);
                error->info_length = strlen(error->info);
            }
            return false;
        }
        total_expected_indices += face_definitions[i].num_indices;
        if (face_definitions[i].num_indices == 3) {
            triangle_count++;
        }
    }
    
    if (total_expected_indices != num_indices) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Face definitions expect %zu indices but got %zu", total_expected_indices, num_indices);
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate index data
    uint32_t *index_data = (uint32_t*)malloc(num_indices * sizeof(uint32_t));
    if (!index_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate index data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy index data
    memcpy(index_data, indices, num_indices * sizeof(uint32_t));
    
    // Allocate face data
    ufbx_face *faces = (ufbx_face*)malloc(num_faces * sizeof(ufbx_face));
    if (!faces) {
        free(index_data);
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate face data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy face definitions
    memcpy(faces, face_definitions, num_faces * sizeof(ufbx_face));
    
    // Set mesh data
    mesh->vertex_indices.data = index_data;
    mesh->vertex_indices.count = num_indices;
    mesh->faces.data = faces;
    mesh->faces.count = num_faces;
    mesh->num_faces = num_faces;
    mesh->num_triangles = triangle_count;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set mesh face indices (uniform topology - backward compatibility)
bool ufbx_set_mesh_indices(ufbx_mesh *mesh, const uint32_t *indices, size_t num_indices, size_t vertices_per_face, ufbx_error *error) {
    if (!mesh || !indices || num_indices == 0 || vertices_per_face == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, indices, index count, or vertices per face");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Validate that indices can be evenly divided by vertices_per_face
    if (num_indices % vertices_per_face != 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Index count (%zu) not divisible by vertices per face (%zu)", num_indices, vertices_per_face);
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Create uniform face definitions
    size_t num_faces = num_indices / vertices_per_face;
    ufbx_face *face_definitions = (ufbx_face*)malloc(num_faces * sizeof(ufbx_face));
    if (!face_definitions) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate face definitions");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Initialize uniform face definitions
    for (size_t i = 0; i < num_faces; i++) {
        face_definitions[i].index_begin = i * vertices_per_face;
        face_definitions[i].num_indices = vertices_per_face;
    }
    
    // Use the variable topology function
    bool result = ufbx_set_mesh_faces(mesh, indices, num_indices, face_definitions, num_faces, error);
    free(face_definitions);
    
    return result;
}


// Set material property
bool ufbx_set_material_property(ufbx_material *material, const char *property_name, 
                                int type, const void *value) {
    if (!material || !property_name || !value) {
        return false;
    }
    
    // Handle common material properties
    if (strcmp(property_name, "DiffuseColor") == 0) {
        material->pbr.base_color.value_vec3 = *(const ufbx_vec3*)value;
        material->pbr.base_color.has_value = true;
        return true;
    } else if (strcmp(property_name, "Roughness") == 0) {
        material->pbr.roughness.value_real = *(const ufbx_real*)value;
        material->pbr.roughness.has_value = true;
        return true;
    } else if (strcmp(property_name, "Metallic") == 0) {
        material->pbr.metalness.value_real = *(const ufbx_real*)value;
        material->pbr.metalness.has_value = true;
        return true;
    }
    
    // TODO: Support more material properties and custom properties
    return false;
}

// Forward declarations for writer functions (implemented in ufbx_export_writer.c)
extern ufbx_error ufbx_export_to_file_impl(const ufbx_export_scene *scene, const char *filename, 
                                           const ufbx_export_opts *opts);
extern ufbx_export_result ufbx_export_to_memory_impl(const ufbx_export_scene *scene, 
                                                     const ufbx_export_opts *opts);

// Validation function
static ufbx_error ufbx_validate_export_scene(const ufbx_export_scene *scene) {
    ufbx_error error = { UFBX_ERROR_NONE };
    
    if (!scene) {
        error.type = UFBX_ERROR_UNKNOWN;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Scene is NULL");
        error.info_length = strlen(error.info);
        return error;
    }
    
    // Validate scene metadata
    if (!scene->metadata.creator) {
        error.type = UFBX_ERROR_UNKNOWN;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Scene metadata creator is NULL");
        error.info_length = strlen(error.info);
        return error;
    }
    
    // Validate coordinate system
    if (scene->settings.axes.up == UFBX_COORDINATE_AXIS_UNKNOWN ||
        scene->settings.axes.front == UFBX_COORDINATE_AXIS_UNKNOWN ||
        scene->settings.axes.right == UFBX_COORDINATE_AXIS_UNKNOWN) {
        error.type = UFBX_ERROR_UNKNOWN;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Invalid coordinate system in scene settings");
        error.info_length = strlen(error.info);
        return error;
    }
    
    // Validate unit scale
    if (scene->settings.unit_meters <= 0.0) {
        error.type = UFBX_ERROR_UNKNOWN;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Invalid unit scale in scene settings");
        error.info_length = strlen(error.info);
        return error;
    }
    
    return error;
}


// Set mesh normals with error handling
bool ufbx_set_mesh_normals(ufbx_mesh *mesh, const ufbx_vec3 *normals, size_t num_normals, ufbx_error *error) {
    if (!mesh || !normals || num_normals == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, normals, or normal count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate normal data
    ufbx_vec3 *normal_data = (ufbx_vec3*)malloc(num_normals * sizeof(ufbx_vec3));
    if (!normal_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate normal data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy normal data
    memcpy(normal_data, normals, num_normals * sizeof(ufbx_vec3));
    
    // Set mesh normal data
    mesh->vertex_normal.values.data = normal_data;
    mesh->vertex_normal.values.count = num_normals;
    mesh->vertex_normal.exists = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set mesh UV coordinates with error handling
bool ufbx_set_mesh_uvs(ufbx_mesh *mesh, const ufbx_vec2 *uvs, size_t num_uvs, ufbx_error *error) {
    if (!mesh || !uvs || num_uvs == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh, UVs, or UV count");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate UV data
    ufbx_vec2 *uv_data = (ufbx_vec2*)malloc(num_uvs * sizeof(ufbx_vec2));
    if (!uv_data) {
        if (error) {
            error->type = UFBX_ERROR_OUT_OF_MEMORY;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate UV data");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Copy UV data
    memcpy(uv_data, uvs, num_uvs * sizeof(ufbx_vec2));
    
    // Set mesh UV data
    mesh->vertex_uv.values.data = uv_data;
    mesh->vertex_uv.values.count = num_uvs;
    mesh->vertex_uv.exists = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set node transform with error handling
void ufbx_set_node_transform(ufbx_node *node, const ufbx_transform *transform, ufbx_error *error) {
    if (!node || !transform) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid node or transform");
            error->info_length = strlen(error->info);
        }
        return;
    }
    
    node->local_transform = *transform;
    // TODO: Convert transform to matrix properly when ufbx is fully linked
    // For now, set identity matrix
    node->node_to_parent = ufbx_identity_matrix;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
}

// Attach mesh to node with error handling
bool ufbx_attach_mesh_to_node(ufbx_node *node, ufbx_mesh *mesh, ufbx_error *error) {
    if (!node || !mesh) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid node or mesh");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    node->mesh = mesh;
    node->attrib = &mesh->element;
    node->attrib_type = UFBX_ELEMENT_MESH;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Attach material to mesh with surface index
bool ufbx_attach_material_to_mesh(ufbx_mesh *mesh, ufbx_material *material, int surface_index, ufbx_error *error) {
    if (!mesh || !material) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid mesh or material");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Allocate materials array if not already allocated
    if (!mesh->materials.data) {
        mesh->materials.data = (ufbx_material**)malloc(sizeof(ufbx_material*));
        if (!mesh->materials.data) {
            if (error) {
                error->type = UFBX_ERROR_OUT_OF_MEMORY;
                snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate materials array");
                error->info_length = strlen(error->info);
            }
            return false;
        }
    }
    
    // Set the material
    mesh->materials.data[0] = material;
    mesh->materials.count = 1;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set material albedo color
bool ufbx_set_material_albedo(ufbx_material *material, ufbx_real r, ufbx_real g, ufbx_real b, ufbx_real a, ufbx_error *error) {
    if (!material) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid material");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    material->pbr.base_color.value_vec4 = (ufbx_vec4){ r, g, b, a };
    material->pbr.base_color.has_value = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set material metallic and roughness
bool ufbx_set_material_metallic_roughness(ufbx_material *material, ufbx_real metallic, ufbx_real roughness, ufbx_error *error) {
    if (!material) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid material");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    material->pbr.metalness.value_real = metallic;
    material->pbr.metalness.has_value = true;
    material->pbr.roughness.value_real = roughness;
    material->pbr.roughness.has_value = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Set material emission
bool ufbx_set_material_emission(ufbx_material *material, ufbx_real r, ufbx_real g, ufbx_real b, ufbx_error *error) {
    if (!material) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid material");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    material->pbr.emission_color.value_vec3 = (ufbx_vec3){ r, g, b };
    material->pbr.emission_color.has_value = true;
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Add a texture to the scene
ufbx_texture *ufbx_add_texture(ufbx_export_scene *scene, const char *name) {
    if (!scene || !name) {
        return NULL;
    }
    
    ufbxi_export_scene *scene_imp = (ufbxi_export_scene*)scene;
    
    // Allocate new texture
    ufbx_texture *texture = (ufbx_texture*)calloc(1, sizeof(ufbx_texture));
    if (!texture) {
        return NULL;
    }
    
    // Initialize texture
    texture->element.name.data = strdup(name);
    texture->element.name.length = strlen(name);
    texture->element.type = UFBX_ELEMENT_TEXTURE;
    
    return texture;
}

// Set texture data
bool ufbx_set_texture_data(ufbx_texture *texture, const void *data, size_t size, const char *format, ufbx_error *error) {
    if (!texture || !data || size == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid texture, data, or size");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // TODO: Implement texture data storage
    // For now, just mark as successful
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Attach texture to material
bool ufbx_attach_texture_to_material(ufbx_material *material, ufbx_texture *texture, const char *property_name, ufbx_error *error) {
    if (!material || !texture || !property_name) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid material, texture, or property name");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Attach texture based on property name
    if (strcmp(property_name, "BaseColor") == 0 || strcmp(property_name, "DiffuseColor") == 0) {
        material->pbr.base_color.texture = texture;
    } else if (strcmp(property_name, "NormalMap") == 0) {
        material->pbr.normal_map.texture = texture;
    } else if (strcmp(property_name, "Metallic") == 0) {
        material->pbr.metalness.texture = texture;
    } else if (strcmp(property_name, "Roughness") == 0) {
        material->pbr.roughness.texture = texture;
    }
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}

// Get export buffer size
size_t ufbx_get_export_size(const ufbx_export_scene *scene, const ufbx_export_opts *opts, ufbx_error *error) {
    if (!scene) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Scene is NULL");
            error->info_length = strlen(error->info);
        }
        return 0;
    }
    
    // TODO: Calculate actual export size based on scene content
    // For now, return a reasonable estimate
    size_t estimated_size = 1024 * 1024; // 1MB base size
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return estimated_size;
}

// Export scene to memory buffer
size_t ufbx_export_to_memory(const ufbx_export_scene *scene, void *buffer, size_t buffer_size, const ufbx_export_opts *opts, ufbx_error *error) {
    if (!scene || !buffer || buffer_size == 0) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid scene, buffer, or buffer size");
            error->info_length = strlen(error->info);
        }
        return 0;
    }
    
    // Validate scene structure
    ufbx_error validation_error = ufbx_validate_export_scene(scene);
    if (validation_error.type != UFBX_ERROR_NONE) {
        if (error) *error = validation_error;
        return 0;
    }
    
    // Use the writer implementation to export to memory
    ufbx_export_result result = ufbx_export_to_memory_impl(scene, opts);
    if (result.error.type != UFBX_ERROR_NONE) {
        if (error) *error = result.error;
        return 0;
    }
    
    // Check if buffer is large enough
    if (result.size > buffer_size) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Buffer too small: need %zu bytes, got %zu", result.size, buffer_size);
            error->info_length = strlen(error->info);
        }
        free(result.data);
        return 0;
    }
    
    // Copy data to user buffer
    memcpy(buffer, result.data, result.size);
    free(result.data);
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return result.size;
}

// Export scene to file
bool ufbx_export_to_file(const ufbx_export_scene *scene, const char *filename, const ufbx_export_opts *opts, ufbx_error *error) {
    if (!scene || !filename) {
        if (error) {
            error->type = UFBX_ERROR_UNKNOWN;
            snprintf(error->info, UFBX_ERROR_INFO_LENGTH, "Invalid scene or filename");
            error->info_length = strlen(error->info);
        }
        return false;
    }
    
    // Validate scene structure
    ufbx_error validation_error = ufbx_validate_export_scene(scene);
    if (validation_error.type != UFBX_ERROR_NONE) {
        if (error) *error = validation_error;
        return false;
    }
    
    // Use the writer implementation to export to file
    ufbx_error export_error = ufbx_export_to_file_impl(scene, filename, opts);
    if (export_error.type != UFBX_ERROR_NONE) {
        if (error) *error = export_error;
        return false;
    }
    
    if (error) {
        error->type = UFBX_ERROR_NONE;
    }
    
    return true;
}
