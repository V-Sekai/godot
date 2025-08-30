#include "ufbx_export.h"
#include "ufbx.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// FBX Version Constants
#define UFBX_FBX_VERSION_7400 7400
#define UFBX_FBX_VERSION_7500 7500

// ASCII writer state
typedef struct {
    char *data;
    size_t size;
    size_t capacity;
    int indent_level;
    ufbx_error error;
    bool has_error;
    uint32_t version;
} ufbx_ascii_writer;

// ASCII writer helper functions
static bool ufbx_ascii_ensure_capacity(ufbx_ascii_writer *writer, size_t needed) {
    if (writer->has_error) {
        return false;
    }
    
    size_t required = writer->size + needed;
    if (required <= writer->capacity) {
        return true;
    }
    
    size_t new_capacity = writer->capacity ? writer->capacity * 2 : 1024;
    while (new_capacity < required) {
        new_capacity *= 2;
    }
    
    char *new_data = (char*)realloc(writer->data, new_capacity);
    if (!new_data) {
        writer->has_error = true;
        writer->error.type = UFBX_ERROR_OUT_OF_MEMORY;
        snprintf(writer->error.info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate memory for ASCII writer");
        writer->error.info_length = strlen(writer->error.info);
        return false;
    }
    
    writer->data = new_data;
    writer->capacity = new_capacity;
    return true;
}

static bool ufbx_ascii_write_string(ufbx_ascii_writer *writer, const char *str) {
    if (!str) {
        return true;
    }
    size_t len = strlen(str);
    if (!ufbx_ascii_ensure_capacity(writer, len)) {
        return false;
    }
    
    memcpy(writer->data + writer->size, str, len);
    writer->size += len;
    return true;
}

static bool ufbx_ascii_write_char(ufbx_ascii_writer *writer, char c) {
    if (!ufbx_ascii_ensure_capacity(writer, 1)) {
        return false;
    }
    writer->data[writer->size++] = c;
    return true;
}

static bool ufbx_ascii_write_newline(ufbx_ascii_writer *writer) {
    return ufbx_ascii_write_char(writer, '\n');
}

static bool ufbx_ascii_write_indent(ufbx_ascii_writer *writer) {
    for (int i = 0; i < writer->indent_level; i++) {
        if (!ufbx_ascii_write_string(writer, "    ")) {
            return false; // 4 spaces per indent
        }
    }
    return true;
}

static bool ufbx_ascii_write_property_i64(ufbx_ascii_writer *writer, int64_t value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%lld", (long long)value);
    return ufbx_ascii_write_string(writer, buffer);
}

static bool ufbx_ascii_write_property_f64(ufbx_ascii_writer *writer, double value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.6g", value);
    return ufbx_ascii_write_string(writer, buffer);
}

static bool ufbx_ascii_write_property_string(ufbx_ascii_writer *writer, const char *str) {
    if (!ufbx_ascii_write_char(writer, '"')) {
        return false;
    }
    if (str) {
        // TODO: Escape special characters if needed
        if (!ufbx_ascii_write_string(writer, str)) {
            return false;
        }
    }
    return ufbx_ascii_write_char(writer, '"');
}

static bool ufbx_ascii_write_property_array_f64(ufbx_ascii_writer *writer, const double *values, size_t count) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "*%zu", count);
    if (!ufbx_ascii_write_string(writer, buffer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    
    writer->indent_level++;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "a: ")) {
        return false;
    }
    
    for (size_t i = 0; i < count; i++) {
        if (i > 0) {
            if (!ufbx_ascii_write_char(writer, ',')) {
                return false;
            }
        }
        if (!ufbx_ascii_write_property_f64(writer, values[i])) {
            return false;
        }
    }
    
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    writer->indent_level--;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    return ufbx_ascii_write_char(writer, '}');
}

static bool ufbx_ascii_write_property_array_i32(ufbx_ascii_writer *writer, const int32_t *values, size_t count) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "*%zu", count);
    if (!ufbx_ascii_write_string(writer, buffer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    
    writer->indent_level++;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "a: ")) {
        return false;
    }
    
    for (size_t i = 0; i < count; i++) {
        if (i > 0) {
            if (!ufbx_ascii_write_char(writer, ',')) {
                return false;
            }
        }
        if (!ufbx_ascii_write_property_i64(writer, values[i])) {
            return false;
        }
    }
    
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    writer->indent_level--;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    return ufbx_ascii_write_char(writer, '}');
}

static bool ufbx_ascii_write_property_array_i64(ufbx_ascii_writer *writer, const int64_t *values, size_t count) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "*%zu", count);
    if (!ufbx_ascii_write_string(writer, buffer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    
    writer->indent_level++;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "a: ")) {
        return false;
    }
    
    for (size_t i = 0; i < count; i++) {
        if (i > 0) {
            if (!ufbx_ascii_write_char(writer, ',')) {
                return false;
            }
        }
        if (!ufbx_ascii_write_property_i64(writer, values[i])) {
            return false;
        }
    }
    
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    writer->indent_level--;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    return ufbx_ascii_write_char(writer, '}');
}

static bool ufbx_ascii_write_node_begin(ufbx_ascii_writer *writer, const char *name) {
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, name)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ": ")) {
        return false;
    }
    return true;
}

static bool ufbx_ascii_write_node_end(ufbx_ascii_writer *writer) {
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_char(writer, '}')) {
        return false;
    }
    return ufbx_ascii_write_newline(writer);
}

// ASCII FBX structure writers
static bool ufbx_ascii_write_header(ufbx_ascii_writer *writer, uint32_t version) {
    // Write FBX magic comment
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "; FBX %u.%u.%u project file\n", 
             version / 1000, (version / 100) % 10, (version / 10) % 10);
    if (!ufbx_ascii_write_string(writer, buffer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "; Created by ufbx_export (ASCII only)\n\n")) {
        return false;
    }
    return true;
}

static bool ufbx_ascii_write_scene_info(ufbx_ascii_writer *writer, const ufbx_export_scene *scene) {
    if (!ufbx_ascii_write_node_begin(writer, "FBXHeaderExtension")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // FBXHeaderVersion
    if (!ufbx_ascii_write_node_begin(writer, "FBXHeaderVersion")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 1003)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // FBXVersion
    if (!ufbx_ascii_write_node_begin(writer, "FBXVersion")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, writer->version)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // Creator
    if (!ufbx_ascii_write_node_begin(writer, "Creator")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, scene->metadata.creator)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

static bool ufbx_ascii_write_global_settings(ufbx_ascii_writer *writer, const ufbx_export_scene *scene) {
    if (!ufbx_ascii_write_node_begin(writer, "GlobalSettings")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Version
    if (!ufbx_ascii_write_node_begin(writer, "Version")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 1000)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // Properties70
    if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // UpAxis
    if (!ufbx_ascii_write_node_begin(writer, "P")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "UpAxis")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "int")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Integer")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, scene->settings.axes.up == UFBX_COORDINATE_AXIS_POSITIVE_Y ? 1 : 2)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // UnitScaleFactor
    if (!ufbx_ascii_write_node_begin(writer, "P")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "UnitScaleFactor")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "double")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Number")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, scene->settings.unit_meters * 100.0)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    writer->indent_level--;
    if (!ufbx_ascii_write_node_end(writer)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

static bool ufbx_ascii_write_objects(ufbx_ascii_writer *writer, const ufbx_export_scene *scene) {
    const ufbxi_export_scene *scene_imp = (const ufbxi_export_scene*)scene;
    
    if (!ufbx_ascii_write_node_begin(writer, "Objects")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Write all nodes
    for (size_t i = 0; i < scene_imp->num_nodes; i++) {
        const ufbx_node *node = scene_imp->nodes[i];
        
        if (!ufbx_ascii_write_node_begin(writer, "Model")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, node->element.name.data)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        // Determine correct node type: LimbNode for bones, Mesh for geometry nodes, Null for empties
        const char *node_type = "Null"; // Default to Null/empty
        if (node->mesh) {
            node_type = "Mesh";
        } else if (node->bone || strstr(node->element.name.data, "Bone") != NULL) {
            node_type = "LimbNode";
        }
        if (!ufbx_ascii_write_property_string(writer, node_type)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, " {\n")) {
            return false;
        }
        writer->indent_level++;
        
        // Version
        if (!ufbx_ascii_write_node_begin(writer, "Version")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 232)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // Properties70
        if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, "{\n")) {
            return false;
        }
        writer->indent_level++;
        
        // Lcl Translation
        if (!ufbx_ascii_write_node_begin(writer, "P")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "Lcl Translation")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "Lcl Translation")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "A")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_f64(writer, node->local_transform.translation.x)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_f64(writer, node->local_transform.translation.y)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_f64(writer, node->local_transform.translation.z)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    
    // Write all meshes
    for (size_t i = 0; i < scene_imp->num_meshes; i++) {
        const ufbx_mesh *mesh = scene_imp->meshes[i];
        
        if (!ufbx_ascii_write_node_begin(writer, "Geometry")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, mesh->element.element_id)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, mesh->element.name.data)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "Mesh")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, " {\n")) {
            return false;
        }
        writer->indent_level++;
        
        // Vertices
        if (mesh->vertices.count > 0) {
            if (!ufbx_ascii_write_node_begin(writer, "Vertices")) {
                return false;
            }
            
            // Convert vertices to double array
            double *vertex_data = (double*)malloc(mesh->vertices.count * 3 * sizeof(double));
            if (vertex_data) {
                for (size_t v = 0; v < mesh->vertices.count; v++) {
                    vertex_data[v * 3 + 0] = mesh->vertices.data[v].x;
                    vertex_data[v * 3 + 1] = mesh->vertices.data[v].y;
                    vertex_data[v * 3 + 2] = mesh->vertices.data[v].z;
                }
                if (!ufbx_ascii_write_property_array_f64(writer, vertex_data, mesh->vertices.count * 3)) {
                    free(vertex_data);
                    return false;
                }
                free(vertex_data);
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
        
        // PolygonVertexIndex
        if (mesh->vertex_indices.count > 0) {
            if (!ufbx_ascii_write_node_begin(writer, "PolygonVertexIndex")) {
                return false;
            }
            
            // Convert indices to int32 array with FBX polygon encoding
            int32_t *index_data = (int32_t*)malloc(mesh->vertex_indices.count * sizeof(int32_t));
            if (index_data) {
                // Use actual face data to determine polygon boundaries
                if (mesh->num_faces > 0 && mesh->faces.data) {
                    size_t current_index = 0;
                    for (size_t f = 0; f < mesh->num_faces; f++) {
                        ufbx_face face = mesh->faces.data[f];
                        for (size_t fi = 0; fi < face.num_indices; fi++) {
                            if (face.index_begin + fi < mesh->vertex_indices.count) {
                                int32_t index = (int32_t)mesh->vertex_indices.data[face.index_begin + fi];
                                // Mark last index of each face as negative (FBX polygon end marker)
                                if (fi == face.num_indices - 1) {
                                    index = -(index + 1);
                                }
                                index_data[current_index++] = index;
                            }
                        }
                    }
                }
                if (!ufbx_ascii_write_property_array_i32(writer, index_data, mesh->vertex_indices.count)) {
                    free(index_data);
                    return false;
                }
                free(index_data);
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    
    // Write all materials
    for (size_t i = 0; i < scene_imp->num_materials; i++) {
        const ufbx_material *material = scene_imp->materials[i];
        
        if (!ufbx_ascii_write_node_begin(writer, "Material")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, material->element.element_id)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, material->element.name.data)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, " {\n")) {
            return false;
        }
        writer->indent_level++;
        
        // Version
        if (!ufbx_ascii_write_node_begin(writer, "Version")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 102)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // ShadingModel
        if (!ufbx_ascii_write_node_begin(writer, "ShadingModel")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "phong")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // MultiLayer
        if (!ufbx_ascii_write_node_begin(writer, "MultiLayer")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 0)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // Properties70
        if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, "{\n")) {
            return false;
        }
        writer->indent_level++;
        
        // DiffuseColor
        if (!ufbx_ascii_write_node_begin(writer, "P")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "DiffuseColor")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "Color")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "A")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_f64(writer, material->pbr.base_color.value_vec3.x)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_f64(writer, material->pbr.base_color.value_vec3.y)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_f64(writer, material->pbr.base_color.value_vec3.z)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    
    // Write all animation stacks
    for (size_t i = 0; i < scene_imp->num_anim_stacks; i++) {
        const ufbx_anim_stack *anim_stack = scene_imp->anim_stacks[i];
        
        if (!ufbx_ascii_write_node_begin(writer, "AnimationStack")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, anim_stack->element.element_id)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, anim_stack->element.name.data)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, " {\n")) {
            return false;
        }
        writer->indent_level++;
        
        // Properties70
        if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, "{\n")) {
            return false;
        }
        writer->indent_level++;
        
        // LocalStart
        if (!ufbx_ascii_write_node_begin(writer, "P")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "LocalStart")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "KTime")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "Time")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, (int64_t)(anim_stack->time_begin * 46186158000LL))) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // LocalStop
        if (!ufbx_ascii_write_node_begin(writer, "P")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "LocalStop")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "KTime")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "Time")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, (int64_t)(anim_stack->time_end * 46186158000LL))) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    
    // Write all animation layers
    for (size_t i = 0; i < scene_imp->num_anim_layers; i++) {
        const ufbx_anim_layer *anim_layer = scene_imp->anim_layers[i];
        
        if (!ufbx_ascii_write_node_begin(writer, "AnimationLayer")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, anim_layer->element.element_id)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, anim_layer->element.name.data)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, " {\n")) {
            return false;
        }
        writer->indent_level++;
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    
    // Write all animation curves
    for (size_t i = 0; i < scene_imp->num_anim_curves; i++) {
        const ufbx_anim_curve *anim_curve = scene_imp->anim_curves[i];
        
        if (!ufbx_ascii_write_node_begin(writer, "AnimationCurve")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, anim_curve->element.element_id)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, anim_curve->element.name.data)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, " {\n")) {
            return false;
        }
        writer->indent_level++;
        
        // Default
        if (!ufbx_ascii_write_node_begin(writer, "Default")) {
            return false;
        }
        if (!ufbx_ascii_write_property_f64(writer, 0.0)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // KeyVer
        if (!ufbx_ascii_write_node_begin(writer, "KeyVer")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 4009)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        if (anim_curve->keyframes.count > 0) {
            // KeyTime
            if (!ufbx_ascii_write_node_begin(writer, "KeyTime")) {
                return false;
            }
            
            // Convert keyframe times to int64 array
            int64_t *time_data = (int64_t*)malloc(anim_curve->keyframes.count * sizeof(int64_t));
            if (time_data) {
                for (size_t t = 0; t < anim_curve->keyframes.count; t++) {
                    time_data[t] = (int64_t)(anim_curve->keyframes.data[t].time * 46186158000LL);
                }
                if (!ufbx_ascii_write_property_array_i64(writer, time_data, anim_curve->keyframes.count)) {
                    free(time_data);
                    return false;
                }
                free(time_data);
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
            
            // KeyValueFloat
            if (!ufbx_ascii_write_node_begin(writer, "KeyValueFloat")) {
                return false;
            }
            
            // Convert keyframe values to double array
            double *value_data = (double*)malloc(anim_curve->keyframes.count * sizeof(double));
            if (value_data) {
                for (size_t v = 0; v < anim_curve->keyframes.count; v++) {
                    value_data[v] = anim_curve->keyframes.data[v].value;
                }
                if (!ufbx_ascii_write_property_array_f64(writer, value_data, anim_curve->keyframes.count)) {
                    free(value_data);
                    return false;
                }
                free(value_data);
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    
    // Write all skin deformers (FIXED: Now implemented!)
    for (size_t i = 0; i < scene_imp->scene.skin_deformers.count; i++) {
        const ufbx_skin_deformer *skin = scene_imp->scene.skin_deformers.data[i];
        
        if (!ufbx_ascii_write_node_begin(writer, "Deformer")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, skin->element.element_id)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, skin->element.name.data)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "Skin")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, " {\n")) {
            return false;
        }
        writer->indent_level++;
        
        // Version
        if (!ufbx_ascii_write_node_begin(writer, "Version")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 101)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // Link_DeformAcuracy
        if (!ufbx_ascii_write_node_begin(writer, "Link_DeformAcuracy")) {
            return false;
        }
        if (!ufbx_ascii_write_property_f64(writer, 50.0)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    
    // Write all skin clusters (FIXED: Now implemented!)
    for (size_t i = 0; i < scene_imp->scene.skin_clusters.count; i++) {
        const ufbx_skin_cluster *cluster = scene_imp->scene.skin_clusters.data[i];
        
        if (!ufbx_ascii_write_node_begin(writer, "Deformer")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, cluster->element.element_id)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, cluster->element.name.data)) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "Cluster")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, " {\n")) {
            return false;
        }
        writer->indent_level++;
        
        // Version
        if (!ufbx_ascii_write_node_begin(writer, "Version")) {
            return false;
        }
        if (!ufbx_ascii_write_property_i64(writer, 100)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // UserData
        if (!ufbx_ascii_write_node_begin(writer, "UserData")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, "")) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // Indexes (vertex indices)
        if (cluster->vertices.count > 0) {
            if (!ufbx_ascii_write_node_begin(writer, "Indexes")) {
                return false;
            }
            if (!ufbx_ascii_write_property_array_i32(writer, (const int32_t*)cluster->vertices.data, cluster->vertices.count)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
        
        // Weights (vertex weights)
        if (cluster->weights.count > 0) {
            if (!ufbx_ascii_write_node_begin(writer, "Weights")) {
                return false;
            }
            if (!ufbx_ascii_write_property_array_f64(writer, (const double*)cluster->weights.data, cluster->weights.count)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
        
        // Transform (geometry_to_bone matrix)
        if (!ufbx_ascii_write_node_begin(writer, "Transform")) {
            return false;
        }
        double transform_data[16] = {
            cluster->geometry_to_bone.m00, cluster->geometry_to_bone.m10, cluster->geometry_to_bone.m20, 0.0,
            cluster->geometry_to_bone.m01, cluster->geometry_to_bone.m11, cluster->geometry_to_bone.m21, 0.0,
            cluster->geometry_to_bone.m02, cluster->geometry_to_bone.m12, cluster->geometry_to_bone.m22, 0.0,
            cluster->geometry_to_bone.m03, cluster->geometry_to_bone.m13, cluster->geometry_to_bone.m23, 1.0
        };
        if (!ufbx_ascii_write_property_array_f64(writer, transform_data, 16)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // TransformLink (bind_to_world matrix)
        if (!ufbx_ascii_write_node_begin(writer, "TransformLink")) {
            return false;
        }
        double transform_link_data[16] = {
            cluster->bind_to_world.m00, cluster->bind_to_world.m10, cluster->bind_to_world.m20, 0.0,
            cluster->bind_to_world.m01, cluster->bind_to_world.m11, cluster->bind_to_world.m21, 0.0,
            cluster->bind_to_world.m02, cluster->bind_to_world.m12, cluster->bind_to_world.m22, 0.0,
            cluster->bind_to_world.m03, cluster->bind_to_world.m13, cluster->bind_to_world.m23, 1.0
        };
        if (!ufbx_ascii_write_property_array_f64(writer, transform_link_data, 16)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        writer->indent_level--;
        if (!ufbx_ascii_write_node_end(writer)) {
            return false;
        }
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

static bool ufbx_ascii_write_connections(ufbx_ascii_writer *writer, const ufbx_export_scene *scene) {
    const ufbxi_export_scene *scene_imp = (const ufbxi_export_scene*)scene;
    
    if (!ufbx_ascii_write_node_begin(writer, "Connections")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Connect nodes to root
    for (size_t i = 0; i < scene_imp->num_nodes; i++) {
        const ufbx_node *node = scene_imp->nodes[i];
        if (!node->parent) { // Root level nodes
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, 0)) {
                return false; // Root node ID
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        } else {
            // Connect to parent node
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->parent->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
        
        // Connect mesh to node if it exists
        if (node->mesh) {
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->mesh->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
    }
    
    // Connect materials to meshes
    for (size_t i = 0; i < scene_imp->num_meshes; i++) {
        const ufbx_mesh *mesh = scene_imp->meshes[i];
        if (mesh->materials.count > 0 && mesh->materials.data) {
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, mesh->materials.data[0]->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, mesh->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
    }
    
    // Connect animation layers to animation stacks
    for (size_t i = 0; i < scene_imp->num_anim_layers; i++) {
        const ufbx_anim_layer *layer = scene_imp->anim_layers[i];
        
        // Find the parent animation stack
        for (size_t j = 0; j < scene_imp->num_anim_stacks; j++) {
            const ufbx_anim_stack *stack = scene_imp->anim_stacks[j];
            
            // Check if this layer belongs to this stack
            bool belongs_to_stack = false;
            for (size_t k = 0; k < stack->layers.count; k++) {
                if (stack->layers.data[k] == layer) {
                    belongs_to_stack = true;
                    break;
                }
            }
            
            if (belongs_to_stack) {
                if (!ufbx_ascii_write_node_begin(writer, "C")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_string(writer, "OO")) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, layer->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, stack->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_newline(writer)) {
                    return false;
                }
                break;
            }
        }
    }
    
    // Connect skin deformers to meshes
    for (size_t i = 0; i < scene_imp->num_meshes; i++) {
        const ufbx_mesh *mesh = scene_imp->meshes[i];
        if (mesh->skin_deformers.count > 0 && mesh->skin_deformers.data) {
            for (size_t j = 0; j < mesh->skin_deformers.count; j++) {
                const ufbx_skin_deformer *skin = mesh->skin_deformers.data[j];
                if (!ufbx_ascii_write_node_begin(writer, "C")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_string(writer, "OO")) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, skin->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, mesh->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_newline(writer)) {
                    return false;
                }
            }
        }
    }
    
    // Connect skin clusters to skin deformers and to bone nodes
    for (size_t i = 0; i < scene_imp->scene.skin_deformers.count; i++) {
        const ufbx_skin_deformer *skin = scene_imp->scene.skin_deformers.data[i];
        
        for (size_t j = 0; j < skin->clusters.count; j++) {
            const ufbx_skin_cluster *cluster = skin->clusters.data[j];
            
            // Connect cluster to skin deformer
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, cluster->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, skin->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
            
            // Connect cluster to bone node
            if (cluster->bone_node) {
                if (!ufbx_ascii_write_node_begin(writer, "C")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_string(writer, "OO")) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, cluster->bone_node->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, cluster->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_newline(writer)) {
                    return false;
                }
            }
        }
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

// Main export function implementation - ASCII only
ufbx_error ufbx_export_to_file_impl(const ufbx_export_scene *scene, const char *filename, 
                                    const ufbx_export_opts *opts) {
    ufbx_error error = { UFBX_ERROR_NONE };
    
    if (!scene || !filename) {
        error.type = UFBX_ERROR_UNKNOWN;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Invalid scene or filename");
        error.info_length = strlen(error.info);
        return error;
    }
    
    // Initialize ASCII writer
    ufbx_ascii_writer writer = { 0 };
    writer.version = opts && opts->fbx_version ? opts->fbx_version : UFBX_FBX_VERSION_7400;
    
    // Write ASCII FBX header
    if (!ufbx_ascii_write_header(&writer, writer.version)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write scene info
    if (!ufbx_ascii_write_scene_info(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write global settings
    if (!ufbx_ascii_write_global_settings(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write Objects
    if (!ufbx_ascii_write_objects(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write Connections
    if (!ufbx_ascii_write_connections(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write to file
    FILE *file = fopen(filename, "w"); // Text mode for ASCII
    if (!file) {
        error.type = UFBX_ERROR_FILE_NOT_FOUND;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Could not open file for writing");
        error.info_length = strlen(error.info);
        goto cleanup;
    }
    
    size_t written = fwrite(writer.data, 1, writer.size, file);
    fclose(file);
    
    if (written != writer.size) {
        error.type = UFBX_ERROR_IO;
        snprintf(error.info, UFBX_ERROR_INFO_LENGTH, "Failed to write complete file");
        error.info_length = strlen(error.info);
        goto cleanup;
    }
    
cleanup:
    free(writer.data);
    return error;
}

ufbx_export_result ufbx_export_to_memory_impl(const ufbx_export_scene *scene, 
                                              const ufbx_export_opts *opts) {
    ufbx_export_result result = { 0 };
    
    if (!scene) {
        result.error.type = UFBX_ERROR_UNKNOWN;
        snprintf(result.error.info, UFBX_ERROR_INFO_LENGTH, "Invalid scene");
        result.error.info_length = strlen(result.error.info);
        return result;
    }
    
    // Initialize ASCII writer
    ufbx_ascii_writer writer = { 0 };
    writer.version = opts && opts->fbx_version ? opts->fbx_version : UFBX_FBX_VERSION_7400;
    
    // Write ASCII FBX header
    if (!ufbx_ascii_write_header(&writer, writer.version)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write scene info
    if (!ufbx_ascii_write_scene_info(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write global settings
    if (!ufbx_ascii_write_global_settings(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write Objects
    if (!ufbx_ascii_write_objects(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write Connections
    if (!ufbx_ascii_write_connections(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Return the data
    result.data = writer.data;
    result.size = writer.size;
    writer.data = NULL; // Transfer ownership
    
cleanup:
    free(writer.data);
    return result;
}
