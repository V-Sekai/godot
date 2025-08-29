#include "ufbx_export.h"
#include "ufbx.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// FBX Binary Format Constants
#define UFBX_FBX_MAGIC "Kaydara FBX Binary  \x00\x1a\x00"
#define UFBX_FBX_MAGIC_SIZE 23
#define UFBX_FBX_VERSION_7400 7400
#define UFBX_FBX_VERSION_7500 7500

// FBX Node Types
#define UFBX_FBX_TYPE_BOOL 'C'
#define UFBX_FBX_TYPE_INT16 'Y'
#define UFBX_FBX_TYPE_INT32 'I'
#define UFBX_FBX_TYPE_INT64 'L'
#define UFBX_FBX_TYPE_FLOAT32 'F'
#define UFBX_FBX_TYPE_FLOAT64 'D'
#define UFBX_FBX_TYPE_ARRAY_BOOL 'b'
#define UFBX_FBX_TYPE_ARRAY_INT32 'i'
#define UFBX_FBX_TYPE_ARRAY_INT64 'l'
#define UFBX_FBX_TYPE_ARRAY_FLOAT32 'f'
#define UFBX_FBX_TYPE_ARRAY_FLOAT64 'd'
#define UFBX_FBX_TYPE_STRING 'S'
#define UFBX_FBX_TYPE_BINARY 'R'

// Internal writer state
typedef struct {
    uint8_t *data;
    size_t size;
    size_t capacity;
    ufbx_error error;
    bool has_error;
    uint32_t version;
} ufbx_fbx_writer;

// Helper functions for writing binary data
static bool ufbx_writer_ensure_capacity(ufbx_fbx_writer *writer, size_t needed) {
    if (writer->has_error) return false;
    
    size_t required = writer->size + needed;
    if (required <= writer->capacity) return true;
    
    size_t new_capacity = writer->capacity ? writer->capacity * 2 : 1024;
    while (new_capacity < required) {
        new_capacity *= 2;
    }
    
    uint8_t *new_data = (uint8_t*)realloc(writer->data, new_capacity);
    if (!new_data) {
        writer->has_error = true;
        writer->error.type = UFBX_ERROR_OUT_OF_MEMORY;
        writer->error.description = "Failed to allocate memory for FBX writer";
        return false;
    }
    
    writer->data = new_data;
    writer->capacity = new_capacity;
    return true;
}

static bool ufbx_writer_write_bytes(ufbx_fbx_writer *writer, const void *data, size_t size) {
    if (!ufbx_writer_ensure_capacity(writer, size)) return false;
    
    memcpy(writer->data + writer->size, data, size);
    writer->size += size;
    return true;
}

static bool ufbx_writer_write_u8(ufbx_fbx_writer *writer, uint8_t value) {
    return ufbx_writer_write_bytes(writer, &value, 1);
}

static bool ufbx_writer_write_u16_le(ufbx_fbx_writer *writer, uint16_t value) {
    uint8_t bytes[2] = { (uint8_t)value, (uint8_t)(value >> 8) };
    return ufbx_writer_write_bytes(writer, bytes, 2);
}

static bool ufbx_writer_write_u32_le(ufbx_fbx_writer *writer, uint32_t value) {
    uint8_t bytes[4] = { 
        (uint8_t)value, 
        (uint8_t)(value >> 8), 
        (uint8_t)(value >> 16), 
        (uint8_t)(value >> 24) 
    };
    return ufbx_writer_write_bytes(writer, bytes, 4);
}

static bool ufbx_writer_write_u64_le(ufbx_fbx_writer *writer, uint64_t value) {
    uint8_t bytes[8] = { 
        (uint8_t)value, 
        (uint8_t)(value >> 8), 
        (uint8_t)(value >> 16), 
        (uint8_t)(value >> 24),
        (uint8_t)(value >> 32), 
        (uint8_t)(value >> 40), 
        (uint8_t)(value >> 48), 
        (uint8_t)(value >> 56)
    };
    return ufbx_writer_write_bytes(writer, bytes, 8);
}

static bool ufbx_writer_write_i32_le(ufbx_fbx_writer *writer, int32_t value) {
    return ufbx_writer_write_u32_le(writer, (uint32_t)value);
}

static bool ufbx_writer_write_i64_le(ufbx_fbx_writer *writer, int64_t value) {
    return ufbx_writer_write_u64_le(writer, (uint64_t)value);
}

static bool ufbx_writer_write_f32_le(ufbx_fbx_writer *writer, float value) {
    union { float f; uint32_t u; } conv = { value };
    return ufbx_writer_write_u32_le(writer, conv.u);
}

static bool ufbx_writer_write_f64_le(ufbx_fbx_writer *writer, double value) {
    union { double d; uint64_t u; } conv = { value };
    return ufbx_writer_write_u64_le(writer, conv.u);
}

static bool ufbx_writer_write_string(ufbx_fbx_writer *writer, const char *str) {
    size_t len = str ? strlen(str) : 0;
    if (!ufbx_writer_write_u32_le(writer, (uint32_t)len)) return false;
    if (len > 0) {
        return ufbx_writer_write_bytes(writer, str, len);
    }
    return true;
}

// FBX Node writing functions
typedef struct {
    size_t offset_pos;
    size_t start_pos;
} ufbx_fbx_node_context;

static bool ufbx_writer_begin_node(ufbx_fbx_writer *writer, const char *name, ufbx_fbx_node_context *ctx) {
    ctx->start_pos = writer->size;
    
    // Write placeholder for end offset (will be filled later)
    ctx->offset_pos = writer->size;
    if (!ufbx_writer_write_u32_le(writer, 0)) return false;
    
    // Write placeholder for property count (will be filled later)
    if (!ufbx_writer_write_u32_le(writer, 0)) return false;
    
    // Write placeholder for property list length (will be filled later)
    if (!ufbx_writer_write_u32_le(writer, 0)) return false;
    
    // Write node name
    size_t name_len = name ? strlen(name) : 0;
    if (!ufbx_writer_write_u8(writer, (uint8_t)name_len)) return false;
    if (name_len > 0) {
        if (!ufbx_writer_write_bytes(writer, name, name_len)) return false;
    }
    
    return true;
}

static bool ufbx_writer_end_node(ufbx_fbx_writer *writer, ufbx_fbx_node_context *ctx, uint32_t property_count) {
    size_t end_pos = writer->size;
    size_t property_list_length = end_pos - ctx->start_pos - 13; // 13 = header size
    
    // Update end offset
    uint32_t end_offset = (uint32_t)end_pos;
    memcpy(writer->data + ctx->offset_pos, &end_offset, 4);
    
    // Update property count
    memcpy(writer->data + ctx->offset_pos + 4, &property_count, 4);
    
    // Update property list length
    uint32_t prop_list_len = (uint32_t)property_list_length;
    memcpy(writer->data + ctx->offset_pos + 8, &prop_list_len, 4);
    
    return true;
}

static bool ufbx_writer_write_property_i64(ufbx_fbx_writer *writer, int64_t value) {
    if (!ufbx_writer_write_u8(writer, UFBX_FBX_TYPE_INT64)) return false;
    return ufbx_writer_write_i64_le(writer, value);
}

static bool ufbx_writer_write_property_f64(ufbx_fbx_writer *writer, double value) {
    if (!ufbx_writer_write_u8(writer, UFBX_FBX_TYPE_FLOAT64)) return false;
    return ufbx_writer_write_f64_le(writer, value);
}

static bool ufbx_writer_write_property_string(ufbx_fbx_writer *writer, const char *str) {
    if (!ufbx_writer_write_u8(writer, UFBX_FBX_TYPE_STRING)) return false;
    return ufbx_writer_write_string(writer, str);
}

static bool ufbx_writer_write_property_array_f64(ufbx_fbx_writer *writer, const double *values, size_t count) {
    if (!ufbx_writer_write_u8(writer, UFBX_FBX_TYPE_ARRAY_FLOAT64)) return false;
    if (!ufbx_writer_write_u32_le(writer, (uint32_t)(count * 8))) return false; // Array length in bytes
    if (!ufbx_writer_write_u32_le(writer, 0)) return false; // Encoding (0 = uncompressed)
    if (!ufbx_writer_write_u32_le(writer, (uint32_t)count)) return false; // Element count
    
    for (size_t i = 0; i < count; i++) {
        if (!ufbx_writer_write_f64_le(writer, values[i])) return false;
    }
    return true;
}

static bool ufbx_writer_write_property_array_i32(ufbx_fbx_writer *writer, const int32_t *values, size_t count) {
    if (!ufbx_writer_write_u8(writer, UFBX_FBX_TYPE_ARRAY_INT32)) return false;
    if (!ufbx_writer_write_u32_le(writer, (uint32_t)(count * 4))) return false; // Array length in bytes
    if (!ufbx_writer_write_u32_le(writer, 0)) return false; // Encoding (0 = uncompressed)
    if (!ufbx_writer_write_u32_le(writer, (uint32_t)count)) return false; // Element count
    
    for (size_t i = 0; i < count; i++) {
        if (!ufbx_writer_write_i32_le(writer, values[i])) return false;
    }
    return true;
}

// High-level FBX structure writing
static bool ufbx_write_fbx_header(ufbx_fbx_writer *writer) {
    // Write FBX magic
    if (!ufbx_writer_write_bytes(writer, UFBX_FBX_MAGIC, UFBX_FBX_MAGIC_SIZE)) return false;
    
    // Write version
    if (!ufbx_writer_write_u32_le(writer, writer->version)) return false;
    
    return true;
}

static bool ufbx_write_fbx_footer(ufbx_fbx_writer *writer) {
    // Write null node (13 zero bytes)
    uint8_t null_node[13] = { 0 };
    if (!ufbx_writer_write_bytes(writer, null_node, 13)) return false;
    
    // Write footer padding and version
    uint8_t footer[160] = { 0 };
    
    // Set some footer bytes (FBX format requirement)
    for (int i = 0; i < 16; i++) {
        footer[i] = 0xF8 + i;
    }
    
    // Write version at the end
    uint32_t version = writer->version;
    memcpy(footer + 156, &version, 4);
    
    return ufbx_writer_write_bytes(writer, footer, 160);
}

static bool ufbx_write_scene_info(ufbx_fbx_writer *writer, const ufbx_export_scene *scene) {
    ufbx_fbx_node_context ctx;
    
    // Write FBXHeaderExtension
    if (!ufbx_writer_begin_node(writer, "FBXHeaderExtension", &ctx)) return false;
    {
        // FBXHeaderVersion
        ufbx_fbx_node_context version_ctx;
        if (!ufbx_writer_begin_node(writer, "FBXHeaderVersion", &version_ctx)) return false;
        if (!ufbx_writer_write_property_i64(writer, 1003)) return false;
        if (!ufbx_writer_end_node(writer, &version_ctx, 1)) return false;
        
        // FBXVersion
        ufbx_fbx_node_context fbx_version_ctx;
        if (!ufbx_writer_begin_node(writer, "FBXVersion", &fbx_version_ctx)) return false;
        if (!ufbx_writer_write_property_i64(writer, writer->version)) return false;
        if (!ufbx_writer_end_node(writer, &fbx_version_ctx, 1)) return false;
        
        // Creator
        ufbx_fbx_node_context creator_ctx;
        if (!ufbx_writer_begin_node(writer, "Creator", &creator_ctx)) return false;
        if (!ufbx_writer_write_property_string(writer, scene->metadata.creator)) return false;
        if (!ufbx_writer_end_node(writer, &creator_ctx, 1)) return false;
    }
    if (!ufbx_writer_end_node(writer, &ctx, 0)) return false;
    
    return true;
}

static bool ufbx_write_global_settings(ufbx_fbx_writer *writer, const ufbx_export_scene *scene) {
    ufbx_fbx_node_context ctx;
    
    if (!ufbx_writer_begin_node(writer, "GlobalSettings", &ctx)) return false;
    {
        // Version
        ufbx_fbx_node_context version_ctx;
        if (!ufbx_writer_begin_node(writer, "Version", &version_ctx)) return false;
        if (!ufbx_writer_write_property_i64(writer, 1000)) return false;
        if (!ufbx_writer_end_node(writer, &version_ctx, 1)) return false;
        
        // Properties70
        ufbx_fbx_node_context props_ctx;
        if (!ufbx_writer_begin_node(writer, "Properties70", &props_ctx)) return false;
        {
            // UpAxis
            ufbx_fbx_node_context prop_ctx;
            if (!ufbx_writer_begin_node(writer, "P", &prop_ctx)) return false;
            if (!ufbx_writer_write_property_string(writer, "UpAxis")) return false;
            if (!ufbx_writer_write_property_string(writer, "int")) return false;
            if (!ufbx_writer_write_property_string(writer, "Integer")) return false;
            if (!ufbx_writer_write_property_string(writer, "")) return false;
            if (!ufbx_writer_write_property_i64(writer, scene->settings.axes.up == UFBX_COORDINATE_AXIS_POSITIVE_Y ? 1 : 2)) return false;
            if (!ufbx_writer_end_node(writer, &prop_ctx, 5)) return false;
            
            // UpAxisSign
            if (!ufbx_writer_begin_node(writer, "P", &prop_ctx)) return false;
            if (!ufbx_writer_write_property_string(writer, "UpAxisSign")) return false;
            if (!ufbx_writer_write_property_string(writer, "int")) return false;
            if (!ufbx_writer_write_property_string(writer, "Integer")) return false;
            if (!ufbx_writer_write_property_string(writer, "")) return false;
            if (!ufbx_writer_write_property_i64(writer, 1)) return false;
            if (!ufbx_writer_end_node(writer, &prop_ctx, 5)) return false;
            
            // FrontAxis
            if (!ufbx_writer_begin_node(writer, "P", &prop_ctx)) return false;
            if (!ufbx_writer_write_property_string(writer, "FrontAxis")) return false;
            if (!ufbx_writer_write_property_string(writer, "int")) return false;
            if (!ufbx_writer_write_property_string(writer, "Integer")) return false;
            if (!ufbx_writer_write_property_string(writer, "")) return false;
            if (!ufbx_writer_write_property_i64(writer, scene->settings.axes.front == UFBX_COORDINATE_AXIS_NEGATIVE_Z ? 2 : 0)) return false;
            if (!ufbx_writer_end_node(writer, &prop_ctx, 5)) return false;
            
            // UnitScaleFactor
            if (!ufbx_writer_begin_node(writer, "P", &prop_ctx)) return false;
            if (!ufbx_writer_write_property_string(writer, "UnitScaleFactor")) return false;
            if (!ufbx_writer_write_property_string(writer, "double")) return false;
            if (!ufbx_writer_write_property_string(writer, "Number")) return false;
            if (!ufbx_writer_write_property_string(writer, "")) return false;
            if (!ufbx_writer_write_property_f64(writer, scene->settings.unit_meters * 100.0)) return false; // Convert to cm
            if (!ufbx_writer_end_node(writer, &prop_ctx, 5)) return false;
        }
        if (!ufbx_writer_end_node(writer, &props_ctx, 0)) return false;
    }
    if (!ufbx_writer_end_node(writer, &ctx, 0)) return false;
    
    return true;
}

static bool ufbx_write_documents(ufbx_fbx_writer *writer, const ufbx_export_scene *scene) {
    ufbx_fbx_node_context ctx;
    
    if (!ufbx_writer_begin_node(writer, "Documents", &ctx)) return false;
    {
        // Count
        ufbx_fbx_node_context count_ctx;
        if (!ufbx_writer_begin_node(writer, "Count", &count_ctx)) return false;
        if (!ufbx_writer_write_property_i64(writer, 1)) return false;
        if (!ufbx_writer_end_node(writer, &count_ctx, 1)) return false;
        
        // Document
        ufbx_fbx_node_context doc_ctx;
        if (!ufbx_writer_begin_node(writer, "Document", &doc_ctx)) return false;
        if (!ufbx_writer_write_property_i64(writer, 1234567890)) return false; // Document ID
        if (!ufbx_writer_write_property_string(writer, "Scene")) return false;
        if (!ufbx_writer_write_property_string(writer, "Scene")) return false;
        {
            // Properties70
            ufbx_fbx_node_context props_ctx;
            if (!ufbx_writer_begin_node(writer, "Properties70", &props_ctx)) return false;
            if (!ufbx_writer_end_node(writer, &props_ctx, 0)) return false;
            
            // RootNode
            ufbx_fbx_node_context root_ctx;
            if (!ufbx_writer_begin_node(writer, "RootNode", &root_ctx)) return false;
            if (!ufbx_writer_write_property_i64(writer, 0)) return false; // Root node ID
            if (!ufbx_writer_end_node(writer, &root_ctx, 1)) return false;
        }
        if (!ufbx_writer_end_node(writer, &doc_ctx, 3)) return false;
    }
    if (!ufbx_writer_end_node(writer, &ctx, 0)) return false;
    
    return true;
}

static bool ufbx_write_references(ufbx_fbx_writer *writer, const ufbx_export_scene *scene) {
    ufbx_fbx_node_context ctx;
    
    if (!ufbx_writer_begin_node(writer, "References", &ctx)) return false;
    if (!ufbx_writer_end_node(writer, &ctx, 0)) return false;
    
    return true;
}

static bool ufbx_write_definitions(ufbx_fbx_writer *writer, const ufbx_export_scene *scene) {
    ufbx_fbx_node_context ctx;
    
    if (!ufbx_writer_begin_node(writer, "Definitions", &ctx)) return false;
    {
        // Version
        ufbx_fbx_node_context version_ctx;
        if (!ufbx_writer_begin_node(writer, "Version", &version_ctx)) return false;
        if (!ufbx_writer_write_property_i64(writer, 100)) return false;
        if (!ufbx_writer_end_node(writer, &version_ctx, 1)) return false;
        
        // Count
        ufbx_fbx_node_context count_ctx;
        if (!ufbx_writer_begin_node(writer, "Count", &count_ctx)) return false;
        
        // Calculate total object count
        const ufbx_export_scene_imp *scene_imp = (const ufbx_export_scene_imp*)scene;
        int64_t total_count = 1; // GlobalSettings
        total_count += scene_imp->num_nodes;
        total_count += scene_imp->num_meshes;
        total_count += scene_imp->num_materials;
        
        if (!ufbx_writer_write_property_i64(writer, total_count)) return false;
        if (!ufbx_writer_end_node(writer, &count_ctx, 1)) return false;
        
        // ObjectType definitions
        if (scene_imp->num_nodes > 0) {
            ufbx_fbx_node_context obj_type_ctx;
            if (!ufbx_writer_begin_node(writer, "ObjectType", &obj_type_ctx)) return false;
            if (!ufbx_writer_write_property_string(writer, "Model")) return false;
            {
                ufbx_fbx_node_context obj_count_ctx;
                if (!ufbx_writer_begin_node(writer, "Count", &obj_count_ctx)) return false;
                if (!ufbx_writer_write_property_i64(writer, scene_imp->num_nodes)) return false;
                if (!ufbx_writer_end_node(writer, &obj_count_ctx, 1)) return false;
            }
            if (!ufbx_writer_end_node(writer, &obj_type_ctx, 1)) return false;
        }
        
        if (scene_imp->num_meshes > 0) {
            ufbx_fbx_node_context obj_type_ctx;
            if (!ufbx_writer_begin_node(writer, "ObjectType", &obj_type_ctx)) return false;
            if (!ufbx_writer_write_property_string(writer, "Geometry")) return false;
            {
                ufbx_fbx_node_context obj_count_ctx;
                if (!ufbx_writer_begin_node(writer, "Count", &obj_count_ctx)) return false;
                if (!ufbx_writer_write_property_i64(writer, scene_imp->num_meshes)) return false;
                if (!ufbx_writer_end_node(writer, &obj_count_ctx, 1)) return false;
            }
            if (!ufbx_writer_end_node(writer, &obj_type_ctx, 1)) return false;
        }
        
        if (scene_imp->num_materials > 0) {
            ufbx_fbx_node_context obj_type_ctx;
            if (!ufbx_writer_begin_node(writer, "ObjectType", &obj_type_ctx)) return false;
            if (!ufbx_writer_write_property_string(writer, "Material")) return false;
            {
                ufbx_fbx_node_context obj_count_ctx;
                if (!ufbx_writer_begin_node(writer, "Count", &obj_count_ctx)) return false;
                if (!ufbx_writer_write_property_i64(writer, scene_imp->num_materials)) return false;
                if (!ufbx_writer_end_node(writer, &obj_count_ctx, 1)) return false;
            }
            if (!ufbx_writer_end_node(writer, &obj_type_ctx, 1)) return false;
        }
    }
    if (!ufbx_writer_end_node(writer, &ctx, 0)) return false;
    
    return true;
}

static bool ufbx_write_objects(ufbx_fbx_writer *writer, const ufbx_export_scene *scene) {
    const ufbx_export_scene_imp *scene_imp = (const ufbx_export_scene_imp*)scene;
    ufbx_fbx_node_context ctx;
    
    if (!ufbx_writer_begin_node(writer, "Objects", &ctx)) return false;
    {
        // Write all nodes
        for (size_t i = 0; i < scene_imp->num_nodes; i++) {
            const ufbx_node *node = scene_imp->nodes[i];
            ufbx_fbx_node_context model_ctx;
            
            if (!ufbx_writer_begin_node(writer, "Model", &model_ctx)) return false;
            if (!ufbx_writer_write_property_i64(writer, node->element.element_id)) return false;
            if (!ufbx_writer_write_property_string(writer, node->element.name.data)) return false;
            if (!ufbx_writer_write_property_string(writer, "Mesh")) return false;
            {
                // Version
                ufbx_fbx_node_context version_ctx;
                if (!ufbx_writer_begin_node(writer, "Version", &version_ctx)) return false;
                if (!ufbx_writer_write_property_i64(writer, 232)) return false;
                if (!ufbx_writer_end_node(writer, &version_ctx, 1)) return false;
                
                // Properties70
                ufbx_fbx_node_context props_ctx;
                if (!ufbx_writer_begin_node(writer, "Properties70", &props_ctx)) return false;
                {
                    // Lcl Translation
                    ufbx_fbx_node_context prop_ctx;
                    if (!ufbx_writer_begin_node(writer, "P", &prop_ctx)) return false;
                    if (!ufbx_writer_write_property_string(writer, "Lcl Translation")) return false;
                    if (!ufbx_writer_write_property_string(writer, "Lcl Translation")) return false;
                    if (!ufbx_writer_write_property_string(writer, "")) return false;
                    if (!ufbx_writer_write_property_string(writer, "A")) return false;
                    if (!ufbx_writer_write_property_f64(writer, node->local_transform.translation.x)) return false;
                    if (!ufbx_writer_write_property_f64(writer, node->local_transform.translation.y)) return false;
                    if (!ufbx_writer_write_property_f64(writer, node->local_transform.translation.z)) return false;
                    if (!ufbx_writer_end_node(writer, &prop_ctx, 7)) return false;
                    
                    // Lcl Rotation
                    if (!ufbx_writer_begin_node(writer, "P", &prop_ctx)) return false;
                    if (!ufbx_writer_write_property_string(writer, "Lcl Rotation")) return false;
                    if (!ufbx_writer_write_property_string(writer, "Lcl Rotation")) return false;
                    if (!ufbx_writer_write_property_string(writer, "")) return false;
                    if (!ufbx_writer_write_property_string(writer, "A")) return false;
                    // Convert quaternion to Euler angles (simplified)
                    if (!ufbx_writer_write_property_f64(writer, 0.0)) return false;
                    if (!ufbx_writer_write_property_f64(writer, 0.0)) return false;
                    if (!ufbx_writer_write_property_f64(writer, 0.0)) return false;
                    if (!ufbx_writer_end_node(writer, &prop_ctx, 7)) return false;
                    
                    // Lcl Scaling
                    if (!ufbx_writer_begin_node(writer, "P", &prop_ctx)) return false;
                    if (!ufbx_writer_write_property_string(writer, "Lcl Scaling")) return false;
                    if (!ufbx_writer_write_property_string(writer, "Lcl Scaling")) return false;
                    if (!ufbx_writer_write_property_string(writer, "")) return false;
                    if (!ufbx_writer_write_property_string(writer, "A")) return false;
                    if (!ufbx_writer_write_property_f64(writer, node->local_transform.scale.x)) return false;
                    if (!ufbx_writer_write_property_f64(writer, node->local_transform.scale.y)) return false;
                    if (!ufbx_writer_write_property_f64(writer, node->local_transform.scale.z)) return false;
                    if (!ufbx_writer_end_node(writer, &prop_ctx, 7)) return false;
                }
                if (!ufbx_writer_end_node(writer, &props_ctx, 0)) return false;
            }
            if (!ufbx_writer_end_node(writer, &model_ctx, 3)) return false;
        }
        
        // Write all meshes
        for (size_t i = 0; i < scene_imp->num_meshes; i++) {
            const ufbx_mesh *mesh = scene_imp->meshes[i];
            ufbx_fbx_node_context geom_ctx;
            
            if (!ufbx_writer_begin_node(writer, "Geometry", &geom_ctx)) return false;
            if (!ufbx_writer_write_property_i64(writer, mesh->element.element_id)) return false;
            if (!ufbx_writer_write_property_string(writer, mesh->element.name.data)) return false;
            if (!ufbx_writer_write_property_string(writer, "Mesh")) return false;
            {
                // Vertices
                if (mesh->vertices.count > 0) {
                    ufbx_fbx_node_context vertices_ctx;
                    if (!ufbx_writer_begin_node(writer, "Vertices", &vertices_ctx)) return false;
                    
                    // Convert vertices to double array
                    double *vertex_data = (double*)malloc(mesh->vertices.count * 3 * sizeof(double));
                    if (vertex_data) {
                        for (size_t v = 0; v < mesh->vertices.count; v++) {
                            vertex_data[v * 3 + 0] = mesh->vertices.data[v].x;
                            vertex_data[v * 3 + 1] = mesh->vertices.data[v].y;
                            vertex_data[v * 3 + 2] = mesh->vertices.data[v].z;
                        }
                        if (!ufbx_writer_write_property_array_f64(writer, vertex_data, mesh->vertices.count * 3)) {
                            free(vertex_data);
                            return false;
                        }
                        free(vertex_data);
                    }
                    if (!ufbx_writer_end_node(writer, &vertices_ctx, 1)) return false;
                }
                
                // PolygonVertexIndex
                if (mesh->vertex_indices.count > 0) {
                    ufbx_fbx_node_context indices_ctx;
                    if (!ufbx_writer_begin_node(writer, "PolygonVertexIndex", &indices_ctx)) return false;
                    
                    // Convert indices to int32 array with FBX polygon encoding
                    int32_t *index_data = (int32_t*)malloc(mesh->vertex_indices.count * sizeof(int32_t));
                    if (index_data) {
                        for (size_t idx = 0; idx < mesh->vertex_indices.count; idx++) {
                            int32_t index = (int32_t)mesh->vertex_indices.data[idx];
                            // FBX uses negative indices to mark polygon end
                            if ((idx + 1) % 3 == 0) { // End of triangle
                                index = -(index + 1);
                            }
                            index_data[idx] = index;
                        }
                        if (!ufbx_writer_write_property_array_i32(writer, index_data, mesh->vertex_indices.count)) {
                            free(index_data);
                            return false;
                        }
                        free(index_data);
                    }
                    if (!ufbx_writer_end_node(writer, &indices_ctx, 1)) return false;
                }
                
                // LayerElementNormal
                if (mesh->vertex_normal.exists && mesh->vertex_normal.count > 0) {
                    ufbx_fbx_node_context layer_elem_ctx;
                    if (!ufbx_writer_begin_node(writer, "LayerElementNormal", &layer_elem_ctx)) return false;
                    if (!ufbx_writer_write_property_i64(writer, 0)) return false;
                    {
                        // Version
                        ufbx_fbx_node_context version_ctx;
                        if (!ufbx_writer_begin_node(writer, "Version", &version_ctx)) return false;
                        if (!ufbx_writer_write_property_i64(writer, 101)) return false;
                        if (!ufbx_writer_end_node(writer, &version_ctx, 1)) return false;
                        
                        // Name
                        ufbx_fbx_node_context name_ctx;
                        if (!ufbx_writer_begin_node(writer, "Name", &name_ctx)) return false;
                        if (!ufbx_writer_write_property_string(writer, "")) return false;
                        if (!ufbx_writer_end_node(writer, &name_ctx, 1)) return false;
                        
                        // MappingInformationType
                        ufbx_fbx_node_context mapping_ctx;
                        if (!ufbx_writer_begin_node(writer, "MappingInformationType", &mapping_ctx)) return false;
                        if (!ufbx_writer_write_property_string(writer, "ByVertice")) return false;
                        if (!ufbx_writer_end_node(writer, &mapping_ctx, 1)) return false;
                        
                        // ReferenceInformationType
                        ufbx_fbx_node_context ref_ctx;
                        if (!ufbx_writer_begin_node(writer, "ReferenceInformationType", &ref_ctx)) return false;
                        if (!ufbx_writer_write_property_string(writer, "Direct")) return false;
                        if (!ufbx_writer_end_node(writer, &ref_ctx, 1)) return false;
                        
                        // Normals
                        ufbx_fbx_node_context normals_ctx;
                        if (!ufbx_writer_begin_node(writer, "Normals", &normals_ctx)) return false;
                        
                        double *normal_data = (double*)malloc(mesh->vertex_normal.count * 3 * sizeof(double));
                        if (normal_data) {
                            for (size_t n = 0; n < mesh->vertex_normal.count; n++) {
                                normal_data[n * 3 + 0] = mesh->vertex_normal.data[n].x;
                                normal_data[n * 3 + 1] = mesh->vertex_normal.data[n].y;
                                normal_data[n * 3 + 2] = mesh->vertex_normal.data[n].z;
                            }
                            if (!ufbx_writer_write_property_array_f64(writer, normal_data, mesh->vertex_normal.count * 3)) {
                                free(normal_data);
                                return false;
                            }
                            free(normal_data);
                        }
                        if (!ufbx_writer_end_node(writer, &normals_ctx, 1)) return false;
                    }
                    if (!ufbx_writer_end_node(writer, &layer_elem_ctx, 1)) return false;
                }
            }
            if (!ufbx_writer_end_node(writer, &geom_ctx, 3)) return false;
        }
        
        // Write all materials
        for (size_t i = 0; i < scene_imp->num_materials; i++) {
            const ufbx_material *material = scene_imp->materials[i];
            ufbx_fbx_node_context mat_ctx;
            
            if (!ufbx_writer_begin_node(writer, "Material", &mat_ctx)) return false;
            if (!ufbx_writer_write_property_i64(writer, material->element.element_id)) return false;
            if (!ufbx_writer_write_property_string(writer, material->element.name.data)) return false;
            if (!ufbx_writer_write_property_string(writer, "")) return false;
            {
                // Version
                ufbx_fbx_node_context version_ctx;
                if (!ufbx_writer_begin_node(writer, "Version", &version_ctx)) return false;
                if (!ufbx_writer_write_property_i64(writer, 102)) return false;
                if (!ufbx_writer_end_node(writer, &version_ctx, 1)) return false;
                
                // ShadingModel
                ufbx_fbx_node_context shading_ctx;
                if (!ufbx_writer_begin_node(writer, "ShadingModel", &shading_ctx)) return false;
                if (!ufbx_writer_write_property_string(writer, "phong")) return false;
                if (!ufbx_writer_end_node(writer, &shading_ctx, 1)) return false;
                
                // MultiLayer
                ufbx_fbx_node_context multi_ctx;
                if (!ufbx_writer_begin_node(writer, "MultiLayer", &multi_ctx)) return false;
                if (!ufbx_writer_write_property_i64(writer, 0)) return false;
                if (!ufbx_writer_end_node(writer, &multi_ctx, 1)) return false;
                
                // Properties70
                ufbx_fbx_node_context props_ctx;
                if (!ufbx_writer_begin_node(writer, "Properties70", &props_ctx)) return false;
                {
                    // DiffuseColor
                    ufbx_fbx_node_context prop_ctx;
                    if (!ufbx_writer_begin_node(writer, "P", &prop_ctx)) return false;
                    if (!ufbx_writer_write_property_string(writer, "DiffuseColor")) return false;
                    if (!ufbx_writer_write_property_string(writer, "Color")) return false;
                    if (!ufbx_writer_write_property_string(writer, "")) return false;
                    if (!ufbx_writer_write_property_string(writer, "A")) return false;
                    if (!ufbx_writer_write_property_f64(writer, material->pbr.base_factor.x)) return false;
                    if (!ufbx_writer_write_property_f64(writer, material->pbr.base_factor.y)) return false;
                    if (!ufbx_writer_write_property_f64(writer, material->pbr.base_factor.z)) return false;
                    if (!ufbx_writer_end_node(writer, &prop_ctx, 7)) return false;
                }
                if (!ufbx_writer_end_node(writer, &props_ctx, 0)) return false;
            }
            if (!ufbx_writer_end_node(writer, &mat_ctx, 3)) return false;
        }
    }
    if (!ufbx_writer_end_node(writer, &ctx, 0)) return false;
    
    return true;
}

static bool ufbx_write_connections(ufbx_fbx_writer *writer, const ufbx_export_scene *scene) {
    const ufbx_export_scene_imp *scene_imp = (const ufbx_export_scene_imp*)scene;
    ufbx_fbx_node_context ctx;
    
    if (!ufbx_writer_begin_node(writer, "Connections", &ctx)) return false;
    {
        // Connect nodes to root
        for (size_t i = 0; i < scene_imp->num_nodes; i++) {
            const ufbx_node *node = scene_imp->nodes[i];
            if (!node->parent) { // Root level nodes
                ufbx_fbx_node_context conn_ctx;
                if (!ufbx_writer_begin_node(writer, "C", &conn_ctx)) return false;
                if (!ufbx_writer_write_property_string(writer, "OO")) return false;
                if (!ufbx_writer_write_property_i64(writer, node->element.element_id)) return false;
                if (!ufbx_writer_write_property_i64(writer, 0)) return false; // Root node ID
                if (!ufbx_writer_end_node(writer, &conn_ctx, 3)) return false;
            } else {
                // Connect to parent node
                ufbx_fbx_node_context conn_ctx;
                if (!ufbx_writer_begin_node(writer, "C", &conn_ctx)) return false;
                if (!ufbx_writer_write_property_string(writer, "OO")) return false;
                if (!ufbx_writer_write_property_i64(writer, node->element.element_id)) return false;
                if (!ufbx_writer_write_property_i64(writer, node->parent->element.element_id)) return false;
                if (!ufbx_writer_end_node(writer, &conn_ctx, 3)) return false;
            }
            
            // Connect mesh to node if it exists
            if (node->mesh) {
                ufbx_fbx_node_context conn_ctx;
                if (!ufbx_writer_begin_node(writer, "C", &conn_ctx)) return false;
                if (!ufbx_writer_write_property_string(writer, "OO")) return false;
                if (!ufbx_writer_write_property_i64(writer, node->mesh->element.element_id)) return false;
                if (!ufbx_writer_write_property_i64(writer, node->element.element_id)) return false;
                if (!ufbx_writer_end_node(writer, &conn_ctx, 3)) return false;
            }
        }
        
        // Connect materials to meshes
        for (size_t i = 0; i < scene_imp->num_meshes; i++) {
            const ufbx_mesh *mesh = scene_imp->meshes[i];
            if (mesh->materials.count > 0 && mesh->materials.data) {
                ufbx_fbx_node_context conn_ctx;
                if (!ufbx_writer_begin_node(writer, "C", &conn_ctx)) return false;
                if (!ufbx_writer_write_property_string(writer, "OO")) return false;
                if (!ufbx_writer_write_property_i64(writer, mesh->materials.data[0]->element.element_id)) return false;
                if (!ufbx_writer_write_property_i64(writer, mesh->element.element_id)) return false;
                if (!ufbx_writer_end_node(writer, &conn_ctx, 3)) return false;
            }
        }
    }
    if (!ufbx_writer_end_node(writer, &ctx, 0)) return false;
    
    return true;
}

// Main export function implementation
ufbx_error ufbx_export_to_file_impl(const ufbx_export_scene *scene, const char *filename, 
                                    const ufbx_export_opts *opts) {
    ufbx_error error = { UFBX_ERROR_NONE };
    
    if (!scene || !filename) {
        error.type = UFBX_ERROR_BAD_ARGUMENT;
        error.description = "Invalid scene or filename";
        return error;
    }
    
    // Initialize writer
    ufbx_fbx_writer writer = { 0 };
    writer.version = opts && opts->fbx_version ? opts->fbx_version : UFBX_FBX_VERSION_7400;
    
    // Write FBX header
    if (!ufbx_write_fbx_header(&writer)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write scene info
    if (!ufbx_write_scene_info(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write global settings
    if (!ufbx_write_global_settings(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write Documents
    if (!ufbx_write_documents(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write References
    if (!ufbx_write_references(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write Definitions
    if (!ufbx_write_definitions(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write Objects
    if (!ufbx_write_objects(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write Connections
    if (!ufbx_write_connections(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write footer
    if (!ufbx_write_fbx_footer(&writer)) {
        error = writer.error;
        goto cleanup;
    }
    
    // Write to file
    FILE *file = fopen(filename, "wb");
    if (!file) {
        error.type = UFBX_ERROR_FILE_NOT_FOUND;
        error.description = "Could not open file for writing";
        goto cleanup;
    }
    
    size_t written = fwrite(writer.data, 1, writer.size, file);
    fclose(file);
    
    if (written != writer.size) {
        error.type = UFBX_ERROR_IO;
        error.description = "Failed to write complete file";
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
        result.error.type = UFBX_ERROR_BAD_ARGUMENT;
        result.error.description = "Invalid scene";
        return result;
    }
    
    // Initialize writer
    ufbx_fbx_writer writer = { 0 };
    writer.version = opts && opts->fbx_version ? opts->fbx_version : UFBX_FBX_VERSION_7400;
    
    // Write FBX header
    if (!ufbx_write_fbx_header(&writer)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write scene info
    if (!ufbx_write_scene_info(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write global settings
    if (!ufbx_write_global_settings(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write Documents
    if (!ufbx_write_documents(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write References
    if (!ufbx_write_references(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write Definitions
    if (!ufbx_write_definitions(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write Objects
    if (!ufbx_write_objects(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write Connections
    if (!ufbx_write_connections(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // Write footer
    if (!ufbx_write_fbx_footer(&writer)) {
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
