#include "ufbx_export.h"
#include "ufbx_ascii_writer.h"
#include "ufbx_fbx_structures.h"
#include "ufbx_geometry_writer.h"
#include "ufbx_animation_writer.h"
#include "ufbx_material_writer.h"
#include "ufbx_connections.h"
#include <stdio.h>
#include <string.h>

static bool ufbx_ascii_write_objects(ufbx_ascii_writer *writer, const ufbx_export_scene *scene) {
    const ufbxi_export_scene *scene_imp = (const ufbxi_export_scene*)scene;
    
    if (!ufbx_ascii_write_node_begin(writer, "Objects")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Write all nodes using modular writer
    for (size_t i = 0; i < scene_imp->num_nodes; i++) {
        if (!ufbx_ascii_write_model_node(writer, scene_imp->nodes[i])) {
            return false;
        }
    }
    
    // Write all meshes using modular writer
    for (size_t i = 0; i < scene_imp->num_meshes; i++) {
        if (!ufbx_ascii_write_geometry(writer, scene_imp->meshes[i])) {
            return false;
        }
    }
    
    // Write all materials using modular writer
    for (size_t i = 0; i < scene_imp->num_materials; i++) {
        if (!ufbx_ascii_write_material(writer, scene_imp->materials[i])) {
            return false;
        }
    }
    
    // Write all bones using modular writer
    for (size_t i = 0; i < scene_imp->scene.bones.count; i++) {
        if (!ufbx_ascii_write_bone_attribute(writer, scene_imp->scene.bones.data[i])) {
            return false;
        }
    }
    
    // Write all animation stacks using modular writer
    for (size_t i = 0; i < scene_imp->num_anim_stacks; i++) {
        if (!ufbx_ascii_write_animation_stack(writer, scene_imp->anim_stacks[i])) {
            return false;
        }
    }
    
    // Write all animation layers using modular writer
    for (size_t i = 0; i < scene_imp->num_anim_layers; i++) {
        if (!ufbx_ascii_write_animation_layer(writer, scene_imp->anim_layers[i])) {
            return false;
        }
    }
    
    // Write all animation values using modular writer
    for (size_t i = 0; i < scene_imp->num_anim_values; i++) {
        if (!ufbx_ascii_write_animation_curve_node(writer, scene_imp->anim_values[i])) {
            return false;
        }
    }
    
    // Write all animation curves using modular writer
    for (size_t i = 0; i < scene_imp->num_anim_curves; i++) {
        if (!ufbx_ascii_write_animation_curve(writer, scene_imp->anim_curves[i])) {
            return false;
        }
    }
    
    // Write all skin deformers using modular writer
    for (size_t i = 0; i < scene_imp->scene.skin_deformers.count; i++) {
        if (!ufbx_ascii_write_skin_deformer(writer, scene_imp->scene.skin_deformers.data[i])) {
            return false;
        }
    }
    
    // Write all skin clusters using modular writer
    for (size_t i = 0; i < scene_imp->scene.skin_clusters.count; i++) {
        if (!ufbx_ascii_write_skin_cluster(writer, scene_imp->scene.skin_clusters.data[i])) {
            return false;
        }
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

// Main export function implementation - ASCII only (now using modular approach)
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
    
    // Write ASCII FBX using modular approach
    if (!ufbx_ascii_write_header(&writer, writer.version)) {
        error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_scene_info(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_global_settings(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_documents_section(&writer)) {
        error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_definitions_section(&writer)) {
        error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_objects(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_connections(&writer, scene)) {
        error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_takes_section(&writer)) {
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
    
    // Write ASCII FBX using modular approach - MATCH FILE EXPORT STRUCTURE
    if (!ufbx_ascii_write_header(&writer, writer.version)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_scene_info(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_global_settings(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    // CRITICAL FIX: Add missing sections that were causing corruption
    if (!ufbx_ascii_write_documents_section(&writer)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_definitions_section(&writer)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_objects(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_connections(&writer, scene)) {
        result.error = writer.error;
        goto cleanup;
    }
    
    if (!ufbx_ascii_write_takes_section(&writer)) {
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
