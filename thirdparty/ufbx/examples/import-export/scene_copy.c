#include "scene_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Copy scene data from imported scene to export scene
bool copy_scene_data(ufbx_scene *source_scene, ufbx_export_scene *export_scene)
{
    printf("Copying scene data from imported scene to export scene...\n");
    ufbx_error error;
    
    // Copy all nodes and build hierarchy
    printf("  Copying %zu nodes...\n", source_scene->nodes.count);
    for (size_t i = 0; i < source_scene->nodes.count; i++) {
        ufbx_node *src_node = source_scene->nodes.data[i];
        
        // Skip root node as it's automatically created
        if (!src_node->parent) {
            continue;
        }
        
        // Find parent in export scene (or NULL for root children)
        ufbx_node *parent_node = NULL;
        if (src_node->parent && src_node->parent->parent) {
            // TODO: Look up parent by name/ID - for now just use NULL (direct root children)
            parent_node = NULL;
        }
        
        // Create node in export scene
        ufbx_node *export_node = ufbx_add_node(export_scene, src_node->name.data, parent_node);
        if (!export_node) {
            printf("    Failed to add node: %s\n", src_node->name.data);
            return false;
        }
        
        // Copy transform
        ufbx_set_node_transform(export_node, &src_node->local_transform, &error);
        if (error.type != UFBX_ERROR_NONE) {
            print_error(&error, "Failed to set node transform");
            return false;
        }
        
        printf("    Added node: %s\n", src_node->name.data);
    }
    
    // Copy all meshes
    printf("  Copying %zu meshes...\n", source_scene->meshes.count);
    for (size_t i = 0; i < source_scene->meshes.count; i++) {
        ufbx_mesh *src_mesh = source_scene->meshes.data[i];
        
        // Create mesh in export scene
        ufbx_mesh *export_mesh = ufbx_add_mesh(export_scene, src_mesh->name.data);
        if (!export_mesh) {
            printf("    Failed to add mesh: %s\n", src_mesh->name.data);
            return false;
        }
        
        // Copy vertex data
        if (src_mesh->vertices.count > 0) {
            bool success = ufbx_set_mesh_vertices(export_mesh, src_mesh->vertices.data, 
                                                 src_mesh->vertices.count, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh vertices");
                return false;
            }
        }
        
        // Copy face/index data
        if (src_mesh->vertex_indices.count > 0 && src_mesh->num_faces > 0) {
            bool success = ufbx_set_mesh_faces(export_mesh, src_mesh->vertex_indices.data,
                                              src_mesh->vertex_indices.count,
                                              src_mesh->faces.data, src_mesh->num_faces, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh faces");
                return false;
            }
        }
        
        // Copy normals if present
        if (src_mesh->vertex_normal.exists && src_mesh->vertex_normal.values.count > 0) {
            bool success = ufbx_set_mesh_normals(export_mesh, src_mesh->vertex_normal.values.data,
                                                 src_mesh->vertex_normal.values.count, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh normals");
                return false;
            }
        }
        
        // Copy UVs if present
        if (src_mesh->vertex_uv.exists && src_mesh->vertex_uv.values.count > 0) {
            bool success = ufbx_set_mesh_uvs(export_mesh, src_mesh->vertex_uv.values.data,
                                             src_mesh->vertex_uv.values.count, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh UVs");
                return false;
            }
        }
        
        printf("    Added mesh: %s (%zu vertices, %zu faces)\n", 
               src_mesh->name.data, src_mesh->vertices.count, src_mesh->num_faces);
    }
    
    // Copy all materials
    printf("  Copying %zu materials...\n", source_scene->materials.count);
    for (size_t i = 0; i < source_scene->materials.count; i++) {
        ufbx_material *src_material = source_scene->materials.data[i];
        
        // Create material in export scene
        ufbx_material *export_material = ufbx_add_material(export_scene, src_material->name.data);
        if (!export_material) {
            printf("    Failed to add material: %s\n", src_material->name.data);
            return false;
        }
        
        // Copy material properties
        if (src_material->pbr.base_color.has_value) {
            ufbx_vec4 color = src_material->pbr.base_color.value_vec4;
            bool success = ufbx_set_material_albedo(export_material, color.x, color.y, color.z, color.w, &error);
            if (!success) {
                print_error(&error, "Failed to set material albedo");
                return false;
            }
        }
        
        if (src_material->pbr.metalness.has_value && src_material->pbr.roughness.has_value) {
            bool success = ufbx_set_material_metallic_roughness(export_material, 
                                                               src_material->pbr.metalness.value_real,
                                                               src_material->pbr.roughness.value_real, &error);
            if (!success) {
                print_error(&error, "Failed to set material metallic/roughness");
                return false;
            }
        }
        
        printf("    Added material: %s\n", src_material->name.data);
    }
    
    printf("Scene data copied successfully!\n");
    return true;
}
