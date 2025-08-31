#include "scene_copy_common.h"
#include <stdio.h>
#include <stdlib.h>

// Copy meshes with full vertex attribute support
bool copy_meshes(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                 mesh_mapping **mesh_mappings)
{
    printf("  Copying %zu meshes...\n", source_scene->meshes.count);
    
    if (source_scene->meshes.count == 0) {
        *mesh_mappings = NULL;
        return true;
    }
    
    *mesh_mappings = malloc(source_scene->meshes.count * sizeof(mesh_mapping));
    if (!*mesh_mappings) {
        fprintf(stderr, "Failed to allocate mesh mapping array\n");
        return false;
    }
    
    for (size_t i = 0; i < source_scene->meshes.count; i++) {
        ufbx_mesh *src_mesh = source_scene->meshes.data[i];
        
        // Create mesh in export scene
        ufbx_mesh *export_mesh = ufbx_add_mesh(export_scene, src_mesh->name.data);
        if (!export_mesh) {
            printf("    Failed to add mesh: %s\n", src_mesh->name.data);
            return false;
        }
        
        // Store mapping
        (*mesh_mappings)[i].src_mesh = src_mesh;
        (*mesh_mappings)[i].export_mesh = export_mesh;
        
        // Copy vertex data
        if (src_mesh->vertices.count > 0) {
            ufbx_error vertices_error = {0};
            bool success = ufbx_set_mesh_vertices(export_mesh, src_mesh->vertices.data, 
                                                 src_mesh->vertices.count, &vertices_error);
            if (!success) {
                print_error(&vertices_error, "Failed to set mesh vertices");
                return false;
            }
        }
        
        // Copy face/index data with validation
        if (src_mesh->vertex_indices.count > 0 && src_mesh->num_faces > 0) {
            // Validate faces first - skip faces with less than 3 vertices
            bool has_valid_faces = false;
            for (size_t face_idx = 0; face_idx < src_mesh->num_faces; face_idx++) {
                ufbx_face face = src_mesh->faces.data[face_idx];
                if (face.num_indices >= 3) {
                    has_valid_faces = true;
                    break;
                }
            }
            
            if (has_valid_faces) {
                ufbx_error faces_error = {0};
                bool success = ufbx_set_mesh_faces(export_mesh, src_mesh->vertex_indices.data,
                                                  src_mesh->vertex_indices.count,
                                                  src_mesh->faces.data, src_mesh->num_faces, &faces_error);
                if (!success) {
                    printf("    Warning: Failed to set mesh faces for '%s' - may contain invalid faces\n", src_mesh->name.data);
                    // Don't fail entirely, just skip the problematic face data
                }
            } else {
                printf("    Warning: Mesh '%s' has no valid faces (all faces have < 3 vertices)\n", src_mesh->name.data);
            }
        }
        
        // Copy normals if present
        if (src_mesh->vertex_normal.exists && src_mesh->vertex_normal.values.count > 0) {
            ufbx_error normals_error = {0};
            bool success = ufbx_set_mesh_normals(export_mesh, src_mesh->vertex_normal.values.data,
                                                 src_mesh->vertex_normal.values.count, &normals_error);
            if (!success) {
                print_error(&normals_error, "Failed to set mesh normals");
                return false;
            }
        }
        
        // Copy UVs if present
        if (src_mesh->vertex_uv.exists && src_mesh->vertex_uv.values.count > 0) {
            ufbx_error uvs_error = {0};
            bool success = ufbx_set_mesh_uvs(export_mesh, src_mesh->vertex_uv.values.data,
                                             src_mesh->vertex_uv.values.count, &uvs_error);
            if (!success) {
                print_error(&uvs_error, "Failed to set mesh UVs");
                return false;
            }
        }
        
        printf("    Added mesh: %s (%zu vertices, %zu faces)\n", 
               src_mesh->name.data, src_mesh->vertices.count, src_mesh->num_faces);
    }
    
    return true;
}

// Attach materials to meshes (should be called after materials are copied)
bool attach_materials_to_meshes(ufbx_scene *source_scene, mesh_mapping *mesh_mappings, material_mapping *material_mappings)
{
    printf("  Attaching materials to meshes...\n");
    
    for (size_t i = 0; i < source_scene->meshes.count; i++) {
        ufbx_mesh *src_mesh = source_scene->meshes.data[i];
        ufbx_mesh *export_mesh = mesh_mappings[i].export_mesh;
        
        // Attach materials to mesh
        for (size_t j = 0; j < src_mesh->materials.count; j++) {
            ufbx_material *src_mat = src_mesh->materials.data[j];
            // Find corresponding export material
            for (size_t k = 0; k < source_scene->materials.count; k++) {
                if (source_scene->materials.data[k] == src_mat) {
                    ufbx_error attach_error = {0};
                    bool success = ufbx_attach_material_to_mesh(export_mesh, material_mappings[k].export_material, &attach_error);
                    if (!success) {
                        print_error(&attach_error, "Failed to attach material to mesh");
                        return false;
                    }
                    break;
                }
            }
        }
    }
    
    return true;
}
