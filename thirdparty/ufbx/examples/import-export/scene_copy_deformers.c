#include "scene_copy_common.h"
#include <stdio.h>
#include <stdlib.h>

// Copy skin deformers with proper cluster data
bool copy_skin_deformers(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                         node_mapping *node_mappings, size_t num_mappings,
                         mesh_mapping *mesh_mappings, skin_mapping **skin_mappings)
{
    printf("  Copying %zu skin deformers...\n", source_scene->skin_deformers.count);
    ufbx_error error;
    
    if (source_scene->skin_deformers.count == 0) {
        *skin_mappings = NULL;
        return true;
    }
    
    *skin_mappings = malloc(source_scene->skin_deformers.count * sizeof(skin_mapping));
    if (!*skin_mappings) {
        fprintf(stderr, "Failed to allocate skin mapping array\n");
        return false;
    }
    
    for (size_t i = 0; i < source_scene->skin_deformers.count; i++) {
        ufbx_skin_deformer *src_skin = source_scene->skin_deformers.data[i];
        
        ufbx_skin_deformer *export_skin = ufbx_add_skin_deformer(export_scene, src_skin->name.data);
        if (!export_skin) {
            printf("    Failed to add skin deformer: %s\n", src_skin->name.data);
            continue;
        }
        
        // Store mapping
        (*skin_mappings)[i].src_skin = src_skin;
        (*skin_mappings)[i].export_skin = export_skin;
        
        // Copy skin clusters
        for (size_t j = 0; j < src_skin->clusters.count; j++) {
            ufbx_skin_cluster *src_cluster = src_skin->clusters.data[j];
            
            // Find corresponding bone node
            ufbx_node *export_bone = NULL;
            for (size_t k = 0; k < num_mappings; k++) {
                if (node_mappings[k].src_node == src_cluster->bone_node) {
                    export_bone = node_mappings[k].export_node;
                    break;
                }
            }
            
            if (export_bone) {
                ufbx_skin_cluster *export_cluster = ufbx_add_skin_cluster(export_scene, export_skin, export_bone, src_cluster->name.data);
                if (!export_cluster) {
                    continue;
                }
                
                // Copy cluster transform matrices - use correct matrix members
                ufbx_error transform_error = {0};
                bool success = ufbx_set_skin_cluster_transform(export_cluster, 
                                                             &src_cluster->geometry_to_bone,
                                                             &src_cluster->bind_to_world, &transform_error);
                if (!success) {
                    print_error(&transform_error, "Failed to set skin cluster transform");
                    continue;
                }
                
                // Copy cluster vertex indices and weights
                if (src_cluster->vertices.count > 0 && src_cluster->weights.count > 0) {
                    ufbx_error vertices_error = {0};
                    success = ufbx_set_skin_cluster_vertices(export_cluster,
                                                           src_cluster->vertices.data,
                                                           src_cluster->weights.data,
                                                           src_cluster->vertices.count, &vertices_error);
                    if (!success) {
                        print_error(&vertices_error, "Failed to set skin cluster vertices");
                    }
                }
            }
        }
        
        // Copy skin weights (global weights for the skin deformer)
        if (src_skin->weights.count > 0) {
            ufbx_error weights_error = {0};
            bool success = ufbx_set_skin_weights(export_skin, src_skin->weights.data, src_skin->weights.count, &weights_error);
            if (!success) {
                print_error(&weights_error, "Failed to set skin weights");
            }
        }
        
        // Attach skin to meshes
        for (size_t j = 0; j < source_scene->meshes.count; j++) {
            ufbx_mesh *src_mesh = source_scene->meshes.data[j];
            for (size_t k = 0; k < src_mesh->skin_deformers.count; k++) {
                if (src_mesh->skin_deformers.data[k] == src_skin) {
                    ufbx_mesh *export_mesh = NULL;
                    // Find corresponding export mesh
                    for (size_t m = 0; m < source_scene->meshes.count; m++) {
                        if (source_scene->meshes.data[m] == src_mesh) {
                            export_mesh = mesh_mappings[m].export_mesh;
                            break;
                        }
                    }
                    if (export_mesh) {
                        ufbx_error attach_error = {0};
                        bool success = ufbx_attach_skin_to_mesh(export_mesh, export_skin, &attach_error);
                        if (!success) {
                            print_error(&attach_error, "Failed to attach skin to mesh");
                        }
                    }
                    break;
                }
            }
        }
        
        printf("    Added skin deformer: %s\n", src_skin->name.data);
    }
    
    return true;
}

bool copy_blend_deformers(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                          mesh_mapping *mesh_mappings, blend_mapping **blend_mappings)
{
    printf("  Copying %zu blend deformers...\n", source_scene->blend_deformers.count);
    ufbx_error error;
    
    if (source_scene->blend_deformers.count == 0) {
        *blend_mappings = NULL;
        return true;
    }
    
    *blend_mappings = malloc(source_scene->blend_deformers.count * sizeof(blend_mapping));
    if (!*blend_mappings) {
        fprintf(stderr, "Failed to allocate blend mapping array\n");
        return false;
    }
    
    for (size_t i = 0; i < source_scene->blend_deformers.count; i++) {
        ufbx_blend_deformer *src_blend = source_scene->blend_deformers.data[i];
        
        ufbx_blend_deformer *export_blend = ufbx_add_blend_deformer(export_scene, src_blend->name.data);
        if (!export_blend) {
            printf("    Failed to add blend deformer: %s\n", src_blend->name.data);
            continue;
        }
        
        // Store mapping
        (*blend_mappings)[i].src_blend = src_blend;
        (*blend_mappings)[i].export_blend = export_blend;
        
        // Copy blend channels
        for (size_t j = 0; j < src_blend->channels.count; j++) {
            ufbx_blend_channel *src_channel = src_blend->channels.data[j];
            
            ufbx_blend_channel *export_channel = ufbx_add_blend_channel(export_scene, export_blend, src_channel->name.data);
            if (!export_channel) {
                continue;
            }
        }
        
        printf("    Added blend deformer: %s\n", src_blend->name.data);
    }
    
    return true;
}
