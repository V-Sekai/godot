#include "scene_copy_common.h"
#include <stdio.h>
#include <stdlib.h>

// Copy all nodes from source to export scene
bool copy_nodes(ufbx_scene *source_scene, ufbx_export_scene *export_scene, 
                node_mapping **node_mappings, size_t *num_mappings)
{
    printf("  Copying %zu nodes...\n", source_scene->nodes.count);
    ufbx_error error;
    
    *node_mappings = malloc(source_scene->nodes.count * sizeof(node_mapping));
    if (!*node_mappings) {
        fprintf(stderr, "Failed to allocate node mapping array\n");
        return false;
    }
    
    *num_mappings = 0;
    
    for (size_t i = 0; i < source_scene->nodes.count; i++) {
        ufbx_node *src_node = source_scene->nodes.data[i];
        ufbx_node *export_node;
        
        // Skip root node as it's automatically created
        if (!src_node->parent) {
            export_node = export_scene->root_node;
        } else {
            export_node = ufbx_add_node(export_scene, src_node->name.data, NULL);
            if (!export_node) {
                printf("    Failed to add node: %s\n", src_node->name.data);
                return false;
            }
        }
        
        // Copy transform
        ufbx_error transform_error = {0};
        ufbx_set_node_transform(export_node, &src_node->local_transform, &transform_error);
        if (transform_error.type != UFBX_ERROR_NONE) {
            print_error(&transform_error, "Failed to set node transform");
            return false;
        }
        
        // Store mapping
        (*node_mappings)[*num_mappings].src_node = src_node;
        (*node_mappings)[*num_mappings].export_node = export_node;
        (*num_mappings)++;
        
        printf("    Added node: %s\n", src_node->name.data);
    }
    
    return true;
}

// Set up parent-child relationships between nodes
bool setup_node_hierarchy(node_mapping *node_mappings, size_t num_mappings)
{
    printf("  Setting up node hierarchy...\n");
    
    for (size_t i = 0; i < num_mappings; i++) {
        ufbx_node *src_node = node_mappings[i].src_node;
        ufbx_node *export_node = node_mappings[i].export_node;
        
        if (src_node->parent) {
            // Find parent in mappings
            ufbx_node *export_parent = NULL;
            for (size_t j = 0; j < num_mappings; j++) {
                if (node_mappings[j].src_node == src_node->parent) {
                    export_parent = node_mappings[j].export_node;
                    break;
                }
            }
            
            if (export_parent) {
                // TODO: Implement proper parent setting when API is available
                // For now we track the relationship but can't set it directly
                printf("    Would set parent of '%s' to '%s'\n", 
                       export_node->name.data, export_parent->name.data);
            }
        }
    }
    
    return true;
}

bool attach_elements_to_nodes(ufbx_scene *source_scene, node_mapping *node_mappings, size_t num_mappings,
                              mesh_mapping *mesh_mappings)
{
    printf("  Attaching elements to nodes...\n");
    ufbx_error error;
    
    // Attach meshes to nodes
    for (size_t i = 0; i < source_scene->nodes.count; i++) {
        ufbx_node *src_node = source_scene->nodes.data[i];
        if (src_node->mesh) {
            // Find the corresponding export node
            ufbx_node *export_node = NULL;
            for (size_t j = 0; j < num_mappings; j++) {
                if (node_mappings[j].src_node == src_node) {
                    export_node = node_mappings[j].export_node;
                    break;
                }
            }
            
            if (export_node) {
                // Find the corresponding export mesh
                ufbx_mesh *export_mesh = NULL;
                for (size_t j = 0; j < source_scene->meshes.count; j++) {
                    if (source_scene->meshes.data[j] == src_node->mesh) {
                        export_mesh = mesh_mappings[j].export_mesh;
                        break;
                    }
                }
                
                if (export_mesh) {
                    ufbx_error attach_error = {0};
                    bool success = ufbx_attach_mesh_to_node(export_node, export_mesh, &attach_error);
                    if (!success) {
                        print_error(&attach_error, "Failed to attach mesh to node");
                        return false;
                    }
                    printf("    Attached mesh '%s' to node '%s'\n", 
                           export_mesh->name.data, export_node->name.data);
                }
            }
        }
    }
    
    return true;
}
