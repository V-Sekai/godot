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
        
        // Skip root node as it's automatically created
        if (!src_node->parent) {
            // Map root node
            (*node_mappings)[*num_mappings].src_node = src_node;
            (*node_mappings)[*num_mappings].export_node = export_scene->root_node;
            (*num_mappings)++;
            printf("    Mapped root node: %s\n", src_node->name.data ? src_node->name.data : "(unnamed root)");
            continue;
        }
        
        // Create all nodes first - parent relationships handled later
        ufbx_node *export_node = ufbx_add_node(export_scene, src_node->name, NULL);
        if (!export_node) {
            printf("    Failed to add node: %s\n", src_node->name.data);
            return false;
        }
        
        // Copy complete transform data
        ufbx_set_node_transform(export_node, &src_node->local_transform, &error);
        if (error.type != UFBX_ERROR_NONE) {
            print_error(&error, "Failed to set node transform");
            return false;
        }
        
        // Copy available transform properties
        
        // Copy visibility and basic properties
        export_node->visible = src_node->visible;
        export_node->rotation_order = src_node->rotation_order;
        export_node->inherit_mode = src_node->inherit_mode;
        
        // Copy geometry transform if present
        if (src_node->has_geometry_transform) {
            export_node->geometry_transform = src_node->geometry_transform;
            export_node->has_geometry_transform = true;
        }
        
        // Copy available transform helper matrices
        export_node->node_to_parent = src_node->node_to_parent;
        export_node->node_to_world = src_node->node_to_world;
        export_node->euler_rotation = src_node->euler_rotation;
        
        printf("    Enhanced transform copying for node: %s (inheritance, matrices)\n", src_node->name.data);
        
        // Copy bone if present
        if (src_node->bone) {
            ufbx_bone *export_bone = ufbx_add_bone(export_scene, export_node, src_node->bone->element.name);
            if (export_bone) {
                ufbx_error bone_error = {0};
                bool bone_success = ufbx_set_bone_properties(export_bone, src_node->bone->relative_length, &bone_error);
                if (!bone_success) {
                    print_error(&bone_error, "Failed to set bone properties");
                }
                printf("    Added bone to node: %s\n", src_node->name.data);
            }
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
                // Set parent pointer directly (following ufbx.c pattern)
                export_node->parent = export_parent;
                
                // Add to parent's children list
                size_t new_count = export_parent->children.count + 1;
                ufbx_node **new_children = (ufbx_node**)realloc(export_parent->children.data, new_count * sizeof(ufbx_node*));
                if (new_children) {
                    new_children[export_parent->children.count] = export_node;
                    export_parent->children.data = new_children;
                    export_parent->children.count = new_count;
                    printf("    Set parent of '%s' to '%s'\n", 
                           export_node->name.data, export_parent->name.data);
                } else {
                    printf("    Failed to allocate children array for parent '%s'\n", export_parent->name.data);
                }
            }
        }
    }
    
    return true;
}

bool attach_elements_to_nodes(ufbx_scene *source_scene, node_mapping *node_mappings, size_t num_mappings,
                              mesh_mapping *mesh_mappings)
{
    printf("  Attaching elements to nodes...\n");
    
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
