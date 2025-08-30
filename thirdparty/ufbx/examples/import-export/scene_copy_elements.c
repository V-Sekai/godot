#include "scene_copy_common.h"
#include <stdio.h>
#include <stdlib.h>

// Copy lights and cameras with all their properties
bool copy_lights_and_cameras(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                             node_mapping *node_mappings, size_t num_mappings)
{
    printf("  Copying lights and cameras...\n");
    ufbx_error error;
    
    // Copy lights
    printf("    Copying %zu lights...\n", source_scene->lights.count);
    for (size_t i = 0; i < source_scene->lights.count; i++) {
        ufbx_light *src_light = source_scene->lights.data[i];
        
        ufbx_light *export_light = ufbx_add_light(export_scene, src_light->name.data);
        if (!export_light) {
            printf("      Failed to add light: %s\n", src_light->name.data);
            continue;
        }
        
        // Copy light properties
        ufbx_error light_error = {0};
        bool light_success = ufbx_set_light_properties(export_light,
                                                       src_light->type,
                                                       src_light->color,
                                                       src_light->intensity, &light_error);
        if (!light_success) {
            print_error(&light_error, "Failed to set light properties");
            continue;
        }
        
        // Copy advanced light settings
        ufbx_error advanced_error = {0};
        bool advanced_success = ufbx_set_light_advanced_properties(export_light,
                                                                  src_light->inner_angle,
                                                                  src_light->outer_angle,
                                                                  src_light->cast_shadows, &advanced_error);
        if (!advanced_success) {
            print_error(&advanced_error, "Failed to set advanced light properties");
        }
        
        // Attach light to corresponding node
        for (size_t j = 0; j < num_mappings; j++) {
            if (node_mappings[j].src_node->light == src_light) {
                ufbx_error attach_error = {0};
                bool attach_success = ufbx_attach_light_to_node(node_mappings[j].export_node, export_light, &attach_error);
                if (!attach_success) {
                    print_error(&attach_error, "Failed to attach light to node");
                } else {
                    printf("      Attached light '%s' to node '%s'\n", 
                           src_light->name.data, node_mappings[j].export_node->name.data);
                }
                break;
            }
        }
        
        printf("      Added light: %s\n", src_light->name.data);
    }
    
    // Copy cameras
    printf("    Copying %zu cameras...\n", source_scene->cameras.count);
    for (size_t i = 0; i < source_scene->cameras.count; i++) {
        ufbx_camera *src_camera = source_scene->cameras.data[i];
        
        ufbx_camera *export_camera = ufbx_add_camera(export_scene, src_camera->name.data);
        if (!export_camera) {
            printf("      Failed to add camera: %s\n", src_camera->name.data);
            continue;
        }
        
        // Copy camera properties
        ufbx_error camera_error = {0};
        bool camera_success = ufbx_set_camera_properties(export_camera,
                                                         src_camera->projection_mode,
                                                         src_camera->field_of_view_x,
                                                         src_camera->field_of_view_y, &camera_error);
        if (!camera_success) {
            print_error(&camera_error, "Failed to set camera properties");
            continue;
        }
        
        // Copy clipping planes and other settings
        ufbx_error clipping_error = {0};
        bool clipping_success = ufbx_set_camera_clipping(export_camera,
                                                        src_camera->near_plane,
                                                        src_camera->far_plane, &clipping_error);
        if (!clipping_success) {
            print_error(&clipping_error, "Failed to set camera clipping planes");
        }
        
        // Attach camera to corresponding node
        for (size_t j = 0; j < num_mappings; j++) {
            if (node_mappings[j].src_node->camera == src_camera) {
                ufbx_error attach_error = {0};
                bool attach_success = ufbx_attach_camera_to_node(node_mappings[j].export_node, export_camera, &attach_error);
                if (!attach_success) {
                    print_error(&attach_error, "Failed to attach camera to node");
                } else {
                    printf("      Attached camera '%s' to node '%s'\n", 
                           src_camera->name.data, node_mappings[j].export_node->name.data);
                }
                break;
            }
        }
        
        printf("      Added camera: %s\n", src_camera->name.data);
    }
    
    return true;
}

// Copy constraints between nodes
bool copy_constraints(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                      node_mapping *node_mappings, size_t num_mappings)
{
    printf("  Copying %zu constraints...\n", source_scene->constraints.count);
    ufbx_error error;
    
    for (size_t i = 0; i < source_scene->constraints.count; i++) {
        ufbx_constraint *src_constraint = source_scene->constraints.data[i];
        
        // Find constrained node and target nodes
        ufbx_node *export_constrained = NULL;
        for (size_t j = 0; j < num_mappings; j++) {
            if (node_mappings[j].src_node == src_constraint->constrained_object) {
                export_constrained = node_mappings[j].export_node;
                break;
            }
        }
        
        if (!export_constrained) {
            printf("    Could not find constrained node for constraint: %s\n", src_constraint->name.data);
            continue;
        }
        
        ufbx_constraint *export_constraint = ufbx_add_constraint(export_scene, src_constraint->name.data, 
                                                               export_constrained, src_constraint->type);
        if (!export_constraint) {
            printf("    Failed to add constraint: %s\n", src_constraint->name.data);
            continue;
        }
        
        // Copy constraint targets
        for (size_t j = 0; j < src_constraint->targets.count; j++) {
            ufbx_constraint_target *src_target = &src_constraint->targets.data[j];
            
            // Find export target node
            ufbx_node *export_target = NULL;
            for (size_t k = 0; k < num_mappings; k++) {
                if (node_mappings[k].src_node == src_target->node) {
                    export_target = node_mappings[k].export_node;
                    break;
                }
            }
            
            if (export_target) {
                ufbx_error target_error = {0};
                bool target_success = ufbx_add_constraint_target(export_constraint, export_target, src_target->weight, &target_error);
                if (!target_success) {
                    print_error(&target_error, "Failed to add constraint target");
                } else {
                    printf("      Added constraint target: %s -> %s (weight: %.2f)\n",
                           export_constrained->name.data, export_target->name.data, src_target->weight);
                }
            }
        }
        
        printf("    Added constraint: %s\n", src_constraint->name.data);
    }
    
    return true;
}

// Copy user-defined properties and custom attributes
bool copy_user_properties(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                          node_mapping *node_mappings, size_t num_mappings)
{
    printf("  Copying user properties...\n");
    
    // Copy custom properties from nodes
    for (size_t i = 0; i < num_mappings; i++) {
        ufbx_node *src_node = node_mappings[i].src_node;
        ufbx_node *export_node = node_mappings[i].export_node;
        
        // Copy user-defined properties
        for (size_t j = 0; j < src_node->all_props.props.count; j++) {
            ufbx_prop *src_prop = &src_node->all_props.props.data[j];
            
            // Skip built-in properties, only copy custom ones
            if (src_prop->name.length > 0 && src_prop->name.data[0] != '_' && 
                strncmp(src_prop->name.data, "Lcl", 3) != 0) {
                
                ufbx_error prop_error = {0};
                bool prop_success = ufbx_set_node_user_property(export_node, 
                                                               src_prop->name.data,
                                                               &src_prop->value, &prop_error);
                if (!prop_success) {
                    print_error(&prop_error, "Failed to set user property");
                } else {
                    printf("    Copied user property '%s' to node '%s'\n", 
                           src_prop->name.data, export_node->name.data);
                }
            }
        }
    }
    
    // Copy scene-level metadata and settings
    if (source_scene->metadata.creator.length > 0) {
        ufbx_error creator_error = {0};
        bool creator_success = ufbx_set_scene_metadata(export_scene, "Creator", 
                                                      source_scene->metadata.creator.data, &creator_error);
        if (!creator_success) {
            print_error(&creator_error, "Failed to set scene creator metadata");
        } else {
            printf("    Copied scene creator: %s\n", source_scene->metadata.creator.data);
        }
    }
    
    return true;
}
