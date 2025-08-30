#include "scene_copy_common.h"
#include <stdio.h>
#include <stdlib.h>

// Copy animation data with proper connections
bool copy_animations(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                     node_mapping *node_mappings, size_t num_mappings,
                     anim_stack_mapping **stack_mappings, anim_layer_mapping **layer_mappings)
{
    printf("  Copying %zu animation stacks...\n", source_scene->anim_stacks.count);
    ufbx_error error;
    
    if (source_scene->anim_stacks.count == 0) {
        *stack_mappings = NULL;
        *layer_mappings = NULL;
        return true;
    }
    
    *stack_mappings = malloc(source_scene->anim_stacks.count * sizeof(anim_stack_mapping));
    if (!*stack_mappings) {
        fprintf(stderr, "Failed to allocate animation stack mapping array\n");
        return false;
    }
    
    // Count total layers for allocation
    size_t total_layers = 0;
    for (size_t i = 0; i < source_scene->anim_stacks.count; i++) {
        total_layers += source_scene->anim_stacks.data[i]->layers.count;
    }
    
    *layer_mappings = malloc(total_layers * sizeof(anim_layer_mapping));
    if (!*layer_mappings) {
        fprintf(stderr, "Failed to allocate animation layer mapping array\n");
        return false;
    }
    
    size_t layer_mapping_index = 0;
    
    for (size_t i = 0; i < source_scene->anim_stacks.count; i++) {
        ufbx_anim_stack *src_stack = source_scene->anim_stacks.data[i];
        
        // Create animation stack
        ufbx_anim_stack *export_stack = ufbx_add_animation(export_scene, src_stack->name.data);
        if (!export_stack) {
            printf("    Failed to add animation stack: %s\n", src_stack->name.data);
            return false;
        }
        
        // Store stack mapping
        (*stack_mappings)[i].src_stack = src_stack;
        (*stack_mappings)[i].export_stack = export_stack;
        
        // Set time range
        ufbx_error time_range_error = {0};
        bool success = ufbx_set_anim_stack_time_range(export_stack, src_stack->time_begin, src_stack->time_end, &time_range_error);
        if (!success) {
            print_error(&time_range_error, "Failed to set animation time range");
            return false;
        }
        
        // Copy animation layers
        for (size_t j = 0; j < src_stack->layers.count; j++) {
            ufbx_anim_layer *src_layer = src_stack->layers.data[j];
            
            ufbx_anim_layer *export_layer = ufbx_add_anim_layer(export_scene, export_stack, src_layer->name.data);
            if (!export_layer) {
                printf("    Failed to add animation layer: %s\n", src_layer->name.data);
                continue;
            }
            
            // Store layer mapping
            (*layer_mappings)[layer_mapping_index].src_layer = src_layer;
            (*layer_mappings)[layer_mapping_index].export_layer = export_layer;
            layer_mapping_index++;
            
            // Copy animation values and connect them to properties
            for (size_t k = 0; k < src_layer->anim_values.count; k++) {
                ufbx_anim_value *src_value = src_layer->anim_values.data[k];
                
                ufbx_anim_value *export_value = ufbx_add_anim_value(export_scene, export_layer, src_value->name.data);
                if (!export_value) {
                    continue;
                }
                
                // Find the target node for this animation value
                // We need to look at the connections from the animation layer to find what this animates
                ufbx_node *target_node = NULL;
                const char *target_property = NULL;
                
                // Look through animation properties to find what this value animates
                for (size_t prop_idx = 0; prop_idx < src_layer->anim_props.count; prop_idx++) {
                    ufbx_anim_prop *anim_prop = &src_layer->anim_props.data[prop_idx];
                    if (anim_prop->anim_value == src_value) {
                        // Found the target element and property
                        if (anim_prop->element->type == UFBX_ELEMENT_NODE) {
                            // Find the corresponding export node
                            ufbx_node *src_target = (ufbx_node*)anim_prop->element;
                            for (size_t node_idx = 0; node_idx < num_mappings; node_idx++) {
                                if (node_mappings[node_idx].src_node == src_target) {
                                    target_node = node_mappings[node_idx].export_node;
                                    target_property = anim_prop->prop_name.data;
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
                
                // Connect the animation value to the target property
                if (target_node && target_property) {
                    ufbx_error connect_error = {0};
                    success = ufbx_connect_anim_prop(export_scene, export_layer, 
                                                   (ufbx_element*)target_node, target_property, export_value, &connect_error);
                    if (!success) {
                        print_error(&connect_error, "Failed to connect animation property");
                        // Continue with other properties
                    }
                }
                
                // Copy curves for each component
                for (int comp = 0; comp < 3; comp++) { // X, Y, Z components
                    ufbx_anim_curve *src_curve = src_value->curves[comp];
                    if (src_curve && src_curve->keyframes.count > 0) {
                        ufbx_anim_curve *export_curve = ufbx_add_anim_curve(export_scene, export_value, comp);
                        if (export_curve) {
                            // Set extrapolation modes
                            ufbx_error extrap_error = {0};
                            success = ufbx_set_anim_curve_extrapolation(export_curve, 
                                                                       src_curve->pre_extrapolation,
                                                                       src_curve->post_extrapolation, &extrap_error);
                            if (!success) {
                                print_error(&extrap_error, "Failed to set animation extrapolation");
                            }
                            
                            // Copy keyframes
                            ufbx_error keyframes_error = {0};
                            success = ufbx_set_anim_curve_keyframes(export_curve, 
                                                                   src_curve->keyframes.data,
                                                                   src_curve->keyframes.count, &keyframes_error);
                            if (!success) {
                                print_error(&keyframes_error, "Failed to set animation keyframes");
                            }
                        }
                    }
                }
            }
        }
        
        printf("    Added animation stack: %s\n", src_stack->name.data);
    }
    
    return true;
}
