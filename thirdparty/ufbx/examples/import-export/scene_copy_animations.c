#include "scene_copy_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Copy animation data with proper connections
bool copy_animations(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                     node_mapping *node_mappings, size_t num_mappings,
                     anim_stack_mapping **stack_mappings, anim_layer_mapping **layer_mappings)
{
    printf("  Copying %zu animation stacks...\n", source_scene->anim_stacks.count);
    
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
        
        // Copy animation layers with proper ufbx patterns
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
            
            // Copy layer properties for comprehensive preservation (THIS IS CRITICAL!)
            export_layer->weight = src_layer->weight;
            export_layer->weight_is_animated = src_layer->weight_is_animated;
            export_layer->blended = src_layer->blended;
            export_layer->additive = src_layer->additive;
            export_layer->compose_rotation = src_layer->compose_rotation;
            export_layer->compose_scale = src_layer->compose_scale;
            
            printf("    Enhanced layer copying: %s (weight=%.3f, blended=%d, additive=%d)\n", 
                   src_layer->name.data, src_layer->weight, src_layer->blended, src_layer->additive);
            
            // Copy animation values following ufbx.c connection patterns
            for (size_t k = 0; k < src_layer->anim_values.count; k++) {
                ufbx_anim_value *src_value = src_layer->anim_values.data[k];
                
                ufbx_anim_value *export_value = ufbx_add_anim_value(export_scene, export_layer, src_value->name.data);
                if (!export_value) {
                    continue;
                }
                
                // Copy default value
                export_value->default_value = src_value->default_value;
                
                // Copy curves with proper component handling (following ufbx.c patterns)
                for (size_t comp = 0; comp < 3; comp++) { // Only X,Y,Z components to match ufbx.c
                    ufbx_anim_curve *src_curve = src_value->curves[comp];
                    if (src_curve && src_curve->keyframes.count > 0) {
                        ufbx_anim_curve *export_curve = ufbx_add_anim_curve(export_scene, export_value, (int)comp);
                        if (export_curve) {
                            // Copy keyframes exactly as they are (no enhancements that might break format)
                            ufbx_error curve_error = {0};
                            bool curve_success = ufbx_set_anim_curve_keyframes(export_curve, 
                                                                             src_curve->keyframes.data,
                                                                             src_curve->keyframes.count, &curve_error);
                            if (!curve_success) {
                                print_error(&curve_error, "Failed to set animation keyframes");
                                continue;
                            }
                            
                            // Copy ALL curve metadata for comprehensive preservation
                            export_curve->pre_extrapolation = src_curve->pre_extrapolation;
                            export_curve->post_extrapolation = src_curve->post_extrapolation;
                            export_curve->min_value = src_curve->min_value;
                            export_curve->max_value = src_curve->max_value;
                            export_curve->min_time = src_curve->min_time;
                            export_curve->max_time = src_curve->max_time;
                            
                            printf("      Enhanced curve component %zu with %zu keyframes (extrapolation: %d->%d)\n", 
                                   comp, src_curve->keyframes.count, 
                                   src_curve->pre_extrapolation.mode, src_curve->post_extrapolation.mode);
                        }
                    }
                }
                
                // Connect animation to the CORRECT element by finding the original source
                // Don't connect to all nodes - find the specific node this animation belongs to
                ufbx_node *target_node = NULL;
                
                // Find which node in the original scene this animation value belongs to
                for (size_t anim_prop_idx = 0; anim_prop_idx < src_layer->anim_props.count; anim_prop_idx++) {
                    ufbx_anim_prop *src_prop = &src_layer->anim_props.data[anim_prop_idx];
                    if (src_prop->anim_value == src_value) {
                        // Found the original connection - find the corresponding export node
                        for (size_t node_idx = 0; node_idx < num_mappings; node_idx++) {
                            if (node_mappings[node_idx].src_node == (ufbx_node*)src_prop->element) {
                                target_node = node_mappings[node_idx].export_node;
                                break;
                            }
                        }
                        break;
                    }
                }
                
                if (target_node) {
                    // Connect based on animation value name patterns from ufbx.c
                    const char *prop_name = NULL;
                    if (strcmp(src_value->name.data, "T") == 0) {
                        prop_name = "Lcl Translation";
                    } else if (strcmp(src_value->name.data, "R") == 0) {
                        prop_name = "Lcl Rotation";
                    } else if (strcmp(src_value->name.data, "S") == 0) {
                        prop_name = "Lcl Scaling";
                    }
                    
                    if (prop_name) {
                        ufbx_error connect_error = {0};
                        bool connect_success = ufbx_connect_anim_prop(export_scene, export_layer, 
                                                                    (ufbx_element*)target_node, 
                                                                    prop_name, export_value, &connect_error);
                        if (!connect_success) {
                            printf("      Warning: Failed to connect animation '%s' to node '%s' property '%s'\n", 
                                   src_value->name.data, target_node->element.name.data, prop_name);
                        } else {
                            printf("      Connected animation '%s' to node '%s' property '%s'\n", 
                                   src_value->name.data, target_node->element.name.data, prop_name);
                        }
                    }
                } else {
                    printf("      Warning: Could not find target node for animation value '%s'\n", src_value->name.data);
                }
                
                printf("      Added animation value: %s\n", src_value->name.data);
            }
        }
        
        printf("    Added animation stack: %s\n", src_stack->name.data);
    }
    
    return true;
}
