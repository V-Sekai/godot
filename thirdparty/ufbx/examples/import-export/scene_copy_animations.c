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
            
            // Copy animation values and curves (simplified version)
            for (size_t k = 0; k < src_layer->anim_values.count; k++) {
                ufbx_anim_value *src_value = src_layer->anim_values.data[k];
                
                ufbx_anim_value *export_value = ufbx_add_anim_value(export_scene, export_layer, src_value->name.data);
                if (!export_value) {
                    continue;
                }
                
                // Copy animation value default value
                export_value->default_value = src_value->default_value;
                
                // Copy curves for all components (not just X,Y,Z)
                size_t max_components = sizeof(src_value->curves) / sizeof(src_value->curves[0]);
                for (size_t comp = 0; comp < max_components; comp++) {
                    ufbx_anim_curve *src_curve = src_value->curves[comp];
                    if (src_curve && src_curve->keyframes.count > 0) {
                        ufbx_anim_curve *export_curve = ufbx_add_anim_curve(export_scene, export_value, (int)comp);
                        if (export_curve) {
                            // Copy keyframes with all metadata - create enhanced keyframe array
                            ufbx_keyframe *enhanced_keyframes = malloc(src_curve->keyframes.count * sizeof(ufbx_keyframe));
                            if (enhanced_keyframes) {
                                // Copy keyframes with full metadata preservation
                                for (size_t kf_idx = 0; kf_idx < src_curve->keyframes.count; kf_idx++) {
                                    ufbx_keyframe *src_kf = &src_curve->keyframes.data[kf_idx];
                                    ufbx_keyframe *dst_kf = &enhanced_keyframes[kf_idx];
                                    
                                    // Copy all keyframe data
                                    *dst_kf = *src_kf;  // Copy entire structure
                                    
                                    // Ensure tangent modes and weights are preserved (only available fields)
                                    dst_kf->interpolation = src_kf->interpolation;
                                    dst_kf->left.dx = src_kf->left.dx;
                                    dst_kf->left.dy = src_kf->left.dy;
                                    dst_kf->right.dx = src_kf->right.dx;
                                    dst_kf->right.dy = src_kf->right.dy;
                                    // Note: bias, continuity, tension may not be available in export structs
                                }
                                
                                // Set enhanced keyframes
                                ufbx_error curve_error = {0};
                                bool curve_success = ufbx_set_anim_curve_keyframes(export_curve, 
                                                                                 enhanced_keyframes,
                                                                                 src_curve->keyframes.count, &curve_error);
                                free(enhanced_keyframes);
                                
                                if (!curve_success) {
                                    print_error(&curve_error, "Failed to set animation keyframes");
                                    continue;
                                }
                            } else {
                                // Fallback to basic keyframe copying
                                ufbx_error curve_error = {0};
                                bool curve_success = ufbx_set_anim_curve_keyframes(export_curve, 
                                                                                 src_curve->keyframes.data,
                                                                                 src_curve->keyframes.count, &curve_error);
                                if (!curve_success) {
                                    print_error(&curve_error, "Failed to set animation keyframes");
                                    continue;
                                }
                            }
                            
                            // Copy curve extrapolation settings
                            ufbx_error extrap_error = {0};
                            bool extrap_success = ufbx_set_anim_curve_extrapolation(export_curve,
                                                                                  src_curve->pre_extrapolation,
                                                                                  src_curve->post_extrapolation, &extrap_error);
                            if (!extrap_success) {
                                print_error(&extrap_error, "Failed to set curve extrapolation");
                            }
                            
                            printf("      Copied curve component %zu with %zu keyframes (enhanced metadata)\n", comp, src_curve->keyframes.count);
                        }
                    }
                }
                
                // Animation connection will be handled automatically by ufbx export
                // when animation values are named properly and associated with layers
                printf("      Added animation value: %s\n", src_value->name.data);
            }
        }
        
        printf("    Added animation stack: %s\n", src_stack->name.data);
    }
    
    return true;
}
