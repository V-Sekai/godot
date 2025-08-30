#include "scene_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Generate synthetic FBX with animation but NO connections to isolate the issue
bool generate_animation_no_connections(const char *output_file)
{
    printf("Generating animation FBX (no connections): %s\n", output_file);
    
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *scene = ufbx_create_scene(&create_opts);
    if (!scene) {
        fprintf(stderr, "Failed to create export scene\n");
        return false;
    }
    
    // Create animated node
    ufbx_node *anim_node = ufbx_add_node(scene, "AnimatedNode", NULL);
    if (!anim_node) {
        fprintf(stderr, "Failed to add animated node\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Create animation stack
    ufbx_anim_stack *stack = ufbx_add_animation(scene, "TestAnimation");
    if (!stack) {
        fprintf(stderr, "Failed to add animation stack\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Set time range
    ufbx_error error = { 0 };
    bool success = ufbx_set_anim_stack_time_range(stack, 0.0, 1.0, &error);
    if (!success) {
        print_error(&error, "Failed to set animation time range");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Create animation layer
    ufbx_anim_layer *layer = ufbx_add_anim_layer(scene, stack, "BaseLayer");
    if (!layer) {
        fprintf(stderr, "Failed to add animation layer\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Create animation value for translation
    ufbx_anim_value *anim_value = ufbx_add_anim_value(scene, layer, "T");
    if (!anim_value) {
        fprintf(stderr, "Failed to add animation value\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Create animation curve for X translation
    ufbx_anim_curve *curve = ufbx_add_anim_curve(scene, anim_value, 0);
    if (!curve) {
        fprintf(stderr, "Failed to add animation curve\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Add simple keyframes
    ufbx_keyframe keyframes[2] = {
        { .time = 0.0, .value = 0.0, .interpolation = UFBX_INTERPOLATION_LINEAR },
        { .time = 1.0, .value = 5.0, .interpolation = UFBX_INTERPOLATION_LINEAR }
    };
    
    success = ufbx_set_anim_curve_keyframes(curve, keyframes, 2, &error);
    if (!success) {
        print_error(&error, "Failed to set animation keyframes");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // SKIP animation connections to test if that's the issue
    printf("    Skipping animation connections to isolate issue...\n");
    
    // Export
    ufbx_export_opts export_opts = {
        .ascii_format = true,
        .fbx_version = 7400,
    };
    
    success = ufbx_export_to_file(scene, output_file, &export_opts, &error);
    if (!success) {
        print_error(&error, "Failed to export animation FBX");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    ufbx_free_export_scene(scene);
    printf("✓ Generated animation FBX (no connections) successfully\n");
    return true;
}

int main()
{
    return generate_animation_no_connections("data/synthetic/animation_no_connections.fbx") ? 0 : 1;
}
