#include "scene_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test just animation stack (no layers, values, curves)
bool test_just_stack(const char *output_file)
{
    printf("Testing just animation stack: %s\n", output_file);
    
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *scene = ufbx_create_scene(&create_opts);
    if (!scene) {
        fprintf(stderr, "Failed to create export scene\n");
        return false;
    }
    
    // Create node
    ufbx_node *node = ufbx_add_node(scene, "TestNode", NULL);
    if (!node) {
        fprintf(stderr, "Failed to add node\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Create ONLY animation stack (no layers, values, curves)
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
    
    // Export immediately (no layers, values, curves)
    ufbx_export_opts export_opts = {
        .ascii_format = true,
        .fbx_version = 7400,
    };
    
    success = ufbx_export_to_file(scene, output_file, &export_opts, &error);
    if (!success) {
        print_error(&error, "Failed to export stack-only FBX");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    ufbx_free_export_scene(scene);
    printf("✓ Generated stack-only FBX successfully\n");
    return true;
}

int main()
{
    return test_just_stack("data/synthetic/just_stack.fbx") ? 0 : 1;
}
