#include "scene_utils.h"
#include <stdio.h>
#include <stdlib.h>

int test_roundtrip(const char *input_file, const char *output_file)
{
    printf("=== FBX Roundtrip Test ===\n");
    printf("Input file: %s\n", input_file);
    printf("Output file: %s\n", output_file);
    printf("\n");

    // Load the FBX file
    printf("Loading FBX file...\n");
    ufbx_load_opts load_opts = {
        .load_external_files = true,
        .ignore_missing_external_files = true,
        .generate_missing_normals = true,
        .target_axes = {
            .right = UFBX_COORDINATE_AXIS_POSITIVE_X,
            .up = UFBX_COORDINATE_AXIS_POSITIVE_Y,
            .front = UFBX_COORDINATE_AXIS_POSITIVE_Z,
        },
        .target_unit_meters = 1.0f,
    };

    ufbx_error error;
    ufbx_scene *scene = ufbx_load_file(input_file, &load_opts, &error);
    
    if (!scene) {
        print_error(&error, "Failed to load FBX file");
        return 1;
    }
    
    printf("Successfully loaded FBX file!\n");
    print_warnings(scene);
    print_scene_info(scene);
    printf("\n");

    // Create an export scene
    printf("Creating export scene...\n");
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *export_scene = ufbx_create_scene(&create_opts);
    if (!export_scene) {
        fprintf(stderr, "Failed to create export scene\n");
        ufbx_free_scene(scene);
        return 1;
    }
    printf("Export scene created successfully!\n");
    
    // Copy scene data from imported scene to export scene
    if (!copy_scene_data(scene, export_scene)) {
        fprintf(stderr, "Failed to copy scene data\n");
        ufbx_free_export_scene(export_scene);
        ufbx_free_scene(scene);
        return 1;
    }

    // Export the scene back to FBX
    printf("Exporting FBX file...\n");
    ufbx_export_opts export_opts = {
        .ascii_format = true,  // Use ASCII for easier debugging
        .fbx_version = 7400,
    };

    bool export_success = ufbx_export_to_file(export_scene, output_file, &export_opts, &error);
    if (!export_success) {
        print_error(&error, "Failed to export FBX file");
        ufbx_free_export_scene(export_scene);
        ufbx_free_scene(scene);
        return 1;
    }

    printf("Successfully exported FBX file!\n");
    printf("\n");

    // Clean up
    ufbx_free_export_scene(export_scene);
    ufbx_free_scene(scene);

    printf("=== Roundtrip Test Complete ===\n");
    printf("Input:  %s\n", input_file);
    printf("Output: %s\n", output_file);
    
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.fbx> <output.fbx>\n", argv[0]);
        fprintf(stderr, "Example: %s ./data/huesitos.fbx huesitos_roundtrip.fbx\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    return test_roundtrip(input_file, output_file);
}
