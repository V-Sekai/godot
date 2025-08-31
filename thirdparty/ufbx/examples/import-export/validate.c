#include "../../ufbx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void print_error(const ufbx_error *error, const char *description)
{
    char buffer[1024];
    ufbx_format_error(buffer, sizeof(buffer), error);
    fprintf(stderr, "Error: %s\n%s\n", description, buffer);
}

typedef struct {
    size_t nodes;
    size_t meshes;
    size_t materials;
    size_t animations;
    size_t bones;
    size_t skin_deformers;
    size_t blend_deformers;
    size_t total_vertices;
    size_t total_faces;
} scene_metrics;

void get_scene_metrics(ufbx_scene *scene, scene_metrics *metrics)
{
    memset(metrics, 0, sizeof(scene_metrics));
    
    metrics->nodes = scene->nodes.count;
    metrics->meshes = scene->meshes.count;
    metrics->materials = scene->materials.count;
    metrics->animations = scene->anim_stacks.count;
    metrics->bones = scene->bones.count;
    metrics->skin_deformers = scene->skin_deformers.count;
    metrics->blend_deformers = scene->blend_deformers.count;
    
    // Count total vertices and faces
    for (size_t i = 0; i < scene->meshes.count; i++) {
        ufbx_mesh *mesh = scene->meshes.data[i];
        metrics->total_vertices += mesh->num_vertices;
        metrics->total_faces += mesh->num_faces;
    }
}

void print_metrics_comparison(const scene_metrics *input, const scene_metrics *output)
{
    printf("=== Roundtrip Validation ===\n");
    printf("Data preservation check:\n");
    
    bool perfect_match = true;
    
    printf("  Nodes:          %zu -> %zu", input->nodes, output->nodes);
    if (input->nodes == output->nodes) {
        printf(" ✓\n");
    } else {
        printf(" ✗ MISMATCH\n");
        perfect_match = false;
    }
    
    printf("  Meshes:         %zu -> %zu", input->meshes, output->meshes);
    if (input->meshes == output->meshes) {
        printf(" ✓\n");
    } else {
        printf(" ✗ MISMATCH\n");
        perfect_match = false;
    }
    
    printf("  Materials:      %zu -> %zu", input->materials, output->materials);
    if (input->materials == output->materials) {
        printf(" ✓\n");
    } else {
        printf(" ✗ MISMATCH\n");
        perfect_match = false;
    }
    
    printf("  Animations:     %zu -> %zu", input->animations, output->animations);
    if (input->animations == output->animations) {
        printf(" ✓\n");
    } else {
        printf(" ✗ MISMATCH\n");
        perfect_match = false;
    }
    
    printf("  Bones:          %zu -> %zu", input->bones, output->bones);
    if (input->bones == output->bones) {
        printf(" ✓\n");
    } else {
        printf(" ✗ MISMATCH\n");
        perfect_match = false;
    }
    
    printf("  Skin deformers: %zu -> %zu", input->skin_deformers, output->skin_deformers);
    if (input->skin_deformers == output->skin_deformers) {
        printf(" ✓\n");
    } else {
        printf(" ✗ MISMATCH\n");
        perfect_match = false;
    }
    
    printf("  Blend deformers:%zu -> %zu", input->blend_deformers, output->blend_deformers);
    if (input->blend_deformers == output->blend_deformers) {
        printf(" ✓\n");
    } else {
        printf(" ✗ MISMATCH\n");
        perfect_match = false;
    }
    
    printf("  Total vertices: %zu -> %zu", input->total_vertices, output->total_vertices);
    if (input->total_vertices == output->total_vertices) {
        printf(" ✓\n");
    } else {
        printf(" ✗ MISMATCH\n");
        perfect_match = false;
    }
    
    printf("  Total faces:    %zu -> %zu", input->total_faces, output->total_faces);
    if (input->total_faces == output->total_faces) {
        printf(" ✓\n");
    } else {
        printf(" ✗ MISMATCH\n");
        perfect_match = false;
    }
    
    printf("\n");
    if (perfect_match) {
        printf("🎉 ROUNDTRIP VALIDATION: PERFECT MATCH\n");
        printf("All scene data was preserved correctly!\n");
    } else {
        printf("⚠️  ROUNDTRIP VALIDATION: DATA LOSS DETECTED\n");
        printf("Some scene data was not preserved during roundtrip.\n");
    }
}

int validate_roundtrip(const char *input_file, const char *output_file)
{
    printf("=== FBX Roundtrip Validation ===\n");
    printf("Input file:  %s\n", input_file);
    printf("Output file: %s\n", output_file);
    printf("\n");

    // Load both files with verbose error reporting settings
    ufbx_load_opts load_opts = {
        .load_external_files = true,
        .ignore_missing_external_files = true,
        .generate_missing_normals = true,
        .strict = true,  // Enable strict mode for detailed error reporting
        .disable_quirks = true,  // Disable exporter-specific quirks
        .retain_dom = true,  // Keep raw document structure for debugging
        .index_error_handling = UFBX_INDEX_ERROR_HANDLING_ABORT_LOADING,  // Fail on bad indices
        .unicode_error_handling = UFBX_UNICODE_ERROR_HANDLING_ABORT_LOADING,  // Fail on Unicode errors
        .target_axes = {
            .right = UFBX_COORDINATE_AXIS_POSITIVE_X,
            .up = UFBX_COORDINATE_AXIS_POSITIVE_Y,
            .front = UFBX_COORDINATE_AXIS_POSITIVE_Z,
        },
        .target_unit_meters = 1.0f,
    };

    ufbx_error error;
    
    // Load input file
    printf("Loading input file...\n");
    ufbx_scene *input_scene = ufbx_load_file(input_file, &load_opts, &error);
    if (!input_scene) {
        print_error(&error, "Failed to load input FBX file");
        return 1;
    }
    
    // Load output file
    printf("Loading output file...\n");
    printf("  File path: %s\n", output_file);
    
    // Check if output file exists and get its size
    FILE *test_file = fopen(output_file, "rb");
    if (!test_file) {
        fprintf(stderr, "ERROR: Output file does not exist or cannot be opened\n");
        ufbx_free_scene(input_scene);
        return 1;
    }
    
    fseek(test_file, 0, SEEK_END);
    long file_size = ftell(test_file);
    fclose(test_file);
    printf("  File size: %ld bytes\n", file_size);
    
    if (file_size == 0) {
        fprintf(stderr, "ERROR: Output file is empty\n");
        ufbx_free_scene(input_scene);
        return 1;
    }
    
    // Try to load with more detailed error reporting
    ufbx_scene *output_scene = ufbx_load_file(output_file, &load_opts, &error);
    if (!output_scene) {
        printf("ERROR: Failed to load output FBX file\n");
        printf("  Error type: %d\n", error.type);
        printf("  Error description: %.*s\n", (int)error.description.length, error.description.data);
        printf("  Stack size: %u\n", error.stack_size);
        for (size_t i = 0; i < error.stack_size; i++) {
            ufbx_error_frame *frame = &error.stack[i];
            printf("    Frame %zu: %.*s (%u)\n", i, (int)frame->description.length, frame->description.data, frame->source_line);
        }
        
        // Try to read the beginning of the file to see what's there
        printf("\n  First 200 bytes of output file:\n");
        test_file = fopen(output_file, "rb");
        if (test_file) {
            char buffer[201];
            size_t bytes_read = fread(buffer, 1, 200, test_file);
            buffer[bytes_read] = '\0';
            printf("    \"%s\"\n", buffer);
            fclose(test_file);
        }
        
        print_error(&error, "Failed to load output FBX file");
        ufbx_free_scene(input_scene);
        return 1;
    }
    
    printf("Both files loaded successfully!\n");
    printf("\n");
    
    // Get metrics for both scenes
    scene_metrics input_metrics, output_metrics;
    get_scene_metrics(input_scene, &input_metrics);
    get_scene_metrics(output_scene, &output_metrics);
    
    // Compare and print results
    print_metrics_comparison(&input_metrics, &output_metrics);
    
    // Cleanup
    ufbx_free_scene(input_scene);
    ufbx_free_scene(output_scene);
    
    // Return success/failure based on data preservation
    bool data_preserved = (
        input_metrics.nodes == output_metrics.nodes &&
        input_metrics.meshes == output_metrics.meshes &&
        input_metrics.materials == output_metrics.materials &&
        input_metrics.animations == output_metrics.animations &&
        input_metrics.bones == output_metrics.bones &&
        input_metrics.total_vertices == output_metrics.total_vertices &&
        input_metrics.total_faces == output_metrics.total_faces
    );
    
    return data_preserved ? 0 : 1;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.fbx> <output.fbx>\n", argv[0]);
        fprintf(stderr, "Example: %s ./data/huesitos.fbx huesitos_roundtrip.fbx\n", argv[0]);
        fprintf(stderr, "\nThis tool validates that a roundtrip preserved scene data correctly.\n");
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    return validate_roundtrip(input_file, output_file);
}
