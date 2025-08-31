#include "../../ufbx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Borrowed from testing_utils.h for data comparison
typedef struct {
    size_t num;
    double sum;
    double max;
} ufbxt_diff_error;

// Progress tracking context
typedef struct {
    const char *stage;
    size_t bytes_processed;
    size_t total_bytes;
    bool failed;
    char error_context[512];
} progress_context;

// Validation tier levels
typedef enum {
    VALIDATION_TIER_STRICT = 0,
    VALIDATION_TIER_STANDARD,
    VALIDATION_TIER_LENIENT,
    VALIDATION_TIER_RECOVERY,
    VALIDATION_TIER_COUNT
} validation_tier;

const char* tier_names[] = {
    "STRICT",
    "STANDARD", 
    "LENIENT",
    "RECOVERY"
};

// Progress callback to track loading progress and identify failure points
ufbx_progress_result progress_callback(void *user, const ufbx_progress *progress)
{
    progress_context *ctx = (progress_context*)user;
    ctx->bytes_processed = progress->bytes_read;
    ctx->total_bytes = progress->bytes_total;
    
    printf("    Progress [%s]: %zu/%zu bytes (%.1f%%)\n", 
           ctx->stage,
           ctx->bytes_processed, 
           ctx->total_bytes,
           ctx->total_bytes > 0 ? (100.0 * ctx->bytes_processed / ctx->total_bytes) : 0.0);
    
    return UFBX_PROGRESS_CONTINUE;
}

void print_error_detailed(const ufbx_error *error, const char *description)
{
    char buffer[2048];
    ufbx_format_error(buffer, sizeof(buffer), error);
    printf("ERROR: %s\n", description);
    printf("  Full error details:\n");
    printf("  %s\n", buffer);
    
    if (error->stack_size > 0) {
        printf("  Error stack trace:\n");
        for (size_t i = 0; i < error->stack_size; i++) {
            const ufbx_error_frame *frame = &error->stack[i];
            printf("    Frame %zu: %.*s (line %u)\n", 
                   i, (int)frame->description.length, frame->description.data, frame->source_line);
        }
    }
}

// Helper functions for data comparison
void assert_close_real(ufbxt_diff_error *p_err, ufbx_real a, ufbx_real b, ufbx_real threshold)
{
    ufbx_real err = (ufbx_real)fabs(a - b);
    if (err > threshold) {
        printf("      Real value mismatch: %.6f vs %.6f (diff: %.6f, threshold: %.6f)\n", 
               a, b, err, threshold);
    }
    p_err->num++;
    p_err->sum += err;
    if (err > p_err->max) p_err->max = err;
}

void assert_close_vec3(ufbxt_diff_error *p_err, ufbx_vec3 a, ufbx_vec3 b, ufbx_real threshold)
{
    assert_close_real(p_err, a.x, b.x, threshold);
    assert_close_real(p_err, a.y, b.y, threshold);
    assert_close_real(p_err, a.z, b.z, threshold);
}

void assert_close_vec2(ufbxt_diff_error *p_err, ufbx_vec2 a, ufbx_vec2 b, ufbx_real threshold)
{
    assert_close_real(p_err, a.x, b.x, threshold);
    assert_close_real(p_err, a.y, b.y, threshold);
}

// Try to load a scene with different validation tiers
ufbx_scene* try_load_with_tier(const char *filename, validation_tier tier, ufbx_error *error, progress_context *progress_ctx)
{
    ufbx_load_opts load_opts = { 0 };
    
    // Set progress callback
    load_opts.progress_cb.fn = progress_callback;
    load_opts.progress_cb.user = progress_ctx;
    load_opts.progress_interval_hint = 1024; // Report every 1KB
    
    switch (tier) {
        case VALIDATION_TIER_STRICT:
            progress_ctx->stage = "STRICT";
            load_opts.strict = true;
            load_opts.disable_quirks = true;
            load_opts.index_error_handling = UFBX_INDEX_ERROR_HANDLING_ABORT_LOADING;
            load_opts.unicode_error_handling = UFBX_UNICODE_ERROR_HANDLING_ABORT_LOADING;
            load_opts.retain_dom = true; // Keep raw structure for debugging
            break;
            
        case VALIDATION_TIER_STANDARD:
            progress_ctx->stage = "STANDARD";
            load_opts.load_external_files = true;
            load_opts.ignore_missing_external_files = true;
            load_opts.generate_missing_normals = true;
            break;
            
        case VALIDATION_TIER_LENIENT:
            progress_ctx->stage = "LENIENT";
            load_opts.load_external_files = true;
            load_opts.ignore_missing_external_files = true;
            load_opts.generate_missing_normals = true;
            load_opts.connect_broken_elements = true;
            load_opts.allow_nodes_out_of_root = true;
            break;
            
        case VALIDATION_TIER_RECOVERY:
            progress_ctx->stage = "RECOVERY";
            load_opts.load_external_files = false;
            load_opts.ignore_missing_external_files = true;
            load_opts.generate_missing_normals = true;
            load_opts.connect_broken_elements = true;
            load_opts.allow_nodes_out_of_root = true;
            load_opts.allow_unsafe = true;
            load_opts.index_error_handling = UFBX_INDEX_ERROR_HANDLING_CLAMP;
            load_opts.unicode_error_handling = UFBX_UNICODE_ERROR_HANDLING_REPLACEMENT_CHARACTER;
            break;
            
        default:
            return NULL;
    }
    
    // Common target settings for consistency
    load_opts.target_axes.right = UFBX_COORDINATE_AXIS_POSITIVE_X;
    load_opts.target_axes.up = UFBX_COORDINATE_AXIS_POSITIVE_Y;
    load_opts.target_axes.front = UFBX_COORDINATE_AXIS_POSITIVE_Z;
    load_opts.target_unit_meters = 1.0f;
    
    return ufbx_load_file(filename, &load_opts, error);
}

// Enhanced scene comparison using existing utilities
bool compare_scenes_detailed(ufbx_scene *input_scene, ufbx_scene *output_scene, ufbxt_diff_error *total_error)
{
    bool data_preserved = true;
    ufbxt_diff_error local_error = { 0 };
    
    printf("  === Detailed Scene Comparison ===\n");
    
    // Compare basic counts first
    printf("  Scene element counts:\n");
    printf("    Nodes:           %zu -> %zu", input_scene->nodes.count, output_scene->nodes.count);
    if (input_scene->nodes.count == output_scene->nodes.count) {
        printf(" ✓\n");
    } else {
        printf(" ✗ COUNT MISMATCH\n");
        data_preserved = false;
    }
    
    printf("    Meshes:          %zu -> %zu", input_scene->meshes.count, output_scene->meshes.count);
    if (input_scene->meshes.count == output_scene->meshes.count) {
        printf(" ✓\n");
    } else {
        printf(" ✗ COUNT MISMATCH\n");
        data_preserved = false;
    }
    
    printf("    Materials:       %zu -> %zu", input_scene->materials.count, output_scene->materials.count);
    if (input_scene->materials.count == output_scene->materials.count) {
        printf(" ✓\n");
    } else {
        printf(" ✗ COUNT MISMATCH\n");
        data_preserved = false;
    }
    
    printf("    Animations:      %zu -> %zu", input_scene->anim_stacks.count, output_scene->anim_stacks.count);
    if (input_scene->anim_stacks.count == output_scene->anim_stacks.count) {
        printf(" ✓\n");
    } else {
        printf(" ✗ COUNT MISMATCH\n");
        data_preserved = false;
    }
    
    printf("    Bones:           %zu -> %zu", input_scene->bones.count, output_scene->bones.count);
    if (input_scene->bones.count == output_scene->bones.count) {
        printf(" ✓\n");
    } else {
        printf(" ✗ COUNT MISMATCH\n");
        data_preserved = false;
    }
    
    printf("    Skin deformers:  %zu -> %zu", input_scene->skin_deformers.count, output_scene->skin_deformers.count);
    if (input_scene->skin_deformers.count == output_scene->skin_deformers.count) {
        printf(" ✓\n");
    } else {
        printf(" ✗ COUNT MISMATCH\n");
        data_preserved = false;
    }
    
    printf("    Blend deformers: %zu -> %zu", input_scene->blend_deformers.count, output_scene->blend_deformers.count);
    if (input_scene->blend_deformers.count == output_scene->blend_deformers.count) {
        printf(" ✓\n");
    } else {
        printf(" ✗ COUNT MISMATCH\n");
        data_preserved = false;
    }
    
    // Detailed mesh data comparison
    if (input_scene->meshes.count == output_scene->meshes.count && input_scene->meshes.count > 0) {
        printf("\n  === Detailed Mesh Data Comparison ===\n");
        
        for (size_t i = 0; i < input_scene->meshes.count; i++) {
            ufbx_mesh *input_mesh = input_scene->meshes.data[i];
            ufbx_mesh *output_mesh = output_scene->meshes.data[i];
            
            printf("  Mesh %zu (\"%.*s\"):\n", i, 
                   (int)input_mesh->name.length, input_mesh->name.data);
            
            printf("    Vertices:     %zu -> %zu", input_mesh->num_vertices, output_mesh->num_vertices);
            if (input_mesh->num_vertices == output_mesh->num_vertices) {
                printf(" ✓\n");
                
                // Compare actual vertex data if counts match
                if (input_mesh->vertices.count == output_mesh->vertices.count && 
                    input_mesh->vertices.count > 0) {
                    printf("    Vertex data comparison:\n");
                    ufbxt_diff_error vertex_error = { 0 };
                    
                    size_t compare_count = input_mesh->vertices.count;
                    if (compare_count > 100) compare_count = 100; // Sample first 100 for performance
                    
                    for (size_t vi = 0; vi < compare_count; vi++) {
                        assert_close_vec3(&vertex_error, 
                                        input_mesh->vertices.data[vi], 
                                        output_mesh->vertices.data[vi], 
                                        0.001f);
                    }
                    
                    if (vertex_error.num > 0) {
                        double avg_error = vertex_error.sum / vertex_error.num;
                        printf("      Position diff: avg %.6f, max %.6f (%zu samples)\n", 
                               avg_error, vertex_error.max, vertex_error.num);
                        
                        if (vertex_error.max > 0.001) {
                            printf("      ⚠️  Vertex precision loss detected\n");
                        }
                    }
                }
            } else {
                printf(" ✗ COUNT MISMATCH\n");
                data_preserved = false;
            }
            
            printf("    Faces:        %zu -> %zu", input_mesh->num_faces, output_mesh->num_faces);
            if (input_mesh->num_faces == output_mesh->num_faces) {
                printf(" ✓\n");
            } else {
                printf(" ✗ COUNT MISMATCH\n");
                data_preserved = false;
            }
            
            // Compare normals if both exist
            bool input_has_normals = input_mesh->vertex_normal.exists && input_mesh->vertex_normal.values.count > 0;
            bool output_has_normals = output_mesh->vertex_normal.exists && output_mesh->vertex_normal.values.count > 0;
            
            printf("    Normals:      %s -> %s", 
                   input_has_normals ? "YES" : "NO",
                   output_has_normals ? "YES" : "NO");
            
            if (input_has_normals == output_has_normals) {
                printf(" ✓\n");
                
                // Compare normal data if both exist
                if (input_has_normals && output_has_normals &&
                    input_mesh->vertex_normal.values.count == output_mesh->vertex_normal.values.count) {
                    ufbxt_diff_error normal_error = { 0 };
                    
                    size_t compare_count = input_mesh->vertex_normal.values.count;
                    if (compare_count > 100) compare_count = 100; // Sample for performance
                    
                    for (size_t ni = 0; ni < compare_count; ni++) {
                        assert_close_vec3(&normal_error,
                                        input_mesh->vertex_normal.values.data[ni],
                                        output_mesh->vertex_normal.values.data[ni],
                                        0.01f); // More tolerance for normals
                    }
                    
                    if (normal_error.num > 0) {
                        double avg_error = normal_error.sum / normal_error.num;
                        printf("      Normal diff: avg %.6f, max %.6f (%zu samples)\n", 
                               avg_error, normal_error.max, normal_error.num);
                    }
                }
            } else {
                printf(" ✗ NORMAL PRESENCE MISMATCH\n");
                data_preserved = false;
            }
            
            // Compare UV data
            bool input_has_uvs = input_mesh->vertex_uv.exists && input_mesh->vertex_uv.values.count > 0;
            bool output_has_uvs = output_mesh->vertex_uv.exists && output_mesh->vertex_uv.values.count > 0;
            
            printf("    UVs:          %s -> %s", 
                   input_has_uvs ? "YES" : "NO",
                   output_has_uvs ? "YES" : "NO");
            
            if (input_has_uvs == output_has_uvs) {
                printf(" ✓\n");
                
                // Compare UV data if both exist
                if (input_has_uvs && output_has_uvs &&
                    input_mesh->vertex_uv.values.count == output_mesh->vertex_uv.values.count) {
                    ufbxt_diff_error uv_error = { 0 };
                    
                    size_t compare_count = input_mesh->vertex_uv.values.count;
                    if (compare_count > 100) compare_count = 100; // Sample for performance
                    
                    for (size_t ui = 0; ui < compare_count; ui++) {
                        assert_close_vec2(&uv_error,
                                        input_mesh->vertex_uv.values.data[ui],
                                        output_mesh->vertex_uv.values.data[ui],
                                        0.001f);
                    }
                    
                    if (uv_error.num > 0) {
                        double avg_error = uv_error.sum / uv_error.num;
                        printf("      UV diff: avg %.6f, max %.6f (%zu samples)\n", 
                               avg_error, uv_error.max, uv_error.num);
                    }
                }
            } else {
                printf(" ✗ UV PRESENCE MISMATCH\n");
                data_preserved = false;
            }
        }
    }
    
    // Compare animation data if both scenes have animations
    if (input_scene->anim_stacks.count == output_scene->anim_stacks.count && 
        input_scene->anim_stacks.count > 0) {
        printf("\n  === Animation Data Comparison ===\n");
        
        for (size_t i = 0; i < input_scene->anim_stacks.count; i++) {
            ufbx_anim_stack *input_stack = input_scene->anim_stacks.data[i];
            ufbx_anim_stack *output_stack = output_scene->anim_stacks.data[i];
            
            printf("  Animation %zu (\"%.*s\"):\n", i,
                   (int)input_stack->name.length, input_stack->name.data);
            
            printf("    Duration:     %.3f -> %.3f", 
                   input_stack->time_end - input_stack->time_begin,
                   output_stack->time_end - output_stack->time_begin);
            
            ufbxt_diff_error time_error = { 0 };
            assert_close_real(&time_error, 
                            input_stack->time_end - input_stack->time_begin,
                            output_stack->time_end - output_stack->time_begin,
                            0.01f);
            
            if (time_error.max > 0.01f) {
                printf(" ⚠️  TIME MISMATCH\n");
                data_preserved = false;
            } else {
                printf(" ✓\n");
            }
            
            printf("    Layers:       %zu -> %zu", 
                   input_stack->layers.count, output_stack->layers.count);
            if (input_stack->layers.count == output_stack->layers.count) {
                printf(" ✓\n");
            } else {
                printf(" ✗ LAYER COUNT MISMATCH\n");
                data_preserved = false;
            }
        }
    }
    
    // Accumulate total error statistics
    if (total_error && local_error.num > 0) {
        total_error->num += local_error.num;
        total_error->sum += local_error.sum;
        if (local_error.max > total_error->max) {
            total_error->max = local_error.max;
        }
    }
    
    return data_preserved;
}

// Progressive validation with fallback
int validate_roundtrip_enhanced(const char *input_file, const char *output_file)
{
    printf("=== Enhanced FBX Roundtrip Validation ===\n");
    printf("Input file:  %s\n", input_file);
    printf("Output file: %s\n", output_file);
    printf("\n");

    ufbx_error error;
    progress_context progress_ctx = { 0 };
    
    // Load input file (always use standard settings for input)
    printf("Loading input file with STANDARD settings...\n");
    progress_ctx.stage = "INPUT";
    ufbx_scene *input_scene = try_load_with_tier(input_file, VALIDATION_TIER_STANDARD, &error, &progress_ctx);
    
    if (!input_scene) {
        print_error_detailed(&error, "Failed to load input FBX file");
        return 1;
    }
    
    printf("✓ Input file loaded successfully!\n");
    if (input_scene->metadata.warnings.count > 0) {
        printf("  Input warnings: %zu\n", input_scene->metadata.warnings.count);
        for (size_t i = 0; i < input_scene->metadata.warnings.count; i++) {
            ufbx_warning warning = input_scene->metadata.warnings.data[i];
            printf("    %s", warning.description.data);
            if (warning.count > 1) {
                printf(" (x%zu)", warning.count);
            }
            printf("\n");
        }
    }
    printf("\n");
    
    // Check if output file exists and get basic info
    FILE *test_file = fopen(output_file, "rb");
    if (!test_file) {
        printf("ERROR: Output file does not exist or cannot be opened\n");
        ufbx_free_scene(input_scene);
        return 1;
    }
    
    fseek(test_file, 0, SEEK_END);
    long file_size = ftell(test_file);
    fclose(test_file);
    printf("Output file size: %ld bytes\n", file_size);
    
    if (file_size == 0) {
        printf("ERROR: Output file is empty\n");
        ufbx_free_scene(input_scene);
        return 1;
    }
    
    // Try progressive loading of output file
    printf("\nTrying progressive loading of output file...\n");
    ufbx_scene *output_scene = NULL;
    validation_tier successful_tier = VALIDATION_TIER_COUNT;
    
    for (validation_tier tier = VALIDATION_TIER_STRICT; tier < VALIDATION_TIER_COUNT; tier++) {
        printf("  Attempting %s loading...\n", tier_names[tier]);
        
        progress_ctx.failed = false;
        progress_ctx.bytes_processed = 0;
        progress_ctx.total_bytes = 0;
        
        output_scene = try_load_with_tier(output_file, tier, &error, &progress_ctx);
        
        if (output_scene) {
            successful_tier = tier;
            printf("  ✓ SUCCESS with %s loading\n", tier_names[tier]);
            break;
        } else {
            printf("  ✗ FAILED with %s loading\n", tier_names[tier]);
            printf("    Progress reached: %zu/%zu bytes (%.1f%%)\n",
                   progress_ctx.bytes_processed, progress_ctx.total_bytes,
                   progress_ctx.total_bytes > 0 ? (100.0 * progress_ctx.bytes_processed / progress_ctx.total_bytes) : 0.0);
            print_error_detailed(&error, "Loading failed");
            printf("\n");
        }
    }
    
    if (!output_scene) {
        printf("💥 CRITICAL: Output file could not be loaded with any validation tier!\n");
        printf("This indicates a serious structural problem with the exported FBX file.\n");
        ufbx_free_scene(input_scene);
        return 1;
    }
    
    printf("\n📊 Loading Summary:\n");
    printf("  Input file:  Loaded with STANDARD settings\n");
    printf("  Output file: Loaded with %s settings\n", tier_names[successful_tier]);
    
    if (successful_tier > VALIDATION_TIER_STANDARD) {
        printf("  ⚠️  Output file required relaxed loading settings!\n");
        printf("      This indicates export quality issues that should be investigated.\n");
    }
    
    printf("\n");
    
    // Perform detailed data comparison
    ufbxt_diff_error total_error = { 0 };
    bool data_preserved = compare_scenes_detailed(input_scene, output_scene, &total_error);
    
    // Final validation summary
    printf("\n=== Enhanced Validation Summary ===\n");
    if (data_preserved) {
        printf("🎉 VALIDATION PASSED: Data preserved correctly\n");
        if (total_error.num > 0) {
            double avg_error = total_error.sum / total_error.num;
            printf("   Data precision: avg %.6f, max %.6f (%zu comparisons)\n", 
                   avg_error, total_error.max, total_error.num);
        }
    } else {
        printf("⚠️  VALIDATION FAILED: Data loss or structural changes detected\n");
        if (total_error.num > 0) {
            double avg_error = total_error.sum / total_error.num;
            printf("   Data differences: avg %.6f, max %.6f (%zu comparisons)\n", 
                   avg_error, total_error.max, total_error.num);
        }
    }
    
    printf("   Loading tier required: %s\n", tier_names[successful_tier]);
    
    // Cleanup
    ufbx_free_scene(input_scene);
    ufbx_free_scene(output_scene);
    
    return data_preserved ? 0 : 1;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.fbx> <output.fbx>\n", argv[0]);
        fprintf(stderr, "Example: %s ./data/huesitos.fbx huesitos_roundtrip.fbx\n", argv[0]);
        fprintf(stderr, "\nEnhanced validation tool with progressive loading and detailed data comparison.\n");
        fprintf(stderr, "This tool provides comprehensive analysis of roundtrip data preservation.\n");
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];

    return validate_roundtrip_enhanced(input_file, output_file);
}
