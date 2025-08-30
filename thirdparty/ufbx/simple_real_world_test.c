#include "ufbx_export.h"
#include "ufbx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Simple test with files we know exist
static const char* test_files[] = {
    "blender_279_default_6100_ascii.fbx",
    "blender_279_default_7400_binary.fbx", 
    "blender_282_suzanne_7400_binary.fbx"
};

static const size_t num_test_files = sizeof(test_files) / sizeof(test_files[0]);

// Convert loaded scene to export scene (simplified)
static ufbx_export_scene *convert_to_export_scene(ufbx_scene *loaded_scene) {
    ufbx_export_scene *export_scene = ufbx_create_scene(NULL);
    if (!export_scene) return NULL;
    
    // Add root node
    ufbx_node *root_node = ufbx_add_node(export_scene, "ConvertedRoot", NULL);
    if (!root_node) {
        ufbx_free_export_scene(export_scene);
        return NULL;
    }
    
    // Convert first mesh if available
    if (loaded_scene->meshes.count > 0) {
        ufbx_mesh *orig_mesh = loaded_scene->meshes.data[0];
        
        // Add mesh node
        ufbx_node *mesh_node = ufbx_add_node(export_scene, orig_mesh->element.name.data, root_node);
        if (!mesh_node) {
            ufbx_free_export_scene(export_scene);
            return NULL;
        }
        
        // Create new mesh
        ufbx_mesh *new_mesh = ufbx_add_mesh(export_scene, orig_mesh->element.name.data);
        if (!new_mesh) {
            ufbx_free_export_scene(export_scene);
            return NULL;
        }
        
        // Copy mesh data if available
        if (orig_mesh->vertex_position.exists && orig_mesh->vertex_position.values.count > 0) {
            ufbx_error error;
            bool success = ufbx_set_mesh_vertices(new_mesh, orig_mesh->vertex_position.values.data, 
                                                 orig_mesh->vertex_position.values.count, &error);
            if (!success) {
                printf("      Warning: Failed to set mesh vertices: %s\n", error.description.data);
            }
        }
        
        // Attach mesh to node
        ufbx_error error;
        ufbx_attach_mesh_to_node(mesh_node, new_mesh, &error);
        
        // Add material if available
        if (loaded_scene->materials.count > 0) {
            ufbx_material *orig_material = loaded_scene->materials.data[0];
            ufbx_material *new_material = ufbx_add_material(export_scene, orig_material->element.name.data);
            if (new_material) {
                ufbx_set_material_albedo(new_material, 0.8f, 0.2f, 0.1f, 1.0f, &error);
                ufbx_attach_material_to_mesh(new_mesh, new_material, 0, &error);
            }
        }
    }
    
    return export_scene;
}

int main() {
    printf("🚀 SIMPLE REAL-WORLD FBX ASCII EXPORT TEST\n");
    printf("Testing ASCII export with known existing files\n\n");
    
    int passed = 0;
    int failed = 0;
    
    for (size_t i = 0; i < num_test_files; i++) {
        printf("🧪 Testing: %s\n", test_files[i]);
        
        char input_path[512];
        snprintf(input_path, sizeof(input_path), "data/%s", test_files[i]);
        
        // Load original file
        printf("  📂 Loading original file...\n");
        ufbx_load_opts load_opts = { 0 };
        ufbx_error load_error;
        ufbx_scene *original_scene = ufbx_load_file(input_path, &load_opts, &load_error);
        
        if (!original_scene) {
            printf("    ❌ Failed to load: %s\n", load_error.description.data);
            failed++;
            continue;
        }
        
        printf("    ✅ Loaded: %zu nodes, %zu meshes, %zu materials\n",
               original_scene->nodes.count, original_scene->meshes.count, original_scene->materials.count);
        
        // Convert to export scene
        printf("  🔄 Converting to export scene...\n");
        ufbx_export_scene *export_scene = convert_to_export_scene(original_scene);
        if (!export_scene) {
            printf("    ❌ Failed to convert to export scene\n");
            ufbx_free_scene(original_scene);
            failed++;
            continue;
        }
        
        printf("    ✅ Converted: %zu nodes, %zu meshes, %zu materials\n",
               export_scene->nodes.count, export_scene->meshes.count, export_scene->materials.count);
        
        // Export to ASCII
        printf("  📤 Exporting to ASCII...\n");
        ufbx_export_opts export_opts = { 0 };
        export_opts.ascii_format = true;
        export_opts.fbx_version = 7400;
        export_opts.export_materials = true;
        
        char output_path[512];
        snprintf(output_path, sizeof(output_path), "/tmp/test_%zu_ascii.fbx", i);
        
        ufbx_error export_error;
        bool export_success = ufbx_export_to_file(export_scene, output_path, &export_opts, &export_error);
        
        if (!export_success) {
            printf("    ❌ Export failed: %s\n", export_error.description.data);
            ufbx_free_export_scene(export_scene);
            ufbx_free_scene(original_scene);
            failed++;
            continue;
        }
        
        printf("    ✅ ASCII export completed\n");
        
        // Re-import ASCII file
        printf("  📥 Re-importing ASCII file...\n");
        ufbx_scene *roundtrip_scene = ufbx_load_file(output_path, &load_opts, &load_error);
        
        if (!roundtrip_scene) {
            printf("    ❌ Re-import failed: %s\n", load_error.description.data);
            ufbx_free_export_scene(export_scene);
            ufbx_free_scene(original_scene);
            failed++;
            continue;
        }
        
        printf("    ✅ Re-import successful: %zu nodes, %zu meshes, %zu materials\n",
               roundtrip_scene->nodes.count, roundtrip_scene->meshes.count, roundtrip_scene->materials.count);
        
        // Basic validation
        bool validation_passed = true;
        if (roundtrip_scene->nodes.count == 0) {
            printf("    ❌ No nodes in roundtrip\n");
            validation_passed = false;
        }
        
        if (validation_passed) {
            printf("  ✅ PASSED: %s\n", test_files[i]);
            passed++;
        } else {
            printf("  ❌ FAILED: %s\n", test_files[i]);
            failed++;
        }
        
        // Cleanup
        ufbx_free_export_scene(export_scene);
        ufbx_free_scene(original_scene);
        ufbx_free_scene(roundtrip_scene);
        
        printf("\n");
    }
    
    printf("📊 RESULTS: %d passed, %d failed\n", passed, failed);
    
    if (passed == num_test_files) {
        printf("🏆 SUCCESS: ASCII export working with real-world files!\n");
        return 0;
    } else {
        printf("⚠️  Some tests failed\n");
        return 1;
    }
}
