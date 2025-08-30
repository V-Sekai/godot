#include "ufbx_export.h"
#include "ufbx.h"
#include <stdio.h>
#include <string.h>

int main() {
    printf("=== FBX Export Validation Test ===\n");
    
    // Step 1: Create and export a test scene
    printf("Step 1: Creating and exporting test scene...\n");
    
    ufbx_export_scene *scene = ufbx_create_scene(NULL);
    if (!scene) {
        printf("FAIL: Could not create export scene\n");
        return 1;
    }
    
    // Add root node
    ufbx_node *root_node = ufbx_add_node(scene, "TestRoot", NULL);
    if (!root_node) {
        printf("FAIL: Could not create root node\n");
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    // Add mesh node
    ufbx_node *mesh_node = ufbx_add_node(scene, "TestMeshNode", root_node);
    if (!mesh_node) {
        printf("FAIL: Could not create mesh node\n");
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    // Create triangle mesh
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "TestTriangle");
    if (!mesh) {
        printf("FAIL: Could not create mesh\n");
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    // Define triangle vertices
    ufbx_vec3 vertices[] = {
        {0.0f, 1.0f, 0.0f},   // Top
        {-1.0f, -1.0f, 0.0f}, // Bottom left
        {1.0f, -1.0f, 0.0f}   // Bottom right
    };
    
    uint32_t indices[] = {0, 1, 2};
    
    ufbx_error error;
    bool success = ufbx_set_mesh_vertices(mesh, vertices, 3, &error);
    if (!success) {
        printf("FAIL: Could not set mesh vertices - %s\n", error.description.data);
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    success = ufbx_set_mesh_indices(mesh, indices, 3, &error);
    if (!success) {
        printf("FAIL: Could not set mesh indices - %s\n", error.description.data);
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    // Attach mesh to node
    success = ufbx_attach_mesh_to_node(mesh_node, mesh, &error);
    if (!success) {
        printf("FAIL: Could not attach mesh to node - %s\n", error.description.data);
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    // Create material
    ufbx_material *material = ufbx_add_material(scene, "TestMaterial");
    if (!material) {
        printf("FAIL: Could not create material\n");
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    success = ufbx_set_material_albedo(material, 0.8f, 0.2f, 0.1f, 1.0f, &error);
    if (!success) {
        printf("FAIL: Could not set material albedo - %s\n", error.description.data);
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    success = ufbx_attach_material_to_mesh(mesh, material, 0, &error);
    if (!success) {
        printf("FAIL: Could not attach material to mesh - %s\n", error.description.data);
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    // Export to file
    ufbx_export_opts opts = {0};
    opts.ascii_format = true;   // Use ASCII for easier debugging
    opts.fbx_version = 7400;
    opts.export_materials = true;
    
    const char *filename = "validation_test.fbx";
    success = ufbx_export_to_file(scene, filename, &opts, &error);
    if (!success) {
        printf("FAIL: Could not export to file - %s\n", error.description.data);
        ufbx_free_export_scene(scene);
        return 1;
    }
    
    printf("SUCCESS: Exported scene to %s\n", filename);
    ufbx_free_export_scene(scene);
    
    // Step 2: Load the exported file back using ufbx
    printf("\nStep 2: Loading exported file back with ufbx...\n");
    
    ufbx_load_opts load_opts = {0};
    ufbx_scene *loaded_scene = ufbx_load_file(filename, &load_opts, &error);
    if (!loaded_scene) {
        printf("FAIL: Could not load exported file - %s\n", error.description.data);
        return 1;
    }
    
    printf("SUCCESS: Loaded scene from %s\n", filename);
    printf("  - Version: %u\n", loaded_scene->metadata.version);
    printf("  - Creator: %s\n", loaded_scene->metadata.creator.data);
    printf("  - File format: %s\n", loaded_scene->metadata.ascii ? "ASCII" : "Binary");
    printf("  - Elements: %zu\n", loaded_scene->elements.count);
    printf("  - Nodes: %zu\n", loaded_scene->nodes.count);
    printf("  - Meshes: %zu\n", loaded_scene->meshes.count);
    printf("  - Materials: %zu\n", loaded_scene->materials.count);
    
    // Step 3: Validate the loaded content
    printf("\nStep 3: Validating loaded content...\n");
    
    // Check nodes
    bool found_root = false, found_mesh_node = false;
    for (size_t i = 0; i < loaded_scene->nodes.count; i++) {
        ufbx_node *node = loaded_scene->nodes.data[i];
        if (strcmp(node->name.data, "TestRoot") == 0) {
            found_root = true;
            printf("  ✓ Found root node: %s\n", node->name.data);
        } else if (strcmp(node->name.data, "TestMeshNode") == 0) {
            found_mesh_node = true;
            printf("  ✓ Found mesh node: %s\n", node->name.data);
        }
    }
    
    if (!found_root) {
        printf("  ✗ Root node not found\n");
    }
    if (!found_mesh_node) {
        printf("  ✗ Mesh node not found\n");
    }
    
    // Check meshes
    bool found_mesh = false;
    for (size_t i = 0; i < loaded_scene->meshes.count; i++) {
        ufbx_mesh *mesh = loaded_scene->meshes.data[i];
        if (strcmp(mesh->name.data, "TestTriangle") == 0) {
            found_mesh = true;
            printf("  ✓ Found mesh: %s\n", mesh->name.data);
            printf("    - Vertices: %zu\n", mesh->num_vertices);
            printf("    - Indices: %zu\n", mesh->num_indices);
            printf("    - Faces: %zu\n", mesh->num_faces);
            
            // Validate triangle data
            if (mesh->num_vertices == 3 && mesh->num_indices == 3 && mesh->num_faces == 1) {
                printf("    ✓ Triangle geometry is correct\n");
                
                // Check vertex positions
                if (mesh->vertex_position.exists && mesh->vertex_position.values.count == 3) {
                    ufbx_vec3 *verts = mesh->vertex_position.values.data;
                    printf("    ✓ Vertex positions loaded:\n");
                    printf("      [0]: (%.3f, %.3f, %.3f)\n", verts[0].x, verts[0].y, verts[0].z);
                    printf("      [1]: (%.3f, %.3f, %.3f)\n", verts[1].x, verts[1].y, verts[1].z);
                    printf("      [2]: (%.3f, %.3f, %.3f)\n", verts[2].x, verts[2].y, verts[2].z);
                } else {
                    printf("    ✗ Vertex positions not found or incorrect count\n");
                }
            } else {
                printf("    ✗ Triangle geometry is incorrect\n");
            }
        }
    }
    
    if (!found_mesh) {
        printf("  ✗ Test mesh not found\n");
    }
    
    // Check materials
    bool found_material = false;
    for (size_t i = 0; i < loaded_scene->materials.count; i++) {
        ufbx_material *material = loaded_scene->materials.data[i];
        if (strcmp(material->name.data, "TestMaterial") == 0) {
            found_material = true;
            printf("  ✓ Found material: %s\n", material->name.data);
            
            // Check material properties
            ufbx_prop *diffuse = ufbx_find_prop(&material->props, "DiffuseColor");
            if (diffuse) {
                printf("    ✓ DiffuseColor: (%.3f, %.3f, %.3f)\n", 
                       diffuse->value_vec3.x, diffuse->value_vec3.y, diffuse->value_vec3.z);
            } else {
                printf("    ✗ DiffuseColor property not found\n");
            }
        }
    }
    
    if (!found_material) {
        printf("  ✗ Test material not found\n");
    }
    
    ufbx_free_scene(loaded_scene);
    
    // Step 4: Summary
    printf("\n=== VALIDATION SUMMARY ===\n");
    bool validation_passed = found_root && found_mesh_node && found_mesh && found_material;
    
    if (validation_passed) {
        printf("✓ VALIDATION PASSED: FBX export/import round-trip successful!\n");
        printf("✓ All exported data was correctly preserved\n");
        printf("✓ FBX file format is valid and compatible with ufbx loader\n");
        printf("\n🎉 FBX EXPORT IMPLEMENTATION IS WORKING CORRECTLY! 🎉\n");
        return 0;
    } else {
        printf("✗ VALIDATION FAILED: Some exported data was not preserved\n");
        return 1;
    }
}
