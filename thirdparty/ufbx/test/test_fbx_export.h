#ifndef UFBXT_TEST_FBX_EXPORT_H
#define UFBXT_TEST_FBX_EXPORT_H

#include "../ufbx_export.h"
#include "testing_utils.h"

// Test basic scene creation and export
UFBXT_TEST(test_export_basic_scene)
#if UFBXT_IMPL
{
    // Create a basic export scene
    ufbx_export_scene *scene = ufbx_create_scene(NULL);
    ufbxt_assert(scene != NULL);
    
    // Add a root node
    ufbx_node *root_node = ufbx_add_node(scene, "Root", NULL);
    ufbxt_assert(root_node != NULL);
    ufbxt_assert(strcmp(root_node->element.name.data, "Root") == 0);
    
    // Add a mesh node
    ufbx_node *mesh_node = ufbx_add_node(scene, "TestMesh", root_node);
    ufbxt_assert(mesh_node != NULL);
    ufbxt_assert(mesh_node->parent == root_node);
    
    // Create a simple cube mesh
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "CubeMesh");
    ufbxt_assert(mesh != NULL);
    
    // Define cube vertices
    ufbx_vec3 vertices[] = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},  // Back face
        {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}       // Front face
    };
    
    // Define cube indices (triangles)
    uint32_t indices[] = {
        // Back face
        0, 1, 2, 2, 3, 0,
        // Front face
        4, 6, 5, 6, 4, 7,
        // Left face
        0, 4, 7, 7, 3, 0,
        // Right face
        1, 5, 6, 6, 2, 1,
        // Bottom face
        0, 1, 5, 5, 4, 0,
        // Top face
        3, 7, 6, 6, 2, 3
    };
    
    // Set mesh data
    ufbx_error error;
    bool success = ufbx_set_mesh_vertices(mesh, vertices, 8, &error);
    ufbxt_assert(success);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_mesh_indices(mesh, indices, 36, &error);
    ufbxt_assert(success);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    // Attach mesh to node
    success = ufbx_attach_mesh_to_node(mesh_node, mesh, &error);
    ufbxt_assert(success);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    // Test export size calculation
    size_t export_size = ufbx_get_export_size(scene, NULL, &error);
    ufbxt_assert(export_size > 0);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    // Clean up
    ufbx_free_export_scene(scene);
}
#endif

// Test material creation and properties
UFBXT_TEST(test_export_materials)
#if UFBXT_IMPL
{
    ufbx_export_scene *scene = ufbx_create_scene(NULL);
    ufbxt_assert(scene != NULL);
    
    // Create a material
    ufbx_material *material = ufbx_add_material(scene, "TestMaterial");
    ufbxt_assert(material != NULL);
    ufbxt_assert(strcmp(material->element.name.data, "TestMaterial") == 0);
    
    // Set material properties
    ufbx_error error;
    bool success = ufbx_set_material_albedo(material, 0.8f, 0.2f, 0.1f, 1.0f, &error);
    ufbxt_assert(success);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    ufbxt_assert(material->pbr.base_color.has_value);
    
    success = ufbx_set_material_metallic_roughness(material, 0.7f, 0.3f, &error);
    ufbxt_assert(success);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    ufbxt_assert(material->pbr.metalness.has_value);
    ufbxt_assert(material->pbr.roughness.has_value);
    
    success = ufbx_set_material_emission(material, 0.1f, 0.1f, 0.2f, &error);
    ufbxt_assert(success);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    ufbxt_assert(material->pbr.emission_color.has_value);
    
    ufbx_free_export_scene(scene);
}
#endif

// Test mesh with normals and UVs
UFBXT_TEST(test_export_mesh_attributes)
#if UFBXT_IMPL
{
    ufbx_export_scene *scene = ufbx_create_scene(NULL);
    ufbxt_assert(scene != NULL);
    
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "AttributeMesh");
    ufbxt_assert(mesh != NULL);
    
    // Simple triangle
    ufbx_vec3 vertices[] = {
        {0, 1, 0}, {-1, -1, 0}, {1, -1, 0}
    };
    
    ufbx_vec3 normals[] = {
        {0, 0, 1}, {0, 0, 1}, {0, 0, 1}
    };
    
    ufbx_vec2 uvs[] = {
        {0.5f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}
    };
    
    uint32_t indices[] = {0, 1, 2};
    
    ufbx_error error;
    
    // Set all mesh attributes
    bool success = ufbx_set_mesh_vertices(mesh, vertices, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_mesh_normals(mesh, normals, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_mesh_uvs(mesh, uvs, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_mesh_indices(mesh, indices, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    // Verify mesh data was set correctly
    ufbxt_assert(mesh->vertices.count == 3);
    ufbxt_assert(mesh->vertex_normal.exists);
    ufbxt_assert(mesh->vertex_normal.values.count == 3);
    ufbxt_assert(mesh->vertex_uv.exists);
    ufbxt_assert(mesh->vertex_uv.values.count == 3);
    ufbxt_assert(mesh->vertex_indices.count == 3);
    
    ufbx_free_export_scene(scene);
}
#endif

// Test texture creation and attachment
UFBXT_TEST(test_export_textures)
#if UFBXT_IMPL
{
    ufbx_export_scene *scene = ufbx_create_scene(NULL);
    ufbxt_assert(scene != NULL);
    
    // Create material and texture
    ufbx_material *material = ufbx_add_material(scene, "TexturedMaterial");
    ufbxt_assert(material != NULL);
    
    ufbx_texture *texture = ufbx_add_texture(scene, "TestTexture");
    ufbxt_assert(texture != NULL);
    ufbxt_assert(strcmp(texture->element.name.data, "TestTexture") == 0);
    
    // Create dummy texture data (simple 2x2 RGB)
    uint8_t texture_data[] = {
        255, 0, 0,    255, 255, 0,    // Red, Yellow
        0, 255, 0,    0, 0, 255       // Green, Blue
    };
    
    ufbx_error error;
    bool success = ufbx_set_texture_data(texture, texture_data, sizeof(texture_data), "rgb", &error);
    ufbxt_assert(success);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    // Attach texture to material
    success = ufbx_attach_texture_to_material(material, texture, "BaseColor", &error);
    ufbxt_assert(success);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    ufbx_free_export_scene(scene);
}
#endif

// Test scene hierarchy
UFBXT_TEST(test_export_hierarchy)
#if UFBXT_IMPL
{
    ufbx_export_scene *scene = ufbx_create_scene(NULL);
    ufbxt_assert(scene != NULL);
    
    // Create hierarchy: Root -> Parent -> Child1, Child2
    ufbx_node *root = ufbx_add_node(scene, "Root", NULL);
    ufbxt_assert(root != NULL);
    
    ufbx_node *parent = ufbx_add_node(scene, "Parent", root);
    ufbxt_assert(parent != NULL);
    ufbxt_assert(parent->parent == root);
    
    ufbx_node *child1 = ufbx_add_node(scene, "Child1", parent);
    ufbxt_assert(child1 != NULL);
    ufbxt_assert(child1->parent == parent);
    
    ufbx_node *child2 = ufbx_add_node(scene, "Child2", parent);
    ufbxt_assert(child2 != NULL);
    ufbxt_assert(child2->parent == parent);
    
    // Set transforms
    ufbx_transform parent_transform = {
        {0, 2, 0},        // translation
        {0, 0, 0, 1},     // rotation (identity)
        {1, 1, 1}         // scale
    };
    
    ufbx_transform child1_transform = {
        {-1, 0, 0},       // translation
        {0, 0, 0, 1},     // rotation
        {0.5f, 0.5f, 0.5f} // scale
    };
    
    ufbx_error error;
    ufbx_set_node_transform(parent, &parent_transform, &error);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    ufbx_set_node_transform(child1, &child1_transform, &error);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    ufbx_free_export_scene(scene);
}
#endif

// Test error handling
UFBXT_TEST(test_export_error_handling)
#if UFBXT_IMPL
{
    ufbx_error error;
    
    // Test NULL scene
    size_t size = ufbx_get_export_size(NULL, NULL, &error);
    ufbxt_assert(size == 0);
    ufbxt_assert(error.type == UFBX_ERROR_UNKNOWN);
    
    // Test NULL mesh for vertex setting
    bool success = ufbx_set_mesh_vertices(NULL, NULL, 0, &error);
    ufbxt_assert(!success);
    ufbxt_assert(error.type == UFBX_ERROR_UNKNOWN);
    
    // Test NULL material for property setting
    success = ufbx_set_material_albedo(NULL, 1.0f, 1.0f, 1.0f, 1.0f, &error);
    ufbxt_assert(!success);
    ufbxt_assert(error.type == UFBX_ERROR_UNKNOWN);
    
    // Test invalid file export
    success = ufbx_export_to_file(NULL, "test.fbx", NULL, &error);
    ufbxt_assert(!success);
    ufbxt_assert(error.type == UFBX_ERROR_UNKNOWN);
}
#endif

// Test ASCII export format (when implemented)
UFBXT_TEST(test_export_ascii_format)
#if UFBXT_IMPL
{
    ufbx_export_scene *scene = ufbx_create_scene(NULL);
    ufbxt_assert(scene != NULL);
    
    // Create simple scene
    ufbx_node *root = ufbx_add_node(scene, "ASCIITest", NULL);
    ufbxt_assert(root != NULL);
    
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "ASCIIMesh");
    ufbxt_assert(mesh != NULL);
    
    // Simple triangle
    ufbx_vec3 vertices[] = {{0, 1, 0}, {-1, -1, 0}, {1, -1, 0}};
    uint32_t indices[] = {0, 1, 2};
    
    ufbx_error error;
    bool success = ufbx_set_mesh_vertices(mesh, vertices, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_mesh_indices(mesh, indices, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_attach_mesh_to_node(root, mesh, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    // Configure export options for ASCII
    ufbx_export_opts opts = {0};
    opts.ascii_format = true;
    opts.fbx_version = 7400;
    
    // Test export (will fail until writer is implemented)
    success = ufbx_export_to_file(scene, "ascii_test.fbx", &opts, &error);
    // For now, we expect this to fail with UNKNOWN (not implemented)
    ufbxt_assert(!success);
    ufbxt_assert(error.type == UFBX_ERROR_UNKNOWN);
    
    ufbx_free_export_scene(scene);
}
#endif

// Test complex scene with multiple elements
UFBXT_TEST(test_export_complex_scene)
#if UFBXT_IMPL
{
    ufbx_export_scene *scene = ufbx_create_scene(NULL);
    ufbxt_assert(scene != NULL);
    
    // Create scene hierarchy
    ufbx_node *root = ufbx_add_node(scene, "ComplexScene", NULL);
    ufbxt_assert(root != NULL);
    
    // Add multiple mesh nodes
    for (int i = 0; i < 3; i++) {
        char node_name[32];
        char mesh_name[32];
        char material_name[32];
        
        snprintf(node_name, sizeof(node_name), "Node%d", i);
        snprintf(mesh_name, sizeof(mesh_name), "Mesh%d", i);
        snprintf(material_name, sizeof(material_name), "Material%d", i);
        
        ufbx_node *node = ufbx_add_node(scene, node_name, root);
        ufbxt_assert(node != NULL);
        
        ufbx_mesh *mesh = ufbx_add_mesh(scene, mesh_name);
        ufbxt_assert(mesh != NULL);
        
        ufbx_material *material = ufbx_add_material(scene, material_name);
        ufbxt_assert(material != NULL);
        
        // Set different transforms for each node
        ufbx_transform transform = {
            {(float)i * 2.0f, 0, 0},  // translation
            {0, 0, 0, 1},             // rotation
            {1, 1, 1}                 // scale
        };
        
        ufbx_error error;
        ufbx_set_node_transform(node, &transform, &error);
        ufbxt_assert(error.type == UFBX_ERROR_NONE);
        
        // Create simple quad for each mesh
        ufbx_vec3 vertices[] = {
            {-0.5f, -0.5f, 0}, {0.5f, -0.5f, 0}, {0.5f, 0.5f, 0}, {-0.5f, 0.5f, 0}
        };
        uint32_t indices[] = {0, 1, 2, 2, 3, 0};
        
        bool success = ufbx_set_mesh_vertices(mesh, vertices, 4, &error);
        ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
        
        success = ufbx_set_mesh_indices(mesh, indices, 6, &error);
        ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
        
        success = ufbx_attach_mesh_to_node(node, mesh, &error);
        ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
        
        success = ufbx_attach_material_to_mesh(mesh, material, 0, &error);
        ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    }
    
    // Test export size calculation for complex scene
    ufbx_error error;
    size_t export_size = ufbx_get_export_size(scene, NULL, &error);
    ufbxt_assert(export_size > 0);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    ufbx_free_export_scene(scene);
}
#endif

// Test comprehensive scene building workflow
UFBXT_TEST(test_export_complete_workflow)
#if UFBXT_IMPL
{
    // Create scene with all components
    ufbx_export_scene *scene = ufbx_create_scene(NULL);
    ufbxt_assert(scene != NULL);
    
    // Create hierarchy
    ufbx_node *root_node = ufbx_add_node(scene, "Root", NULL);
    ufbxt_assert(root_node != NULL);
    
    ufbx_node *child_node = ufbx_add_node(scene, "Child", root_node);
    ufbxt_assert(child_node != NULL);
    
    // Create mesh with full data
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "TestMesh");
    ufbxt_assert(mesh != NULL);
    
    ufbx_vec3 vertices[] = {
        {0.0f, 1.0f, 0.0f}, {-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}
    };
    uint32_t indices[] = {0, 1, 2};
    ufbx_vec3 normals[] = {
        {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}
    };
    ufbx_vec2 uvs[] = {
        {0.5f, 1.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}
    };
    
    ufbx_error error;
    bool success = ufbx_set_mesh_vertices(mesh, vertices, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_mesh_indices(mesh, indices, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_mesh_normals(mesh, normals, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_mesh_uvs(mesh, uvs, 3, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    // Attach mesh to node
    success = ufbx_attach_mesh_to_node(child_node, mesh, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    // Create and configure material
    ufbx_material *material = ufbx_add_material(scene, "TestMaterial");
    ufbxt_assert(material != NULL);
    
    success = ufbx_set_material_albedo(material, 0.8f, 0.2f, 0.1f, 1.0f, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_material_metallic_roughness(material, 0.7f, 0.3f, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    success = ufbx_set_material_emission(material, 0.1f, 0.1f, 0.2f, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    // Attach material to mesh
    success = ufbx_attach_material_to_mesh(mesh, material, 0, &error);
    ufbxt_assert(success && error.type == UFBX_ERROR_NONE);
    
    // Test export size calculation
    size_t export_size = ufbx_get_export_size(scene, NULL, &error);
    ufbxt_assert(export_size > 0);
    ufbxt_assert(error.type == UFBX_ERROR_NONE);
    
    // Test export attempts (expected to fail until writer is implemented)
    ufbx_export_opts opts = {0};
    opts.ascii_format = true;
    opts.fbx_version = 7400;
    
    success = ufbx_export_to_file(scene, "test_output.fbx", &opts, &error);
    ufbxt_assert(!success);
    ufbxt_assert(error.type == UFBX_ERROR_UNKNOWN); // Not implemented yet
    
    char buffer[1024];
    size_t written = ufbx_export_to_memory(scene, buffer, sizeof(buffer), &opts, &error);
    ufbxt_assert(written == 0);
    ufbxt_assert(error.type == UFBX_ERROR_UNKNOWN); // Not implemented yet
    
    ufbx_free_export_scene(scene);
}
#endif

// Test memory management
UFBXT_TEST(test_export_memory_management)
#if UFBXT_IMPL
{
    // Test multiple scene creation and cleanup
    for (int i = 0; i < 10; i++) {
        ufbx_export_scene *scene = ufbx_create_scene(NULL);
        ufbxt_assert(scene != NULL);
        
        // Add some content
        ufbx_node *node = ufbx_add_node(scene, "TestNode", NULL);
        ufbxt_assert(node != NULL);
        
        ufbx_mesh *mesh = ufbx_add_mesh(scene, "TestMesh");
        ufbxt_assert(mesh != NULL);
        
        ufbx_material *material = ufbx_add_material(scene, "TestMaterial");
        ufbxt_assert(material != NULL);
        
        // Clean up
        ufbx_free_export_scene(scene);
    }
}
#endif

#endif // UFBXT_TEST_FBX_EXPORT_H
