#include "scene_utils.h"
#include <stdio.h>
#include <stdlib.h>

// Generate a minimal synthetic FBX with just root + one node
bool generate_minimal_node(const char *output_file)
{
    printf("Generating minimal node FBX: %s\n", output_file);
    
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *scene = ufbx_create_scene(&create_opts);
    if (!scene) {
        fprintf(stderr, "Failed to create export scene\n");
        return false;
    }
    
    // Create just one child node
    ufbx_node *test_node = ufbx_add_node(scene, "TestNode", NULL);
    if (!test_node) {
        fprintf(stderr, "Failed to add test node\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Set a simple transform
    ufbx_transform transform = { 0 };
    transform.translation.x = 1.0f;
    transform.translation.y = 2.0f;
    transform.translation.z = 3.0f;
    transform.rotation.w = 1.0f;
    transform.scale.x = transform.scale.y = transform.scale.z = 1.0f;
    
    ufbx_error error = { 0 };
    ufbx_set_node_transform(test_node, &transform, &error);
    if (error.type != UFBX_ERROR_NONE) {
        print_error(&error, "Failed to set node transform");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Export with ASCII format
    ufbx_export_opts export_opts = {
        .ascii_format = true,
        .fbx_version = 7400,
    };
    
    bool success = ufbx_export_to_file(scene, output_file, &export_opts, &error);
    if (!success) {
        print_error(&error, "Failed to export minimal FBX");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    ufbx_free_export_scene(scene);
    printf("✓ Generated minimal node FBX successfully\n");
    return true;
}

// Generate synthetic FBX with mesh
bool generate_simple_mesh(const char *output_file)
{
    printf("Generating simple mesh FBX: %s\n", output_file);
    
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *scene = ufbx_create_scene(&create_opts);
    if (!scene) {
        fprintf(stderr, "Failed to create export scene\n");
        return false;
    }
    
    // Create mesh node
    ufbx_node *mesh_node = ufbx_add_node(scene, "MeshNode", NULL);
    if (!mesh_node) {
        fprintf(stderr, "Failed to add mesh node\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Create a simple cube mesh
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "CubeMesh");
    if (!mesh) {
        fprintf(stderr, "Failed to add mesh\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Simple cube vertices
    ufbx_vec3 vertices[8] = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1,  1}, {1, -1,  1}, {1, 1,  1}, {-1, 1,  1}
    };
    
    ufbx_error error = { 0 };
    bool success = ufbx_set_mesh_vertices(mesh, vertices, 8, &error);
    if (!success) {
        print_error(&error, "Failed to set mesh vertices");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Simple cube faces (6 quads)
    uint32_t indices[24] = {
        0,1,2,3, // Front face
        4,7,6,5, // Back face
        0,4,5,1, // Bottom face
        2,6,7,3, // Top face
        0,3,7,4, // Left face
        1,5,6,2  // Right face
    };
    
    success = ufbx_set_mesh_indices(mesh, indices, 24, 4, &error);
    if (!success) {
        print_error(&error, "Failed to set mesh indices");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Attach mesh to node
    success = ufbx_attach_mesh_to_node(mesh_node, mesh, &error);
    if (!success) {
        print_error(&error, "Failed to attach mesh to node");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Export
    ufbx_export_opts export_opts = {
        .ascii_format = true,
        .fbx_version = 7400,
    };
    
    success = ufbx_export_to_file(scene, output_file, &export_opts, &error);
    if (!success) {
        print_error(&error, "Failed to export mesh FBX");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    ufbx_free_export_scene(scene);
    printf("✓ Generated simple mesh FBX successfully\n");
    return true;
}

// Generate synthetic FBX with mesh + material
bool generate_mesh_material(const char *output_file)
{
    printf("Generating mesh + material FBX: %s\n", output_file);
    
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *scene = ufbx_create_scene(&create_opts);
    if (!scene) {
        fprintf(stderr, "Failed to create export scene\n");
        return false;
    }
    
    // Create material first
    ufbx_material *material = ufbx_add_material(scene, "TestMaterial");
    if (!material) {
        fprintf(stderr, "Failed to add material\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Set material properties
    ufbx_error error = { 0 };
    bool success = ufbx_set_material_albedo(material, 0.8f, 0.2f, 0.2f, 1.0f, &error);
    if (!success) {
        print_error(&error, "Failed to set material albedo");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Create mesh
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "CubeMesh");
    if (!mesh) {
        fprintf(stderr, "Failed to add mesh\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Simple cube vertices
    ufbx_vec3 vertices[8] = {
        {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
        {-1, -1,  1}, {1, -1,  1}, {1, 1,  1}, {-1, 1,  1}
    };
    
    success = ufbx_set_mesh_vertices(mesh, vertices, 8, &error);
    if (!success) {
        print_error(&error, "Failed to set mesh vertices");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    uint32_t indices[24] = {
        0,1,2,3, 4,7,6,5, 0,4,5,1, 2,6,7,3, 0,3,7,4, 1,5,6,2
    };
    
    success = ufbx_set_mesh_indices(mesh, indices, 24, 4, &error);
    if (!success) {
        print_error(&error, "Failed to set mesh indices");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Attach material to mesh
    success = ufbx_attach_material_to_mesh(mesh, material, 0, &error);
    if (!success) {
        print_error(&error, "Failed to attach material to mesh");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Create mesh node and attach mesh
    ufbx_node *mesh_node = ufbx_add_node(scene, "MeshNode", NULL);
    if (!mesh_node) {
        fprintf(stderr, "Failed to add mesh node\n");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    success = ufbx_attach_mesh_to_node(mesh_node, mesh, &error);
    if (!success) {
        print_error(&error, "Failed to attach mesh to node");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    // Export
    ufbx_export_opts export_opts = {
        .ascii_format = true,
        .fbx_version = 7400,
    };
    
    success = ufbx_export_to_file(scene, output_file, &export_opts, &error);
    if (!success) {
        print_error(&error, "Failed to export mesh+material FBX");
        ufbx_free_export_scene(scene);
        return false;
    }
    
    ufbx_free_export_scene(scene);
    printf("✓ Generated mesh + material FBX successfully\n");
    return true;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <test_type>\n", argv[0]);
        fprintf(stderr, "test_type: minimal, mesh, material\n");
        return 1;
    }
    
    const char *test_type = argv[1];
    
    if (strcmp(test_type, "minimal") == 0) {
        return generate_minimal_node("data/synthetic/minimal_node.fbx") ? 0 : 1;
    } else if (strcmp(test_type, "mesh") == 0) {
        return generate_simple_mesh("data/synthetic/simple_mesh.fbx") ? 0 : 1;
    } else if (strcmp(test_type, "material") == 0) {
        return generate_mesh_material("data/synthetic/mesh_material.fbx") ? 0 : 1;
    } else {
        fprintf(stderr, "Unknown test type: %s\n", test_type);
        return 1;
    }
}
