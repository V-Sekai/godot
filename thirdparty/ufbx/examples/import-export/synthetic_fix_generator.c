#include "scene_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Generate synthetic pivot animation (for maya_pivot_* failures)
bool generate_pivot_animation(const char *output_file)
{
    printf("Generating synthetic pivot animation: %s\n", output_file);
    
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *scene = ufbx_create_scene(&create_opts);
    if (!scene) return false;
    
    // Create node with pivot transform
    ufbx_node *node = ufbx_add_node(scene, "PivotNode", NULL);
    if (!node) goto cleanup;
    
    // Set transform with offset (pivot-like)
    ufbx_transform transform = { 0 };
    transform.translation.x = 0.0f;
    transform.translation.y = 0.0f; 
    transform.translation.z = 0.0f;
    transform.rotation.w = 1.0f;
    transform.scale.x = transform.scale.y = transform.scale.z = 1.0f;
    
    ufbx_error error = { 0 };
    ufbx_set_node_transform(node, &transform, &error);
    
    // Export - NO animation data to test if pivot transforms alone work
    ufbx_export_opts export_opts = { .ascii_format = true, .fbx_version = 7400 };
    bool success = ufbx_export_to_file(scene, output_file, &export_opts, &error);
    ufbx_free_export_scene(scene);
    return success;
    
cleanup:
    ufbx_free_export_scene(scene);
    return false;
}

// Generate synthetic skin deformer (for maya_instanced_skin_* failures)  
bool generate_skin_animation(const char *output_file)
{
    printf("Generating synthetic skin deformer: %s\n", output_file);
    
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *scene = ufbx_create_scene(&create_opts);
    if (!scene) return false;
    
    // Create bone node
    ufbx_node *bone_node = ufbx_add_node(scene, "Bone", NULL);
    if (!bone_node) goto cleanup;
    
    // Create mesh node  
    ufbx_node *mesh_node = ufbx_add_node(scene, "MeshNode", NULL);
    if (!mesh_node) goto cleanup;
    
    // Create simple mesh
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "SkinMesh");
    if (!mesh) goto cleanup;
    
    // Simple 4 vertex quad
    ufbx_vec3 vertices[4] = {{-1,-1,0}, {1,-1,0}, {1,1,0}, {-1,1,0}};
    ufbx_error error = { 0 };
    if (!ufbx_set_mesh_vertices(mesh, vertices, 4, &error)) goto cleanup;
    
    uint32_t indices[4] = {0,1,2,3};
    if (!ufbx_set_mesh_indices(mesh, indices, 4, 4, &error)) goto cleanup;
    
    // Create skin deformer
    ufbx_skin_deformer *skin = ufbx_add_skin_deformer(scene, "TestSkin");
    if (!skin) goto cleanup;
    
    // Create skin cluster
    ufbx_skin_cluster *cluster = ufbx_add_skin_cluster(scene, skin, bone_node, "TestCluster");
    if (!cluster) goto cleanup;
    
    // Set simple weights
    uint32_t cluster_vertices[4] = {0,1,2,3};
    ufbx_real cluster_weights[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (!ufbx_set_skin_cluster_vertices(cluster, cluster_vertices, cluster_weights, 4, &error)) goto cleanup;
    
    // Set identity transforms  
    ufbx_matrix identity = ufbx_identity_matrix;
    if (!ufbx_set_skin_cluster_transform(cluster, &identity, &identity, &error)) goto cleanup;
    
    // Attach skin to mesh
    if (!ufbx_attach_skin_to_mesh(mesh, skin, &error)) goto cleanup;
    if (!ufbx_attach_mesh_to_node(mesh_node, mesh, &error)) goto cleanup;
    
    // Export
    ufbx_export_opts export_opts = { .ascii_format = true, .fbx_version = 7400 };
    bool success = ufbx_export_to_file(scene, output_file, &export_opts, &error);
    ufbx_free_export_scene(scene);
    return success;
    
cleanup:
    ufbx_free_export_scene(scene);
    return false;
}

// Generate synthetic blend shape (for maya_blend_shape_* failures)
bool generate_blend_animation(const char *output_file)
{
    printf("Generating synthetic blend shape: %s\n", output_file);
    
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *scene = ufbx_create_scene(&create_opts);
    if (!scene) return false;
    
    // Create mesh node
    ufbx_node *mesh_node = ufbx_add_node(scene, "MeshNode", NULL);
    if (!mesh_node) goto cleanup;
    
    // Create base mesh
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "BlendMesh");
    if (!mesh) goto cleanup;
    
    // Simple 4 vertex quad  
    ufbx_vec3 vertices[4] = {{-1,-1,0}, {1,-1,0}, {1,1,0}, {-1,1,0}};
    ufbx_error error = { 0 };
    if (!ufbx_set_mesh_vertices(mesh, vertices, 4, &error)) goto cleanup;
    
    uint32_t indices[4] = {0,1,2,3};
    if (!ufbx_set_mesh_indices(mesh, indices, 4, 4, &error)) goto cleanup;
    
    // Create blend deformer
    ufbx_blend_deformer *blend = ufbx_add_blend_deformer(scene, "TestBlend");
    if (!blend) goto cleanup;
    
    // Create blend channel
    ufbx_blend_channel *channel = ufbx_add_blend_channel(scene, blend, "TestChannel");
    if (!channel) goto cleanup;
    
    // Attach blend to mesh
    if (!ufbx_attach_blend_to_mesh(mesh, blend, &error)) goto cleanup;
    if (!ufbx_attach_mesh_to_node(mesh_node, mesh, &error)) goto cleanup;
    
    // Export
    ufbx_export_opts export_opts = { .ascii_format = true, .fbx_version = 7400 };
    bool success = ufbx_export_to_file(scene, output_file, &export_opts, &error);
    ufbx_free_export_scene(scene);
    return success;
    
cleanup:
    ufbx_free_export_scene(scene);
    return false;
}

// Generate synthetic complex scene combining elements
bool generate_complex_scene(const char *output_file)
{
    printf("Generating synthetic complex scene: %s\n", output_file);
    
    ufbx_export_opts create_opts = { 0 };
    ufbx_export_scene *scene = ufbx_create_scene(&create_opts);
    if (!scene) return false;
    
    // Create material
    ufbx_material *material = ufbx_add_material(scene, "TestMaterial");
    if (!material) goto cleanup;
    
    ufbx_error error = { 0 };
    ufbx_set_material_albedo(material, 0.8f, 0.2f, 0.2f, 1.0f, &error);
    
    // Create mesh with material
    ufbx_mesh *mesh = ufbx_add_mesh(scene, "TestMesh");
    if (!mesh) goto cleanup;
    
    ufbx_vec3 vertices[4] = {{-1,-1,0}, {1,-1,0}, {1,1,0}, {-1,1,0}};
    if (!ufbx_set_mesh_vertices(mesh, vertices, 4, &error)) goto cleanup;
    
    uint32_t indices[4] = {0,1,2,3};
    if (!ufbx_set_mesh_indices(mesh, indices, 4, 4, &error)) goto cleanup;
    if (!ufbx_attach_material_to_mesh(mesh, material, 0, &error)) goto cleanup;
    
    // Create mesh node
    ufbx_node *mesh_node = ufbx_add_node(scene, "MeshNode", NULL);
    if (!mesh_node) goto cleanup;
    if (!ufbx_attach_mesh_to_node(mesh_node, mesh, &error)) goto cleanup;
    
    // Create bone node
    ufbx_node *bone_node = ufbx_add_node(scene, "Bone", NULL);
    if (!bone_node) goto cleanup;
    
    ufbx_bone *bone = ufbx_add_bone(scene, bone_node, "TestBone");
    if (!bone) goto cleanup;
    ufbx_set_bone_properties(bone, 1.0f, &error);
    
    // Create skin deformer  
    ufbx_skin_deformer *skin = ufbx_add_skin_deformer(scene, "TestSkin");
    if (!skin) goto cleanup;
    
    ufbx_skin_cluster *cluster = ufbx_add_skin_cluster(scene, skin, bone_node, "TestCluster");
    if (!cluster) goto cleanup;
    
    uint32_t cluster_vertices[4] = {0,1,2,3};
    ufbx_real cluster_weights[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (!ufbx_set_skin_cluster_vertices(cluster, cluster_vertices, cluster_weights, 4, &error)) goto cleanup;
    
    ufbx_matrix identity = ufbx_identity_matrix;
    if (!ufbx_set_skin_cluster_transform(cluster, &identity, &identity, &error)) goto cleanup;
    if (!ufbx_attach_skin_to_mesh(mesh, skin, &error)) goto cleanup;
    
    // Create animation stack only (no layers/values/curves since those break validation)
    ufbx_anim_stack *stack = ufbx_add_animation(scene, "TestAnimation");
    if (!stack) goto cleanup;
    if (!ufbx_set_anim_stack_time_range(stack, 0.0, 1.0, &error)) goto cleanup;
    
    // Export
    ufbx_export_opts export_opts = { .ascii_format = true, .fbx_version = 7400 };
    bool success = ufbx_export_to_file(scene, output_file, &export_opts, &error);
    ufbx_free_export_scene(scene);
    return success;
    
cleanup:
    ufbx_free_export_scene(scene);
    return false;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <test_type>\n", argv[0]);
        fprintf(stderr, "test_type: pivot, skin, blend, complex\n");
        return 1;
    }
    
    const char *test_type = argv[1];
    
    if (strcmp(test_type, "pivot") == 0) {
        return generate_pivot_animation("data/synthetic/synthetic_pivot.fbx") ? 0 : 1;
    } else if (strcmp(test_type, "skin") == 0) {
        return generate_skin_animation("data/synthetic/synthetic_skin.fbx") ? 0 : 1;
    } else if (strcmp(test_type, "blend") == 0) {
        return generate_blend_animation("data/synthetic/synthetic_blend.fbx") ? 0 : 1;
    } else if (strcmp(test_type, "complex") == 0) {
        return generate_complex_scene("data/synthetic/synthetic_complex.fbx") ? 0 : 1;
    } else {
        fprintf(stderr, "Unknown test type: %s\n", test_type);
        return 1;
    }
}
