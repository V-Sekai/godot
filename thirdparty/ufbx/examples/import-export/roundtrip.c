#include "../../ufbx.h"
#include "../../ufbx_export.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_error(const ufbx_error *error, const char *description)
{
    char buffer[1024];
    ufbx_format_error(buffer, sizeof(buffer), error);
    fprintf(stderr, "Error: %s\n%s\n", description, buffer);
}

void print_warnings(ufbx_scene *scene)
{
    if (scene->metadata.warnings.count > 0) {
        printf("Warnings during load:\n");
        for (size_t i = 0; i < scene->metadata.warnings.count; i++) {
            ufbx_warning warning = scene->metadata.warnings.data[i];
            printf("  %s", warning.description.data);
            if (warning.count > 1) {
                printf(" (x%zu)", warning.count);
            }
            printf("\n");
        }
    }
}

void print_scene_info(ufbx_scene *scene)
{
    printf("Scene info:\n");
    printf("  Nodes: %zu\n", scene->nodes.count);
    printf("  Meshes: %zu\n", scene->meshes.count);
    printf("  Materials: %zu\n", scene->materials.count);
    printf("  Textures: %zu\n", scene->textures.count);
    printf("  Animations: %zu\n", scene->anim_stacks.count);
    printf("  Bones: %zu\n", scene->bones.count);
    printf("  Skin deformers: %zu\n", scene->skin_deformers.count);
    printf("  Blend deformers: %zu\n", scene->blend_deformers.count);
    printf("  Blend channels: %zu\n", scene->blend_channels.count);
    
    // Check for skeletons
    size_t skeleton_count = 0;
    for (size_t i = 0; i < scene->nodes.count; i++) {
        ufbx_node *node = scene->nodes.data[i];
        if (node->bone) {
            skeleton_count++;
        }
    }
    printf("  Skeleton nodes: %zu\n", skeleton_count);
    
    if (scene->meshes.count > 0) {
        printf("Mesh details:\n");
        for (size_t i = 0; i < scene->meshes.count; i++) {
            ufbx_mesh *mesh = scene->meshes.data[i];
            printf("  Mesh %zu: %zu vertices, %zu faces", i, mesh->num_vertices, mesh->num_faces);
            if (mesh->name.length > 0) {
                printf(", name: \"%.*s\"", (int)mesh->name.length, mesh->name.data);
            }
            printf("\n");
            
            // Check skin deformers
            if (mesh->skin_deformers.count > 0) {
                printf("    Skin deformers: %zu\n", mesh->skin_deformers.count);
                for (size_t j = 0; j < mesh->skin_deformers.count; j++) {
                    ufbx_skin_deformer *skin = mesh->skin_deformers.data[j];
                    printf("      Skin %zu: %zu clusters, %zu weights\n", j, skin->clusters.count, skin->weights.count);
                }
            }
            
            // Check blend deformers
            if (mesh->blend_deformers.count > 0) {
                printf("    Blend deformers: %zu\n", mesh->blend_deformers.count);
                for (size_t j = 0; j < mesh->blend_deformers.count; j++) {
                    ufbx_blend_deformer *blend = mesh->blend_deformers.data[j];
                    printf("      Blend %zu: %zu channels\n", j, blend->channels.count);
                }
            }
        }
    }
}

// Copy scene data from imported scene to export scene
bool copy_scene_data(ufbx_scene *source_scene, ufbx_export_scene *export_scene)
{
    printf("Copying scene data from imported scene to export scene...\n");
    ufbx_error error;
    
    // Copy all nodes and build hierarchy
    printf("  Copying %zu nodes...\n", source_scene->nodes.count);
    for (size_t i = 0; i < source_scene->nodes.count; i++) {
        ufbx_node *src_node = source_scene->nodes.data[i];
        
        // Skip root node as it's automatically created
        if (!src_node->parent) {
            continue;
        }
        
        // Find parent in export scene (or NULL for root children)
        ufbx_node *parent_node = NULL;
        if (src_node->parent && src_node->parent->parent) {
            // TODO: Look up parent by name/ID - for now just use NULL (direct root children)
            parent_node = NULL;
        }
        
        // Create node in export scene
        ufbx_node *export_node = ufbx_add_node(export_scene, src_node->name.data, parent_node);
        if (!export_node) {
            printf("    Failed to add node: %s\n", src_node->name.data);
            return false;
        }
        
        // Copy transform
        ufbx_set_node_transform(export_node, &src_node->local_transform, &error);
        if (error.type != UFBX_ERROR_NONE) {
            print_error(&error, "Failed to set node transform");
            return false;
        }
        
        printf("    Added node: %s\n", src_node->name.data);
    }
    
    // Copy all meshes
    printf("  Copying %zu meshes...\n", source_scene->meshes.count);
    for (size_t i = 0; i < source_scene->meshes.count; i++) {
        ufbx_mesh *src_mesh = source_scene->meshes.data[i];
        
        // Create mesh in export scene
        ufbx_mesh *export_mesh = ufbx_add_mesh(export_scene, src_mesh->name.data);
        if (!export_mesh) {
            printf("    Failed to add mesh: %s\n", src_mesh->name.data);
            return false;
        }
        
        // Copy vertex data
        if (src_mesh->vertices.count > 0) {
            bool success = ufbx_set_mesh_vertices(export_mesh, src_mesh->vertices.data, 
                                                 src_mesh->vertices.count, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh vertices");
                return false;
            }
        }
        
        // Copy face/index data
        if (src_mesh->vertex_indices.count > 0 && src_mesh->num_faces > 0) {
            bool success = ufbx_set_mesh_faces(export_mesh, src_mesh->vertex_indices.data,
                                              src_mesh->vertex_indices.count,
                                              src_mesh->faces.data, src_mesh->num_faces, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh faces");
                return false;
            }
        }
        
        // Copy normals if present
        if (src_mesh->vertex_normal.exists && src_mesh->vertex_normal.values.count > 0) {
            bool success = ufbx_set_mesh_normals(export_mesh, src_mesh->vertex_normal.values.data,
                                                 src_mesh->vertex_normal.values.count, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh normals");
                return false;
            }
        }
        
        // Copy UVs if present
        if (src_mesh->vertex_uv.exists && src_mesh->vertex_uv.values.count > 0) {
            bool success = ufbx_set_mesh_uvs(export_mesh, src_mesh->vertex_uv.values.data,
                                             src_mesh->vertex_uv.values.count, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh UVs");
                return false;
            }
        }
        
        printf("    Added mesh: %s (%zu vertices, %zu faces)\n", 
               src_mesh->name.data, src_mesh->vertices.count, src_mesh->num_faces);
    }
    
    // Copy all materials
    printf("  Copying %zu materials...\n", source_scene->materials.count);
    for (size_t i = 0; i < source_scene->materials.count; i++) {
        ufbx_material *src_material = source_scene->materials.data[i];
        
        // Create material in export scene
        ufbx_material *export_material = ufbx_add_material(export_scene, src_material->name.data);
        if (!export_material) {
            printf("    Failed to add material: %s\n", src_material->name.data);
            return false;
        }
        
        // Copy material properties
        if (src_material->pbr.base_color.has_value) {
            ufbx_vec4 color = src_material->pbr.base_color.value_vec4;
            bool success = ufbx_set_material_albedo(export_material, color.x, color.y, color.z, color.w, &error);
            if (!success) {
                print_error(&error, "Failed to set material albedo");
                return false;
            }
        }
        
        if (src_material->pbr.metalness.has_value && src_material->pbr.roughness.has_value) {
            bool success = ufbx_set_material_metallic_roughness(export_material, 
                                                               src_material->pbr.metalness.value_real,
                                                               src_material->pbr.roughness.value_real, &error);
            if (!success) {
                print_error(&error, "Failed to set material metallic/roughness");
                return false;
            }
        }
        
        printf("    Added material: %s\n", src_material->name.data);
    }
    
    printf("Scene data copied successfully!\n");
    return true;
}

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
