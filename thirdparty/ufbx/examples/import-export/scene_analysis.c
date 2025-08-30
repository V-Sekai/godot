#include "scene_utils.h"
#include <stdio.h>

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
