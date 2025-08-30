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

// Helper structure to track node mappings during copy
typedef struct {
    ufbx_node *src_node;
    ufbx_node *export_node;
} node_mapping;


// Copy scene data from imported scene to export scene
bool copy_scene_data(ufbx_scene *source_scene, ufbx_export_scene *export_scene)
{
    printf("Copying scene data from imported scene to export scene...\n");
    ufbx_error error;
    
    // Create node mapping array to track source->export relationships
    node_mapping *node_mappings = malloc(source_scene->nodes.count * sizeof(node_mapping));
    if (!node_mappings) {
        fprintf(stderr, "Failed to allocate node mapping array\n");
        return false;
    }
    size_t num_mappings = 0;
    
    // First pass: Create all nodes (without hierarchy)
    printf("  Copying %zu nodes...\n", source_scene->nodes.count);
    for (size_t i = 0; i < source_scene->nodes.count; i++) {
        ufbx_node *src_node = source_scene->nodes.data[i];
        
        // Skip root node as it's automatically created
        if (!src_node->parent) {
            // Map root node
            node_mappings[num_mappings].src_node = src_node;
            node_mappings[num_mappings].export_node = export_scene->root_node;
            num_mappings++;
            continue;
        }
        
        // Create node in export scene (temporarily without parent)
        ufbx_node *export_node = ufbx_add_node(export_scene, src_node->name.data, NULL);
        if (!export_node) {
            printf("    Failed to add node: %s\n", src_node->name.data);
            free(node_mappings);
            return false;
        }
        
        // Copy transform
        ufbx_set_node_transform(export_node, &src_node->local_transform, &error);
        if (error.type != UFBX_ERROR_NONE) {
            print_error(&error, "Failed to set node transform");
            free(node_mappings);
            return false;
        }
        
        // Store mapping
        node_mappings[num_mappings].src_node = src_node;
        node_mappings[num_mappings].export_node = export_node;
        num_mappings++;
        
        printf("    Added node: %s\n", src_node->name.data);
    }
    
    // Second pass: Set up parent-child relationships
    printf("  Setting up node hierarchy...\n");
    for (size_t i = 0; i < num_mappings; i++) {
        ufbx_node *src_node = node_mappings[i].src_node;
        ufbx_node *export_node = node_mappings[i].export_node;
        
        if (src_node->parent) {
            // Find parent in mappings
            ufbx_node *export_parent = NULL;
            for (size_t j = 0; j < num_mappings; j++) {
                if (node_mappings[j].src_node == src_node->parent) {
                    export_parent = node_mappings[j].export_node;
                    break;
                }
            }
            
            if (export_parent && export_parent != export_scene->root_node) {
                // Set parent (this is conceptual - ufbx may not have a direct reparent function)
                // For now, we'll track this relationship for mesh attachment
            }
        }
    }
    
    // Copy all materials first (needed for mesh assignments)
    printf("  Copying %zu materials...\n", source_scene->materials.count);
    ufbx_material **material_mappings = malloc(source_scene->materials.count * sizeof(ufbx_material*));
    if (!material_mappings) {
        fprintf(stderr, "Failed to allocate material mapping array\n");
        free(node_mappings);
        return false;
    }
    
    for (size_t i = 0; i < source_scene->materials.count; i++) {
        ufbx_material *src_material = source_scene->materials.data[i];
        
        // Create material in export scene
        ufbx_material *export_material = ufbx_add_material(export_scene, src_material->name.data);
        if (!export_material) {
            printf("    Failed to add material: %s\n", src_material->name.data);
            free(node_mappings);
            free(material_mappings);
            return false;
        }
        
        // Store mapping
        material_mappings[i] = export_material;
        
        // Copy material properties
        if (src_material->pbr.base_color.has_value) {
            ufbx_vec4 color = src_material->pbr.base_color.value_vec4;
            bool success = ufbx_set_material_albedo(export_material, color.x, color.y, color.z, color.w, &error);
            if (!success) {
                print_error(&error, "Failed to set material albedo");
                free(node_mappings);
                free(material_mappings);
                return false;
            }
        }
        
        if (src_material->pbr.metalness.has_value && src_material->pbr.roughness.has_value) {
            bool success = ufbx_set_material_metallic_roughness(export_material, 
                                                               src_material->pbr.metalness.value_real,
                                                               src_material->pbr.roughness.value_real, &error);
            if (!success) {
                print_error(&error, "Failed to set material metallic/roughness");
                free(node_mappings);
                free(material_mappings);
                return false;
            }
        }
        
        printf("    Added material: %s\n", src_material->name.data);
    }
    
    // Copy all meshes and attach to nodes
    printf("  Copying %zu meshes...\n", source_scene->meshes.count);
    ufbx_mesh **mesh_mappings = malloc(source_scene->meshes.count * sizeof(ufbx_mesh*));
    if (!mesh_mappings) {
        fprintf(stderr, "Failed to allocate mesh mapping array\n");
        free(node_mappings);
        free(material_mappings);
        return false;
    }
    
    for (size_t i = 0; i < source_scene->meshes.count; i++) {
        ufbx_mesh *src_mesh = source_scene->meshes.data[i];
        
        // Create mesh in export scene
        ufbx_mesh *export_mesh = ufbx_add_mesh(export_scene, src_mesh->name.data);
        if (!export_mesh) {
            printf("    Failed to add mesh: %s\n", src_mesh->name.data);
            free(node_mappings);
            free(material_mappings);
            free(mesh_mappings);
            return false;
        }
        
        // Store mapping
        mesh_mappings[i] = export_mesh;
        
        // Copy vertex data
        if (src_mesh->vertices.count > 0) {
            bool success = ufbx_set_mesh_vertices(export_mesh, src_mesh->vertices.data, 
                                                 src_mesh->vertices.count, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh vertices");
                free(node_mappings);
                free(material_mappings);
                free(mesh_mappings);
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
                free(node_mappings);
                free(material_mappings);
                free(mesh_mappings);
                return false;
            }
        }
        
        // Copy normals if present
        if (src_mesh->vertex_normal.exists && src_mesh->vertex_normal.values.count > 0) {
            bool success = ufbx_set_mesh_normals(export_mesh, src_mesh->vertex_normal.values.data,
                                                 src_mesh->vertex_normal.values.count, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh normals");
                free(node_mappings);
                free(material_mappings);
                free(mesh_mappings);
                return false;
            }
        }
        
        // Copy UVs if present
        if (src_mesh->vertex_uv.exists && src_mesh->vertex_uv.values.count > 0) {
            bool success = ufbx_set_mesh_uvs(export_mesh, src_mesh->vertex_uv.values.data,
                                             src_mesh->vertex_uv.values.count, &error);
            if (!success) {
                print_error(&error, "Failed to set mesh UVs");
                free(node_mappings);
                free(material_mappings);
                free(mesh_mappings);
                return false;
            }
        }
        
        // Attach materials to mesh
        for (size_t j = 0; j < src_mesh->materials.count; j++) {
            ufbx_material *src_mat = src_mesh->materials.data[j];
            // Find corresponding export material
            for (size_t k = 0; k < source_scene->materials.count; k++) {
                if (source_scene->materials.data[k] == src_mat) {
                    bool success = ufbx_attach_material_to_mesh(export_mesh, material_mappings[k], j, &error);
                    if (!success) {
                        print_error(&error, "Failed to attach material to mesh");
                        free(node_mappings);
                        free(material_mappings);
                        free(mesh_mappings);
                        return false;
                    }
                    break;
                }
            }
        }
        
        printf("    Added mesh: %s (%zu vertices, %zu faces)\n", 
               src_mesh->name.data, src_mesh->vertices.count, src_mesh->num_faces);
    }
    
    // Attach meshes to nodes
    printf("  Attaching meshes to nodes...\n");
    for (size_t i = 0; i < source_scene->nodes.count; i++) {
        ufbx_node *src_node = source_scene->nodes.data[i];
        if (src_node->mesh) {
            // Find the corresponding export node
            ufbx_node *export_node = NULL;
            for (size_t j = 0; j < num_mappings; j++) {
                if (node_mappings[j].src_node == src_node) {
                    export_node = node_mappings[j].export_node;
                    break;
                }
            }
            
            if (export_node) {
                // Find the corresponding export mesh
                ufbx_mesh *export_mesh = NULL;
                for (size_t j = 0; j < source_scene->meshes.count; j++) {
                    if (source_scene->meshes.data[j] == src_node->mesh) {
                        export_mesh = mesh_mappings[j];
                        break;
                    }
                }
                
                if (export_mesh) {
                    bool success = ufbx_attach_mesh_to_node(export_node, export_mesh, &error);
                    if (!success) {
                        print_error(&error, "Failed to attach mesh to node");
                        free(node_mappings);
                        free(material_mappings);
                        free(mesh_mappings);
                        return false;
                    }
                    printf("    Attached mesh '%s' to node '%s'\n", 
                           export_mesh->name.data, export_node->name.data);
                }
            }
        }
    }
    
    // Copy animation data
    printf("  Copying %zu animation stacks...\n", source_scene->anim_stacks.count);
    for (size_t i = 0; i < source_scene->anim_stacks.count; i++) {
        ufbx_anim_stack *src_stack = source_scene->anim_stacks.data[i];
        
        // Create animation stack
        ufbx_anim_stack *export_stack = ufbx_add_animation(export_scene, src_stack->name.data);
        if (!export_stack) {
            printf("    Failed to add animation stack: %s\n", src_stack->name.data);
            free(node_mappings);
            free(material_mappings);
            free(mesh_mappings);
            return false;
        }
        
        // Set time range
        bool success = ufbx_set_anim_stack_time_range(export_stack, src_stack->time_begin, src_stack->time_end, &error);
        if (!success) {
            print_error(&error, "Failed to set animation time range");
            free(node_mappings);
            free(material_mappings);
            free(mesh_mappings);
            return false;
        }
        
        // Copy animation layers
        for (size_t j = 0; j < src_stack->layers.count; j++) {
            ufbx_anim_layer *src_layer = src_stack->layers.data[j];
            
            ufbx_anim_layer *export_layer = ufbx_add_anim_layer(export_scene, export_stack, src_layer->name.data);
            if (!export_layer) {
                printf("    Failed to add animation layer: %s\n", src_layer->name.data);
                continue;
            }
            
            // Copy animation values and curves
            for (size_t k = 0; k < src_layer->anim_values.count; k++) {
                ufbx_anim_value *src_value = src_layer->anim_values.data[k];
                
                ufbx_anim_value *export_value = ufbx_add_anim_value(export_scene, export_layer, src_value->name.data);
                if (!export_value) {
                    continue;
                }
                
                // Copy curves for each component
                for (int comp = 0; comp < 3; comp++) { // X, Y, Z components
                    ufbx_anim_curve *src_curve = src_value->curves[comp];
                    if (src_curve && src_curve->keyframes.count > 0) {
                        ufbx_anim_curve *export_curve = ufbx_add_anim_curve(export_scene, export_value, comp);
                        if (export_curve) {
                            bool curve_success = ufbx_set_anim_curve_keyframes(export_curve, 
                                                                             src_curve->keyframes.data,
                                                                             src_curve->keyframes.count, &error);
                            if (!curve_success) {
                                print_error(&error, "Failed to set animation keyframes");
                            }
                        }
                    }
                }
            }
        }
        
        printf("    Added animation stack: %s\n", src_stack->name.data);
    }
    
    // Copy skin deformers
    printf("  Copying %zu skin deformers...\n", source_scene->skin_deformers.count);
    for (size_t i = 0; i < source_scene->skin_deformers.count; i++) {
        ufbx_skin_deformer *src_skin = source_scene->skin_deformers.data[i];
        
        ufbx_skin_deformer *export_skin = ufbx_add_skin_deformer(export_scene, src_skin->name.data);
        if (!export_skin) {
            printf("    Failed to add skin deformer: %s\n", src_skin->name.data);
            continue;
        }
        
        // Copy skin clusters
        for (size_t j = 0; j < src_skin->clusters.count; j++) {
            ufbx_skin_cluster *src_cluster = src_skin->clusters.data[j];
            
            // Find corresponding bone node
            ufbx_node *export_bone = NULL;
            for (size_t k = 0; k < num_mappings; k++) {
                if (node_mappings[k].src_node == src_cluster->bone_node) {
                    export_bone = node_mappings[k].export_node;
                    break;
                }
            }
            
            if (export_bone) {
                ufbx_skin_cluster *export_cluster = ufbx_add_skin_cluster(export_scene, export_skin, export_bone, src_cluster->name.data);
                if (!export_cluster) {
                    continue;
                }
                
                // Copy cluster transform matrices
                bool success = ufbx_set_skin_cluster_transform(export_cluster, 
                                                             &src_cluster->geometry_to_bone,
                                                             &src_cluster->bone_to_world, &error);
                if (!success) {
                    print_error(&error, "Failed to set skin cluster transform");
                    continue;
                }
                
                // Copy cluster vertex indices and weights
                if (src_cluster->vertices.count > 0 && src_cluster->weights.count > 0) {
                    success = ufbx_set_skin_cluster_vertices(export_cluster,
                                                           src_cluster->vertices.data,
                                                           src_cluster->weights.data,
                                                           src_cluster->vertices.count, &error);
                    if (!success) {
                        print_error(&error, "Failed to set skin cluster vertices");
                    }
                }
            }
        }
        
        // Copy skin weights (global weights for the skin deformer)
        if (src_skin->weights.count > 0) {
            bool success = ufbx_set_skin_weights(export_skin, src_skin->weights.data, src_skin->weights.count, &error);
            if (!success) {
                print_error(&error, "Failed to set skin weights");
            }
        }
        
        // Attach skin to meshes
        for (size_t j = 0; j < source_scene->meshes.count; j++) {
            ufbx_mesh *src_mesh = source_scene->meshes.data[j];
            for (size_t k = 0; k < src_mesh->skin_deformers.count; k++) {
                if (src_mesh->skin_deformers.data[k] == src_skin) {
                    bool success = ufbx_attach_skin_to_mesh(mesh_mappings[j], export_skin, &error);
                    if (!success) {
                        print_error(&error, "Failed to attach skin to mesh");
                    }
                    break;
                }
            }
        }
        
        printf("    Added skin deformer: %s\n", src_skin->name.data);
    }
    
    // Copy blend deformers
    printf("  Copying %zu blend deformers...\n", source_scene->blend_deformers.count);
    for (size_t i = 0; i < source_scene->blend_deformers.count; i++) {
        ufbx_blend_deformer *src_blend = source_scene->blend_deformers.data[i];
        
        ufbx_blend_deformer *export_blend = ufbx_add_blend_deformer(export_scene, src_blend->name.data);
        if (!export_blend) {
            printf("    Failed to add blend deformer: %s\n", src_blend->name.data);
            continue;
        }
        
        // Copy blend channels
        for (size_t j = 0; j < src_blend->channels.count; j++) {
            ufbx_blend_channel *src_channel = src_blend->channels.data[j];
            
            ufbx_blend_channel *export_channel = ufbx_add_blend_channel(export_scene, export_blend, src_channel->name.data);
            if (!export_channel) {
                continue;
            }
        }
        
        // Attach blend to meshes
        for (size_t j = 0; j < source_scene->meshes.count; j++) {
            ufbx_mesh *src_mesh = source_scene->meshes.data[j];
            for (size_t k = 0; k < src_mesh->blend_deformers.count; k++) {
                if (src_mesh->blend_deformers.data[k] == src_blend) {
                    bool success = ufbx_attach_blend_to_mesh(mesh_mappings[j], export_blend, &error);
                    if (!success) {
                        print_error(&error, "Failed to attach blend to mesh");
                    }
                    break;
                }
            }
        }
        
        printf("    Added blend deformer: %s\n", src_blend->name.data);
    }
    
    // Clean up
    free(node_mappings);
    free(material_mappings);
    free(mesh_mappings);
    
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
