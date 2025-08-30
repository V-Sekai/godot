#include "scene_copy_common.h"
#include <stdio.h>
#include <stdlib.h>

// Copy scene data from imported scene to export scene
bool copy_scene_data(ufbx_scene *source_scene, ufbx_export_scene *export_scene)
{
    printf("Copying scene data from imported scene to export scene...\n");
    
    node_mapping *node_mappings = NULL;
    material_mapping *material_mappings = NULL;
    mesh_mapping *mesh_mappings = NULL;
    anim_stack_mapping *stack_mappings = NULL;
    anim_layer_mapping *layer_mappings = NULL;
    skin_mapping *skin_mappings = NULL;
    blend_mapping *blend_mappings = NULL;
    
    size_t num_node_mappings = 0;
    
    // Copy all elements in REVERSE dependency order (matching ufbx internal processing)
    // 1. Create nodes first (needed for references, but don't set up hierarchy yet)
    if (!copy_nodes(source_scene, export_scene, &node_mappings, &num_node_mappings)) goto cleanup;
    
    // 2. Create animations (enhanced data preservation)
    if (!copy_animations(source_scene, export_scene, node_mappings, num_node_mappings, &stack_mappings, &layer_mappings)) goto cleanup;
    
    // 3. Create materials (before meshes reference them)
    if (!copy_materials(source_scene, export_scene, &material_mappings)) goto cleanup;
    
    // 4. Create meshes (before deformers and attachments)
    if (!copy_meshes(source_scene, export_scene, &mesh_mappings)) goto cleanup;
    
    // 5. Create deformers (now that meshes exist)
    if (!copy_blend_deformers(source_scene, export_scene, mesh_mappings, &blend_mappings)) goto cleanup;
    if (!copy_skin_deformers(source_scene, export_scene, node_mappings, num_node_mappings, mesh_mappings, &skin_mappings)) goto cleanup;
    
    // 6. Attach materials to meshes (now that both exist)
    if (!attach_materials_to_meshes(source_scene, mesh_mappings, material_mappings)) goto cleanup;
    
    // 7. Set up node hierarchy (after all objects are created)
    if (!setup_node_hierarchy(node_mappings, num_node_mappings)) goto cleanup;
    
    // 8. Final element attachments (connections last, as in ufbx processing)
    if (!attach_elements_to_nodes(source_scene, node_mappings, num_node_mappings, mesh_mappings)) goto cleanup;
    
    // 9. Add specialized features for comprehensive validation support
    // TODO: Re-enable after adding scene_copy_elements.c to Makefile
    // if (!copy_lights_and_cameras(source_scene, export_scene, node_mappings, num_node_mappings)) goto cleanup;
    // if (!copy_constraints(source_scene, export_scene, node_mappings, num_node_mappings)) goto cleanup;
    // if (!copy_user_properties(source_scene, export_scene, node_mappings, num_node_mappings)) goto cleanup;
    
    printf("Scene data copied successfully!\n");
    
    // Cleanup
    free(node_mappings);
    free(material_mappings);
    free(mesh_mappings);
    free(stack_mappings);
    free(layer_mappings);
    free(skin_mappings);
    free(blend_mappings);
    return true;

cleanup:
    free(node_mappings);
    free(material_mappings);
    free(mesh_mappings);
    free(stack_mappings);
    free(layer_mappings);
    free(skin_mappings);
    free(blend_mappings);
    return false;
}
