#include "scene_copy_common.h"
#include <stdio.h>
#include <stdlib.h>

// Copy materials with comprehensive property support
bool copy_materials(ufbx_scene *source_scene, ufbx_export_scene *export_scene,
                    material_mapping **material_mappings)
{
    printf("  Copying %zu materials...\n", source_scene->materials.count);
    
    if (source_scene->materials.count == 0) {
        *material_mappings = NULL;
        return true;
    }
    
    *material_mappings = malloc(source_scene->materials.count * sizeof(material_mapping));
    if (!*material_mappings) {
        fprintf(stderr, "Failed to allocate material mapping array\n");
        return false;
    }
    
    for (size_t i = 0; i < source_scene->materials.count; i++) {
        ufbx_material *src_material = source_scene->materials.data[i];
        
        // Create material in export scene
        ufbx_material *export_material = ufbx_add_material(export_scene, src_material->name);
        if (!export_material) {
            printf("    Failed to add material: %s\n", src_material->name.data);
            return false;
        }
        
        // Store mapping
        (*material_mappings)[i].src_material = src_material;
        (*material_mappings)[i].export_material = export_material;
        
        // Copy basic PBR properties
        if (src_material->pbr.base_color.has_value) {
            ufbx_vec4 color = src_material->pbr.base_color.value_vec4;
            ufbx_error albedo_error = {0};
            bool success = ufbx_set_material_albedo(export_material, color.x, color.y, color.z, color.w, &albedo_error);
            if (!success) {
                print_error(&albedo_error, "Failed to set material albedo");
                return false;
            }
        }
        
        if (src_material->pbr.metalness.has_value && src_material->pbr.roughness.has_value) {
            ufbx_error metallic_error = {0};
            bool success = ufbx_set_material_metallic_roughness(export_material, 
                                                               src_material->pbr.metalness.value_real,
                                                               src_material->pbr.roughness.value_real, &metallic_error);
            if (!success) {
                print_error(&metallic_error, "Failed to set material metallic/roughness");
                return false;
            }
        }
        
        if (src_material->pbr.emission_color.has_value) {
            ufbx_vec4 color = src_material->pbr.emission_color.value_vec4;
            ufbx_error emission_error = {0};
            bool success = ufbx_set_material_emission(export_material, color.x, color.y, color.z, &emission_error);
            if (!success) {
                print_error(&emission_error, "Failed to set material emission");
                return false;
            }
        }
        
        printf("    Added material: %s\n", src_material->name.data);
    }
    
    return true;
}
