#include "ufbx_connections.h"

bool ufbx_ascii_write_connections(ufbx_ascii_writer *writer, const ufbx_export_scene *scene) {
    const ufbxi_export_scene *scene_imp = (const ufbxi_export_scene*)scene;
    
    if (!ufbx_ascii_write_node_begin(writer, "Connections")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Connect nodes to root or parent
    for (size_t i = 0; i < scene_imp->num_nodes; i++) {
        const ufbx_node *node = scene_imp->nodes[i];
        if (!node->parent) { // Root level nodes
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, 0)) {
                return false; // Root node ID
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        } else {
            // Connect to parent node
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->parent->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
        
        // Connect mesh to node if it exists
        if (node->mesh) {
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->mesh->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
        
        // Connect bone to node if it exists
        if (node->bone) {
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->bone->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
    }
    
    // Connect materials to nodes (not meshes directly)
    for (size_t i = 0; i < scene_imp->num_nodes; i++) {
        const ufbx_node *node = scene_imp->nodes[i];
        if (node->mesh && node->mesh->materials.count > 0 && node->mesh->materials.data) {
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->mesh->materials.data[0]->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
    }
    
    // Connect animation layers to animation stacks
    for (size_t i = 0; i < scene_imp->num_anim_layers; i++) {
        const ufbx_anim_layer *layer = scene_imp->anim_layers[i];
        
        // Find the parent animation stack
        for (size_t j = 0; j < scene_imp->num_anim_stacks; j++) {
            const ufbx_anim_stack *stack = scene_imp->anim_stacks[j];
            
            // Check if this layer belongs to this stack
            bool belongs_to_stack = false;
            for (size_t k = 0; k < stack->layers.count; k++) {
                if (stack->layers.data[k] == layer) {
                    belongs_to_stack = true;
                    break;
                }
            }
            
            if (belongs_to_stack) {
                if (!ufbx_ascii_write_node_begin(writer, "C")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_string(writer, "OO")) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, layer->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, stack->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_newline(writer)) {
                    return false;
                }
                break;
            }
        }
    }
    
    // Connect animation values to animation layers
    for (size_t i = 0; i < scene_imp->num_anim_layers; i++) {
        const ufbx_anim_layer *layer = scene_imp->anim_layers[i];
        
        // Connect all animation values that belong to this layer
        for (size_t j = 0; j < layer->anim_values.count; j++) {
            const ufbx_anim_value *anim_value = layer->anim_values.data[j];
            
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, anim_value->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, layer->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
    }
    
    // Connect animation curves to animation values
    for (size_t i = 0; i < scene_imp->num_anim_values; i++) {
        const ufbx_anim_value *anim_value = scene_imp->anim_values[i];
        
        // Connect all curves that belong to this animation value
        for (int comp = 0; comp < 3; comp++) {
            if (anim_value->curves[comp]) {
                const ufbx_anim_curve *curve = anim_value->curves[comp];
                
                if (!ufbx_ascii_write_node_begin(writer, "C")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_string(writer, "OP")) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, curve->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, anim_value->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                // Property name for component (d|X, d|Y, d|Z)
                const char *prop_names[] = {"d|X", "d|Y", "d|Z"};
                if (!ufbx_ascii_write_property_string(writer, prop_names[comp])) {
                    return false;
                }
                if (!ufbx_ascii_write_newline(writer)) {
                    return false;
                }
            }
        }
    }
    
    // Connect animation values to node properties
    for (size_t i = 0; i < scene_imp->num_anim_layers; i++) {
        const ufbx_anim_layer *layer = scene_imp->anim_layers[i];
        
        // Connect animation properties
        for (size_t j = 0; j < layer->anim_props.count; j++) {
            const ufbx_anim_prop *anim_prop = &layer->anim_props.data[j];
            
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OP")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, anim_prop->anim_value->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, anim_prop->element->element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, anim_prop->prop_name.data)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
        }
    }
    
    // Connect skin deformers to meshes
    for (size_t i = 0; i < scene_imp->num_meshes; i++) {
        const ufbx_mesh *mesh = scene_imp->meshes[i];
        if (mesh->skin_deformers.count > 0 && mesh->skin_deformers.data) {
            for (size_t j = 0; j < mesh->skin_deformers.count; j++) {
                const ufbx_skin_deformer *skin = mesh->skin_deformers.data[j];
                if (!ufbx_ascii_write_node_begin(writer, "C")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_string(writer, "OO")) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, skin->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, mesh->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_newline(writer)) {
                    return false;
                }
            }
        }
    }
    
    // Connect skin clusters to skin deformers and to bone nodes
    for (size_t i = 0; i < scene_imp->scene.skin_deformers.count; i++) {
        const ufbx_skin_deformer *skin = scene_imp->scene.skin_deformers.data[i];
        
        for (size_t j = 0; j < skin->clusters.count; j++) {
            const ufbx_skin_cluster *cluster = skin->clusters.data[j];
            
            // Connect cluster to skin deformer
            if (!ufbx_ascii_write_node_begin(writer, "C")) {
                return false;
            }
            if (!ufbx_ascii_write_property_string(writer, "OO")) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, cluster->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
            if (!ufbx_ascii_write_property_i64(writer, skin->element.element_id)) {
                return false;
            }
            if (!ufbx_ascii_write_newline(writer)) {
                return false;
            }
            
            // Connect cluster to bone node
            if (cluster->bone_node) {
                if (!ufbx_ascii_write_node_begin(writer, "C")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_string(writer, "OO")) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, cluster->bone_node->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_string(writer, ", ")) {
                    return false;
                }
                if (!ufbx_ascii_write_property_i64(writer, cluster->element.element_id)) {
                    return false;
                }
                if (!ufbx_ascii_write_newline(writer)) {
                    return false;
                }
            }
        }
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}
