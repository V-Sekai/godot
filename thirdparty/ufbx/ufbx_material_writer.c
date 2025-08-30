#include "ufbx_material_writer.h"
#include <stdlib.h>

bool ufbx_ascii_write_model_node(ufbx_ascii_writer *writer, const ufbx_node *node) {
    if (!ufbx_ascii_write_node_begin(writer, "Model")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, node->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, node->element.name.data)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    // Determine correct node type: LimbNode for bones, Mesh for geometry nodes, Null for empties
    const char *node_type = "Null"; // Default to Null/empty
    if (node->mesh) {
        node_type = "Mesh";
    } else if (node->bone) {
        node_type = "LimbNode";
    }
    if (!ufbx_ascii_write_property_string(writer, node_type)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Version
    if (!ufbx_ascii_write_node_begin(writer, "Version")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 232)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // Properties70
    if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Lcl Translation
    if (!ufbx_ascii_write_node_begin(writer, "P")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Lcl Translation")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Lcl Translation")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "A")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, node->local_transform.translation.x)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, node->local_transform.translation.y)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, node->local_transform.translation.z)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    writer->indent_level--;
    if (!ufbx_ascii_write_node_end(writer)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

bool ufbx_ascii_write_material(ufbx_ascii_writer *writer, const ufbx_material *material) {
    if (!ufbx_ascii_write_node_begin(writer, "Material")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, material->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, material->element.name.data)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Version
    if (!ufbx_ascii_write_node_begin(writer, "Version")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 102)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // ShadingModel
    if (!ufbx_ascii_write_node_begin(writer, "ShadingModel")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "lambert")) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // MultiLayer
    if (!ufbx_ascii_write_node_begin(writer, "MultiLayer")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 0)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // Properties70
    if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // DiffuseColor
    if (!ufbx_ascii_write_node_begin(writer, "P")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "DiffuseColor")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Color")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "A")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, material->pbr.base_color.value_vec3.x)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, material->pbr.base_color.value_vec3.y)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, material->pbr.base_color.value_vec3.z)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    writer->indent_level--;
    if (!ufbx_ascii_write_node_end(writer)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

bool ufbx_ascii_write_bone_attribute(ufbx_ascii_writer *writer, const ufbx_bone *bone) {
    if (!ufbx_ascii_write_node_begin(writer, "NodeAttribute")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, bone->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, bone->element.name.data)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "LimbNode")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    writer->indent_level++;
    
    // TypeFlags
    if (!ufbx_ascii_write_node_begin(writer, "TypeFlags")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Skeleton")) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // Properties70
    if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Size (relative length)
    if (!ufbx_ascii_write_node_begin(writer, "P")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Size")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "double")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Number")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, bone->relative_length)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    writer->indent_level--;
    if (!ufbx_ascii_write_node_end(writer)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

bool ufbx_ascii_write_skin_deformer(ufbx_ascii_writer *writer, const ufbx_skin_deformer *skin) {
    if (!ufbx_ascii_write_node_begin(writer, "Deformer")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, skin->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, skin->element.name.data)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Skin")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Version
    if (!ufbx_ascii_write_node_begin(writer, "Version")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 101)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // Link_DeformAcuracy
    if (!ufbx_ascii_write_node_begin(writer, "Link_DeformAcuracy")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, 50.0)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

bool ufbx_ascii_write_skin_cluster(ufbx_ascii_writer *writer, const ufbx_skin_cluster *cluster) {
    if (!ufbx_ascii_write_node_begin(writer, "Deformer")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, cluster->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, cluster->element.name.data)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Cluster")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Version
    if (!ufbx_ascii_write_node_begin(writer, "Version")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 100)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // UserData
    if (!ufbx_ascii_write_node_begin(writer, "UserData")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "")) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // Indexes (vertex indices)
    if (cluster->vertices.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "Indexes")) {
            return false;
        }
        if (!ufbx_ascii_write_property_array_i32(writer, (const int32_t*)cluster->vertices.data, cluster->vertices.count)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
    }
    
    // Weights (vertex weights)
    if (cluster->weights.count > 0) {
        if (!ufbx_ascii_write_node_begin(writer, "Weights")) {
            return false;
        }
        if (!ufbx_ascii_write_property_array_f64(writer, (const double*)cluster->weights.data, cluster->weights.count)) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
    }
    
    // Transform (geometry_to_bone matrix)
    if (!ufbx_ascii_write_node_begin(writer, "Transform")) {
        return false;
    }
    double transform_data[16] = {
        cluster->geometry_to_bone.m00, cluster->geometry_to_bone.m10, cluster->geometry_to_bone.m20, 0.0,
        cluster->geometry_to_bone.m01, cluster->geometry_to_bone.m11, cluster->geometry_to_bone.m21, 0.0,
        cluster->geometry_to_bone.m02, cluster->geometry_to_bone.m12, cluster->geometry_to_bone.m22, 0.0,
        cluster->geometry_to_bone.m03, cluster->geometry_to_bone.m13, cluster->geometry_to_bone.m23, 1.0
    };
    if (!ufbx_ascii_write_property_array_f64(writer, transform_data, 16)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // TransformLink (bind_to_world matrix)
    if (!ufbx_ascii_write_node_begin(writer, "TransformLink")) {
        return false;
    }
    double transform_link_data[16] = {
        cluster->bind_to_world.m00, cluster->bind_to_world.m10, cluster->bind_to_world.m20, 0.0,
        cluster->bind_to_world.m01, cluster->bind_to_world.m11, cluster->bind_to_world.m21, 0.0,
        cluster->bind_to_world.m02, cluster->bind_to_world.m12, cluster->bind_to_world.m22, 0.0,
        cluster->bind_to_world.m03, cluster->bind_to_world.m13, cluster->bind_to_world.m23, 1.0
    };
    if (!ufbx_ascii_write_property_array_f64(writer, transform_link_data, 16)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}
