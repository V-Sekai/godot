#include "ufbx_animation_writer.h"
#include <stdlib.h>

bool ufbx_ascii_write_animation_stack(ufbx_ascii_writer *writer, const ufbx_anim_stack *anim_stack) {
    if (!ufbx_ascii_write_node_begin(writer, "AnimationStack")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, anim_stack->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, anim_stack->element.name.data)) {
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
    
    // Properties70
    if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // LocalStart
    if (!ufbx_ascii_write_node_begin(writer, "P")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "LocalStart")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "KTime")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Time")) {
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
    if (!ufbx_ascii_write_property_i64(writer, (int64_t)(anim_stack->time_begin * 46186158000LL))) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // LocalStop
    if (!ufbx_ascii_write_node_begin(writer, "P")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "LocalStop")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "KTime")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Time")) {
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
    if (!ufbx_ascii_write_property_i64(writer, (int64_t)(anim_stack->time_end * 46186158000LL))) {
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

bool ufbx_ascii_write_animation_layer(ufbx_ascii_writer *writer, const ufbx_anim_layer *anim_layer) {
    if (!ufbx_ascii_write_node_begin(writer, "AnimationLayer")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, anim_layer->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, anim_layer->element.name.data)) {
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
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

bool ufbx_ascii_write_animation_curve_node(ufbx_ascii_writer *writer, const ufbx_anim_value *anim_value) {
    if (!ufbx_ascii_write_node_begin(writer, "AnimationCurveNode")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, anim_value->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, anim_value->element.name.data)) {
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
    
    // Properties70 with default values
    if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Default values for each component
    const char *components[] = {"d|X", "d|Y", "d|Z"};
    for (int comp = 0; comp < 3; comp++) {
        if (!ufbx_ascii_write_node_begin(writer, "P")) {
            return false;
        }
        if (!ufbx_ascii_write_property_string(writer, components[comp])) {
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
        if (!ufbx_ascii_write_property_string(writer, "A")) {
            return false;
        }
        if (!ufbx_ascii_write_string(writer, ", ")) {
            return false;
        }
        if (!ufbx_ascii_write_property_f64(writer, anim_value->default_value.v[comp])) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
    }
    
    writer->indent_level--;
    if (!ufbx_ascii_write_node_end(writer)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

bool ufbx_ascii_write_animation_curve(ufbx_ascii_writer *writer, const ufbx_anim_curve *anim_curve) {
    if (!ufbx_ascii_write_node_begin(writer, "AnimationCurve")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, anim_curve->element.element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, anim_curve->element.name.data)) {
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
    
    // Default
    if (!ufbx_ascii_write_node_begin(writer, "Default")) {
        return false;
    }
    if (!ufbx_ascii_write_property_f64(writer, 0.0)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // KeyVer
    if (!ufbx_ascii_write_node_begin(writer, "KeyVer")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 4009)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    if (anim_curve->keyframes.count > 0) {
        // KeyTime
        if (!ufbx_ascii_write_node_begin(writer, "KeyTime")) {
            return false;
        }
        
        // Convert keyframe times to int64 array - IMPROVED ERROR HANDLING
        int64_t *time_data = (int64_t*)malloc(anim_curve->keyframes.count * sizeof(int64_t));
        if (!time_data) {
            return false; // Out of memory
        }
        
        for (size_t t = 0; t < anim_curve->keyframes.count; t++) {
            time_data[t] = (int64_t)(anim_curve->keyframes.data[t].time * 46186158000LL);
        }
        
        bool success = ufbx_ascii_write_property_array_i64(writer, time_data, anim_curve->keyframes.count);
        free(time_data);
        if (!success) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // KeyValueFloat
        if (!ufbx_ascii_write_node_begin(writer, "KeyValueFloat")) {
            return false;
        }
        
        // Convert keyframe values to double array - IMPROVED ERROR HANDLING
        double *value_data = (double*)malloc(anim_curve->keyframes.count * sizeof(double));
        if (!value_data) {
            return false; // Out of memory
        }
        
        for (size_t v = 0; v < anim_curve->keyframes.count; v++) {
            value_data[v] = anim_curve->keyframes.data[v].value;
        }
        
        bool success2 = ufbx_ascii_write_property_array_f64(writer, value_data, anim_curve->keyframes.count);
        free(value_data);
        if (!success2) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // KeyAttrFlags (CRITICAL FOR VALIDATION!)
        if (!ufbx_ascii_write_node_begin(writer, "KeyAttrFlags")) {
            return false;
        }
        
        // Generate interpolation flags based on keyframe interpolation - IMPROVED ERROR HANDLING
        int32_t *flag_data = (int32_t*)malloc(anim_curve->keyframes.count * sizeof(int32_t));
        if (!flag_data) {
            return false; // Out of memory
        }
        
        for (size_t f = 0; f < anim_curve->keyframes.count; f++) {
            // Use linear interpolation flag (1028) as default
            flag_data[f] = 1028; // Linear interpolation
        }
        
        bool success3 = ufbx_ascii_write_property_array_i32(writer, flag_data, anim_curve->keyframes.count);
        free(flag_data);
        if (!success3) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // KeyAttrDataFloat (CRITICAL TANGENT DATA FOR VALIDATION!)
        if (!ufbx_ascii_write_node_begin(writer, "KeyAttrDataFloat")) {
            return false;
        }
        
        // Generate tangent data (8 floats per keyframe: slopes, weights, velocities) - IMPROVED ERROR HANDLING
        double *attr_data = (double*)malloc(anim_curve->keyframes.count * 8 * sizeof(double));
        if (!attr_data) {
            return false; // Out of memory
        }
        
        for (size_t a = 0; a < anim_curve->keyframes.count; a++) {
            const ufbx_keyframe *kf = &anim_curve->keyframes.data[a];
            // Default tangent data for linear interpolation
            attr_data[a * 8 + 0] = kf->right.dx;  // RightSlope
            attr_data[a * 8 + 1] = kf->right.dx;  // NextLeftSlope  
            attr_data[a * 8 + 2] = 218434821.0;  // RightWeight (0.333333 as int bits)
            attr_data[a * 8 + 3] = 0.0;          // Padding
            attr_data[a * 8 + 4] = kf->left.dx;  // RightVelocity
            attr_data[a * 8 + 5] = kf->left.dx;  // NextLeftVelocity
            attr_data[a * 8 + 6] = 218434821.0;  // NextLeftWeight
            attr_data[a * 8 + 7] = 0.0;          // Padding
        }
        
        bool success4 = ufbx_ascii_write_property_array_f64(writer, attr_data, anim_curve->keyframes.count * 8);
        free(attr_data);
        if (!success4) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // KeyAttrRefCount (CRITICAL REFERENCE COUNT DATA!)
        if (!ufbx_ascii_write_node_begin(writer, "KeyAttrRefCount")) {
            return false;
        }
        
        // Generate reference counts (all 1s for simple case) - IMPROVED ERROR HANDLING
        int32_t *ref_data = (int32_t*)malloc(anim_curve->keyframes.count * sizeof(int32_t));
        if (!ref_data) {
            return false; // Out of memory
        }
        
        for (size_t r = 0; r < anim_curve->keyframes.count; r++) {
            ref_data[r] = 1; // Standard reference count
        }
        
        bool success5 = ufbx_ascii_write_property_array_i32(writer, ref_data, anim_curve->keyframes.count);
        free(ref_data);
        if (!success5) {
            return false;
        }
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}
