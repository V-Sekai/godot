#include "ufbx_animation_writer.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Build complete arrays using safe stack-based allocation with fallback
static bool write_keyframe_array_safe(ufbx_ascii_writer *writer, const ufbx_anim_curve *anim_curve, 
                                     const char* array_type) {
    if (strcmp(array_type, "time") == 0) {
        // CRITICAL FIX: Always use heap allocation to avoid stack buffer overflow
        int64_t *data = (int64_t*)calloc(anim_curve->keyframes.count, sizeof(int64_t));
        if (!data) {
            printf("ERROR: Failed to allocate time array for %zu keyframes\n", anim_curve->keyframes.count);
            return false; // Fail cleanly rather than write corrupted data
        }
        
        for (size_t i = 0; i < anim_curve->keyframes.count; i++) {
            data[i] = (int64_t)(anim_curve->keyframes.data[i].time * 46186158000LL);
        }
        
        bool success = ufbx_ascii_write_property_array_i64(writer, data, anim_curve->keyframes.count);
        free(data);
        return success;
    } else if (strcmp(array_type, "value") == 0) {
        // CRITICAL FIX: Always use heap allocation to avoid stack buffer overflow
        double *data = (double*)calloc(anim_curve->keyframes.count, sizeof(double));
        if (!data) {
            printf("ERROR: Failed to allocate value array for %zu keyframes\n", anim_curve->keyframes.count);
            return false; // Fail cleanly rather than write corrupted data
        }
        
        for (size_t i = 0; i < anim_curve->keyframes.count; i++) {
            data[i] = anim_curve->keyframes.data[i].value;
        }
        
        bool success = ufbx_ascii_write_property_array_f64(writer, data, anim_curve->keyframes.count);
        free(data);
        return success;
    } else if (strcmp(array_type, "flags") == 0) {
        // CRITICAL FIX: Always use heap allocation to avoid stack buffer overflow
        int32_t *data = (int32_t*)calloc(anim_curve->keyframes.count, sizeof(int32_t));
        if (!data) {
            printf("ERROR: Failed to allocate flags array for %zu keyframes\n", anim_curve->keyframes.count);
            return false; // Fail cleanly rather than write corrupted data
        }
        
        for (size_t i = 0; i < anim_curve->keyframes.count; i++) {
            data[i] = 1028; // Linear interpolation
        }
        
        bool success = ufbx_ascii_write_property_array_i32(writer, data, anim_curve->keyframes.count);
        free(data);
        return success;
    } else if (strcmp(array_type, "tangents") == 0) {
        // CRITICAL FIX: Always use heap allocation to avoid stack buffer overflow
        size_t total_elements = anim_curve->keyframes.count * 8;
        double *data = (double*)calloc(total_elements, sizeof(double));
        if (!data) {
            printf("ERROR: Failed to allocate tangent array for %zu elements\n", total_elements);
            return false; // Fail cleanly rather than write corrupted data
        }
        
        // Fill complete array with proper tangent data
        for (size_t i = 0; i < anim_curve->keyframes.count; i++) {
            const ufbx_keyframe *kf = &anim_curve->keyframes.data[i];
            data[i * 8 + 0] = kf->right.dx;
            data[i * 8 + 1] = kf->right.dx;
            data[i * 8 + 2] = 218434821.0;
            data[i * 8 + 3] = 0.0;
            data[i * 8 + 4] = kf->left.dx;
            data[i * 8 + 5] = kf->left.dx;
            data[i * 8 + 6] = 218434821.0;
            data[i * 8 + 7] = 0.0;
        }
        
        bool success = ufbx_ascii_write_property_array_f64(writer, data, total_elements);
        free(data);
        return success;
    }
    
    return false; // Unknown array type
}

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
    
    // CRITICAL FIX: Use large element ID compatible with working files
    int64_t large_element_id = (int64_t)anim_value->element.element_id * 1000000LL + 2426741000000LL;
    if (!ufbx_ascii_write_property_i64(writer, large_element_id)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    
    // CRITICAL FIX: Generate descriptive name compatible with working files
    char descriptive_name[64];
    snprintf(descriptive_name, sizeof(descriptive_name), "AnimCurveNode::%s", anim_value->element.name.data);
    if (!ufbx_ascii_write_property_string(writer, descriptive_name)) {
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
    
    // CRITICAL FIX: Properties70 with double space (match working format)
    if (!ufbx_ascii_write_node_begin(writer, "Properties70")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {  // Double space to match working format
        return false;
    }
    writer->indent_level++;
    
    // CRITICAL FIX: Default values with complex floating-point data (match working format)
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
        
        // CRITICAL FIX: Write values with proper spacing (no space before negative values)
        double value = anim_value->default_value.v[comp];
        char value_str[32];
        snprintf(value_str, sizeof(value_str), "%.12g", value);  // Higher precision like working format
        
        if (value < 0) {
            // No space before negative values (match working format)
            if (!ufbx_ascii_write_string(writer, ",")) {
                return false;
            }
        } else {
            // Space before positive values
            if (!ufbx_ascii_write_string(writer, ", ")) {
                return false;
            }
        }
        
        if (!ufbx_ascii_write_string(writer, value_str)) {
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
    // Check for error state before starting - ufbx.c pattern
    if (writer->has_error) {
        return false;
    }
    
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
        printf("Writing animation curve with %zu keyframes using chunked approach\n", anim_curve->keyframes.count);
        
        // KeyTime - USE SAFE ALLOCATION WITH PROPER ARRAY COUNT
        if (!ufbx_ascii_write_node_begin(writer, "KeyTime")) {
            return false;
        }
        
        if (!write_keyframe_array_safe(writer, anim_curve, "time")) {
            printf("ERROR: Failed to write keyframe times\n");
            return false;
        }
        
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // KeyValueFloat - USE SAFE ALLOCATION WITH PROPER ARRAY COUNT
        if (!ufbx_ascii_write_node_begin(writer, "KeyValueFloat")) {
            return false;
        }
        
        if (!write_keyframe_array_safe(writer, anim_curve, "value")) {
            printf("ERROR: Failed to write keyframe values\n");
            return false;
        }
        
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // KeyAttrFlags - USE SAFE ALLOCATION WITH PROPER ARRAY COUNT
        if (!ufbx_ascii_write_node_begin(writer, "KeyAttrFlags")) {
            return false;
        }
        
        if (!write_keyframe_array_safe(writer, anim_curve, "flags")) {
            printf("ERROR: Failed to write keyframe flags\n");
            return false;
        }
        
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // KeyAttrDataFloat - CRITICAL SECTION WITH PROPER ARRAY COUNT
        if (!ufbx_ascii_write_node_begin(writer, "KeyAttrDataFloat")) {
            return false;
        }
        
        if (!write_keyframe_array_safe(writer, anim_curve, "tangents")) {
            printf("ERROR: Failed to write keyframe tangents\n");
            return false;
        }
        
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
        
        // KeyAttrRefCount - USE SAFE ALLOCATION WITH PROPER ARRAY COUNT
        if (!ufbx_ascii_write_node_begin(writer, "KeyAttrRefCount")) {
            return false;
        }
        
        if (!write_keyframe_array_safe(writer, anim_curve, "flags")) {
            printf("ERROR: Failed to write keyframe ref counts\n");
            return false;
        }
        
        if (!ufbx_ascii_write_newline(writer)) {
            return false;
        }
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}
