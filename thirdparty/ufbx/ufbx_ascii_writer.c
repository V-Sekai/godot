#include "ufbx_ascii_writer.h"
#include <stdio.h>
#include <string.h>

// ASCII writer helper functions
bool ufbx_ascii_ensure_capacity(ufbx_ascii_writer *writer, size_t needed) {
    if (writer->has_error) {
        return false;
    }
    
    size_t required = writer->size + needed;
    if (required <= writer->capacity) {
        return true;
    }
    
    size_t new_capacity = writer->capacity ? writer->capacity * 2 : 1024;
    while (new_capacity < required) {
        new_capacity *= 2;
    }
    
    char *new_data = (char*)realloc(writer->data, new_capacity);
    if (!new_data) {
        writer->has_error = true;
        writer->error.type = UFBX_ERROR_OUT_OF_MEMORY;
        snprintf(writer->error.info, UFBX_ERROR_INFO_LENGTH, "Failed to allocate memory for ASCII writer");
        writer->error.info_length = strlen(writer->error.info);
        return false;
    }
    
    writer->data = new_data;
    writer->capacity = new_capacity;
    return true;
}

bool ufbx_ascii_write_string(ufbx_ascii_writer *writer, const char *str) {
    if (!str) {
        return true;
    }
    size_t len = strlen(str);
    if (!ufbx_ascii_ensure_capacity(writer, len)) {
        return false;
    }
    
    memcpy(writer->data + writer->size, str, len);
    writer->size += len;
    return true;
}

bool ufbx_ascii_write_char(ufbx_ascii_writer *writer, char c) {
    if (!ufbx_ascii_ensure_capacity(writer, 1)) {
        return false;
    }
    writer->data[writer->size++] = c;
    return true;
}

bool ufbx_ascii_write_newline(ufbx_ascii_writer *writer) {
    return ufbx_ascii_write_char(writer, '\n');
}

bool ufbx_ascii_write_indent(ufbx_ascii_writer *writer) {
    for (int i = 0; i < writer->indent_level; i++) {
        if (!ufbx_ascii_write_string(writer, "    ")) {
            return false; // 4 spaces per indent
        }
    }
    return true;
}

bool ufbx_ascii_write_property_i64(ufbx_ascii_writer *writer, int64_t value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%lld", (long long)value);
    return ufbx_ascii_write_string(writer, buffer);
}

bool ufbx_ascii_write_property_f64(ufbx_ascii_writer *writer, double value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.6g", value);
    return ufbx_ascii_write_string(writer, buffer);
}

bool ufbx_ascii_write_property_string(ufbx_ascii_writer *writer, const char *str) {
    if (!ufbx_ascii_write_char(writer, '"')) {
        return false;
    }
    if (str) {
        // TODO: Escape special characters if needed
        if (!ufbx_ascii_write_string(writer, str)) {
            return false;
        }
    }
    return ufbx_ascii_write_char(writer, '"');
}

bool ufbx_ascii_write_property_array_f64(ufbx_ascii_writer *writer, const double *values, size_t count) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "*%zu", count);
    if (!ufbx_ascii_write_string(writer, buffer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    
    writer->indent_level++;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "a: ")) {
        return false;
    }
    
    for (size_t i = 0; i < count; i++) {
        if (i > 0) {
            if (!ufbx_ascii_write_char(writer, ',')) {
                return false;
            }
        }
        if (!ufbx_ascii_write_property_f64(writer, values[i])) {
            return false;
        }
    }
    
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    writer->indent_level--;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    return ufbx_ascii_write_char(writer, '}');
}

bool ufbx_ascii_write_property_array_i32(ufbx_ascii_writer *writer, const int32_t *values, size_t count) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "*%zu", count);
    if (!ufbx_ascii_write_string(writer, buffer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    
    writer->indent_level++;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "a: ")) {
        return false;
    }
    
    for (size_t i = 0; i < count; i++) {
        if (i > 0) {
            if (!ufbx_ascii_write_char(writer, ',')) {
                return false;
            }
        }
        if (!ufbx_ascii_write_property_i64(writer, values[i])) {
            return false;
        }
    }
    
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    writer->indent_level--;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    return ufbx_ascii_write_char(writer, '}');
}

bool ufbx_ascii_write_property_array_i64(ufbx_ascii_writer *writer, const int64_t *values, size_t count) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "*%zu", count);
    if (!ufbx_ascii_write_string(writer, buffer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, " {\n")) {
        return false;
    }
    
    writer->indent_level++;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "a: ")) {
        return false;
    }
    
    for (size_t i = 0; i < count; i++) {
        if (i > 0) {
            if (!ufbx_ascii_write_char(writer, ',')) {
                return false;
            }
        }
        if (!ufbx_ascii_write_property_i64(writer, values[i])) {
            return false;
        }
    }
    
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    writer->indent_level--;
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    return ufbx_ascii_write_char(writer, '}');
}

bool ufbx_ascii_write_node_begin(ufbx_ascii_writer *writer, const char *name) {
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, name)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ": ")) {
        return false;
    }
    return true;
}

bool ufbx_ascii_write_node_end(ufbx_ascii_writer *writer) {
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_indent(writer)) {
        return false;
    }
    if (!ufbx_ascii_write_char(writer, '}')) {
        return false;
    }
    return ufbx_ascii_write_newline(writer);
}
