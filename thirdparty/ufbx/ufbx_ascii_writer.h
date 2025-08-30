#ifndef UFBX_ASCII_WRITER_H
#define UFBX_ASCII_WRITER_H

#include "ufbx_export.h"
#include "ufbx.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// ASCII writer state
typedef struct {
    char *data;
    size_t size;
    size_t capacity;
    int indent_level;
    ufbx_error error;
    bool has_error;
    uint32_t version;
} ufbx_ascii_writer;

// Core ASCII writer functions
bool ufbx_ascii_ensure_capacity(ufbx_ascii_writer *writer, size_t needed);
bool ufbx_ascii_write_string(ufbx_ascii_writer *writer, const char *str);
bool ufbx_ascii_write_char(ufbx_ascii_writer *writer, char c);
bool ufbx_ascii_write_newline(ufbx_ascii_writer *writer);
bool ufbx_ascii_write_indent(ufbx_ascii_writer *writer);

// Property writers
bool ufbx_ascii_write_property_i64(ufbx_ascii_writer *writer, int64_t value);
bool ufbx_ascii_write_property_f64(ufbx_ascii_writer *writer, double value);
bool ufbx_ascii_write_property_string(ufbx_ascii_writer *writer, const char *str);

// Array writers
bool ufbx_ascii_write_property_array_f64(ufbx_ascii_writer *writer, const double *values, size_t count);
bool ufbx_ascii_write_property_array_i32(ufbx_ascii_writer *writer, const int32_t *values, size_t count);
bool ufbx_ascii_write_property_array_i64(ufbx_ascii_writer *writer, const int64_t *values, size_t count);

// Node structure writers
bool ufbx_ascii_write_node_begin(ufbx_ascii_writer *writer, const char *name);
bool ufbx_ascii_write_node_end(ufbx_ascii_writer *writer);

#endif // UFBX_ASCII_WRITER_H
