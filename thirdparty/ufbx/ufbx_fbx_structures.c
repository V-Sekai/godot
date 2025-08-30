#include "ufbx_fbx_structures.h"
#include <stdio.h>

// ASCII FBX structure writers
bool ufbx_ascii_write_header(ufbx_ascii_writer *writer, uint32_t version) {
    // Write FBX magic comment
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "; FBX %u.%u.%u project file\n", 
             version / 1000, (version / 100) % 10, (version / 10) % 10);
    if (!ufbx_ascii_write_string(writer, buffer)) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "; Created by ufbx_export (ASCII only)\n\n")) {
        return false;
    }
    return true;
}

bool ufbx_ascii_write_scene_info(ufbx_ascii_writer *writer, const ufbx_export_scene *scene) {
    if (!ufbx_ascii_write_node_begin(writer, "FBXHeaderExtension")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // FBXHeaderVersion
    if (!ufbx_ascii_write_node_begin(writer, "FBXHeaderVersion")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 1003)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // FBXVersion
    if (!ufbx_ascii_write_node_begin(writer, "FBXVersion")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, writer->version)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // Creator
    if (!ufbx_ascii_write_node_begin(writer, "Creator")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, scene->metadata.creator)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    writer->indent_level--;
    return ufbx_ascii_write_node_end(writer);
}

bool ufbx_ascii_write_global_settings(ufbx_ascii_writer *writer, const ufbx_export_scene *scene) {
    if (!ufbx_ascii_write_node_begin(writer, "GlobalSettings")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, "{\n")) {
        return false;
    }
    writer->indent_level++;
    
    // Version
    if (!ufbx_ascii_write_node_begin(writer, "Version")) {
        return false;
    }
    if (!ufbx_ascii_write_property_i64(writer, 1000)) {
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
    
    // UpAxis
    if (!ufbx_ascii_write_node_begin(writer, "P")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "UpAxis")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "int")) {
        return false;
    }
    if (!ufbx_ascii_write_string(writer, ", ")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "Integer")) {
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
    if (!ufbx_ascii_write_property_i64(writer, scene->settings.axes.up == UFBX_COORDINATE_AXIS_POSITIVE_Y ? 1 : 2)) {
        return false;
    }
    if (!ufbx_ascii_write_newline(writer)) {
        return false;
    }
    
    // UnitScaleFactor
    if (!ufbx_ascii_write_node_begin(writer, "P")) {
        return false;
    }
    if (!ufbx_ascii_write_property_string(writer, "UnitScaleFactor")) {
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
    if (!ufbx_ascii_write_property_f64(writer, scene->settings.unit_meters * 100.0)) {
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

bool ufbx_ascii_write_documents_section(ufbx_ascii_writer *writer) {
    if (!ufbx_ascii_write_string(writer, "Documents: {\n")) {
        return false;
    }
    writer->indent_level++;
    if (!ufbx_ascii_write_indent(writer) || !ufbx_ascii_write_string(writer, "Count: 1\n")) {
        return false;
    }
    writer->indent_level--;
    if (!ufbx_ascii_write_string(writer, "}\n\n")) {
        return false;
    }
    return true;
}

bool ufbx_ascii_write_definitions_section(ufbx_ascii_writer *writer) {
    if (!ufbx_ascii_write_string(writer, "Definitions: {\n")) {
        return false;
    }
    writer->indent_level++;
    if (!ufbx_ascii_write_indent(writer) || !ufbx_ascii_write_string(writer, "Version: 100\n")) {
        return false;
    }
    if (!ufbx_ascii_write_indent(writer) || !ufbx_ascii_write_string(writer, "Count: 5\n")) {
        return false;
    }
    writer->indent_level--;
    if (!ufbx_ascii_write_string(writer, "}\n\n")) {
        return false;
    }
    return true;
}

bool ufbx_ascii_write_takes_section(ufbx_ascii_writer *writer) {
    if (!ufbx_ascii_write_string(writer, "Takes: {\n")) {
        return false;
    }
    writer->indent_level++;
    if (!ufbx_ascii_write_indent(writer) || !ufbx_ascii_write_string(writer, "Current: \"Take 001\"\n")) {
        return false;
    }
    writer->indent_level--;
    if (!ufbx_ascii_write_string(writer, "}\n")) {
        return false;
    }
    return true;
}
