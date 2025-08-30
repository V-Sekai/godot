#ifndef UFBX_GEOMETRY_WRITER_H
#define UFBX_GEOMETRY_WRITER_H

#include "ufbx_ascii_writer.h"

// Geometry writing functions
bool ufbx_ascii_write_geometry(ufbx_ascii_writer *writer, const ufbx_mesh *mesh);
bool ufbx_ascii_write_layer_element_normal(ufbx_ascii_writer *writer, const ufbx_mesh *mesh);
bool ufbx_ascii_write_layer_element_uv(ufbx_ascii_writer *writer, const ufbx_mesh *mesh);
bool ufbx_ascii_write_layer_element_material(ufbx_ascii_writer *writer, const ufbx_mesh *mesh);
bool ufbx_ascii_write_layer_structure(ufbx_ascii_writer *writer, const ufbx_mesh *mesh);

#endif // UFBX_GEOMETRY_WRITER_H
