#ifndef UFBX_FBX_STRUCTURES_H
#define UFBX_FBX_STRUCTURES_H

#include "ufbx_ascii_writer.h"

// FBX Version Constants
#define UFBX_FBX_VERSION_7400 7400
#define UFBX_FBX_VERSION_7500 7500

// FBX structure writers
bool ufbx_ascii_write_header(ufbx_ascii_writer *writer, uint32_t version);
bool ufbx_ascii_write_scene_info(ufbx_ascii_writer *writer, const ufbx_export_scene *scene);
bool ufbx_ascii_write_global_settings(ufbx_ascii_writer *writer, const ufbx_export_scene *scene);
bool ufbx_ascii_write_documents_section(ufbx_ascii_writer *writer);
bool ufbx_ascii_write_definitions_section(ufbx_ascii_writer *writer);
bool ufbx_ascii_write_takes_section(ufbx_ascii_writer *writer);

#endif // UFBX_FBX_STRUCTURES_H
