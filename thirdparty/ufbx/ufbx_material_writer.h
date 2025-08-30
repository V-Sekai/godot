#ifndef UFBX_MATERIAL_WRITER_H
#define UFBX_MATERIAL_WRITER_H

#include "ufbx_ascii_writer.h"

// Material and node writing functions
bool ufbx_ascii_write_model_node(ufbx_ascii_writer *writer, const ufbx_node *node);
bool ufbx_ascii_write_material(ufbx_ascii_writer *writer, const ufbx_material *material);
bool ufbx_ascii_write_bone_attribute(ufbx_ascii_writer *writer, const ufbx_bone *bone);
bool ufbx_ascii_write_skin_deformer(ufbx_ascii_writer *writer, const ufbx_skin_deformer *skin);
bool ufbx_ascii_write_skin_cluster(ufbx_ascii_writer *writer, const ufbx_skin_cluster *cluster);

#endif // UFBX_MATERIAL_WRITER_H
