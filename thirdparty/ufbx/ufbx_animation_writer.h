#ifndef UFBX_ANIMATION_WRITER_H
#define UFBX_ANIMATION_WRITER_H

#include "ufbx_ascii_writer.h"

// Animation writing functions
bool ufbx_ascii_write_animation_stack(ufbx_ascii_writer *writer, const ufbx_anim_stack *anim_stack);
bool ufbx_ascii_write_animation_layer(ufbx_ascii_writer *writer, const ufbx_anim_layer *anim_layer);
bool ufbx_ascii_write_animation_curve_node(ufbx_ascii_writer *writer, const ufbx_anim_value *anim_value);
bool ufbx_ascii_write_animation_curve(ufbx_ascii_writer *writer, const ufbx_anim_curve *anim_curve);

#endif // UFBX_ANIMATION_WRITER_H
