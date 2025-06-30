/**************************************************************************/
/*  shaders.h                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

namespace zylann::voxel {

extern const char *g_block_generator_shader_template_0;
extern const char *g_block_generator_shader_template_1;
extern const char *g_block_generator_shader_template_2;
extern const char *g_block_modifier_shader_template_0;
extern const char *g_block_modifier_shader_template_1;
extern const char *g_detail_gather_hits_shader;
extern const char *g_detail_generator_shader_template_0;
extern const char *g_detail_generator_shader_template_1;
extern const char *g_detail_modifier_shader_template_0;
extern const char *g_detail_modifier_shader_template_1;
extern const char *g_detail_normalmap_shader;
extern const char *g_dilate_normalmap_shader;
extern const char *g_modifier_sphere_shader_snippet;
extern const char *g_modifier_mesh_shader_snippet;
extern const char *g_fast_noise_lite_shader[];

} // namespace zylann::voxel
