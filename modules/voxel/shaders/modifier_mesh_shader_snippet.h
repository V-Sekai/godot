/**************************************************************************/
/*  modifier_mesh_shader_snippet.h                                        */
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

// Generated file

// clang-format off
const char *g_modifier_mesh_shader_snippet =
"\n"
"layout (set = 0, binding = 5) restrict readonly buffer ShapeParams {\n"
"	vec3 model_to_buffer_translation;\n"
"	vec3 model_to_buffer_scale;\n"
"	float isolevel;\n"
"} u_shape_params;\n"
"\n"
"layout (set = 0, binding = 6) uniform sampler3D u_sd_buffer;\n"
"\n"
"float get_sd(vec3 pos) {\n"
"	pos = u_shape_params.model_to_buffer_scale * (u_shape_params.model_to_buffer_translation + pos);\n"
"	return texture(u_sd_buffer, pos).r - u_shape_params.isolevel;\n"
"}\n"
"\n";
// clang-format on
