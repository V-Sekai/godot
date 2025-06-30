/**************************************************************************/
/*  transvoxel_minimal_shader.h                                           */
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
const char *g_transvoxel_minimal_shader =
"shader_type spatial;\n"
"\n"
"// From Voxel Tools API\n"
"uniform int u_transition_mask;\n"
"\n"
"float get_transvoxel_secondary_factor(int idata) {\n"
"	int transition_mask = u_transition_mask & 0xff;\n"
"\n"
"	int cell_border_mask = idata & 63; // Which sides the cell is touching\n"
"	int vertex_border_mask = (idata >> 8) & 63; // Which sides the vertex is touching\n"
"	// If the vertex is near a side where there is a low-resolution neighbor,\n"
"	// move it to secondary position\n"
"	int m = transition_mask & cell_border_mask;\n"
"	float t = float(m != 0);\n"
"	// If the vertex lies on one or more sides, and at least one side has no low-resolution neighbor,\n"
"	// don't move the vertex.\n"
"	t *= float((vertex_border_mask & ~transition_mask) == 0);\n"
"	\n"
"	// Debugging\n"
"	//t *= 0.5 + 0.5 * sin(TIME * 4.0);\n"
"	//t *= 2.0;\n"
"\n"
"	return t;\n"
"}\n"
"\n"
"vec3 get_transvoxel_position(vec3 vertex_pos, vec4 fdata) {\n"
"	int idata = floatBitsToInt(fdata.a);\n"
"\n"
"	// Move vertices to smooth transitions\n"
"	float secondary_factor = get_transvoxel_secondary_factor(idata);\n"
"	vec3 secondary_position = fdata.xyz;\n"
"	vec3 pos = mix(vertex_pos, secondary_position, secondary_factor);\n"
"\n"
"	// If the mesh combines transitions and the vertex belongs to a transition,\n"
"	// when that transition isn't active we change the position of the vertices so\n"
"	// all triangles will be degenerate and won't be visible.\n"
"	// This is an alternative to rendering them separately,\n"
"	// which has less draw calls and less mesh resources to create in Godot.\n"
"	// Ideally I would tweak the index buffer like LOD does but Godot does not\n"
"	// expose anything to use it that way.\n"
"	int itransition = (idata >> 16) & 0xff; // Is the vertex on a transition mesh?\n"
"	float transition_cull = float(itransition == 0 || (itransition & u_transition_mask) != 0);\n"
"	pos *= transition_cull;\n"
"\n"
"	return pos;\n"
"}\n"
"\n"
"void vertex() {\n"
"	VERTEX = get_transvoxel_position(VERTEX, CUSTOM0);\n"
"}\n";
// clang-format on
