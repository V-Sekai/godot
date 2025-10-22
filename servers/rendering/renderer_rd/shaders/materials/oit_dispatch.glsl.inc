/**************************************************************************/
/*  oit_dispatch.glsl.inc                                                 */
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

// OIT Dispatch for Transparent Materials (VR Tile-Based)
// Include in transparent fragment shaders if use_oit_transparency is true

#define TILE_SIZE 16
#define DEPTH_BINS 4
#define MAX_FRAGS 16

layout(std430, binding = 0) restrict buffer OITTileBuffer {
	uint heads[];
	uint min_depths[];
	uint max_depths[];
	uint counts[];
}

layout(std430, binding = 1) restrict buffer OITFragmentBuffer {
	float depths[];
	vec4 colors[]; // vec4 for unpack ease, but pack as u32 in usage
	uint nexts[];
}

layout(std430, binding = 2) restrict buffer OITCounterBuffer {
	uint counters[];
}

uniform bool oit_enabled;
uniform bool oit_multiview_enabled;
uniform int oit_view_count;
uniform int oit_viewport_width;
uniform int oit_viewport_height;
uniform int oit_max_frags;

uint get_tile_index(ivec2 tile, int bin, uint view) {
	uint tiles_x = (oit_viewport_width + TILE_SIZE - 1) / TILE_SIZE;
	uint tiles_per_tile_row = tiles_x; // per tile row
	uint base = (uint(tile.y) * tiles_per_row + uint(tile.x)) * DEPTH_BINS;
	return base + bin * oit_view_count + view;
}

void oit_dispatch(vec3 frag_color, float frag_alpha, float frag_depth) {
	if (!oit_enabled || frag_alpha < 0.01) {
		discard; // Or return if not discard
		return;
	}

	ivec2 frag_coord = ivec2(gl_FragCoord.xy);
	uint view = oit_multiview_enabled ? gl_ViewIndex : 0;
	ivec2 tile = frag_coord / TILE_SIZE;
	int bin = int((1.0 - frag_depth) * DEPTH_BINS);

	uint view_max = oit_max_frags / oit_view_count;
	uint view_offset = view * view_max;
	uint local_offset = atomicAdd(counters[view], 1u);
	if (local_offset >= view_max) {
		return; // Overflow to standard blend
	}
	uint frag_idx = view_offset + local_offset;

	depths[frag_idx] = frag_depth;
	colors[frag_idx] = vec4(frag_color, frag_alpha);
	uint tile_idx = get_tile_index(tile, bin, view);

	uint old_head = atomicExchange(heads[tile_idx], frag_idx);
	nexts[frag_idx] = old_head;

	// Atomic min/max for depth (convert to uint for atomic)
	uint depth_u = floatBitsToUint(frag_depth);
	atomicMin(min_depths[tile_idx], depth_u);
	atomicMax(max_depths[tile_idx], depth_u);
	atomicAdd(counts[tile_idx], 1u);
}

// Call at end of transparent fragment: oit_dispatch(albedo.rgb, alpha, gl_FragCoord.z);
