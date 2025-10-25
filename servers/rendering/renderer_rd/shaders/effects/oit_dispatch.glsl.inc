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

// OIT Fragment Dispatch Include
// For BaseMaterial3D oit_transparency integration

// OIT data structures
struct TileData {
	uint fragment_head;
	uint fragment_count;
	uint min_depth;
	uint max_depth;
};

struct FragmentData {
	uint next;
	uint depth_packed;
	uint color_packed;
	uint padding;
};

// OIT buffers - always declared
layout(std430, set = 1, binding = 37) buffer OITTileBuffer {
	TileData tiles[];
};

layout(std430, set = 1, binding = 38) buffer OITFragmentBuffer {
	FragmentData fragments[];
};

layout(std430, set = 1, binding = 39) buffer OITFragmentCounterBuffer {
	uint fragment_counter[];
};

// OIT parameters (can be passed via push constants or defines)
const int OIT_TILE_SIZE = 16;
const int OIT_DEPTH_BINS = 8;

uint get_tile_index(ivec2 tile_coord, uint depth_bin, uint view_index, uint tiles_per_row, uint view_count) {
	uint tile_idx = tile_coord.y * tiles_per_row + tile_coord.x;
	uint bin_idx = tile_idx * uint(OIT_DEPTH_BINS) + depth_bin;
	return bin_idx * view_count + view_index;
}

uint pack_rgba(vec4 color) {
	uvec4 bytes = uvec4(round(clamp(color, 0.0, 1.0) * 255.0));
	return (bytes.r << 24) | (bytes.g << 16) | (bytes.b << 8) | bytes.a;
}

void oit_collect_fragment(vec4 frag_color, bool oit_enabled, uint screen_width, uint screen_height, uint view_count, uint max_fragments) {
	if (!oit_enabled) {
		return;
	}

	ivec2 tile_coord = ivec2(gl_FragCoord.xy) / OIT_TILE_SIZE;
	uint depth_bin = uint(clamp((1.0 - gl_FragCoord.z) * float(OIT_DEPTH_BINS), 0.0, float(OIT_DEPTH_BINS - 1)));

#ifdef USE_MULTIVIEW
	uint view_idx = uint(ViewIndex);
#else
	uint view_idx = 0;
#endif

	uint tiles_per_row = (screen_width + uint(OIT_TILE_SIZE) - 1u) / uint(OIT_TILE_SIZE);
	uint tile_idx = get_tile_index(tile_coord, depth_bin, view_idx, tiles_per_row, view_count);
	uint frag_idx = atomicAdd(fragment_counter[view_idx], 1u);

	if (frag_idx >= max_fragments) {
		return; // Buffer overflow
	}

	uint depth_packed = floatBitsToUint(gl_FragCoord.z);
	uint color_packed = pack_rgba(frag_color);

	// Linked list prepend
	fragments[frag_idx].next = tiles[tile_idx].fragment_head;
	fragments[frag_idx].depth_packed = depth_packed;
	fragments[frag_idx].color_packed = color_packed;
	tiles[tile_idx].fragment_head = frag_idx;

	atomicMin(tiles[tile_idx].min_depth, depth_packed);
	atomicMax(tiles[tile_idx].max_depth, depth_packed);
	atomicAdd(tiles[tile_idx].fragment_count, 1);
}

// Always define oit_debug_tile_visualization to prevent compilation errors.
// This function is a placeholder for OIT debug visualization.
vec3 oit_debug_tile_visualization(vec2 p_frag_coord, uint p_screen_width, uint p_screen_height) {
	return vec3(0.0, 0.0, 0.0);
}
