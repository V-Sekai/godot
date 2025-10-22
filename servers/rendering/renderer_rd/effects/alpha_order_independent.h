/**************************************************************************/
/*  alpha_order_independent.h (formerly oit.h)                            */
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

#include <cstdint>

#include "servers/rendering/renderer_rd/pipeline_cache_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/oit_blend.glsl.gen.h"
#include "servers/rendering/rendering_server.h"

namespace RendererRD {

const int TILE_SIZE = 8;
const int MAX_FRAGMENTS_PER_TILE = 64;
const int DEPTH_BINS_PER_TILE = 32;

struct TileData {
	uint32_t fragment_head;
	uint32_t min_depth;
	uint32_t max_depth;
	uint32_t fragment_count;
};

struct FragmentData {
	uint32_t next;
	uint32_t depth_packed;
	uint32_t color_packed;
	uint32_t alpha_packed;
};

class AlphaOrderIndependent {
	bool vr_multiview_enabled = false;
	int xr_view_count = 1;

public:
	AlphaOrderIndependent();
	~AlphaOrderIndependent();
	void init();
	void free();
	void update_vr_state();
	void get_tile_count(int p_width, int p_height, uint32_t &r_tile_x, uint32_t &r_tile_y);
	uint32_t _get_tile_buffer_index(uint32_t p_tile_x, uint32_t p_tile_y, uint32_t p_depth_bin, uint32_t p_view, uint32_t p_tiles_per_row, uint32_t p_view_count);
	void _get_tile_coords(uint32_t p_pixel_x, uint32_t p_pixel_y, uint32_t &r_tile_x, uint32_t &r_tile_y);

	void create_alpha_order_independent_buffers(RID &r_tile_buffer, RID &r_fragment_buffer, RID &r_counter_buffer, int p_width, int p_height, bool p_multiview, int p_view_count);
	void free_alpha_order_independent_buffers(RID &r_tile_buffer, RID &r_fragment_buffer, RID &r_counter_buffer);
	void clear_alpha_order_independent_buffers(RID p_tile_buffer, RID p_fragment_buffer, RID p_counter_buffer, int p_width, int p_height, int p_view_count);
	void resolve_alpha_order_independent(RID p_tile_buffer, RID p_fragment_buffer, RID p_dst_texture, RID p_depth_texture, int p_width, int p_height, int p_view);

private:
	struct AlphaOrderIndependentBlend {
		OitBlendShaderRD shader;
		RID shader_version;
		PipelineCacheRD pipeline;
	} blend;

	void _init_alpha_order_independent_blend_pipeline();
};

} // namespace RendererRD
