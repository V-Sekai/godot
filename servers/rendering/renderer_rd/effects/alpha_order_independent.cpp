/**************************************************************************/
/*  alpha_order_independent.cpp (formerly oit.cpp)                        */
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

#include "alpha_order_independent.h"

#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"
#include "servers/rendering/renderer_rd/framebuffer_cache_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server.h"
#include "servers/xr/xr_server.h"

#include "servers/rendering/renderer_rd/shaders/effects/oit_blend.glsl.gen.h"

namespace RendererRD {

struct AlphaOrderIndependentParams {
	int viewport_width;
	int viewport_height;
	int tile_count_x;
	int tile_count_y;
	int tile_size;
	int max_frags;
	int view_count;
	int multiview_enabled;
	uint32_t padding[4]; // std140 requires 48 bytes total (3 vec4s)
};

void AlphaOrderIndependent::update_vr_state() {
	vr_multiview_enabled = false;
	xr_view_count = 1;
#ifdef MODULE_ENABLED_XR
	if (XRServer *const *singleton = XRServer::singleton) {
		XRServer *xr_server = *singleton;
		if (xr_server && xr_server->get_xr_environment() && xr_server->get_xr_environment()->get_multiview_enabled()) {
			vr_multiview_enabled = true;
			xr_view_count = 2;
		}
	}
#endif
}

AlphaOrderIndependent::AlphaOrderIndependent() {
	// Initialize tile-based OIT
	// init(); // Lazy initialization
}

AlphaOrderIndependent::~AlphaOrderIndependent() {
	// Cleanup resources
	free();
}

void AlphaOrderIndependent::init() {
	// Initialize VR state
	update_vr_state();

	// Initialize pipelines
	_init_alpha_order_independent_blend_pipeline();
}

void AlphaOrderIndependent::_init_alpha_order_independent_blend_pipeline() {
	// Initialize OIT blend shader (vertex + fragment for resolve pass)
	Vector<String> modes;
	modes.push_back("\n"); // Base mode

	blend.shader.initialize(modes);
	blend.shader_version = blend.shader.version_create();

	// Setup pipeline with alpha blending enabled
	RD::PipelineColorBlendState::Attachment ba;
	ba.enable_blend = true;
	ba.src_color_blend_factor = RD::BLEND_FACTOR_ONE;
	ba.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	ba.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
	ba.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	ba.color_blend_op = RD::BLEND_OP_ADD;
	ba.alpha_blend_op = RD::BLEND_OP_ADD;

	RD::PipelineColorBlendState blend_state;
	blend_state.attachments.push_back(ba);

	blend.pipeline.setup(
			blend.shader.version_get_shader(blend.shader_version, 0),
			RD::RENDER_PRIMITIVE_TRIANGLES,
			RD::PipelineRasterizationState(),
			RD::PipelineMultisampleState(),
			RD::PipelineDepthStencilState(),
			blend_state,
			0);
}

void AlphaOrderIndependent::free() {
	// Free pipeline resources
	if (blend.shader_version.is_valid()) {
		blend.shader.version_free(blend.shader_version);
		blend.shader_version = RID();
	}

	blend.pipeline.clear();
}

void AlphaOrderIndependent::get_tile_count(int p_width, int p_height, uint32_t &r_tile_x, uint32_t &r_tile_y) {
	// Calculate number of tiles needed to cover viewport
	r_tile_x = (p_width + TILE_SIZE - 1) / TILE_SIZE; // Ceiling division
	r_tile_y = (p_height + TILE_SIZE - 1) / TILE_SIZE;
}

uint32_t AlphaOrderIndependent::_get_tile_buffer_index(uint32_t p_tile_x, uint32_t p_tile_y, uint32_t p_depth_bin, uint32_t p_view, uint32_t p_tiles_per_row, uint32_t p_view_count) {
	// Calculate index into tile buffer
	// Layout: ((tile_y * tiles_per_row + tile_x) * DEPTH_BINS_PER_TILE + depth_bin) * view_count + view
	uint32_t tile_idx = p_tile_y * p_tiles_per_row + p_tile_x;
	uint32_t bin_idx = tile_idx * DEPTH_BINS_PER_TILE + p_depth_bin;
	return bin_idx * p_view_count + p_view;
}

void AlphaOrderIndependent::_get_tile_coords(uint32_t p_pixel_x, uint32_t p_pixel_y, uint32_t &r_tile_x, uint32_t &r_tile_y) {
	// Convert pixel coordinates to tile coordinates
	r_tile_x = p_pixel_x / TILE_SIZE;
	r_tile_y = p_pixel_y / TILE_SIZE;
}

void AlphaOrderIndependent::create_alpha_order_independent_buffers(RID &r_tile_buffer, RID &r_fragment_buffer, RID &r_counter_buffer, int p_width, int p_height, bool p_multiview, int p_view_count) {
	uint32_t tile_count_x, tile_count_y;
	get_tile_count(p_width, p_height, tile_count_x, tile_count_y);

	uint32_t total_tiles = tile_count_x * tile_count_y;
	uint32_t bins = DEPTH_BINS_PER_TILE;
	uint32_t views = p_view_count;
	uint32_t total_tile_entries = total_tiles * bins * views;
	uint32_t max_frags_per_tile = MAX_FRAGMENTS_PER_TILE;
	uint32_t max_fragments = total_tiles * max_frags_per_tile * views;

	// Initialize tile buffer with default values: head = 0xFFFFFFFFu, min/max depth extremes, count = 0
	Vector<TileData> tile_init;
	tile_init.resize(total_tile_entries);
	for (uint32_t i = 0; i < total_tile_entries; i++) {
		tile_init.write[i].fragment_head = 0xFFFFFFFFu;
		tile_init.write[i].min_depth = 0xFFFFFFFFu;
		tile_init.write[i].max_depth = 0u;
		tile_init.write[i].fragment_count = 0u;
	}

	PackedByteArray tile_data;
	tile_data.resize(total_tile_entries * sizeof(TileData));
	memcpy(tile_data.ptrw(), tile_init.ptrw(), tile_data.size());
	r_tile_buffer = RD::get_singleton()->storage_buffer_create(total_tile_entries * sizeof(TileData), tile_data, 0, RD::BUFFER_CREATION_AS_STORAGE_BIT);

	// Fragment buffer: no initial data, size for max
	r_fragment_buffer = RD::get_singleton()->storage_buffer_create(max_fragments * sizeof(FragmentData), PackedByteArray(), 0, RD::BUFFER_CREATION_AS_STORAGE_BIT);

	// Counter buffer: init to 0 per view
	Vector<uint32_t> counters;
	counters.resize(views);
	counters.fill(0);
	PackedByteArray counter_data;
	counter_data.resize(views * sizeof(uint32_t));
	memcpy(counter_data.ptrw(), counters.ptrw(), counter_data.size());
	r_counter_buffer = RD::get_singleton()->storage_buffer_create(views * sizeof(uint32_t), counter_data, 0, RD::BUFFER_CREATION_AS_STORAGE_BIT);

	print_line("OIT: Created buffers for " + itos(total_tiles) + " tiles, " + itos(max_fragments) + " max fragments (views: " + itos(p_view_count) + ", multiview: " + (p_multiview ? "true" : "false") + ")");
}

void AlphaOrderIndependent::free_alpha_order_independent_buffers(RID &r_tile_buffer, RID &r_fragment_buffer, RID &r_counter_buffer) {
	// Free buffer resources using RenderingDevice
	// RenderingDevice handles synchronization automatically, no submit() needed

	if (r_tile_buffer.is_valid()) {
		RD::get_singleton()->free_rid(r_tile_buffer);
		r_tile_buffer = RID();
	}

	if (r_fragment_buffer.is_valid()) {
		RD::get_singleton()->free_rid(r_fragment_buffer);
		r_fragment_buffer = RID();
	}

	if (r_counter_buffer.is_valid()) {
		RD::get_singleton()->free_rid(r_counter_buffer);
		r_counter_buffer = RID();
	}
}

void AlphaOrderIndependent::clear_alpha_order_independent_buffers(RID p_tile_buffer, RID p_fragment_buffer, RID p_counter_buffer, int p_width, int p_height, int p_view_count) {
	if (!p_tile_buffer.is_valid() || !p_counter_buffer.is_valid()) {
		return;
	}

	uint32_t tile_count_x, tile_count_y;
	get_tile_count(p_width, p_height, tile_count_x, tile_count_y);

	uint32_t total_tiles = tile_count_x * tile_count_y;
	uint32_t bins = DEPTH_BINS_PER_TILE;
	uint32_t total_tile_entries = total_tiles * bins * p_view_count;

	// RenderingDevice automatically inserts barriers, explicit barrier() deprecated

	// Clear counters to 0 for each view
	PackedByteArray zero_data;
	zero_data.resize(p_view_count * sizeof(uint32_t));
	zero_data.fill(0);
	RD::get_singleton()->buffer_update(p_counter_buffer, 0, zero_data.size(), zero_data.ptr());

	// Clear tile buffer to initial state
	Vector<TileData> clear_tiles;
	clear_tiles.resize(total_tile_entries);
	for (uint32_t i = 0; i < total_tile_entries; i++) {
		clear_tiles.write[i].fragment_head = 0xFFFFFFFFu;
		clear_tiles.write[i].min_depth = 0xFFFFFFFFu;
		clear_tiles.write[i].max_depth = 0u;
		clear_tiles.write[i].fragment_count = 0u;
	}
	PackedByteArray clear_data;
	clear_data.resize(clear_tiles.size() * sizeof(TileData));
	memcpy(clear_data.ptrw(), clear_tiles.ptrw(), clear_data.size());
	RD::get_singleton()->buffer_update(p_tile_buffer, 0, clear_data.size(), clear_data.ptr());
}

void AlphaOrderIndependent::resolve_alpha_order_independent(RID p_tile_buffer, RID p_fragment_buffer, RID p_dst_texture, RID p_depth_texture, int p_width, int p_height, int p_view) {
	if (!p_tile_buffer.is_valid() || !p_fragment_buffer.is_valid() || !p_dst_texture.is_valid()) {
		return;
	}

	// Lazy initialization
	if (!blend.shader_version.is_valid()) {
		init();
	}

	uint32_t tile_count_x, tile_count_y;
	get_tile_count(p_width, p_height, tile_count_x, tile_count_y);

	RD::get_singleton()->draw_command_begin_label("OIT Resolve");

	// Create params UBO
	AlphaOrderIndependentParams params = {};
	params.viewport_width = p_width;
	params.viewport_height = p_height;
	params.tile_count_x = tile_count_x;
	params.tile_count_y = tile_count_y;
	params.tile_size = TILE_SIZE;
	params.max_frags = MAX_FRAGMENTS_PER_TILE;
	params.view_count = xr_view_count;
	params.multiview_enabled = vr_multiview_enabled ? 1 : 0;
	// padding array is zero-initialized by {} above

	RID params_buffer = RD::get_singleton()->uniform_buffer_create(sizeof(AlphaOrderIndependentParams));
	RD::get_singleton()->buffer_update(params_buffer, 0, sizeof(AlphaOrderIndependentParams), &params);

	// Create uniform set for resolve pass
	Vector<RD::Uniform> uniforms;

	// Binding 0: Opaque texture (depth or color from previous pass)
	if (p_depth_texture.is_valid()) {
		RD::Uniform u_texture;
		u_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
		u_texture.binding = 0;
		// UNIFORM_TYPE_SAMPLER_WITH_TEXTURE requires both sampler and texture RIDs
		u_texture.append_id(MaterialStorage::get_singleton()->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
		u_texture.append_id(p_depth_texture);
		uniforms.push_back(u_texture);
	}

	// Binding 1: Tile buffer
	RD::Uniform u_tile;
	u_tile.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	u_tile.binding = 1;
	u_tile.append_id(p_tile_buffer);
	uniforms.push_back(u_tile);

	// Binding 2: Fragment buffer
	RD::Uniform u_frag;
	u_frag.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
	u_frag.binding = 2;
	u_frag.append_id(p_fragment_buffer);
	uniforms.push_back(u_frag);

	// Binding 3: Params UBO
	RD::Uniform u_params;
	u_params.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
	u_params.binding = 3;
	u_params.append_id(params_buffer);
	uniforms.push_back(u_params);

	// Create uniform set for shader
	RID shader = blend.shader.version_get_shader(blend.shader_version, 0);
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, shader, 0);

	// Get framebuffer format from destination texture
	RID framebuffer = FramebufferCacheRD::get_singleton()->get_cache(p_dst_texture);
	RD::FramebufferFormatID fb_format = RD::get_singleton()->framebuffer_get_format(framebuffer);

	// Begin draw list and bind pipeline
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(framebuffer);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blend.pipeline.get_render_pipeline(RD::INVALID_ID, fb_format));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set, 0);

	// Draw fullscreen triangle (no vertex buffer needed, generated in vertex shader)
	RD::get_singleton()->draw_list_draw(draw_list, false, 1u, 3u);
	RD::get_singleton()->draw_list_end();

	// Cleanup temporary resources
	RD::get_singleton()->free_rid(uniform_set);
	RD::get_singleton()->free_rid(params_buffer);

	RD::get_singleton()->draw_command_end_label();
}

} // namespace RendererRD
