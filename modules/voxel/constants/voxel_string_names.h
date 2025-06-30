/**************************************************************************/
/*  voxel_string_names.h                                                  */
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

#include "../util/containers/fixed_array.h"
#include "../util/godot/core/string_name.h"
#include "../util/math/ortho_basis.h"

namespace zylann::voxel {

class VoxelStringNames {
private:
	static VoxelStringNames *g_singleton;

public:
	static const VoxelStringNames &get_singleton();
	static void create_singleton();
	static void destroy_singleton();

	VoxelStringNames();

	StringName _emerge_block;
	StringName _immerge_block;
	StringName _generate_block;
	StringName _get_used_channels_mask;

	StringName block_loaded;
	StringName block_unloaded;

	StringName mesh_block_entered;
	StringName mesh_block_exited;

	StringName store_colors_in_texture;
	StringName scale;
	StringName enable_baked_lighting;
	StringName pivot_mode;

	StringName u_transition_mask;
	StringName u_block_local_transform;
	StringName u_lod_fade;

	StringName voxel_normalmap_atlas;
	StringName voxel_normalmap_lookup;

	StringName u_voxel_normalmap_atlas;
	StringName u_voxel_cell_lookup;
	StringName u_voxel_cell_size;
	StringName u_voxel_block_size;
	StringName u_voxel_virtual_texture_fade;
	StringName u_voxel_virtual_texture_tile_size;
	StringName u_voxel_virtual_texture_offset_scale;
	StringName u_voxel_lod_info;

#ifdef DEBUG_ENABLED
	StringName _voxel_debug_vt_position;
#endif

	// These are usually in CoreStringNames, but when compiling as a GDExtension, we don't have access to them
	StringName changed;
	StringName frame_post_draw;

#ifdef TOOLS_ENABLED
	StringName Add;
	StringName Remove;
	StringName EditorIcons;
	StringName EditorFonts;
	StringName Pin;
	StringName ExternalLink;
	StringName Search;
	StringName source;
	StringName _dummy_function;
	StringName grab_focus;

	StringName font;
	StringName font_size;
	StringName font_color;
	StringName Label;
	StringName Editor;
#endif

	StringName _rpc_receive_blocks;
	StringName _rpc_receive_area;

	StringName unnamed;
	StringName air;
	StringName cube;

	StringName axis;
	StringName direction;
	StringName rotation;
	StringName x;
	StringName y;
	StringName z;
	StringName negative_x;
	StringName negative_y;
	StringName negative_z;
	StringName positive_x;
	StringName positive_y;
	StringName positive_z;

	FixedArray<StringName, math::ORTHO_ROTATION_COUNT> ortho_rotation_names;
	String ortho_rotation_enum_hint_string;

	StringName compiled;

	StringName _on_async_search_completed;
	StringName async_search_completed;

	StringName file_selected;

	StringName jitter;
	StringName triangle_area_threshold;
	StringName density;
	StringName noise_dimension;
	StringName noise_on_scale;

	StringName add_child;
};

} // namespace zylann::voxel
