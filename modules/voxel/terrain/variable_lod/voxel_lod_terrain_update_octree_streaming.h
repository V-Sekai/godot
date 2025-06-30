/**************************************************************************/
/*  voxel_lod_terrain_update_octree_streaming.h                           */
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

#include "../../storage/voxel_data.h"
#include "voxel_lod_terrain_update_data.h"

namespace zylann::voxel {

// Limitations:
// - Supports only one viewer
// - Still assumes a viewer exists at world origin if there is actually no viewer
// - Does not support viewer flags (separate collision/visual/voxel requirements)

void process_octree_streaming(
		VoxelLodTerrainUpdateData::State &state,
		VoxelData &data,
		Vector3 viewer_pos,
		StdVector<VoxelData::BlockToSave> *data_blocks_to_save,
		StdVector<VoxelLodTerrainUpdateData::BlockToLoad> &data_blocks_to_load,
		const VoxelLodTerrainUpdateData::Settings &settings,
		bool stream_enabled
);

} // namespace zylann::voxel
