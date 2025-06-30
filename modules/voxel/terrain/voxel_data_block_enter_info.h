/**************************************************************************/
/*  voxel_data_block_enter_info.h                                         */
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

#include "../storage/voxel_data_block.h"
#include "../util/godot/classes/object.h"

namespace zylann::voxel {

namespace godot {
class VoxelBuffer;
}

// Information sent with data block entering notifications.
// It is a class for script API convenience.
// You may neither create this object on your own, nor keep a reference to it.
class VoxelDataBlockEnterInfo : public Object {
	GDCLASS(VoxelDataBlockEnterInfo, Object)
public:
	int network_peer_id = -1;
	Vector3i block_position;
	// Shallow copy of the block. We don't use a pointer due to thread-safety, so this information represents only the
	// moment where the block was inserted into the map.
	VoxelDataBlock voxel_block;

private:
	int _b_get_network_peer_id() const;
	Ref<godot::VoxelBuffer> _b_get_voxels() const;
	Vector3i _b_get_position() const;
	int _b_get_lod_index() const;
	bool _b_are_voxels_edited() const;
	// int _b_viewer_id() const;

	static void _bind_methods();
};

} // namespace zylann::voxel
