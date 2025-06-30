/**************************************************************************/
/*  voxel_data_block_enter_info.cpp                                       */
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

#include "voxel_data_block_enter_info.h"
#include "../storage/voxel_buffer_gd.h"

namespace zylann::voxel {

int VoxelDataBlockEnterInfo::_b_get_network_peer_id() const {
	return network_peer_id;
}

Ref<godot::VoxelBuffer> VoxelDataBlockEnterInfo::_b_get_voxels() const {
	ERR_FAIL_COND_V(!voxel_block.has_voxels(), Ref<godot::VoxelBuffer>());
	std::shared_ptr<VoxelBuffer> vbi = voxel_block.get_voxels_shared();
	Ref<godot::VoxelBuffer> vb = godot::VoxelBuffer::create_shared(vbi);
	return vb;
}

Vector3i VoxelDataBlockEnterInfo::_b_get_position() const {
	return block_position;
}

int VoxelDataBlockEnterInfo::_b_get_lod_index() const {
	return voxel_block.get_lod_index();
}

bool VoxelDataBlockEnterInfo::_b_are_voxels_edited() const {
	return voxel_block.is_edited();
}

void VoxelDataBlockEnterInfo::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_network_peer_id"), &VoxelDataBlockEnterInfo::_b_get_network_peer_id);
	ClassDB::bind_method(D_METHOD("get_voxels"), &VoxelDataBlockEnterInfo::_b_get_voxels);
	ClassDB::bind_method(D_METHOD("get_position"), &VoxelDataBlockEnterInfo::_b_get_position);
	ClassDB::bind_method(D_METHOD("get_lod_index"), &VoxelDataBlockEnterInfo::_b_get_lod_index);
	ClassDB::bind_method(D_METHOD("are_voxels_edited"), &VoxelDataBlockEnterInfo::_b_are_voxels_edited);
}

} // namespace zylann::voxel
