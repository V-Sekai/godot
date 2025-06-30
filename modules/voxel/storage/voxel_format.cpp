/**************************************************************************/
/*  voxel_format.cpp                                                      */
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

#include "voxel_format.h"
#include "mixel4.h"

namespace zylann::voxel {

VoxelFormat::VoxelFormat() {
	depths[VoxelBuffer::CHANNEL_TYPE] = VoxelBuffer::DEFAULT_TYPE_CHANNEL_DEPTH;
	depths[VoxelBuffer::CHANNEL_SDF] = VoxelBuffer::DEFAULT_SDF_CHANNEL_DEPTH;
	depths[VoxelBuffer::CHANNEL_COLOR] = VoxelBuffer::DEFAULT_CHANNEL_DEPTH;
	depths[VoxelBuffer::CHANNEL_INDICES] = VoxelBuffer::DEFAULT_INDICES_CHANNEL_DEPTH;
	depths[VoxelBuffer::CHANNEL_WEIGHTS] = VoxelBuffer::DEFAULT_WEIGHTS_CHANNEL_DEPTH;
	depths[VoxelBuffer::CHANNEL_DATA5] = VoxelBuffer::DEFAULT_CHANNEL_DEPTH;
	depths[VoxelBuffer::CHANNEL_DATA6] = VoxelBuffer::DEFAULT_CHANNEL_DEPTH;
	depths[VoxelBuffer::CHANNEL_DATA7] = VoxelBuffer::DEFAULT_CHANNEL_DEPTH;
}

void VoxelFormat::configure_buffer(VoxelBuffer &vb) const {
	// Clear keeping size
	if (vb.get_size() == Vector3i()) {
		vb.clear(this);
	} else {
		vb.create(vb.get_size(), this);
	}
}

VoxelFormat::DepthRange VoxelFormat::get_supported_depths(const VoxelBuffer::ChannelId channel_id) {
	switch (channel_id) {
		case VoxelBuffer::CHANNEL_TYPE:
			return { 1, 2 };
		case VoxelBuffer::CHANNEL_SDF:
			return { 1, 4 };
		case VoxelBuffer::CHANNEL_COLOR:
			return { 1, 4 };
		case VoxelBuffer::CHANNEL_INDICES:
			return { 1, 2 };
		case VoxelBuffer::CHANNEL_WEIGHTS:
			return { 2, 2 };
		case VoxelBuffer::CHANNEL_DATA5:
		case VoxelBuffer::CHANNEL_DATA6:
		case VoxelBuffer::CHANNEL_DATA7:
			return { 1, 4 };
		default:
			ZN_PRINT_ERROR("Unknown channel");
			return { 1, 1 };
	}
}

} // namespace zylann::voxel
