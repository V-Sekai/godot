/**************************************************************************/
/*  voxel_format.h                                                        */
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

#include "voxel_buffer.h"
#include <array>

namespace zylann::voxel {

struct VoxelFormat {
	VoxelFormat();

	void configure_buffer(VoxelBuffer &vb) const;

	bool operator==(const VoxelFormat &other) const {
		return depths == other.depths;
	}

	struct DepthRange {
		uint32_t min;
		uint32_t max;

		inline bool contains(uint32_t i) const {
			return i >= min && i <= max;
		}
	};

	inline uint64_t get_default_raw_value(const VoxelBuffer::ChannelId channel_id) const {
		return VoxelBuffer::get_default_raw_value(channel_id, depths[channel_id]);
	}

	static DepthRange get_supported_depths(const VoxelBuffer::ChannelId channel_id);
	static uint64_t get_default_sdf_raw_value(const VoxelBuffer::Depth depth);

	std::array<VoxelBuffer::Depth, VoxelBuffer::MAX_CHANNELS> depths;
};

} // namespace zylann::voxel
