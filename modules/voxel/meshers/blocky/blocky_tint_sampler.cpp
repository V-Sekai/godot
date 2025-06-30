/**************************************************************************/
/*  blocky_tint_sampler.cpp                                               */
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

#include "blocky_tint_sampler.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/errors.h"
#include "../../util/math/color8.h"

namespace zylann::voxel::blocky {

TintSampler TintSampler::create(const VoxelBuffer &p_voxels, const Mode mode) {
	switch (mode) {
		case MODE_NONE:
			break;

		case MODE_RAW: {
			static constexpr VoxelBuffer::ChannelId channel = VoxelBuffer::CHANNEL_COLOR;
			const VoxelBuffer::Depth depth = p_voxels.get_channel_depth(channel);

			switch (depth) {
				case VoxelBuffer::DEPTH_16_BIT:
					return { [](const TintSampler &self, const Vector3i pos) {
								const uint32_t v = self.voxels.get_voxel(pos, channel);
								return Color(Color8::from_u16(v));
							},
							 p_voxels };

				case VoxelBuffer::DEPTH_32_BIT:
					return { [](const TintSampler &self, const Vector3i pos) {
								const uint32_t v = self.voxels.get_voxel(pos, channel);
								return Color(Color8::from_u32(v));
							},
							 p_voxels };

				default:
					ZN_PRINT_ERROR_ONCE("Color channel depth not supported");
					break;
			}
		} break;

		default:
			ZN_PRINT_ERROR_ONCE("Unknown mode");
			break;
	}

	return { nullptr, p_voxels };
}

} // namespace zylann::voxel::blocky
