/**************************************************************************/
/*  vox_loader.cpp                                                        */
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

#include "vox_loader.h"
#include "../meshers/cubes/voxel_color_palette.h"
#include "../storage/voxel_buffer.h"
#include "vox_data.h"

Error VoxelVoxLoader::load_from_file(String fpath, Ref<VoxelBuffer> voxels,
		Ref<VoxelColorPalette> palette) {
	ERR_FAIL_COND_V(voxels.is_null(), ERR_INVALID_PARAMETER);

	vox::Data data;
	Error load_err = data.load_from_file(fpath);
	ERR_FAIL_COND_V(load_err != OK, load_err);

	const vox::Model &model = data.get_model(0);

	const VoxelBuffer::ChannelId channel = VoxelBuffer::CHANNEL_COLOR;
	Span<const Color8> src_palette = to_span_const(data.get_palette());
	const VoxelBuffer::Depth depth =
			voxels->get_channel_depth(VoxelBuffer::CHANNEL_COLOR);

	Span<uint8_t> dst_raw;
	voxels->create(model.size);
	voxels->decompress_channel(channel);
	CRASH_COND(!voxels->get_channel_raw(channel, dst_raw));

	if (palette.is_valid()) {
		for (size_t i = 0; i < src_palette.size(); ++i) {
			palette->set_color8(i, src_palette[i]);
		}

		switch (depth) {
			case VoxelBuffer::DEPTH_8_BIT: {
				memcpy(dst_raw.data(), model.color_indexes.data(),
						model.color_indexes.size());
			} break;

			case VoxelBuffer::DEPTH_16_BIT: {
				Span<uint16_t> dst = dst_raw.reinterpret_cast_to<uint16_t>();
				for (size_t i = 0; i < dst.size(); ++i) {
					dst[i] = model.color_indexes[i];
				}
			} break;

			default:
				ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Unsupported depth");
				break;
		}

	} else {
		switch (depth) {
			case VoxelBuffer::DEPTH_8_BIT: {
				for (size_t i = 0; i < dst_raw.size(); ++i) {
					const uint8_t ci = model.color_indexes[i];
					dst_raw[i] = src_palette[ci].to_u8();
				}
			} break;

			case VoxelBuffer::DEPTH_16_BIT: {
				Span<uint16_t> dst = dst_raw.reinterpret_cast_to<uint16_t>();
				for (size_t i = 0; i < dst.size(); ++i) {
					const uint8_t ci = model.color_indexes[i];
					dst[i] = src_palette[ci].to_u16();
				}
			} break;

			default:
				ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Unsupported depth");
				break;
		}
	}

	return load_err;
}

void VoxelVoxLoader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_from_file", "fpath", "voxels", "palette"),
			&VoxelVoxLoader::load_from_file);
}
