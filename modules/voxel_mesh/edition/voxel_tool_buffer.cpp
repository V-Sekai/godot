/**************************************************************************/
/*  voxel_tool_buffer.cpp                                                 */
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

#include "voxel_tool_buffer.h"
#include "../storage/voxel_buffer.h"
#include "../util/profiling.h"
#include "funcs.h"

VoxelToolBuffer::VoxelToolBuffer(Ref<VoxelBuffer> vb) {
	ERR_FAIL_COND(vb.is_null());
	_buffer = vb;
}

bool VoxelToolBuffer::is_area_editable(const Box3i &box) const {
	ERR_FAIL_COND_V(_buffer.is_null(), false);
	return Box3i(VoxelVector3i(), _buffer->get_size()).encloses(box);
}

void VoxelToolBuffer::do_sphere(Vector3 center, float radius) {
	ERR_FAIL_COND(_buffer.is_null());

	if (_mode != MODE_TEXTURE_PAINT) {
		// TODO Eventually all specialized voxel tools should use lambda writing
		// functions
		VoxelTool::do_sphere(center, radius);
		return;
	}

	VOXEL_PROFILE_SCOPE();

	Box3i box(VoxelVector3i(center) - VoxelVector3i(Math::floor(radius)),
			VoxelVector3i(Math::ceil(radius) * 2));
	box.clip(Box3i(VoxelVector3i(), _buffer->get_size()));

	_buffer->write_box_2_template<TextureBlendSphereOp, uint16_t, uint16_t>(
			box, VoxelBuffer::CHANNEL_INDICES, VoxelBuffer::CHANNEL_WEIGHTS,
			TextureBlendSphereOp(center, radius, _texture_params), VoxelVector3i());

	_post_edit(box);
}

uint64_t VoxelToolBuffer::_get_voxel(VoxelVector3i pos) const {
	ERR_FAIL_COND_V(_buffer.is_null(), 0);
	return _buffer->get_voxel(pos, _channel);
}

float VoxelToolBuffer::_get_voxel_f(VoxelVector3i pos) const {
	ERR_FAIL_COND_V(_buffer.is_null(), 0);
	return _buffer->get_voxel_f(pos.x, pos.y, pos.z, _channel);
}

void VoxelToolBuffer::_set_voxel(VoxelVector3i pos, uint64_t v) {
	ERR_FAIL_COND(_buffer.is_null());
	return _buffer->set_voxel(v, pos, _channel);
}

void VoxelToolBuffer::_set_voxel_f(VoxelVector3i pos, float v) {
	ERR_FAIL_COND(_buffer.is_null());
	return _buffer->set_voxel_f(v, pos.x, pos.y, pos.z, _channel);
}

void VoxelToolBuffer::_post_edit(const Box3i &box) {
	ERR_FAIL_COND(_buffer.is_null());
	// Nothing special to do
}

void VoxelToolBuffer::set_voxel_metadata(VoxelVector3i pos, Variant meta) {
	ERR_FAIL_COND(_buffer.is_null());
	_buffer->set_voxel_metadata(pos, meta);
}

Variant VoxelToolBuffer::get_voxel_metadata(VoxelVector3i pos) {
	ERR_FAIL_COND_V(_buffer.is_null(), Variant());
	return _buffer->get_voxel_metadata(pos);
}

void VoxelToolBuffer::paste(VoxelVector3i p_pos, Ref<VoxelBuffer> p_voxels,
		uint8_t channels_mask, uint64_t mask_value) {
	ERR_FAIL_COND(_buffer.is_null());
	ERR_FAIL_COND(p_voxels.is_null());

	VoxelBuffer *dst = *_buffer;
	const VoxelBuffer *src = *p_voxels;

	Box3i box(p_pos, p_voxels->get_size());
	const VoxelVector3i min_noclamp = box.pos;
	box.clip(Box3i(VoxelVector3i(), _buffer->get_size()));

	if (channels_mask == 0) {
		channels_mask = (1 << get_channel());
	}

	unsigned int channel_count;
	FixedArray<uint8_t, VoxelBuffer::MAX_CHANNELS> channels =
			VoxelBuffer::mask_to_channels_list(channels_mask, channel_count);

	const VoxelVector3i box_max = box.pos + box.size;

	for (unsigned int ci = 0; ci < channel_count; ++ci) {
		const unsigned int channel_index = channels[ci];

		for (int z = box.pos.z; z < box_max.z; ++z) {
			const int bz = z - min_noclamp.z;

			for (int x = box.pos.x; x < box_max.x; ++x) {
				const int bx = x - min_noclamp.x;

				for (int y = box.pos.y; y < box_max.y; ++y) {
					const int by = y - min_noclamp.y;

					const uint64_t v = src->get_voxel(bx, by, bz, channel_index);
					if (v != mask_value) {
						dst->set_voxel(v, x, y, z, channel_index);

						// Overwrite previous metadata
						dst->set_voxel_metadata(VoxelVector3i(x, y, z), Variant());
					}
				}
			}
		}
	}

	_buffer->copy_voxel_metadata_in_area(
			p_voxels, Box3i(VoxelVector3i(), p_voxels->get_size()), p_pos);
}
