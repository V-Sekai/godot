/**************************************************************************/
/*  voxel_generator_flat.h                                                */
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

#include "../../constants/voxel_constants.h"
#include "../../storage/voxel_buffer.h"
#include "../../storage/voxel_buffer_gd.h"
#include "../../util/thread/rw_lock.h"
#include "../voxel_generator.h"

namespace zylann::voxel {

class VoxelGeneratorFlat : public VoxelGenerator {
	GDCLASS(VoxelGeneratorFlat, VoxelGenerator)

public:
	VoxelGeneratorFlat();
	~VoxelGeneratorFlat();

	void set_channel(VoxelBuffer::ChannelId p_channel);
	VoxelBuffer::ChannelId get_channel() const;
	int get_used_channels_mask() const override;

	Result generate_block(VoxelGenerator::VoxelQueryData input) override;

	void set_voxel_type(int t);
	int get_voxel_type() const;

	void set_height(float h);
	float get_height() const;

protected:
	static void _bind_methods();

private:
	void _b_set_channel(godot::VoxelBuffer::ChannelId p_channel);
	godot::VoxelBuffer::ChannelId _b_get_channel() const;

	struct Parameters {
		VoxelBuffer::ChannelId channel = VoxelBuffer::CHANNEL_SDF;
		int voxel_type = 1;
		float height = 0;
		float iso_scale = 1.f;
	};

	Parameters _parameters;
	RWLock _parameters_lock;
};

} // namespace zylann::voxel
