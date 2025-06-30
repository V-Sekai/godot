/**************************************************************************/
/*  voxel_stream_memory.h                                                 */
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

#include "../constants/voxel_constants.h"
#include "../storage/voxel_buffer.h"
#include "../util/containers/fixed_array.h"
#include "../util/containers/span.h"
#include "../util/containers/std_unordered_map.h"
#include "../util/math/vector3i.h"
#include "../util/memory/memory.h"
#include "../util/thread/mutex.h"
#include "instance_data.h"
#include "voxel_stream.h"

namespace zylann::voxel {

// "fake" stream that just stores copies of the data in memory instead of saving them to the filesystem. May be used for
// testing.
class VoxelStreamMemory : public VoxelStream {
	GDCLASS(VoxelStreamMemory, VoxelStream)
public:
	void load_voxel_blocks(Span<VoxelQueryData> p_blocks) override;
	void save_voxel_blocks(Span<VoxelQueryData> p_blocks) override;
	void load_voxel_block(VoxelQueryData &query_data) override;
	void save_voxel_block(VoxelQueryData &query_data) override;

#ifdef VOXEL_ENABLE_INSTANCER
	bool supports_instance_blocks() const override;
	void load_instance_blocks(Span<InstancesQueryData> out_blocks) override;
	void save_instance_blocks(Span<InstancesQueryData> p_blocks) override;
#endif

	bool supports_loading_all_blocks() const override;
	void load_all_blocks(FullLoadingResult &result) override;

	int get_used_channels_mask() const override;

	int get_lod_count() const override;

	void set_artificial_save_latency_usec(int usec);
	int get_artificial_save_latency_usec() const;

private:
	static void _bind_methods();

	struct VoxelChunk {
		VoxelBuffer voxels;
		VoxelChunk() : voxels(VoxelBuffer::ALLOCATOR_POOL) {}
	};

	struct Lod {
		StdUnorderedMap<Vector3i, VoxelChunk> voxel_blocks;
		StdUnorderedMap<Vector3i, InstanceBlockData> instance_blocks;
		Mutex mutex;
	};

	FixedArray<Lod, constants::MAX_LOD> _lods;
	unsigned int _artificial_save_latency_usec = 0;
};

} // namespace zylann::voxel
