/**************************************************************************/
/*  voxel_stream_cache.cpp                                                */
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

#include "voxel_stream_cache.h"

namespace zylann::voxel {

bool VoxelStreamCache::load_voxel_block(Vector3i position, uint8_t lod_index, VoxelBuffer &out_voxels) {
	const Lod &lod = _cache[lod_index];

	RWLockRead rlock(lod.rw_lock);

	auto it = lod.blocks.find(position);

	if (it == lod.blocks.end()) {
		// Not in cache, will have to query
		return false;

	} else {
		const Block &block = it->second;
		if (!block.has_voxels) {
			// Has a block in cache but there is no voxel data
			return false;
		}
		// In cache, serve it

		// Copying is required since the cache has ownership on its data,
		// and the requests wants us to populate the buffer it provides
		block.voxels.copy_to(out_voxels, true);

		return true;
	}
}

void VoxelStreamCache::save_voxel_block(Vector3i position, uint8_t lod_index, VoxelBuffer &voxels) {
	Lod &lod = _cache[lod_index];
	RWLockWrite wlock(lod.rw_lock);
	auto it = lod.blocks.find(position);

	ZN_ASSERT_RETURN_MSG(
			!Vector3iUtil::is_empty_size(voxels.get_size()), "Saving voxel buffer with empty size is not expected. Bug?"
	);

	if (it == lod.blocks.end()) {
		// Not cached yet, create an entry
		Block b;
		b.position = position;
		b.lod = lod_index;
		// TODO Optimization: if we know the buffer is not shared, we could use move instead
		voxels.copy_to(b.voxels, true);
		b.has_voxels = true;
		lod.blocks.insert(std::make_pair(position, std::move(b)));
		++_count;

	} else {
		// Cached already, overwrite
		voxels.move_to(it->second.voxels);
		it->second.has_voxels = true;
	}
}

#ifdef VOXEL_ENABLE_INSTANCER

bool VoxelStreamCache::load_instance_block(
		Vector3i position,
		uint8_t lod_index,
		UniquePtr<InstanceBlockData> &out_instances
) {
	const Lod &lod = _cache[lod_index];
	lod.rw_lock.read_lock();
	auto it = lod.blocks.find(position);

	if (it == lod.blocks.end()) {
		// Not in cache, will have to query
		lod.rw_lock.read_unlock();
		return false;

	} else {
		// In cache, serve it

		if (it->second.instances == nullptr) {
			out_instances = nullptr;

		} else {
			// Copying is required since the cache has ownership on its data
			out_instances = make_unique_instance<InstanceBlockData>();
			it->second.instances->copy_to(*out_instances);
		}

		lod.rw_lock.read_unlock();
		return true;
	}
}

void VoxelStreamCache::save_instance_block(
		Vector3i position,
		uint8_t lod_index,
		UniquePtr<InstanceBlockData> instances
) {
	Lod &lod = _cache[lod_index];
	RWLockWrite wlock(lod.rw_lock);
	auto it = lod.blocks.find(position);

	if (it == lod.blocks.end()) {
		// Not cached yet, create an entry
		Block b;
		b.position = position;
		b.lod = lod_index;
		b.instances = std::move(instances);
		lod.blocks.insert(std::make_pair(position, std::move(b)));
		++_count;

	} else {
		// Cached already, overwrite
		it->second.instances = std::move(instances);
	}
}

#endif

unsigned int VoxelStreamCache::get_indicative_block_count() const {
	return _count;
}

} // namespace zylann::voxel
