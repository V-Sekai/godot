/**************************************************************************/
/*  voxel_memory_pool.cpp                                                 */
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

#include "voxel_memory_pool.h"
#include "../util/macros.h"
#include "../util/profiling.h"

#include <core/os/os.h>
#include <core/variant/variant.h>

namespace {
VoxelMemoryPool *g_memory_pool = nullptr;
} // namespace

void VoxelMemoryPool::create_singleton() {
	CRASH_COND(g_memory_pool != nullptr);
	g_memory_pool = memnew(VoxelMemoryPool);
}

void VoxelMemoryPool::destroy_singleton() {
	CRASH_COND(g_memory_pool == nullptr);
	VoxelMemoryPool *pool = g_memory_pool;
	g_memory_pool = nullptr;
	memdelete(pool);
}

VoxelMemoryPool *VoxelMemoryPool::get_singleton() {
	CRASH_COND(g_memory_pool == nullptr);
	return g_memory_pool;
}

VoxelMemoryPool::VoxelMemoryPool() {}

VoxelMemoryPool::~VoxelMemoryPool() {
#ifdef TOOLS_ENABLED
	if (OS::get_singleton()->is_stdout_verbose()) {
		debug_print();
	}
#endif
	clear();
}

uint8_t *VoxelMemoryPool::allocate(uint32_t size) {
	VOXEL_PROFILE_SCOPE();
	MutexLock lock(_mutex);
	Pool *pool = get_or_create_pool(size);
	uint8_t *block;
	if (pool->blocks.size() > 0) {
		block = pool->blocks.back();
		pool->blocks.pop_back();
	} else {
		block = (uint8_t *)memalloc(size * sizeof(uint8_t));
	}
	++_used_blocks;
	return block;
}

void VoxelMemoryPool::recycle(uint8_t *block, uint32_t size) {
	MutexLock lock(_mutex);
	Pool *pool =
			_pools[size]; // If not found, entry will be created! It would be an error
	// Check recycling before having allocated
	CRASH_COND(pool == nullptr);
	pool->blocks.push_back(block);
	--_used_blocks;
}

void VoxelMemoryPool::clear() {
	MutexLock lock(_mutex);
	for (const KeyValue<uint32_t, VoxelMemoryPool::Pool *> &E : _pools) {
		Pool *pool = _pools.get(E.key);
		CRASH_COND(pool == nullptr);
		for (auto it = pool->blocks.begin(); it != pool->blocks.end(); ++it) {
			uint8_t *ptr = *it;
			CRASH_COND(ptr == nullptr);
			memfree(ptr);
		}
	}
	_pools.clear();
}

void VoxelMemoryPool::debug_print() {
	MutexLock lock(_mutex);
	print_line("-------- VoxelMemoryPool ----------");
	if (_pools.size() == 0) {
		print_line("No pools created");
	} else {
		for (const KeyValue<uint32_t, Pool *> &E : _pools) {
			Pool *pool = _pools.get(E.key);
			print_line(String("Pool for size {1}: {2} blocks").format(varray(E.key, SIZE_T_TO_VARIANT(pool->blocks.size()))));
		}
	}
}

unsigned int VoxelMemoryPool::debug_get_used_blocks() const {
	MutexLock lock(_mutex);
	return _used_blocks;
}

VoxelMemoryPool::Pool *VoxelMemoryPool::get_or_create_pool(uint32_t size) {
	Pool *pool;
	Pool **ppool = _pools.getptr(size);
	if (ppool == nullptr) {
		pool = memnew(Pool);
		CRASH_COND(pool == nullptr);
		_pools[size] = pool;
	} else {
		pool = *ppool;
		CRASH_COND(pool == nullptr);
	}
	return pool;
}
