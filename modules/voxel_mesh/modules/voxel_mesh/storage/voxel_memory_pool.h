/**************************************************************************/
/*  voxel_memory_pool.h                                                   */
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

#ifndef VOXEL_MEMORY_POOL_H
#define VOXEL_MEMORY_POOL_H

#include "core/os/mutex.h"
#include "core/templates/hash_map.h"

#include <vector>

// Pool based on a scenario where allocated blocks are often the same size.
// A pool of blocks is assigned for each size.
class VoxelMemoryPool {
private:
	struct Pool {
		std::vector<uint8_t *> blocks;
	};

public:
	static void create_singleton();
	static void destroy_singleton();
	static VoxelMemoryPool *get_singleton();

	VoxelMemoryPool();
	~VoxelMemoryPool();

	uint8_t *allocate(uint32_t size);
	void recycle(uint8_t *block, uint32_t size);

	void debug_print();
	unsigned int debug_get_used_blocks() const;

private:
	Pool *get_or_create_pool(uint32_t size);
	void clear();

	HashMap<uint32_t, Pool *> _pools;
	unsigned int _used_blocks = 0;
	Mutex _mutex;
};

#endif // VOXEL_MEMORY_POOL_H
