/**************************************************************************/
/*  gpu_storage_buffer_pool.h                                             */
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

#include "../../util/containers/fixed_array.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/rendering_device.h"
#include <array>

namespace zylann::voxel {

struct GPUStorageBuffer {
	RID rid;
	size_t size = 0;

	inline bool is_null() const {
		// Can't use `is_null()`, core has it but GDExtension doesn't have it
		return !rid.is_valid();
	}

	inline bool is_valid() const {
		return rid.is_valid();
	}
};

// Pools storage buffers of specific sizes so they can be reused.
// Not thread-safe.
class GPUStorageBufferPool {
public:
	GPUStorageBufferPool();
	~GPUStorageBufferPool();

	void clear();
	void set_rendering_device(RenderingDevice *rd);
	GPUStorageBuffer allocate(const PackedByteArray &pba);
	GPUStorageBuffer allocate(uint32_t p_size);
	void recycle(GPUStorageBuffer b);
	void debug_print() const;

private:
	GPUStorageBuffer allocate(uint32_t p_size, const PackedByteArray *pba);

	unsigned int get_pool_index_from_size(uint32_t p_size) const;

	struct Pool {
		StdVector<GPUStorageBuffer> buffers;
		unsigned int used_buffers = 0;
	};

	// Up to roughly 800 Mb with the current size formula
	static const unsigned int POOL_COUNT = 48;

	std::array<uint32_t, POOL_COUNT> _pool_sizes;
	FixedArray<Pool, POOL_COUNT> _pools;
	RenderingDevice *_rendering_device = nullptr;
};

} // namespace zylann::voxel
