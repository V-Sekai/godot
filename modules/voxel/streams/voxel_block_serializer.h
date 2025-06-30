/**************************************************************************/
/*  voxel_block_serializer.h                                              */
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

#include "../util/containers/span.h"
#include "../util/containers/std_vector.h"
#include "../util/godot/macros.h"

#include <cstdint>

ZN_GODOT_FORWARD_DECLARE(class FileAccess)
#ifdef ZN_GODOT_EXTENSION
using namespace godot;
#endif

namespace zylann::voxel {

class VoxelBuffer;

namespace BlockSerializer {

// Latest version, used when serializing
static const uint8_t BLOCK_FORMAT_VERSION = 4;

struct SerializeResult {
	// The lifetime of the pointed object is only valid in the calling thread,
	// until another serialization or deserialization call is made.
	// TODO Eventually figure out allocators so the caller can decide
	const StdVector<uint8_t> &data;
	bool success;

	inline SerializeResult(const StdVector<uint8_t> &p_data, bool p_success) : data(p_data), success(p_success) {}
};

SerializeResult serialize(const VoxelBuffer &voxel_buffer);
bool deserialize(Span<const uint8_t> p_data, VoxelBuffer &out_voxel_buffer);

SerializeResult serialize_and_compress(const VoxelBuffer &voxel_buffer);
bool decompress_and_deserialize(Span<const uint8_t> p_data, VoxelBuffer &out_voxel_buffer);
bool decompress_and_deserialize(FileAccess &f, unsigned int size_to_read, VoxelBuffer &out_voxel_buffer);

// Temporary thread-local buffers for internal use
StdVector<uint8_t> &get_tls_data();
StdVector<uint8_t> &get_tls_compressed_data();

} // namespace BlockSerializer
} // namespace zylann::voxel
