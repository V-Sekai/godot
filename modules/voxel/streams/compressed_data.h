/**************************************************************************/
/*  compressed_data.h                                                     */
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
#include <cstdint>

namespace zylann::voxel::CompressedData {

// Compressed data starts with a single byte telling which compression format is used.
// What follows depends on it.

enum Compression {
	// No compression. All following bytes can be read as-is.
	// Could be used for debugging.
	COMPRESSION_NONE = 0,
	// [deprecated]
	// The next uint32_t will be the size of decompressed data in big endian format.
	// All following bytes are compressed data using LZ4 defaults.
	// This is the fastest compression format.
	COMPRESSION_LZ4_BE = 1,
	// The next uint32_t will be the size of decompressed data (little endian).
	// All following bytes are compressed data using LZ4 defaults.
	// This is the fastest compression format.
	COMPRESSION_LZ4 = 2,
	COMPRESSION_COUNT = 3
};

bool compress(Span<const uint8_t> src, StdVector<uint8_t> &dst, Compression comp);
bool decompress(Span<const uint8_t> src, StdVector<uint8_t> &dst);

} // namespace zylann::voxel::CompressedData
