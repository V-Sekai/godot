/**************************************************************************/
/*  custom_voxel_metadata.h                                               */
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

#include "../../util/containers/span.h"

namespace zylann::voxel {

// Base interface for custom data types.
class ICustomVoxelMetadata {
public:
	virtual ~ICustomVoxelMetadata() {}

	// Gets how many bytes this metadata will occupy when serialized.
	virtual size_t get_serialized_size() const = 0;

	// Serializes this metadata into `dst`. The size of `dst` will be equal or greater than the size returned by
	// `get_serialized_size()`. Returns how many bytes were written.
	virtual size_t serialize(Span<uint8_t> dst) const = 0;

	// Deserializes this metadata from the given bytes.
	// Returns `true` on success, `false` otherwise. `out_read_size` must be assigned to the number of bytes read.
	virtual bool deserialize(Span<const uint8_t> src, uint64_t &out_read_size) = 0;

	virtual ICustomVoxelMetadata *duplicate() = 0;

	// Returns the type index used in metadata tagging (mainly used for debug checks)
	virtual uint8_t get_type_index() const = 0;

	virtual bool equals(const ICustomVoxelMetadata &other) const = 0;
};

} // namespace zylann::voxel
