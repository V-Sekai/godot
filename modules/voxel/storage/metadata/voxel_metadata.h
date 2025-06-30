/**************************************************************************/
/*  voxel_metadata.h                                                      */
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

#include "../../util/memory/memory.h"
// #include "../../util/non_copyable.h"
#include "../../util/containers/span.h"
#include <cstdint>

namespace zylann::voxel {

// Voxel metadata is arbitrary, sparse data that can be attached to particular voxels.
// It is not intended at being an efficient or fast storage method, but rather a versatile one for special cases.
// For example, it can be used to store text, tags, inventory contents, or other complex states attached to a voxel.
// If you need to store smaller data much more frequently, you may rely on a data channel instead.

class ICustomVoxelMetadata;

// Container for one metadata instance. It owns the data.
class VoxelMetadata {
public:
	enum Type : uint8_t { //
		TYPE_EMPTY = 0,
		TYPE_U64 = 1,
		// Reserved predefined types.

		TYPE_CUSTOM_BEGIN = 32,
		// Types equal or greater will implement `ICustomVoxelMetadata`.

		TYPE_APP_SPECIFIC_BEGIN = 40
		// Nothing prevents registering custom types lower than this index, but for convenience, it should be used for
		// application-specific types (aka game-specific). Lower indices can be used for engine-specific integrations.
	};

	static const unsigned int CUSTOM_TYPES_MAX_COUNT = 256 - TYPE_CUSTOM_BEGIN;

	VoxelMetadata() {}

	VoxelMetadata(VoxelMetadata &&other) {
		_type = other._type;
		_data = other._data;
		other._type = TYPE_EMPTY;
	}

	~VoxelMetadata() {
		clear();
	}

	inline void operator=(VoxelMetadata &&other) {
		clear();
		_type = other._type;
		_data = other._data;
		other._type = TYPE_EMPTY;
	}

	void clear();

	inline uint8_t get_type() const {
		return _type;
	}

	void set_u64(const uint64_t &v);
	uint64_t get_u64() const;

	void set_custom(uint8_t type, ICustomVoxelMetadata *custom_data);
	ICustomVoxelMetadata &get_custom();
	const ICustomVoxelMetadata &get_custom() const;

	// Clears this metadata and makes it a duplicate of the given one.
	void copy_from(const VoxelMetadata &src);

	bool equals(const VoxelMetadata &other) const;

private:
	union Data {
		uint64_t u64_data;
		ICustomVoxelMetadata *custom_data;
	};

	uint8_t _type = TYPE_EMPTY;
	Data _data;
};

} // namespace zylann::voxel
