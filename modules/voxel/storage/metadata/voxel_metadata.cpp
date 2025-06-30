/**************************************************************************/
/*  voxel_metadata.cpp                                                    */
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

#include "voxel_metadata.h"
#include "../../util/errors.h"
#include "custom_voxel_metadata.h"

namespace zylann::voxel {

void VoxelMetadata::clear() {
	if (_type >= TYPE_CUSTOM_BEGIN) {
		ZN_DELETE(_data.custom_data);
		_data.custom_data = nullptr;
	}
	_type = TYPE_EMPTY;
}

void VoxelMetadata::set_u64(const uint64_t &v) {
	if (_type != TYPE_U64) {
		clear();
		_type = TYPE_U64;
	}
	_data.u64_data = v;
}

uint64_t VoxelMetadata::get_u64() const {
	ZN_ASSERT(_type == TYPE_U64);
	return _data.u64_data;
}

void VoxelMetadata::set_custom(uint8_t type, ICustomVoxelMetadata *custom_data) {
	ZN_ASSERT(type >= TYPE_CUSTOM_BEGIN);
	clear();
	_type = type;
	_data.custom_data = custom_data;
}

ICustomVoxelMetadata &VoxelMetadata::get_custom() {
	ZN_ASSERT(_type >= TYPE_CUSTOM_BEGIN);
	ZN_ASSERT(_data.custom_data != nullptr);
	return *_data.custom_data;
}

const ICustomVoxelMetadata &VoxelMetadata::get_custom() const {
	ZN_ASSERT(_type >= TYPE_CUSTOM_BEGIN);
	ZN_ASSERT(_data.custom_data != nullptr);
	return *_data.custom_data;
}

void VoxelMetadata::copy_from(const VoxelMetadata &src) {
	clear();
	if (src._type >= TYPE_CUSTOM_BEGIN) {
		ZN_ASSERT(src._data.custom_data != nullptr);
		_data.custom_data = src._data.custom_data->duplicate();
	} else {
		_data = src._data;
	}
	_type = src._type;
}

bool VoxelMetadata::equals(const VoxelMetadata &other) const {
	if (_type != other._type) {
		return false;
	}
	switch (_type) {
		case TYPE_EMPTY:
			return true;
		case TYPE_U64:
			return _data.u64_data == other._data.u64_data;
		default:
			if (_type >= TYPE_CUSTOM_BEGIN) {
				ZN_ASSERT(_data.custom_data != nullptr);
				ZN_ASSERT(other._data.custom_data != nullptr);
				return _data.custom_data->equals(*other._data.custom_data);
			} else {
				ZN_PRINT_ERROR("Non-implemented comparison");
				return false;
			}
	}
}

} // namespace zylann::voxel
