/**************************************************************************/
/*  voxel_metadata_variant.h                                              */
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

#include "../../util/godot/core/variant.h"
#include "custom_voxel_metadata.h"
#include "voxel_metadata.h"

namespace zylann::voxel::godot {

// TODO Not sure if that should be a custom type. Custom types are supposed to be specific to a game?
enum GodotMetadataTypes { //
	METADATA_TYPE_VARIANT = VoxelMetadata::TYPE_CUSTOM_BEGIN
};

// Custom metadata holding a Godot Variant (basically, anything recognized by Godot Engine).
// Serializability depends on the same rules as Godot's `encode_variant`: no invalid objects, no cycles.
class VoxelMetadataVariant : public ICustomVoxelMetadata {
public:
	Variant data;

	size_t get_serialized_size() const override;
	size_t serialize(Span<uint8_t> dst) const override;
	bool deserialize(Span<const uint8_t> src, uint64_t &out_read_size) override;
	ICustomVoxelMetadata *duplicate() override;
	uint8_t get_type_index() const override;
	bool equals(const ICustomVoxelMetadata &other) const override;
};

Variant get_as_variant(const VoxelMetadata &meta);
void set_as_variant(VoxelMetadata &meta, Variant v);

} // namespace zylann::voxel::godot
