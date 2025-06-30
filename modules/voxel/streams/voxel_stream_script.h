/**************************************************************************/
/*  voxel_stream_script.h                                                 */
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

#include "../util/godot/core/gdvirtual.h"
#include "voxel_stream.h"

#ifdef ZN_GODOT_EXTENSION
// GodotCpp wants the full definition of the class in GDVIRTUAL
#include "../storage/voxel_buffer_gd.h"
#endif

namespace zylann::voxel {

// Provides access to a source of paged voxel data, which may load and save.
// Must be implemented in a multi-thread-safe way.
// If you are looking for a more specialized API to generate voxels, use VoxelGenerator.
class VoxelStreamScript : public VoxelStream {
	GDCLASS(VoxelStreamScript, VoxelStream)
public:
	void load_voxel_block(VoxelStream::VoxelQueryData &q) override;
	void save_voxel_block(VoxelStream::VoxelQueryData &q) override;

	int get_used_channels_mask() const override;

protected:
	// TODO Why is it unable to convert `Result` into `Variant` even though a cast is defined in voxel_stream.h???
	GDVIRTUAL3R(int, _load_voxel_block, Ref<godot::VoxelBuffer>, Vector3i, int)
	GDVIRTUAL3(_save_voxel_block, Ref<godot::VoxelBuffer>, Vector3i, int)
	GDVIRTUAL0RC(int, _get_used_channels_mask) // I think `C` means `const`?

	static void _bind_methods();
};

} // namespace zylann::voxel
