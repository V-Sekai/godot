/**************************************************************************/
/*  voxel_generator_script.cpp                                            */
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

#include "voxel_generator_script.h"
#include "../constants/voxel_string_names.h"
#include "../storage/voxel_buffer_gd.h"
#include "../util/godot/check_ref_ownership.h"

namespace zylann::voxel {

VoxelGeneratorScript::VoxelGeneratorScript() {}

VoxelGenerator::Result VoxelGeneratorScript::generate_block(VoxelGenerator::VoxelQueryData input) {
	Result result;

	// Create a temporary wrapper so Godot can pass it to scripts
	Ref<godot::VoxelBuffer> buffer_wrapper(
			memnew(godot::VoxelBuffer(static_cast<godot::VoxelBuffer::Allocator>(input.voxel_buffer.get_allocator())))
	);
	buffer_wrapper.instantiate();
	buffer_wrapper->get_buffer().copy_format(input.voxel_buffer);
	buffer_wrapper->get_buffer().create(input.voxel_buffer.get_size());

	{
		ZN_GODOT_CHECK_REF_COUNT_DOES_NOT_CHANGE(buffer_wrapper);
		if (!GDVIRTUAL_CALL(_generate_block, buffer_wrapper, input.origin_in_voxels, input.lod)) {
			WARN_PRINT_ONCE("VoxelGeneratorScript::_generate_block is unimplemented!");
		}
	}

	// The wrapper is discarded
	buffer_wrapper->get_buffer().move_to(input.voxel_buffer);

	// We may expose this to scripts the day it actually gets used
	// if (ret.get_type() == Variant::DICTIONARY) {
	// 	Dictionary d = ret;
	// 	result.max_lod_hint = d.get("max_lod_hint", false);
	// }

	return result;
}

int VoxelGeneratorScript::get_used_channels_mask() const {
	int mask = 0;
	if (!GDVIRTUAL_CALL(_get_used_channels_mask, mask)) {
		WARN_PRINT_ONCE("VoxelGeneratorScript::_get_used_channels_mask is unimplemented!");
	}
	return mask;
}

void VoxelGeneratorScript::_bind_methods() {
	GDVIRTUAL_BIND(_generate_block, "out_buffer", "origin_in_voxels", "lod");
	GDVIRTUAL_BIND(_get_used_channels_mask);
}

} // namespace zylann::voxel
