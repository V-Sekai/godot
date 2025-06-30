/**************************************************************************/
/*  voxel_tool_buffer.h                                                   */
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

#include "voxel_tool.h"

namespace zylann::voxel {

class VoxelToolBuffer : public VoxelTool {
	GDCLASS(VoxelToolBuffer, VoxelTool)
public:
	VoxelToolBuffer() {}
	VoxelToolBuffer(Ref<godot::VoxelBuffer> vb);

	bool is_area_editable(const Box3i &box) const override;
	void paste(Vector3i p_pos, const VoxelBuffer &src, uint8_t channels_mask) override;
	void paste_masked(
			Vector3i p_pos,
			Ref<godot::VoxelBuffer> p_voxels,
			uint8_t channels_mask,
			uint8_t mask_channel,
			uint64_t mask_value
	) override;

	void set_voxel_metadata(Vector3i pos, Variant meta) override;
	Variant get_voxel_metadata(Vector3i pos) const override;

	void do_sphere(Vector3 center, float radius) override;
	void do_box(Vector3i begin, Vector3i end) override;
	void do_path(Span<const Vector3> positions, Span<const float> radii) override;

protected:
	uint64_t _get_voxel(Vector3i pos) const override;
	float _get_voxel_f(Vector3i pos) const override;
	void _set_voxel(Vector3i pos, uint64_t v) override;
	void _set_voxel_f(Vector3i pos, float v) override;
	void _post_edit(const Box3i &box) override;

private:
	// When compiling with GodotCpp, `_bind_methods` is not optional.
	static void _bind_methods() {}

	Ref<godot::VoxelBuffer> _buffer;
};

} // namespace zylann::voxel
