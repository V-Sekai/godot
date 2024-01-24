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

#ifndef VOXEL_TOOL_BUFFER_H
#define VOXEL_TOOL_BUFFER_H

#include "voxel_tool.h"

class VoxelBuffer;

class VoxelToolBuffer : public VoxelTool {
	GDCLASS(VoxelToolBuffer, VoxelTool)
public:
	VoxelToolBuffer() {}
	VoxelToolBuffer(Ref<VoxelBuffer> vb);

	bool is_area_editable(const Box3i &box) const override;
	void paste(VoxelVector3i p_pos, Ref<VoxelBuffer> p_voxels,
			uint8_t channels_mask, uint64_t mask_value) override;

	void set_voxel_metadata(VoxelVector3i pos, Variant meta) override;
	Variant get_voxel_metadata(VoxelVector3i pos) override;

	void do_sphere(Vector3 center, float radius) override;

protected:
	uint64_t _get_voxel(VoxelVector3i pos) const override;
	float _get_voxel_f(VoxelVector3i pos) const override;
	void _set_voxel(VoxelVector3i pos, uint64_t v) override;
	void _set_voxel_f(VoxelVector3i pos, float v) override;
	void _post_edit(const Box3i &box) override;

private:
	Ref<VoxelBuffer> _buffer;
};

#endif // VOXEL_TOOL_BUFFER_H
