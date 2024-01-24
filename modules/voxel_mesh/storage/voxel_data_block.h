/**************************************************************************/
/*  voxel_data_block.h                                                    */
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

#ifndef VOXEL_DATA_BLOCK_H
#define VOXEL_DATA_BLOCK_H

#include "../storage/voxel_buffer.h"
#include "../util/macros.h"
#include "voxel_ref_count.h"

// Stores loaded voxel data for a chunk of the volume. Mesh and colliders are
// stored separately.
class VoxelDataBlock {
public:
	Ref<VoxelBuffer> voxels;
	const VoxelVector3i position;
	const unsigned int lod_index = 0;
	VoxelRefCount viewers;

	static VoxelDataBlock *create(VoxelVector3i bpos, Ref<VoxelBuffer> buffer,
			unsigned int size, unsigned int p_lod_index) {
		const int bs = size;
		ERR_FAIL_COND_V(buffer.is_null(), nullptr);
		ERR_FAIL_COND_V(buffer->get_size() != VoxelVector3i(bs, bs, bs), nullptr);
		return memnew(VoxelDataBlock(bpos, buffer, p_lod_index));
	}

	void set_modified(bool modified) {
#ifdef TOOLS_ENABLED
		if (_modified == false && modified) {
			PRINT_VERBOSE(String("Marking block {0} as modified")
								  .format(varray(position.to_vec3())));
		}
#endif
		_modified = modified;
	}

	inline bool is_modified() const {
		return _modified;
	}

	void set_needs_lodding(bool need_lodding) {
		_needs_lodding = need_lodding;
	}

	inline bool get_needs_lodding() const {
		return _needs_lodding;
	}

private:
	VoxelDataBlock(VoxelVector3i bpos, Ref<VoxelBuffer> buffer,
			unsigned int p_lod_index) :
			voxels(buffer), position(bpos), lod_index(p_lod_index) {}

	// The block was edited, which requires its LOD counterparts to be recomputed
	bool _needs_lodding = false;

	// Indicates if this block is different from the time it was loaded (should be
	// saved)
	bool _modified = false;
};

#endif // VOXEL_DATA_BLOCK_H
