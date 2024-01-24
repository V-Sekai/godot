/**************************************************************************/
/*  funcs.cpp                                                             */
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

#include "funcs.h"
#include "../util/math/box3i.h"

void copy_3d_region_zxy(Span<uint8_t> dst, VoxelVector3i dst_size,
		VoxelVector3i dst_min, Span<const uint8_t> src,
		VoxelVector3i src_size, VoxelVector3i src_min,
		VoxelVector3i src_max, size_t item_size) {
	//
	VoxelVector3i::sort_min_max(src_min, src_max);
	clip_copy_region(src_min, src_max, src_size, dst_min, dst_size);
	const VoxelVector3i area_size = src_max - src_min;
	if (area_size.x <= 0 || area_size.y <= 0 || area_size.z <= 0) {
		// Degenerate area, we'll not copy anything.
		return;
	}

#ifdef DEBUG_ENABLED
	if (src.data() == dst.data()) {
		ERR_FAIL_COND_MSG(
				Box3i::from_min_max(src_min, src_max)
						.intersects(Box3i::from_min_max(dst_min, dst_min + area_size)),
				"Copy across the same buffer to an overlapping area is not supported");
	}
	ERR_FAIL_COND(area_size.volume() * item_size > dst.size());
	ERR_FAIL_COND(area_size.volume() * item_size > src.size());
#endif

	if (area_size == src_size && area_size == dst_size) {
		// Copy everything
		memcpy(dst.data(), src.data(), dst.size() * item_size);

	} else {
		// Copy area row by row:
		// This offset is how much to move in order to advance by one row (row
		// direction is Y), essentially doing y+1
		const unsigned int src_row_offset = src_size.y * item_size;
		const unsigned int dst_row_offset = dst_size.y * item_size;
		VoxelVector3i pos;
		for (pos.z = 0; pos.z < area_size.z; ++pos.z) {
			pos.x = 0;
			unsigned int src_ri =
					VoxelVector3i(src_min + pos).get_zxy_index(src_size) * item_size;
			unsigned int dst_ri =
					VoxelVector3i(dst_min + pos).get_zxy_index(dst_size) * item_size;
			for (; pos.x < area_size.x; ++pos.x) {
				// TODO Cast src and dst to `restrict` so the optimizer can assume
				// adresses don't overlap,
				//      which might allow to write as a for loop (which may compile as a
				//      `memcpy`)?
				memcpy(&dst[dst_ri], &src[src_ri], area_size.y * item_size);
				src_ri += src_row_offset;
				dst_ri += dst_row_offset;
			}
		}
	}
}
