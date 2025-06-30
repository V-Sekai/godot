/**************************************************************************/
/*  transvoxel_materials_null.h                                           */
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

#include "transvoxel.h"

namespace zylann::voxel::transvoxel::materials {

struct NullProcessor {
	// Called for every 2x2x2 cell containing triangles.
	// The returned value is used to determine if the next cell can reuse vertices from previous cells, when equal.
	inline uint32_t on_cell(const FixedArray<uint32_t, 8> &corner_voxel_indices, const uint8_t case_code) const {
		return 0;
	}
	// Called for every 2x3x3 transition cell containing triangles.
	// Such cells are actually in 2D data-wise, so corners are the same value, so only 9 are passed in.
	// The returned value is used to determine if the next cell can reuse vertices from previous cells, when equal.
	inline uint32_t on_transition_cell(
			const FixedArray<uint32_t, 9> &corner_voxel_indices,
			const uint8_t case_code
	) const {
		return 0;
	}
	// Called one or more times after each `on_cell` for every new vertex, to interpolate and add material data
	inline void on_vertex(const unsigned int v0, const unsigned int v1, const float alpha) const {
		return;
	}
};

} // namespace zylann::voxel::transvoxel::materials
