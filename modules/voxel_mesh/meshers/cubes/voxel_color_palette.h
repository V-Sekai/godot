/**************************************************************************/
/*  voxel_color_palette.h                                                 */
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

#ifndef VOXEL_COLOR_PALETTE_H
#define VOXEL_COLOR_PALETTE_H

#include "../../util/fixed_array.h"
#include "../../util/math/color8.h"
#include <core/io/resource.h>

// Associates small numbers to colors, so colored voxels can be specified using
// less memory.
class VoxelColorPalette : public Resource {
	GDCLASS(VoxelColorPalette, Resource)
public:
	static const unsigned int MAX_COLORS = 256;

	VoxelColorPalette();

	void set_color(int index, Color color);
	Color get_color(int index) const;

	PackedColorArray get_colors() const;
	void set_colors(PackedColorArray colors);

	void clear();

	// Internal

	inline void set_color8(uint8_t i, Color8 c) { _colors[i] = c; }

	inline Color8 get_color8(uint8_t i) const { return _colors[i]; }

private:
	PackedInt32Array _b_get_data() const;
	void _b_set_data(PackedInt32Array colors);

	static void _bind_methods();

	FixedArray<Color8, MAX_COLORS> _colors;
};

#endif // VOXEL_COLOR_PALETTE_H
