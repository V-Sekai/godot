/**************************************************************************/
/*  voxel_color_palette.cpp                                               */
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

#include "voxel_color_palette.h"

VoxelColorPalette::VoxelColorPalette() {
	// Default palette
	_colors[0] = Color8(0, 0, 0, 0);
	_colors[1] = Color8(255, 255, 255, 255);
	for (unsigned int i = 2; i < _colors.size(); ++i) {
		_colors[i] = Color8(0, 0, 0, 255);
	}
}

void VoxelColorPalette::set_color(int index, Color color) {
	ERR_FAIL_INDEX(index, static_cast<int>(_colors.size()));
	_colors[index] = Color8(color);
}

Color VoxelColorPalette::get_color(int index) const {
	ERR_FAIL_INDEX_V(index, static_cast<int>(_colors.size()), Color());
	return _colors[index];
}

PackedColorArray VoxelColorPalette::get_colors() const {
	PackedColorArray colors;
	colors.resize(_colors.size());
	{
		for (unsigned int i = 0; i < _colors.size(); ++i) {
			colors.write[i] = _colors[i];
		}
	}
	return colors;
}

void VoxelColorPalette::set_colors(PackedColorArray colors) {
	// Color count is fixed, but we can't easily prevent Godot from allowing users
	// to set a dynamic array
	ERR_FAIL_COND(colors.size() != static_cast<int>(_colors.size()));
	{
		for (unsigned int i = 0; i < _colors.size(); ++i) {
			_colors[i] = Color8(colors.ptr()[i]);
		}
	}
}

void VoxelColorPalette::clear() {
	for (size_t i = 0; i < _colors.size(); ++i) {
		_colors[i] = Color8();
	}
}

PackedInt32Array VoxelColorPalette::_b_get_data() const {
	PackedInt32Array colors;
	colors.resize(_colors.size());
	{
		for (size_t i = 0; i < _colors.size(); ++i) {
			colors.write[i] = _colors[i].to_u32();
		}
	}
	return colors;
}

void VoxelColorPalette::_b_set_data(PackedInt32Array colors) {
	ERR_FAIL_COND(colors.size() > static_cast<int>(_colors.size()));
	for (int i = 0; i < colors.size(); ++i) {
		_colors[i] = Color8::from_u32(colors[i]);
	}
}

void VoxelColorPalette::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_color", "colors"),
			&VoxelColorPalette::set_colors);
	ClassDB::bind_method(D_METHOD("get_colors"), &VoxelColorPalette::get_colors);
	ClassDB::bind_method(D_METHOD("set_color", "index", "color"),
			&VoxelColorPalette::set_color);
	ClassDB::bind_method(D_METHOD("get_color"), &VoxelColorPalette::get_color);
	ClassDB::bind_method(D_METHOD("set_data", "d"),
			&VoxelColorPalette::_b_set_data);
	ClassDB::bind_method(D_METHOD("get_data"), &VoxelColorPalette::_b_get_data);

	// This is just to allow editing colors in the editor
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_COLOR_ARRAY, "colors",
						 PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR),
			"set_colors", "get_colors");

	ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "data",
						 PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE),
			"set_data", "get_data");

	BIND_CONSTANT(MAX_COLORS);
}
