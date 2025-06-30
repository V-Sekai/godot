/**************************************************************************/
/*  voxel_blocky_attribute_axis.h                                         */
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

#include "voxel_blocky_attribute.h"

namespace zylann::voxel {

// `axis`
class VoxelBlockyAttributeAxis : public VoxelBlockyAttribute {
	GDCLASS(VoxelBlockyAttributeAxis, VoxelBlockyAttribute)
public:
	enum Axis { AXIS_X, AXIS_Y, AXIS_Z, AXIS_COUNT };

	VoxelBlockyAttributeAxis();

	void set_horizontal_only(bool h);
	bool is_horizontal_only() const;

	int from_vec3(Vector3 v) const;

private:
	void update_values();

	static void _bind_methods();

	bool _horizontal_only = false;
	// TODO Corresponding ortho rotations
};

} // namespace zylann::voxel

VARIANT_ENUM_CAST(zylann::voxel::VoxelBlockyAttributeAxis::Axis);
