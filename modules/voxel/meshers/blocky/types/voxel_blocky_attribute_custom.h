/**************************************************************************/
/*  voxel_blocky_attribute_custom.h                                       */
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

class VoxelBlockyAttributeCustom : public VoxelBlockyAttribute {
	GDCLASS(VoxelBlockyAttributeCustom, VoxelBlockyAttribute)
public:
	VoxelBlockyAttributeCustom();

	void set_attribute_name(StringName p_name);
	void set_value_count(int count);
	void set_value_name(int index, StringName p_name);
	void set_default_value(int v);

	// Not exposing rotation like that, because we can't automate this properly at the moment (in the case of rails,
	// there is rotation, but it is uneven as several values have the same rotation but different models). Users
	// will have to rotate the model manually using editor tools, which gives more control.
	// An easier approach is to separate rotation from rail shape as two attributes, but it will waste a few model IDs
	// for straight shapes.
	//
	// void set_is_rotation(bool is_rotation);
	// void set_value_ortho_rotation(int index, int ortho_rotation_index);

private:
	void update_values();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();
};

} // namespace zylann::voxel
