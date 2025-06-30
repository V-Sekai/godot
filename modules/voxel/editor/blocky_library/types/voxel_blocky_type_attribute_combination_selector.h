/**************************************************************************/
/*  voxel_blocky_type_attribute_combination_selector.h                    */
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

#include "../../../meshers/blocky/types/voxel_blocky_type.h"
#include "../../../util/containers/std_vector.h"
#include "../../../util/godot/classes/grid_container.h"

ZN_GODOT_FORWARD_DECLARE(class OptionButton);
ZN_GODOT_FORWARD_DECLARE(class Label);

namespace zylann::voxel {

// Editor with the list of attributes from a specific VoxelBlockyType, allowing to choose a combination
// parametrically.
class VoxelBlockyTypeAttributeCombinationSelector : public GridContainer {
	GDCLASS(VoxelBlockyTypeAttributeCombinationSelector, GridContainer)
public:
	static const char *SIGNAL_COMBINATION_CHANGED;

	VoxelBlockyTypeAttributeCombinationSelector();

	void set_type(Ref<VoxelBlockyType> type);

	VoxelBlockyType::VariantKey get_variant_key() const;

private:
	bool get_preview_attribute_value(const StringName &attrib_name, uint8_t &out_value) const;
	void remove_attribute_editor(unsigned int index);
	void update_attribute_editors();

	bool get_attribute_editor_index(const StringName &attrib_name, unsigned int &out_index) const;

	void _on_type_changed();
	void _on_attribute_editor_value_selected(int value_index, int editor_index);

	static void _bind_methods();

	struct AttributeEditor {
		StringName name;
		uint8_t value;
		Label *label = nullptr;
		OptionButton *selector = nullptr;
		Ref<VoxelBlockyAttribute> attribute_copy;
	};

	Ref<VoxelBlockyType> _type;
	StdVector<AttributeEditor> _attribute_editors;
};

} // namespace zylann::voxel
