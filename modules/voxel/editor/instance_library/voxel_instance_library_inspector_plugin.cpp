/**************************************************************************/
/*  voxel_instance_library_inspector_plugin.cpp                           */
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

#include "voxel_instance_library_inspector_plugin.h"
#include "../../constants/voxel_string_names.h"
#include "control_sizer.h"
#include "voxel_instance_library_editor_plugin.h"
#include "voxel_instance_library_list_editor.h"

namespace zylann::voxel {

bool VoxelInstanceLibraryInspectorPlugin::_zn_can_handle(const Object *p_object) const {
	return Object::cast_to<VoxelInstanceLibrary>(p_object) != nullptr;
}

void VoxelInstanceLibraryInspectorPlugin::_zn_parse_begin(Object *p_object) {
	// TODO How can I make sure the buttons will be at the beginning of the "VoxelInstanceLibrary" category?
	// This is a better place than the Spatial editor toolbar (which would get hidden if you are not in the 3D tab
	// of the editor), but it will appear at the very top of the inspector, even above the "VoxelInstanceLibrary"
	// catgeory of properties. That looks a bit off, and if the class were to be inherited, it would start to be
	// confusing because these buttons are about the property list of "VoxelInstanceLibrary" specifically.
	// I could neither use `parse_property` nor `parse_category`, because when the list is empty,
	// the class returns no properties AND no category.
}

bool VoxelInstanceLibraryInspectorPlugin::_zn_parse_property(
		Object *p_object,
		const Variant::Type p_type,
		const String &p_path,
		const PropertyHint p_hint,
		const String &p_hint_text,
		const BitField<PropertyUsageFlags> p_usage,
		const bool p_wide
) {
	// We use this property as anchor to put our list on top of it
	if (p_path == "_selected_item") {
		Ref<VoxelInstanceLibrary> library(Object::cast_to<VoxelInstanceLibrary>(p_object));

		VoxelInstanceLibraryListEditor *list_editor = memnew(VoxelInstanceLibraryListEditor);
		list_editor->setup(icon_provider, plugin);
		list_editor->set_library(library);
		add_custom_control(list_editor);

		ZN_ControlSizer *sizer = memnew(ZN_ControlSizer);
		sizer->set_target_control(list_editor);
		add_custom_control(sizer);
	}
	return false;
}

} // namespace zylann::voxel
