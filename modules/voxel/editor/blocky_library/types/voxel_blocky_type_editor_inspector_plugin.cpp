/**************************************************************************/
/*  voxel_blocky_type_editor_inspector_plugin.cpp                         */
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

#include "voxel_blocky_type_editor_inspector_plugin.h"
#include "../../../meshers/blocky/types/voxel_blocky_type.h"
#include "../../../util/godot/classes/editor_interface.h"
#include "../../../util/godot/classes/h_box_container.h"
#include "voxel_blocky_type_attribute_combination_selector.h"
#include "voxel_blocky_type_variant_list_editor.h"
#include "voxel_blocky_type_viewer.h"

namespace zylann::voxel {

void VoxelBlockyTypeEditorInspectorPlugin::set_editor_interface(EditorInterface *ed) {
	_editor_interface = ed;
}

void VoxelBlockyTypeEditorInspectorPlugin::set_undo_redo(EditorUndoRedoManager *urm) {
	_undo_redo = urm;
}

bool VoxelBlockyTypeEditorInspectorPlugin::_zn_can_handle(const Object *p_object) const {
	return Object::cast_to<VoxelBlockyType>(p_object) != nullptr;
}

void VoxelBlockyTypeEditorInspectorPlugin::_zn_parse_begin(Object *p_object) {
	const VoxelBlockyType *type_ptr = Object::cast_to<VoxelBlockyType>(p_object);
	ZN_ASSERT_RETURN(type_ptr != nullptr);

	Ref<VoxelBlockyType> type(type_ptr);

	VoxelBlockyTypeAttributeCombinationSelector *selector = memnew(VoxelBlockyTypeAttributeCombinationSelector);
	selector->set_type(type);

	VoxelBlockyTypeViewer *viewer = memnew(VoxelBlockyTypeViewer);
	viewer->set_combination_selector(selector);
	viewer->set_type(type);

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->add_child(viewer);
	hb->add_child(selector);

	add_custom_control(hb);
	return;
}

bool VoxelBlockyTypeEditorInspectorPlugin::_zn_parse_property(
		Object *p_object,
		const Variant::Type p_type,
		const String &p_path,
		const PropertyHint p_hint,
		const String &p_hint_text,
		const BitField<PropertyUsageFlags> p_usage,
		const bool p_wide
) {
	if (p_type != Variant::ARRAY) {
		return false;
	}
	if (p_path != "_variant_models_data") {
		return false;
	}

	const VoxelBlockyType *type_ptr = Object::cast_to<VoxelBlockyType>(p_object);
	ZN_ASSERT_RETURN_V(type_ptr != nullptr, false);

	Ref<VoxelBlockyType> type(type_ptr);
	if (type.is_null()) {
		return false;
	}

	VoxelBlockyTypeVariantListEditor *variant_list_editor = memnew(VoxelBlockyTypeVariantListEditor);
	variant_list_editor->set_type(type);
	variant_list_editor->set_editor_interface(_editor_interface);
	variant_list_editor->set_undo_redo(_undo_redo);
	add_custom_control(variant_list_editor);

	// Removes the property, the custom editor replaces it
	return true;
}

} // namespace zylann::voxel
