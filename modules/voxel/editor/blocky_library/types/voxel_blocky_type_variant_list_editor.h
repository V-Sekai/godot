/**************************************************************************/
/*  voxel_blocky_type_variant_list_editor.h                               */
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
#include "../../../util/godot/classes/v_box_container.h"

ZN_GODOT_FORWARD_DECLARE(class Label);
ZN_GODOT_FORWARD_DECLARE(class EditorResourcePicker);
ZN_GODOT_FORWARD_DECLARE(class GridContainer);
ZN_GODOT_FORWARD_DECLARE(class EditorInterface);
ZN_GODOT_FORWARD_DECLARE(class EditorUndoRedoManager);

namespace zylann::voxel {

// Allows to edit a map of attribute combination and associated models.
// This cannot be exposed as regular properties, therefore it is a custom comtrol.
class VoxelBlockyTypeVariantListEditor : public VBoxContainer {
	GDCLASS(VoxelBlockyTypeVariantListEditor, VBoxContainer)
public:
	VoxelBlockyTypeVariantListEditor();

	void set_type(Ref<VoxelBlockyType> type);
	void set_editor_interface(EditorInterface *ed);
	void set_undo_redo(EditorUndoRedoManager *undo_redo);

private:
	void update_list();

	void _on_type_changed();
	void _on_model_changed(Ref<VoxelBlockyModel> model, int editor_index);
	void _on_model_picker_selected(Ref<VoxelBlockyModel> model, bool inspect);

	static void _bind_methods();

	Ref<VoxelBlockyType> _type;
	Label *_header_label = nullptr;

	struct VariantEditor {
		Label *key_label = nullptr;
		EditorResourcePicker *resource_picker = nullptr;
		VoxelBlockyType::VariantKey key;
	};

	StdVector<VariantEditor> _variant_editors;
	GridContainer *_grid_container = nullptr;
	EditorInterface *_editor_interface = nullptr;
	EditorUndoRedoManager *_undo_redo = nullptr;
};

} // namespace zylann::voxel
