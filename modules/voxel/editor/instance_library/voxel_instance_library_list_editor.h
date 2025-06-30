/**************************************************************************/
/*  voxel_instance_library_list_editor.h                                  */
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

#include "../../terrain/instancing/voxel_instance_library.h"
#include "../../util/godot/classes/h_box_container.h"

ZN_GODOT_FORWARD_DECLARE(class ItemList)
ZN_GODOT_FORWARD_DECLARE(class ConfirmationDialog)
ZN_GODOT_FORWARD_DECLARE(class EditorFileDialog)

namespace zylann::voxel {

class VoxelInstanceLibraryEditorPlugin;

class VoxelInstanceLibraryListEditor : public HBoxContainer {
	GDCLASS(VoxelInstanceLibraryListEditor, HBoxContainer)
public:
	VoxelInstanceLibraryListEditor();

	void setup(const Control *icon_provider, VoxelInstanceLibraryEditorPlugin *plugin);

	void set_library(Ref<VoxelInstanceLibrary> library);

private:
	enum ButtonID { //
		BUTTON_ADD_MULTIMESH_ITEM,
		BUTTON_ADD_SCENE_ITEM,
		BUTTON_REMOVE_ITEM
	};

	void _notification(int p_what);

	void on_list_item_selected(int index);
	void on_button_pressed(int index);
	void on_remove_item_button_pressed();
	void on_remove_item_confirmed();
	void on_open_scene_dialog_file_selected(String fpath);
	// void on_library_item_changed(int id, ChangeType change) override;

	void add_multimesh_item();
	void add_scene_item(String fpath);
	void update_list_from_library();

	static void _bind_methods();

	Ref<VoxelInstanceLibrary> _library;
	ItemList *_item_list = nullptr;
	StdVector<String> _name_cache;

	ButtonID _last_used_button;
	int _item_id_to_remove = -1;

	ConfirmationDialog *_confirmation_dialog = nullptr;
	EditorFileDialog *_open_scene_dialog = nullptr;

	VoxelInstanceLibraryEditorPlugin *_plugin = nullptr;
};

} // namespace zylann::voxel
