/**************************************************************************/
/*  voxel_graph_node_dialog.h                                             */
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

#include "../../generators/graph/voxel_graph_function.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/confirmation_dialog.h"
#include "../../util/godot/core/version.h"
#include "../../util/godot/macros.h"

ZN_GODOT_FORWARD_DECLARE(class Tree);
ZN_GODOT_FORWARD_DECLARE(class LineEdit);
ZN_GODOT_FORWARD_DECLARE(class EditorFileDialog)
ZN_GODOT_FORWARD_DECLARE(class RichTextLabel)
#ifdef ZN_GODOT
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 3
ZN_GODOT_FORWARD_DECLARE(class EditorQuickOpen)
#endif
#endif

namespace zylann::voxel {

// Dialog to pick a graph node type, with categories, search and descriptions
class VoxelGraphNodeDialog : public ConfirmationDialog {
	GDCLASS(VoxelGraphNodeDialog, ConfirmationDialog)
public:
	static const char *SIGNAL_NODE_SELECTED;
	static const char *SIGNAL_FILE_SELECTED;

	VoxelGraphNodeDialog();

	void popup_at_screen_position(Vector2 screen_pos);

private:
	void update_tree(bool autoselect);

	void on_ok_pressed();
	void on_filter_text_changed(String new_text);
	void on_filter_gui_input(Ref<InputEvent> event);
	void on_tree_item_activated();
	void on_tree_item_selected();
	void on_tree_nothing_selected();
	void on_function_file_dialog_file_selected(String fpath);
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 3
	void on_function_quick_open_dialog_quick_open();
#endif
	void on_function_quick_open_dialog_item_selected(String fpath);
	void on_description_label_meta_clicked(Variant meta);

	void _notification(int p_what);

	static void _bind_methods();

	enum SpecialIDs {
		// Preceding IDs are node types
		ID_FUNCTION_BROWSE = pg::VoxelGraphFunction::NODE_TYPE_COUNT,
		ID_FUNCTION_QUICK_OPEN,
		ID_MAX
	};

	struct Item {
		String name;
		String description;
		int category = -1;
		int id = -1;
	};

	StdVector<Item> _items;
	StdVector<String> _category_names;
	Tree *_tree = nullptr;
	LineEdit *_filter_line_edit = nullptr;
	RichTextLabel *_description_label = nullptr;
	EditorFileDialog *_function_file_dialog = nullptr;
#ifdef ZN_GODOT
#if GODOT_VERSION_MAJOR == 4 && GODOT_VERSION_MINOR <= 3
	// TODO GDX: EditorQuickOpen is not exposed!
	EditorQuickOpen *_function_quick_open_dialog = nullptr;
#endif
#endif
};

} // namespace zylann::voxel
