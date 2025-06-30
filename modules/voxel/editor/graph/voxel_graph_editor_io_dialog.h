/**************************************************************************/
/*  voxel_graph_editor_io_dialog.h                                        */
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
#include "../../util/godot/classes/editor_undo_redo_manager.h"
#include "../../util/godot/macros.h"

ZN_GODOT_FORWARD_DECLARE(class ItemList)
ZN_GODOT_FORWARD_DECLARE(class LineEdit)
ZN_GODOT_FORWARD_DECLARE(class OptionButton)
ZN_GODOT_FORWARD_DECLARE(class SpinBox)
ZN_GODOT_FORWARD_DECLARE(class Button)

namespace zylann::voxel {

// Dialog to edit exposed inputs and outputs of a `VoxelGraphFunction`
class VoxelGraphEditorIODialog : public ConfirmationDialog {
	GDCLASS(VoxelGraphEditorIODialog, ConfirmationDialog)
public:
	VoxelGraphEditorIODialog();

	void set_graph(Ref<pg::VoxelGraphFunction> graph);
	void set_undo_redo(EditorUndoRedoManager *undo_redo);

private:
	void set_enabled(bool enabled);

	void _on_auto_generate_button_pressed();
	void _on_ok_pressed();

	void reshow(Ref<pg::VoxelGraphFunction> graph);

	void process();

	void _notification(int p_what);

	struct PortsUI {
		ItemList *item_list = nullptr;
		LineEdit *name = nullptr;
		OptionButton *usage = nullptr;
		SpinBox *default_value = nullptr;
		Button *add = nullptr;
		Button *remove = nullptr;
		Button *move_up = nullptr;
		Button *move_down = nullptr;
		int selected_item = -1;
	};

	static Control *create_ui(PortsUI &ui, String title, bool has_default_values);
	static void set_enabled(PortsUI &ui, bool enabled);
	static void clear(PortsUI &ui);
	void copy_ui_to_data(const PortsUI &ui, StdVector<pg::VoxelGraphFunction::Port> &ports);
	void copy_data_to_ui(PortsUI &ui, const StdVector<pg::VoxelGraphFunction::Port> &ports);
	void process_ui(PortsUI &ui, StdVector<pg::VoxelGraphFunction::Port> &ports);

	static void _bind_methods();

	Ref<pg::VoxelGraphFunction> _graph;
	StdVector<pg::VoxelGraphFunction::Port> _inputs;
	StdVector<pg::VoxelGraphFunction::Port> _outputs;

	PortsUI _inputs_ui;
	PortsUI _outputs_ui;
	Button *_auto_generate_button = nullptr;
	EditorUndoRedoManager *_undo_redo = nullptr;
	bool _reshow_on_undo_redo = true;
};

} // namespace zylann::voxel
