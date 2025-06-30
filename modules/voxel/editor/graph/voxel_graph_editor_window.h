/**************************************************************************/
/*  voxel_graph_editor_window.h                                           */
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

#include "../../util/godot/classes/accept_dialog.h"
#include "../../util/godot/classes/button.h"
#include "../../util/godot/editor_scale.h"

namespace zylann::voxel {

// TODO It would be really nice if we were not forced to use an AcceptDialog for making a window.
// AcceptDialog adds stuff I don't need, but Window is too low level.
class VoxelGraphEditorWindow : public AcceptDialog {
	GDCLASS(VoxelGraphEditorWindow, AcceptDialog)
public:
	VoxelGraphEditorWindow() {
		set_exclusive(false);
		set_close_on_escape(false);
		get_ok_button()->hide();
		set_min_size(Vector2(600, 300) * EDSCALE);
		// I want the window to remain on top of the editor if the editor is given focus. `always_on_top` is the only
		// property allowing that, but it requires `transient` to be `false`. Without `transient`, the window is no
		// longer considered a child and won't give back focus to the editor when closed.
		// So for now, the window will get hidden behind the editor if you click on the editor.
		// you'll have to suffer moving popped out windows out of the editor area if you want to see them both...
		// set_flag(Window::FLAG_ALWAYS_ON_TOP, true);
	}

	// void _notification(int p_what) {
	// 	switch (p_what) {
	// 		case NOTIFICATION_WM_CLOSE_REQUEST:
	// 			call_deferred(SNAME("hide"));
	// 			break;
	// 	}
	// }

	static void _bind_methods() {}
};

} // namespace zylann::voxel
