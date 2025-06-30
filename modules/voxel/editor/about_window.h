/**************************************************************************/
/*  about_window.h                                                        */
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

#include "../util/godot/classes/accept_dialog.h"
#include "../util/godot/macros.h"

ZN_GODOT_FORWARD_DECLARE(class TextureRect);
ZN_GODOT_FORWARD_DECLARE(class RichTextLabel);

namespace zylann::voxel {

class VoxelAboutWindow : public AcceptDialog {
	GDCLASS(VoxelAboutWindow, AcceptDialog)
public:
	VoxelAboutWindow();

	// The same window can be shown by more than one plugin, therefore it is created only once internally.
	// It cannot be created in the initialization of the module because the editor isn't available yet.
	static void create_singleton(Node &base_control);
	static void destroy_singleton();
	static void popup_singleton();

protected:
	void _notification(int p_what);

private:
	void _on_about_rich_text_label_meta_clicked(Variant meta);
	void _on_third_party_list_item_selected(int index);

	static void _bind_methods();

	TextureRect *_icon_texture_rect;
	RichTextLabel *_third_party_rich_text_label;
};

} // namespace zylann::voxel
