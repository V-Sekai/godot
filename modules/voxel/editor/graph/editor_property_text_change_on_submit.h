/**************************************************************************/
/*  editor_property_text_change_on_submit.h                               */
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

#include "../../util/godot/classes/editor_property.h"
#include "../../util/godot/macros.h"

ZN_GODOT_FORWARD_DECLARE(class LineEdit)

namespace zylann {

// The default string editor of the inspector calls the setter of the edited object on every character typed.
// This is not always desired. Instead, this editor should emit a change only when enter is pressed, or when the
// editor looses focus.
// Note: Godot's default string editor for LineEdit is `EditorPropertyText`
class ZN_EditorPropertyTextChangeOnSubmit : public zylann::godot::ZN_EditorProperty {
	GDCLASS(ZN_EditorPropertyTextChangeOnSubmit, zylann::godot::ZN_EditorProperty)
public:
	ZN_EditorPropertyTextChangeOnSubmit();

protected:
	void _zn_update_property() override;

private:
	void _on_line_edit_focus_entered();
	void _on_line_edit_text_changed(String new_text);
	void _on_line_edit_text_submitted(String text);
	void _on_line_edit_focus_exited();

	static void _bind_methods();

	LineEdit *_line_edit = nullptr;
	bool _ignore_changes = false;
	bool _changed = false;
};

} // namespace zylann
