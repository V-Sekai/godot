/**************************************************************************/
/*  editor_property_text_change_on_submit.cpp                             */
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

#include "editor_property_text_change_on_submit.h"
#include "../../util/godot/classes/line_edit.h"

namespace zylann {

ZN_EditorPropertyTextChangeOnSubmit::ZN_EditorPropertyTextChangeOnSubmit() {
	using Self = ZN_EditorPropertyTextChangeOnSubmit;
	_line_edit = memnew(LineEdit);
	add_child(_line_edit);
	add_focusable(_line_edit);
	_line_edit->connect("text_submitted", callable_mp(this, &Self::_on_line_edit_text_submitted));
	_line_edit->connect("text_changed", callable_mp(this, &Self::_on_line_edit_text_changed));
	_line_edit->connect("focus_exited", callable_mp(this, &Self::_on_line_edit_focus_exited));
	_line_edit->connect("focus_entered", callable_mp(this, &Self::_on_line_edit_focus_entered));
}

void ZN_EditorPropertyTextChangeOnSubmit::_zn_update_property() {
	Object *obj = get_edited_object();
	ERR_FAIL_COND(obj == nullptr);
	_ignore_changes = true;
	_line_edit->set_text(obj->get(get_edited_property()));
	_ignore_changes = false;
}

void ZN_EditorPropertyTextChangeOnSubmit::_on_line_edit_focus_entered() {
	_changed = false;
}

void ZN_EditorPropertyTextChangeOnSubmit::_on_line_edit_text_changed(String new_text) {
	if (_ignore_changes) {
		return;
	}
	_changed = true;
}

void ZN_EditorPropertyTextChangeOnSubmit::_on_line_edit_text_submitted(String text) {
	if (_ignore_changes) {
		return;
	}
	// Same behavior as the default `EditorPropertyText`
	if (_line_edit->has_focus()) {
		_line_edit->release_focus();
	}
}

void ZN_EditorPropertyTextChangeOnSubmit::_on_line_edit_focus_exited() {
	if (_changed) {
		_changed = false;

		Object *obj = get_edited_object();
		ERR_FAIL_COND(obj == nullptr);
		String prev_text = obj->get(get_edited_property());

		String text = _line_edit->get_text();

		if (prev_text != text) {
			emit_changed(get_edited_property(), text);
		}
	}
}

void ZN_EditorPropertyTextChangeOnSubmit::_bind_methods() {}

} // namespace zylann
