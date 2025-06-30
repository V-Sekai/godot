/**************************************************************************/
/*  editor_property.cpp                                                   */
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

#include "editor_property.h"
#include "../../containers/fixed_array.h"

namespace zylann::godot {

Span<const Color> editor_property_get_colors(EditorProperty &self) {
	static FixedArray<Color, 4> s_colors;
	const StringName sn_editor("Editor");
	s_colors[0] = self.get_theme_color(StringName("property_color_x"), sn_editor);
	s_colors[1] = self.get_theme_color(StringName("property_color_y"), sn_editor);
	s_colors[2] = self.get_theme_color(StringName("property_color_z"), sn_editor);
	s_colors[3] = self.get_theme_color(StringName("property_color_w"), sn_editor);
	return to_span(s_colors);
}

#if defined(ZN_GODOT)

void ZN_EditorProperty::update_property() {
	_zn_update_property();
}

#elif defined(ZN_GODOT_EXTENSION)

void ZN_EditorProperty::_update_property() {
	_zn_update_property();
}

#endif

void ZN_EditorProperty::_set_read_only(bool p_read_only) {
	_zn_set_read_only(p_read_only);
}

void ZN_EditorProperty::_zn_update_property() {}
void ZN_EditorProperty::_zn_set_read_only(bool p_read_only) {}

} // namespace zylann::godot
