/**************************************************************************/
/*  editor_inspector_plugin.cpp                                           */
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

#include "editor_inspector_plugin.h"

namespace zylann::godot {

#if defined(ZN_GODOT)
bool ZN_EditorInspectorPlugin::can_handle(Object *p_object) {
#elif defined(ZN_GODOT_EXTENSION)
bool ZN_EditorInspectorPlugin::_can_handle(Object *p_object) const {
#endif
	return _zn_can_handle(p_object);
}

#if defined(ZN_GODOT)
void ZN_EditorInspectorPlugin::parse_begin(Object *p_object) {
#elif defined(ZN_GODOT_EXTENSION)
void ZN_EditorInspectorPlugin::_parse_begin(Object *p_object) {
#endif
	_zn_parse_begin(p_object);
}

#if defined(ZN_GODOT)
void ZN_EditorInspectorPlugin::parse_end(Object *p_object) {
#elif defined(ZN_GODOT_EXTENSION)
void ZN_EditorInspectorPlugin::_parse_end(Object *p_object) {
#endif
	_zn_parse_end(p_object);
}

#if defined(ZN_GODOT)
void ZN_EditorInspectorPlugin::parse_group(Object *p_object, const String &p_group) {
#elif defined(ZN_GODOT_EXTENSION)
void ZN_EditorInspectorPlugin::_parse_group(Object *p_object, const String &p_group) {
#endif
	_zn_parse_group(p_object, p_group);
}

#if defined(ZN_GODOT)
bool ZN_EditorInspectorPlugin::parse_property(
		Object *p_object,
		const Variant::Type p_type,
		const String &p_path,
		const PropertyHint p_hint,
		const String &p_hint_text,
		const BitField<PropertyUsageFlags> p_usage,
		const bool p_wide
) {
#elif defined(ZN_GODOT_EXTENSION)
bool ZN_EditorInspectorPlugin::_parse_property(
		Object *p_object,
		Variant::Type p_type,
		const String &p_path,
		PropertyHint p_hint,
		const String &p_hint_text,
		BitField<PropertyUsageFlags> p_usage,
		const bool p_wide
) {
#endif
	return _zn_parse_property(p_object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide);
}

bool ZN_EditorInspectorPlugin::_zn_can_handle(const Object *p_object) const {
	return false;
}

void ZN_EditorInspectorPlugin::_zn_parse_begin(Object *p_object) {}

void ZN_EditorInspectorPlugin::_zn_parse_end(Object *p_object) {}

void ZN_EditorInspectorPlugin::_zn_parse_group(Object *p_object, const String &p_group) {}

bool ZN_EditorInspectorPlugin::_zn_parse_property(
		Object *p_object,
		const Variant::Type p_type,
		const String &p_path,
		const PropertyHint p_hint,
		const String &p_hint_text,
		const BitField<PropertyUsageFlags> p_usage,
		const bool p_wide
) {
	return false;
}

} // namespace zylann::godot
