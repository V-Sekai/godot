/**************************************************************************/
/*  editor_inspector_plugin.h                                             */
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

#if defined(ZN_GODOT)
#include <editor/editor_inspector.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/editor_inspector_plugin.hpp>
using namespace godot;
#endif

namespace zylann::godot {

class ZN_EditorInspectorPlugin : public EditorInspectorPlugin {
	GDCLASS(ZN_EditorInspectorPlugin, EditorInspectorPlugin)
public:
#if defined(ZN_GODOT)
	bool can_handle(Object *p_object) override;
	void parse_begin(Object *p_object) override;
	void parse_end(Object *p_object) override;
	void parse_group(Object *p_object, const String &p_group) override;
	bool parse_property(
			Object *p_object,
			const Variant::Type p_type,
			const String &p_path,
			const PropertyHint p_hint,
			const String &p_hint_text,
			const BitField<PropertyUsageFlags> p_usage,
			const bool p_wide = false
	) override;
#elif defined(ZN_GODOT_EXTENSION)
	bool _can_handle(Object *p_object) const override;
	void _parse_begin(Object *p_object) override;
	void _parse_end(Object *p_object) override;
	void _parse_group(Object *p_object, const String &p_group) override;
	bool _parse_property(
			Object *p_object,
			Variant::Type p_type,
			const String &p_path,
			PropertyHint p_hint,
			const String &p_hint_text,
			BitField<PropertyUsageFlags> p_usage,
			const bool p_wide = false
	) override;
#endif

protected:
	virtual bool _zn_can_handle(const Object *p_object) const;
	virtual void _zn_parse_begin(Object *p_object);
	virtual void _zn_parse_end(Object *p_object);
	virtual void _zn_parse_group(Object *p_object, const String &p_group);
	virtual bool _zn_parse_property(
			Object *p_object,
			const Variant::Type p_type,
			const String &p_path,
			const PropertyHint p_hint,
			const String &p_hint_text,
			const BitField<PropertyUsageFlags> p_usage,
			const bool p_wide
	);

private:
	// When compiling with GodotCpp, `_bind_methods` is not optional
	static void _bind_methods() {}
};

} // namespace zylann::godot
