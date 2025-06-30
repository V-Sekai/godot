/**************************************************************************/
/*  editor_property.h                                                     */
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
#include <godot_cpp/classes/editor_property.hpp>
using namespace godot;
#endif

#include "../../containers/span.h"

namespace zylann::godot {

// This obscure method is actually used to get the XYZW tinting colors for controls that expose coordinates.
// In modules, this is `_get_property_colors`, but it is not exposed in GDExtension.
Span<const Color> editor_property_get_colors(EditorProperty &self);

class ZN_EditorProperty : public EditorProperty {
	GDCLASS(ZN_EditorProperty, EditorProperty)
public:
#if defined(ZN_GODOT)
	void update_property() override;
#elif defined(ZN_GODOT_EXTENSION)
	void _update_property() override;
#endif

#ifdef ZN_GODOT
protected:
#endif
	// This method is protected in core, but public in GDExtension...
	void _set_read_only(bool p_read_only) override;

protected:
	virtual void _zn_update_property();
	virtual void _zn_set_read_only(bool p_read_only);

private:
	// When compiling with GodotCpp, `_bind_methods` is not optional
	static void _bind_methods() {}
};

} // namespace zylann::godot
