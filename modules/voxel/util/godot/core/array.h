/**************************************************************************/
/*  array.h                                                               */
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
#include <core/variant/array.h>

#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/variant/array.hpp>
using namespace godot;

// TODO GDX: GodotCpp does not define `varray` (which is available in modules)
// Here exposed the same as in core, so no namespace

inline godot::Array varray(const godot::Variant &a0) {
	godot::Array a;
	a.append(a0);
	return a;
}

inline godot::Array varray(const godot::Variant &a0, const godot::Variant &a1) {
	godot::Array a;
	a.append(a0);
	a.append(a1);
	return a;
}

inline godot::Array varray(const godot::Variant &a0, const godot::Variant &a1, const godot::Variant &a2) {
	godot::Array a;
	a.append(a0);
	a.append(a1);
	a.append(a2);
	return a;
}

inline godot::Array varray(
		const godot::Variant &a0,
		const godot::Variant &a1,
		const godot::Variant &a2,
		const godot::Variant &a3
) {
	godot::Array a;
	a.append(a0);
	a.append(a1);
	a.append(a2);
	a.append(a3);
	return a;
}

inline godot::Array varray(
		const godot::Variant &a0,
		const godot::Variant &a1,
		const godot::Variant &a2,
		const godot::Variant &a3,
		const godot::Variant &a4
) {
	godot::Array a;
	a.append(a0);
	a.append(a1);
	a.append(a2);
	a.append(a3);
	a.append(a4);
	return a;
}

inline godot::Array varray(
		const godot::Variant &a0,
		const godot::Variant &a1,
		const godot::Variant &a2,
		const godot::Variant &a3,
		const godot::Variant &a4,
		const godot::Variant &a5
) {
	godot::Array a;
	a.append(a0);
	a.append(a1);
	a.append(a2);
	a.append(a3);
	a.append(a4);
	a.append(a5);
	return a;
}

#endif // ZN_GODOT_EXTENSION
