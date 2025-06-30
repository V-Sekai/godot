/**************************************************************************/
/*  dictionary.h                                                          */
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
#include <core/variant/dictionary.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/variant.hpp>
using namespace godot;
#endif

namespace zylann::godot {

template <typename T>
inline bool try_get(const Dictionary &d, const Variant &key, T &out_value) {
#if defined(ZN_GODOT)
	const Variant *v = d.getptr(key);
	if (v == nullptr) {
		return false;
	}
	// TODO There is no easy way to return `false` if the value doesn't have the right type...
	// Because multiple C++ types match Variant types, and Variant types match multiple C++ types, and silently convert
	// between them.
	out_value = *v;
	return true;
#elif defined(ZN_GODOT_EXTENSION)
	Variant v = d.get(key, Variant());
	// TODO GDX: there is no way, in a single lookup, to differentiate an inexistent key and an existing key with the
	// value `null`. So we have to do a second lookup to check what NIL meant.
	if (v.get_type() == Variant::NIL) {
		out_value = T();
		return d.has(key);
	}
	out_value = v;
	return true;
#endif
}

} // namespace zylann::godot
