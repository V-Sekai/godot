/**************************************************************************/
/*  std_string.h                                                          */
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

#include "../memory/std_allocator.h"
#include <string>

#ifdef __GNUC__
#include <string_view>
#endif

namespace zylann {

using StdString = std::basic_string<char, std::char_traits<char>, StdDefaultAllocator<char>>;

struct FwdConstStdString {
	const StdString &s;
	FwdConstStdString(const StdString &p_s) : s(p_s) {}
};

struct FwdMutableStdString {
	StdString &s;
	FwdMutableStdString(StdString &p_s) : s(p_s) {}
};

} // namespace zylann

#ifdef __GNUC__

// Attempt at fixing GCC having trouble dealing with `unordered_map<StdString, V> map;`.
// I couldn't understand why exactly that happens, whether it's a bug or not. In Compiler Explorer, all versions prior
// to GCC 13.1 fail to compile such code, except from 13.1 onwards. Manually defining a hash specialization for our
// alias seems to workaround it.
namespace std {
template <>
struct hash<zylann::StdString> {
	size_t operator()(const zylann::StdString &v) const {
		const std::string_view s(v);
		std::hash<std::string_view> hasher;
		return hasher(s);
	}
};
} // namespace std

#endif
