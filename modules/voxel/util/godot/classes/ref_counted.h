/**************************************************************************/
/*  ref_counted.h                                                         */
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

#include <functional>

#if defined(ZN_GODOT)
#include <core/object/ref_counted.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/classes/ref_counted.hpp>
using namespace godot;
#endif

namespace zylann::godot {

// `(ref1 = ref2).is_valid()` does not work because Ref<T> does not implement an `operator=` returning the value.
// So instead we can write it as `try_get_as(ref2, ref1)`
template <typename From_T, typename To_T>
inline bool try_get_as(const Ref<From_T> &from, Ref<To_T> &to) {
	to = from;
	return to.is_valid();
}

// To allow using Ref<T> as key in Godot's HashMap
template <typename T>
struct RefHasher {
	static _FORCE_INLINE_ uint32_t hash(const Ref<T> &v) {
		return uint32_t(uint64_t(v.ptr())) * (0x9e3779b1L);
	}
};

} // namespace zylann::godot

namespace std {

// For Ref<T> keys in std::unordered_map, hashed by pointer, not by content
template <typename T>
struct hash<Ref<T>> {
	inline size_t operator()(const Ref<T> &v) const {
		return std::hash<const T *>{}(v.ptr());
	}
};

} // namespace std
