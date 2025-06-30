/**************************************************************************/
/*  typed_array.h                                                         */
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
#include <core/variant/typed_array.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/variant/typed_array.hpp>
#endif

#include "../../containers/span.h"
#include <vector>

namespace zylann::godot {

template <typename T>
inline void copy_to(TypedArray<T> &dst, Span<const T> src) {
	dst.resize(src.size());
	for (unsigned int i = 0; i < src.size(); ++i) {
		dst[i] = src[i];
	}
}

template <typename T>
inline void copy_to(TypedArray<T> &dst, Span<const Ref<T>> src) {
	dst.resize(src.size());
	for (unsigned int i = 0; i < src.size(); ++i) {
		dst[i] = src[i];
	}
}

template <typename T, typename TAllocator>
inline void copy_to(std::vector<T, TAllocator> &dst, const TypedArray<T> &src) {
	dst.resize(src.size());
	for (int i = 0; i < src.size(); ++i) {
		dst[i] = src[i];
	}
}

template <typename T, typename TAllocator>
inline void copy_to(std::vector<Ref<T>, TAllocator> &dst, const TypedArray<T> &src) {
	dst.resize(src.size());
	for (int i = 0; i < src.size(); ++i) {
		dst[i] = src[i];
	}
}

template <typename T, typename TAllocator>
inline void copy_range_to(
		std::vector<Ref<T>, TAllocator> &dst,
		const TypedArray<T> &src,
		const int from,
		const int count
) {
	if (count == 0) {
		dst.clear();
		return;
	}
	ZN_ASSERT_RETURN(from >= 0 && from < src.size());
	ZN_ASSERT_RETURN(count >= 0 && from + count <= src.size());
	dst.resize(count);
	const int to = from + count;
	for (int i = from; i < to; ++i) {
		dst[i - from] = src[i];
	}
}

template <typename T>
inline TypedArray<T> to_typed_array(Span<const T> src) {
	TypedArray<T> array;
	copy_to(array, src);
	return array;
}

template <typename T>
inline TypedArray<T> to_typed_array(Span<const Ref<T>> src) {
	TypedArray<T> array;
	copy_to(array, src);
	return array;
}

#if defined(ZN_GODOT)

template <typename T>
Vector<Ref<T>> to_ref_vector(const TypedArray<T> &typed_array) {
	Vector<Ref<T>> refs;
	refs.resize(typed_array.size());
	for (int i = 0; i < typed_array.size(); ++i) {
		refs.write[i] = typed_array[i];
	}
	return refs;
}

#endif

} // namespace zylann::godot
