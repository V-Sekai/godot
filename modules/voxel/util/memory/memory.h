/**************************************************************************/
/*  memory.h                                                              */
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

#include <memory>

// Default new and delete operators.
#if defined(ZN_GODOT)

#include <core/os/memory.h>

// Use Godot's allocator.
// In modules, memnew and memdelete work for anything. However in GDExtension it might not be the case...
#define ZN_NEW(t) memnew(t)
#define ZN_DELETE(t) memdelete(t)
#define ZN_ALLOC(size) memalloc(size)
#define ZN_REALLOC(p, size) memrealloc(p, size)
#define ZN_FREE(p) memfree(p)

#elif defined(ZN_GODOT_EXTENSION)

#include <godot_cpp/core/memory.hpp>

#define ZN_NEW(t) memnew(t)
#define ZN_DELETE(t) ::godot::memdelete(t)
#define ZN_ALLOC(size) memalloc(size)
#define ZN_REALLOC(p, size) memrealloc(p, size)
#define ZN_FREE(p) memfree(p)

#endif

namespace zylann {

// Default, engine-agnostic implementation of unique pointers for this project. Allows to change it in one place.
// Note: array allocations are not used at the moment. Containers are preferred.

template <typename T>
struct DefaultObjectDeleter {
	constexpr DefaultObjectDeleter() noexcept = default;

	// This is required so we can implicitly convert from `UniquePtr<Derived>` to `UniquePtr<Base>`.
	// Looked it up from inside MSVC's STL implementation.
	template <class U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
	DefaultObjectDeleter(const DefaultObjectDeleter<U> &) noexcept {}

	inline void operator()(T *obj) {
		ZN_DELETE(obj);
	}
};

template <typename T>
using UniquePtr = std::unique_ptr<T, DefaultObjectDeleter<T>>;

template <class T, class... Types, std::enable_if_t<!std::is_array_v<T>, int> = 0>
UniquePtr<T> make_unique_instance(Types &&...args) {
	return UniquePtr<T>(ZN_NEW(T(std::forward<Types>(args)...)));
}

// Default, engine-agnostic implementation of shared pointers for this project.

template <class T, class... Types, std::enable_if_t<!std::is_array_v<T>, int> = 0>
inline std::shared_ptr<T> make_shared_instance(Types &&...args) {
	return std::shared_ptr<T>(ZN_NEW(T(std::forward<Types>(args)...)), [](T *p) { ZN_DELETE(p); });
}

} // namespace zylann
