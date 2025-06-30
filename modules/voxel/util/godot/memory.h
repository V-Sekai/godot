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

#if defined(ZN_GODOT)
#include <core/os/memory.h>
#elif defined(ZN_GODOT_EXTENSION)
#include <godot_cpp/core/memory.hpp>
#endif

#include <memory>

namespace zylann::godot {

/*// Creates a shared_ptr which will always use Godot's allocation functions
template <typename T>
inline std::shared_ptr<T> gd_make_shared() {
	// std::make_shared() apparently won't allow us to specify custom new and delete
	return std::shared_ptr<T>(memnew(T), memdelete<T>);
}*/

// For use with smart pointers such as std::unique_ptr
template <typename T>
struct ObjectDeleter {
	inline void operator()(T *obj) {
		memdelete(obj);
	}
};

// Specialization of `std::unique_ptr which always uses Godot's `memdelete()` as deleter.
template <typename T>
using ObjectUniquePtr = std::unique_ptr<T, ObjectDeleter<T>>;

// Creates a `GodotObjectUniquePtr<T>` with an object constructed with `memnew()` inside.
template <typename T>
ObjectUniquePtr<T> make_unique() {
	return ObjectUniquePtr<T>(memnew(T));
}

} // namespace zylann::godot
