/**************************************************************************/
/*  std_allocator.h                                                       */
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

#include <limits>
// #include <new>

#include "../errors.h"
#include "memory.h"

#ifdef DEBUG_ENABLED
#include <atomic>
#endif

namespace zylann {

#ifdef DEBUG_ENABLED
namespace StdDefaultAllocatorCounters {
extern std::atomic_uint64_t g_allocated;
extern std::atomic_uint64_t g_deallocated;
} // namespace StdDefaultAllocatorCounters
#endif

// Default allocator matching standard library requirements.
// When compiling with Godot or GDExtension, it will use Godot's default allocator.
template <class T>
struct StdDefaultAllocator {
	typedef T value_type;

	StdDefaultAllocator() = default;

	template <class U>
	constexpr StdDefaultAllocator(const StdDefaultAllocator<U> &) noexcept {}

	[[nodiscard]] T *allocate(std::size_t n) {
		ZN_ASSERT(n <= std::numeric_limits<std::size_t>::max() / sizeof(T));
		// if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
		// 	throw std::bad_array_new_length();
		// }

		if (T *p = static_cast<T *>(ZN_ALLOC(n * sizeof(T)))) {
#ifdef DEBUG_ENABLED
			StdDefaultAllocatorCounters::g_allocated += n * sizeof(T);
#endif
			return p;
		}

		// throw std::bad_alloc();
		ZN_CRASH_MSG("Bad alloc");
		return nullptr;
	}

	void deallocate(T *p, std::size_t n) noexcept {
#ifdef DEBUG_ENABLED
		StdDefaultAllocatorCounters::g_deallocated += n * sizeof(T);
#endif
		ZN_FREE(p);
	}

	// Note: defining a `rebind` struct is optional as long as the allocator is a template class. It is therefore
	// provided by `allocator_traits`. `rebind` is used by containers to obtain the same allocator with a different T,
	// in order to allocate internal data structures (nodes of linked list, buckets of unordered_map...)
};

template <class T, class U>
bool operator==(const StdDefaultAllocator<T> &, const StdDefaultAllocator<U> &) {
	return true;
}

template <class T, class U>
bool operator!=(const StdDefaultAllocator<T> &, const StdDefaultAllocator<U> &) {
	return false;
}

} // namespace zylann
