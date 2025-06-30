/**************************************************************************/
/*  mutex.h                                                               */
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

// #include "../profiling.h"
#include <mutex>

namespace zylann {

template <class StdMutexT>
class MutexImpl {
	mutable StdMutexT mutex;

public:
	inline void lock() const {
		// ZN_PROFILE_SCOPE();
		mutex.lock();
	}

	inline void unlock() const {
		mutex.unlock();
	}

	inline bool try_lock() const {
		return mutex.try_lock();
	}
};

template <class MutexT>
class MutexLock {
	const MutexT &mutex;

public:
	inline explicit MutexLock(const MutexT &p_mutex) : mutex(p_mutex) {
		mutex.lock();
	}

	inline ~MutexLock() {
		mutex.unlock();
	}
};

using Mutex = MutexImpl<std::recursive_mutex>; // Recursive, for general use
using BinaryMutex = MutexImpl<std::mutex>; // Non-recursive, handle with care

// Note: Godot uses a combination of `extern template` and `_ALWAYS_INLINE_` compiler-specific macros instead of
// `inline`. In that setup, without `_ALWAYS_INLINE_`, GCC does not inline methods in debug builds, which then causes
// `undefined reference` errors when linking. However, considering GCC does inline methods in optimized builds, I don't
// understand what we gain from that setup... so I go with simple `inline`.
//
// Don't instantiate these templates in every file where they are used, do it just once
// extern template class MutexImpl<std::recursive_mutex>;
// extern template class MutexImpl<std::mutex>;
// extern template class MutexLock<MutexImpl<std::recursive_mutex>>;
// extern template class MutexLock<MutexImpl<std::mutex>>;

} // namespace zylann
