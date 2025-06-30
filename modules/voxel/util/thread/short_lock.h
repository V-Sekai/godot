/**************************************************************************/
/*  short_lock.h                                                          */
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

// #define ZN_SHORT_LOCK_IS_MUTEX

#ifdef ZN_SHORT_LOCK_IS_MUTEX
#include "mutex.h"
#else
#include "spin_lock.h"
#endif

namespace zylann {

// A mutex-like primitive that is expected to be locked for short periods of time.
// It can be implemented either with a SpinLock or a Mutex, depending on test results.

#ifdef ZN_SHORT_LOCK_IS_MUTEX
typedef BinaryMutex ShortLock;
#else
typedef SpinLock ShortLock;
#endif

struct ShortLockScope {
	ShortLock &short_lock;
	ShortLockScope(ShortLock &p_sl) : short_lock(p_sl) {
		short_lock.lock();
	}
	~ShortLockScope() {
		short_lock.unlock();
	}
};

} // namespace zylann
