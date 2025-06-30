/**************************************************************************/
/*  rw_lock.h                                                             */
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

#include <shared_mutex>

// #define ZN_PROFILE_RWLOCK
#ifdef ZN_PROFILE_RWLOCK
#include "../profiling.h"
#endif

namespace zylann {

class RWLock {
public:
	// Lock the rwlock, block if locked for write by another thread.
	// WARNING: cannot be locked twice by the same thread, it is undefined behavior.
	void read_lock() const {
#ifdef ZN_PROFILE_RWLOCK
		ZN_PROFILE_SCOPE();
#endif
		_mutex.lock_shared();
	}

	// Unlock the rwlock, let other threads continue
	void read_unlock() const {
		_mutex.unlock_shared();
	}

	// Attempt to lock the rwlock, returns `true` on success, `false` means it can't lock.
	bool read_try_lock() const {
		return _mutex.try_lock_shared();
	}

	// Lock the rwlock, block if locked by someone else
	void write_lock() {
#ifdef ZN_PROFILE_RWLOCK
		ZN_PROFILE_SCOPE();
#endif
		_mutex.lock();
	}

	// Unlock the rwlock, let other thwrites continue
	void write_unlock() {
		_mutex.unlock();
	}

	// Attempt to lock the rwlock, returns `true` on success, `false` means it can't lock.
	bool write_try_lock() {
		return _mutex.try_lock();
	}

private:
	mutable std::shared_timed_mutex _mutex;
};

class RWLockRead {
public:
	RWLockRead(const RWLock &p_lock) : _lock(p_lock) {
		_lock.read_lock();
	}
	~RWLockRead() {
		_lock.read_unlock();
	}

private:
	const RWLock &_lock;
};

class RWLockWrite {
public:
	RWLockWrite(RWLock &p_lock) : _lock(p_lock) {
		_lock.write_lock();
	}
	~RWLockWrite() {
		_lock.write_unlock();
	}

private:
	RWLock &_lock;
};

} // namespace zylann
