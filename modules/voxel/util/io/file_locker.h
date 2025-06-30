/**************************************************************************/
/*  file_locker.h                                                         */
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

#include "../containers/std_unordered_map.h"
#include "../errors.h"
#include "../string/std_string.h"
#include "../thread/mutex.h"
#include "../thread/rw_lock.h"

namespace zylann {

// Performs software locking on paths,
// so that multiple threads (controlled by this module) wanting to access the same file will lock a shared mutex.
class FileLocker {
public:
	void lock_read(const StdString &fpath) {
		lock(fpath, true);
	}

	void lock_write(const StdString &fpath) {
		lock(fpath, false);
	}

	void unlock(const StdString &fpath) {
		unlock_internal(fpath);
	}

private:
	struct File {
		RWLock lock;
		bool read_only;
	};

	void lock(const StdString &fpath, bool read_only) {
		File *fp = nullptr;
		{
			MutexLock lock(_files_mutex);
			// Get or create.
			// Note, we never remove entries from the map
			fp = &_files[fpath];
		}

		if (read_only) {
			fp->lock.read_lock();
			// The read lock was acquired. It means nobody is writing.
			fp->read_only = true;

		} else {
			fp->lock.write_lock();
			// The write lock was acquired. It means only one thread is writing.
			fp->read_only = false;
		}
	}

	void unlock_internal(const StdString &fpath) {
		File *fp = nullptr;
		{
			MutexLock lock(_files_mutex);
			auto it = _files.find(fpath);
			if (it != _files.end()) {
				fp = &it->second;
			}
		}
		ZN_ASSERT_RETURN(fp != nullptr);
		// TODO FileAccess::reopen can have been called, nullifying my efforts to enforce thread sync :|
		// So for now please don't do that

		if (fp->read_only) {
			fp->lock.read_unlock();
		} else {
			fp->lock.write_unlock();
		}
	}

private:
	Mutex _files_mutex;
	StdUnorderedMap<StdString, File> _files;
};

} // namespace zylann
