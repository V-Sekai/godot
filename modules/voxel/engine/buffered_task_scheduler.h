/**************************************************************************/
/*  buffered_task_scheduler.h                                             */
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

#include "../util/containers/std_vector.h"
#include "../util/thread/thread.h"

namespace zylann {

class IThreadedTask;

namespace voxel {

// Helper class to store tasks and schedule them in a single batch
class BufferedTaskScheduler {
public:
	static BufferedTaskScheduler &get_for_current_thread();

	inline void push_main_task(IThreadedTask *task) {
		_main_tasks.push_back(task);
	}

	inline void push_io_task(IThreadedTask *task) {
		_io_tasks.push_back(task);
	}

	inline unsigned int get_main_count() const {
		return _main_tasks.size();
	}

	inline unsigned int get_io_count() const {
		return _io_tasks.size();
	}

	void flush();

	// No destructor! This does not take ownership, it is only a helper. Flush should be called after each use.

private:
	BufferedTaskScheduler();

	bool has_tasks() const {
		return _main_tasks.size() > 0 || _io_tasks.size() > 0;
	}

	StdVector<IThreadedTask *> _main_tasks;
	StdVector<IThreadedTask *> _io_tasks;
	Thread::ID _thread_id;
};

} // namespace voxel
} // namespace zylann
