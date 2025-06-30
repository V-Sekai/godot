/**************************************************************************/
/*  buffered_task_scheduler.cpp                                           */
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

#include "buffered_task_scheduler.h"
#include "../util/containers/span.h"
#include "../util/errors.h"
#include "../util/io/log.h"
#include "voxel_engine.h"

namespace zylann::voxel {

BufferedTaskScheduler::BufferedTaskScheduler() : _thread_id(Thread::get_caller_id()) {}

BufferedTaskScheduler &BufferedTaskScheduler::get_for_current_thread() {
	static thread_local BufferedTaskScheduler tls_task_scheduler;
	if (tls_task_scheduler.has_tasks()) {
		ZN_PRINT_WARNING("Getting BufferedTaskScheduler for a new batch but it already has tasks!");
	}
	return tls_task_scheduler;
}

void BufferedTaskScheduler::flush() {
	ZN_ASSERT(_thread_id == Thread::get_caller_id());
	if (_main_tasks.size() > 0) {
		VoxelEngine::get_singleton().push_async_tasks(to_span(_main_tasks));
	}
	if (_io_tasks.size() > 0) {
		VoxelEngine::get_singleton().push_async_io_tasks(to_span(_io_tasks));
	}
	_main_tasks.clear();
	_io_tasks.clear();
}

} // namespace zylann::voxel
