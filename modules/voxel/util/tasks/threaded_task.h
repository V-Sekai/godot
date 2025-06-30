/**************************************************************************/
/*  threaded_task.h                                                       */
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

#include "task_priority.h"
#include <cstdint>

namespace zylann {

struct ThreadedTaskContext {
	enum Status : uint8_t {
		// The task is complete and will be put in the list of completed tasks by the TaskRunner. It will be deleted
		// later. This is the default status.
		STATUS_COMPLETE = 0,
		// The task is not complete and will be re-run later by the TaskRunner
		STATUS_POSTPONED = 1,
		// The task is not complete and will be re-scheduled by another custom task.
		// The TaskRunner will simply drop its pointer and won't put it in the list of completed tasks.
		// Initially added so we can schedule task B from task A, and have B re-schedule A to use results computed in A
		STATUS_TAKEN_OUT = 2
	};

	// Index of the thread within the runner's pool. Can be used to index arrays as an alternative to thread_local
	// storage.
	const uint8_t thread_index;
	// May be set by the task to signal its status after run
	Status status;
	// Cached priority of the current task. May be useful to copy if the current task spawns other related tasks.
	const TaskPriority task_priority;
	// If this is set to a non-null task, it will run right after the current one on the same thread.
	// By doing so, ownership is given to ThreadedTaskRunner. These tasks must not have been owned by the runner
	// already. Priority of such tasks is not relevant.
	// IThreadedTask *next_immediate_task;

	ThreadedTaskContext(uint8_t p_thread_index, TaskPriority p_priority) :
			thread_index(p_thread_index),
			// By default, if the task does not set this status, it will be considered complete after run
			status(STATUS_COMPLETE),
			task_priority(p_priority) {}

	// To allow scheduling tasks from within tasks, without having to pass it in or use a global
	// ThreadedTaskRunner &runner;
};

// Interface for a task that will run in `ThreadedTaskRunner`.
// The task will run in another thread.
class IThreadedTask {
public:
	virtual ~IThreadedTask() {}

	// Called from within the thread pool
	virtual void run(ThreadedTaskContext &ctx) = 0;

	// Convenience method which can be called by the scheduler of the task (usually on the main thread)
	// in order to apply results. It is not called from the thread pool.
	virtual void apply_result() {};

	// Hints how soon this task will be executed after being scheduled. This is relevant when there are a lot of tasks.
	// Lower values means higher priority.
	// Can change between two calls. The thread pool will poll this value regularly over some time interval.
	virtual TaskPriority get_priority() {
		// Defaulting to maximum priority as it's the most common expectation.
		return TaskPriority::max();
	}

	// May return `true` in order for the thread pool to skip the task
	virtual bool is_cancelled() {
		return false;
	}

	// Gets the name of the task for debug purposes. The returned name's lifetime must span the execution of the engine
	// (usually a string literal).
	virtual const char *get_debug_name() const {
		return "<unnamed>";
	}
};

} // namespace zylann
