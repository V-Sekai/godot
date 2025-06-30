/**************************************************************************/
/*  time_spread_task_runner.h                                             */
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

#include "../containers/fixed_array.h"
#include "../containers/span.h"
#include "../containers/std_queue.h"
#include "../thread/mutex.h"
#include <cstdint>

namespace zylann {

struct TimeSpreadTaskContext {
	// If this is set to `true` by a task,
	// it will be re-scheduled to run again, the next time the runner is processed.
	// Otherwise, the task will be destroyed after it runs.
	bool postpone = false;
};

class ITimeSpreadTask {
public:
	virtual ~ITimeSpreadTask() {}
	virtual void run(TimeSpreadTaskContext &ctx) = 0;
};

// Runs tasks in the caller thread, within a time budget per call. Kind of like coroutines.
class TimeSpreadTaskRunner {
public:
	enum Priority { //
		PRIORITY_NORMAL = 0,
		PRIORITY_LOW = 1,
		PRIORITY_COUNT
	};

	~TimeSpreadTaskRunner();

	// Pushing is thread-safe.
	void push(ITimeSpreadTask *task, Priority priority = PRIORITY_NORMAL);
	void push(Span<ITimeSpreadTask *> tasks, Priority priority = PRIORITY_NORMAL);

	void process(uint64_t time_budget_usec);
	void flush();
	unsigned int get_pending_count() const;

private:
	struct Queue {
		StdQueue<ITimeSpreadTask *> tasks;
		// TODO Optimization: naive thread safety. Should be enough for now.
		BinaryMutex tasks_mutex;
	};
	FixedArray<Queue, PRIORITY_COUNT> _queues;
};

} // namespace zylann
