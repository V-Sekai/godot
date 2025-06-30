/**************************************************************************/
/*  progressive_task_runner.h                                             */
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

#include "../containers/std_queue.h"
#include <cstdint>

namespace zylann {

// TODO It would be really nice if Godot4 Vulkan buffer deallocation was better optimized.
// This is originally to workaround the terribly slow Vulkan buffer deallocation in Godot4.
// It happens on the main thread and causes deferred stutters when a terrain contains a lot of chunks
// and the camera moves fast.
// I hate this workaround because it feels like we are almost not in control of a stable framerate.
// "Make less meshes" is not enough, if it can't be dynamically addressed.

class IProgressiveTask {
public:
	virtual ~IProgressiveTask() {}
	virtual void run() = 0;
};

// Runs a certain amount of tasks per frame such that all tasks should be completed in N seconds.
// This has the effect of spreading the load over time and tends to smooth out CPU spikes.
// This can be used in place of a time-slicing runner when the direct duration of tasks cannot be used as a cost metric.
// This is the case of tasks that delegate their workload to another unreachable system to run later (I'm looking at you
// Godot). It is far from perfect though, and is a last resort solution when optimization and threading are not
// possible. Such tasks may preferably not require low latency in the game, because they will likely run a bit later
// than a time-sliced task.
class ProgressiveTaskRunner {
public:
	~ProgressiveTaskRunner();

	void push(IProgressiveTask *task);
	void process();
	void flush();
	unsigned int get_pending_count() const;

private:
	static const unsigned int MIN_COUNT = 4;
	static const unsigned int COMPLETION_TIME_MSEC = 500;

	StdQueue<IProgressiveTask *> _tasks;
	unsigned int _dequeue_count = MIN_COUNT;
	int64_t _last_process_time_msec = 0;
};

} // namespace zylann
