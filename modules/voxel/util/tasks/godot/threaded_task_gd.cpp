/**************************************************************************/
/*  threaded_task_gd.cpp                                                  */
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

#include "threaded_task_gd.h"

namespace zylann {

// Using a decoupled pattern so we can do a few more safety checks for scripters
class ZN_ThreadedTaskInternal : public IThreadedTask {
public:
	Ref<ZN_ThreadedTask> ref;

	const char *get_debug_name() const override {
		return "ZN_ThreadedTaskInternal";
	}

	void run(ThreadedTaskContext &ctx) override {
		ref->run(ctx.thread_index);
	}

	void apply_result() override {
		// Not exposed. Scripters may prefer to use a `completed` signal instead.
		ref->mark_completed();
		// Clear the reference to break potential circular reference and allow proper cleanup
		ref.unref();
	}

	TaskPriority get_priority() override {
		TaskPriority priority;
		priority.whole = ref->get_priority();
		return priority;
	}

	bool is_cancelled() override {
		return ref->is_cancelled();
	}
};

void ZN_ThreadedTask::run(int thread_index) {
	GDVIRTUAL_CALL(_run, thread_index);
}

int ZN_ThreadedTask::get_priority() {
	int priority = 0;
	if (GDVIRTUAL_CALL(_get_priority, priority)) {
		return priority;
	}
	return 0;
}

bool ZN_ThreadedTask::is_cancelled() {
	bool cancelled = false;
	if (GDVIRTUAL_CALL(_is_cancelled, cancelled)) {
		return cancelled;
	}
	return false;
}

bool ZN_ThreadedTask::is_scheduled() const {
	return _scheduled_task != nullptr;
}

void ZN_ThreadedTask::mark_completed() {
	_scheduled_task = nullptr;
	emit_signal("completed");
}

IThreadedTask *ZN_ThreadedTask::create_task() {
	CRASH_COND(_scheduled_task != nullptr);
	_scheduled_task = memnew(ZN_ThreadedTaskInternal);
	_scheduled_task->ref.reference_ptr(this);
	return _scheduled_task;
}

void ZN_ThreadedTask::_bind_methods() {
	ADD_SIGNAL(MethodInfo("completed"));

	GDVIRTUAL_BIND(_run, "thread_index");
	GDVIRTUAL_BIND(_get_priority);
	GDVIRTUAL_BIND(_is_cancelled);
}

} // namespace zylann
