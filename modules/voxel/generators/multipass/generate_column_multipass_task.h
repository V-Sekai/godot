/**************************************************************************/
/*  generate_column_multipass_task.h                                      */
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

#include "../../util/tasks/threaded_task.h"
#include "voxel_generator_multipass_cb.h"

namespace zylann::voxel {

class BufferedTaskScheduler;

// This task is designed to be scheduled from another.
// Looks up columns in the generator's cache in order to run a pass on a specific column.
// If at least one column isn't found in the map, the task is cancelled, and so should be all its callers.
// Otherwise:
// If a column doesn't fulfills dependency requirements:
//     - If another task is working on that column, the current task is postponed to run later.
//     - Otherwise, a subtask is spawned to work on the dependency.
//       The current task is queued after every subtask spawned this way.
// Otherwise, the task runs the pass, re-schedules its caller, and returns.
//
// One reason to use this pattern instead of "pyramid diffs", is that it can be invoked without assumptions. It will
// return a result if necessary, even if the map is in inconsistent state. We can even decide to override states.
// It actively looks for dependencies, rather than assuming they are loaded by separate logic.
class GenerateColumnMultipassTask : public IThreadedTask {
public:
	GenerateColumnMultipassTask(
			Vector2i p_column_position,
			VoxelFormat p_format,
			uint8_t p_block_size,
			uint8_t p_subpass_index,
			std::shared_ptr<VoxelGeneratorMultipassCBStructs::Internal> p_generator_internal,
			Ref<VoxelGeneratorMultipassCB> p_generator,
			TaskPriority p_priority,
			// When the current task finishes, it will decrement the given counter, and return control to the following
			// caller task when the counter reaches 0.
			IThreadedTask *p_caller,
			std::shared_ptr<std::atomic_int> p_caller_dependency_count
	);

	~GenerateColumnMultipassTask();

	const char *get_debug_name() const override {
		return "GenerateColumnMultipassTask";
	}

	void run(ThreadedTaskContext &ctx) override;

	TaskPriority get_priority() override {
		return _priority;
	}

	// Cancellation cannot use this API for now (it would prevent the task from running) because the task must run in
	// order to re-schedule its caller. Eventually we may find a way to integrate this pattern into the framework.
	// bool is_cancelled() {}

private:
	void schedule_final_block_tasks(
			VoxelGeneratorMultipassCBStructs::Column &column,
			BufferedTaskScheduler &task_scheduler
	);
	void return_to_caller(bool success);

	Vector2i _column_position;
	VoxelFormat _format;
	TaskPriority _priority;
	uint8_t _block_size;
	uint8_t _subpass_index;
	bool _cancelled = false;
	std::shared_ptr<VoxelGeneratorMultipassCBStructs::Internal> _generator_internal;
	Ref<VoxelGeneratorMultipassCB> _generator;
	// Task to execute when `caller_task_dependency_counter` reaches zero.
	// This means the current task was spawned by this one to compute a dependency.
	IThreadedTask *_caller_task = nullptr;
	// Counter to decrement when the task finishes with an outcome equivalent to "the requested block has been
	// processed".
	std::shared_ptr<std::atomic_int> _caller_task_dependency_counter;
	GenerateColumnMultipassTask *_caller_mp_task = nullptr;
};

} // namespace zylann::voxel
