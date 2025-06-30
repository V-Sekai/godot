/**************************************************************************/
/*  generate_block_multipass_cb_task.h                                    */
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

#include "../../engine/ids.h"
#include "../../engine/priority_dependency.h"
#include "../../engine/streaming_dependency.h"
#include "../../util/tasks/threaded_task.h"
#include "../voxel_generator.h"

namespace zylann {

class AsyncDependencyTracker;

namespace voxel {

class VoxelData;

class GenerateBlockMultipassCBTask : public IThreadedTask {
public:
	GenerateBlockMultipassCBTask(const VoxelGenerator::BlockTaskParams &params);
	~GenerateBlockMultipassCBTask();

	const char *get_debug_name() const override {
		return "GenerateBlockMultipassCBTask";
	}

	void run(ThreadedTaskContext &ctx) override;
	TaskPriority get_priority() override;
	bool is_cancelled() override;
	void apply_result() override;

	// Not an input, but can be assigned a reusable instance to avoid allocating one in the task
	std::shared_ptr<VoxelBuffer> voxels;

private:
	void run_cpu_generation();
	void run_stream_saving_and_finish();

	Vector3i _block_position;
	VoxelFormat _format;
	VolumeID _volume_id;
	uint8_t _lod_index;
	uint8_t _block_size;
	bool _drop_beyond_max_distance = true;
	PriorityDependency _priority_dependency;
	std::shared_ptr<StreamingDependency> _stream_dependency; // For saving generator output
	std::shared_ptr<AsyncDependencyTracker> _tracker; // For async edits
	TaskCancellationToken _cancellation_token;

	bool _has_run = false;
	bool _too_far = false;
	uint8_t _stage = 0;
};

} // namespace voxel
} // namespace zylann
