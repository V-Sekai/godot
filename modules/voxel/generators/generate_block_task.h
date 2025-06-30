/**************************************************************************/
/*  generate_block_task.h                                                 */
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

#include "../engine/ids.h"
#include "../engine/priority_dependency.h"
#include "../engine/streaming_dependency.h"
#include "../util/containers/std_vector.h"
#include "../util/tasks/threaded_task.h"

#ifdef VOXEL_ENABLE_GPU
#include "generate_block_gpu_task.h"
#endif

namespace zylann {

class AsyncDependencyTracker;

namespace voxel {

class VoxelData;

// Generic task to procedurally generate a block of voxels in a single pass
class GenerateBlockTask
#ifdef VOXEL_ENABLE_GPU
		: public IGeneratingVoxelsThreadedTask
#else
		: public IThreadedTask
#endif
{
public:
	GenerateBlockTask(const VoxelGenerator::BlockTaskParams &params);
	~GenerateBlockTask();

	const char *get_debug_name() const override {
		return "GenerateBlock";
	}

	void run(ThreadedTaskContext &ctx) override;
	TaskPriority get_priority() override;
	bool is_cancelled() override;
	void apply_result() override;

#ifdef VOXEL_ENABLE_GPU
	void set_gpu_results(StdVector<GenerateBlockGPUTaskResult> &&results) override;
#endif

private:
#ifdef VOXEL_ENABLE_GPU
	void run_gpu_task(zylann::ThreadedTaskContext &ctx);
	void run_gpu_conversion();
#endif
	void run_cpu_generation();
	void run_stream_saving_and_finish();

	// Not an input, but can be assigned a reusable instance to avoid allocating one in the task
	std::shared_ptr<VoxelBuffer> _voxels;

	Vector3i _position;
	VoxelFormat _format;
	VolumeID _volume_id;
	uint8_t _lod_index;
	uint8_t _block_size;
	bool _drop_beyond_max_distance = true;
#ifdef VOXEL_ENABLE_GPU
	bool _use_gpu = false;
#endif
	PriorityDependency _priority_dependency;
	std::shared_ptr<StreamingDependency> _stream_dependency; // For saving generator output
	std::shared_ptr<VoxelData> _data; // Just for modifiers
	std::shared_ptr<AsyncDependencyTracker> _tracker; // For async edits
	TaskCancellationToken _cancellation_token;

	bool _has_run = false;
	bool _too_far = false;
	bool _max_lod_hint = false;
#ifdef VOXEL_ENABLE_GPU
	uint8_t _stage = 0;
	StdVector<GenerateBlockGPUTaskResult> _gpu_generation_results;
#endif
};

} // namespace voxel
} // namespace zylann
