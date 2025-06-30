/**************************************************************************/
/*  generate_instances_block_task.h                                       */
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

#include "../../generators/voxel_generator.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/core/array.h"
#include "../../util/tasks/threaded_task.h"
#include "instancer_task_output_queue.h"
#include "up_mode.h"
#include "voxel_instance_generator.h"

#include <cstdint>
#include <memory>

namespace zylann::voxel {

// TODO Optimize: eventually this should be moved closer to the meshing task, including edited instances
class GenerateInstancesBlockTask : public IThreadedTask {
public:
	Vector3i mesh_block_grid_position;
	uint16_t layer_id;
	uint8_t lod_index;
	uint8_t edited_mask;
	UpMode up_mode;
	float mesh_block_size;
	Array surface_arrays;
	int32_t vertex_range_end = -1;
	int32_t index_range_end = -1;
	Ref<VoxelInstanceGenerator> generator;
	Ref<VoxelGenerator> voxel_generator;
	// Can be pre-populated by edited transforms
	StdVector<Transform3f> transforms;
	std::shared_ptr<InstancerTaskOutputQueue> output_queue;

	const char *get_debug_name() const override {
		return "GenerateInstancesBlock";
	}

	void run(ThreadedTaskContext &ctx) override;
};

} // namespace zylann::voxel
