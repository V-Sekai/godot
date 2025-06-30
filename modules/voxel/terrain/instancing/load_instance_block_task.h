/**************************************************************************/
/*  load_instance_block_task.h                                            */
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
#include "../../streams/voxel_stream.h"
#include "../../util/godot/core/array.h"
#include "../../util/math/vector3i.h"
#include "../../util/tasks/threaded_task.h"
#include "instancer_task_output_queue.h"
#include "up_mode.h"
#include "voxel_instance_library.h"
#include <cstdint>
#include <memory>

namespace zylann::voxel {

struct InstancerQuickReloadingCache;

// Loads all instances of all layers of a specific LOD in a specific chunk
class LoadInstanceChunkTask : public IThreadedTask {
public:
	LoadInstanceChunkTask(
			std::shared_ptr<InstancerTaskOutputQueue> output_queue,
			Ref<VoxelStream> stream,
			Ref<VoxelGenerator> voxel_generator,
			std::shared_ptr<InstancerQuickReloadingCache> quick_reload_cache,
			Ref<VoxelInstanceLibrary> library,
			Array mesh_arrays,
			const int32_t vertex_range_end,
			const int32_t index_range_end,
			const Vector3i grid_position,
			const uint8_t lod_index,
			const uint8_t instance_block_size,
			const uint8_t data_block_size,
			const UpMode up_mode
	);

	const char *get_debug_name() const override {
		return "LoadInstanceChunk";
	}

	void run(ThreadedTaskContext &ctx) override;

private:
	std::shared_ptr<InstancerTaskOutputQueue> _output_queue;
	Ref<VoxelStream> _stream;
	Ref<VoxelGenerator> _voxel_generator;
	std::shared_ptr<InstancerQuickReloadingCache> _quick_reload_cache;
	Ref<VoxelInstanceLibrary> _library;
	Array _mesh_arrays;
	const int32_t _vertex_range_end;
	const int32_t _index_range_end;
	Vector3i _render_grid_position;
	uint8_t _lod_index;
	uint8_t _instance_block_size;
	uint8_t _data_block_size;
	UpMode _up_mode;
};

} // namespace zylann::voxel
