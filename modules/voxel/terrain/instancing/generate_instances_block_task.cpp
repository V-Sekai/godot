/**************************************************************************/
/*  generate_instances_block_task.cpp                                     */
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

#include "generate_instances_block_task.h"
#include "../../util/godot/classes/array_mesh.h"
#include "../../util/profiling.h"

namespace zylann::voxel {

void GenerateInstancesBlockTask::run(ThreadedTaskContext &ctx) {
	ZN_PROFILE_SCOPE();
	ZN_ASSERT_RETURN(generator.is_valid());
	ZN_ASSERT(output_queue != nullptr);

	PackedVector3Array vertices = surface_arrays[ArrayMesh::ARRAY_VERTEX];
	ZN_ASSERT_RETURN(vertices.size() > 0);

	PackedVector3Array normals = surface_arrays[ArrayMesh::ARRAY_NORMAL];
	ZN_ASSERT_RETURN(normals.size() > 0);

	static thread_local StdVector<Transform3f> tls_generated_transforms;
	tls_generated_transforms.clear();

	const uint8_t gen_octant_mask = ~edited_mask;

	generator->generate_transforms(
			tls_generated_transforms,
			mesh_block_grid_position,
			lod_index,
			layer_id,
			surface_arrays,
			vertex_range_end,
			index_range_end,
			up_mode,
			gen_octant_mask,
			mesh_block_size,
			voxel_generator
	);

	for (const Transform3f &t : tls_generated_transforms) {
		transforms.push_back(t);
	}

	{
		MutexLock mlock(output_queue->mutex);
		output_queue->results.push_back(InstanceLoadingTaskOutput());
		InstanceLoadingTaskOutput &o = output_queue->results.back();
		o.layer_id = layer_id;
		o.edited_mask = edited_mask;
		o.render_block_position = mesh_block_grid_position;
		o.transforms = std::move(transforms);
	}
}

} // namespace zylann::voxel
