/**************************************************************************/
/*  free_mesh_task.h                                                      */
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

#include "../engine/voxel_engine.h"
#include "../util/godot/direct_mesh_instance.h"
#include "../util/profiling.h"
#include "../util/tasks/progressive_task_runner.h"

namespace zylann::voxel {

// Had to resort to this in Godot4 because deleting meshes is particularly expensive,
// because of the Vulkan allocator used by the renderer.
// It is a deferred cost (it is not spent at the exact time the Mesh object is destroyed, it happens later), so had to
// use a different type of task to load-balance it. What this task actually does is just to hold a reference on a mesh a
// bit longer, assuming that mesh is no longer used. Then the execution of the task releases that reference.
class FreeMeshTask : public IProgressiveTask {
public:
	static inline void try_add_and_destroy(zylann::godot::DirectMeshInstance &mi) {
		const Mesh *mesh = mi.get_mesh_ptr();
		if (mesh != nullptr && mesh->get_reference_count() == 1) {
			// That instances holds the last reference to this mesh
			add(mi.get_mesh());
		}
		mi.destroy();
	}

	void run() override {
		ZN_PROFILE_SCOPE();
		if (_mesh->get_reference_count() > 1) {
			ZN_PRINT_WARNING(
					"Mesh has more than one ref left, task spreading will not be effective at smoothing "
					"destruction cost"
			);
		}
		_mesh.unref();
	}

private:
	static void add(Ref<Mesh> mesh) {
		ZN_ASSERT(mesh.is_valid());
		FreeMeshTask *task = ZN_NEW(FreeMeshTask(mesh));
		VoxelEngine::get_singleton().push_main_thread_progressive_task(task);
	}

	FreeMeshTask(Ref<Mesh> p_mesh) : _mesh(p_mesh) {}

	Ref<Mesh> _mesh;
};

} // namespace zylann::voxel
