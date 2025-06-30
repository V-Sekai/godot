/**************************************************************************/
/*  voxel_modifier.cpp                                                    */
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

#include "voxel_modifier.h"
#include "../engine/gpu/gpu_task_runner.h"

namespace zylann::voxel {

void VoxelModifier::set_transform(Transform3D t) {
	RWLockWrite wlock(_rwlock);
	if (t == _transform) {
		return;
	}
	_transform = t;
#ifdef VOXEL_ENABLE_GPU
	_shader_data_need_update = true;
#endif
	update_aabb();
}

#ifdef VOXEL_ENABLE_GPU

RID VoxelModifier::get_detail_shader(const BaseGPUResources &base_resources, const Type type) {
	switch (type) {
		case TYPE_SPHERE:
			return base_resources.detail_modifier_sphere_shader.rid;
		case TYPE_MESH:
			return base_resources.detail_modifier_mesh_shader.rid;
		default:
			ZN_PRINT_ERROR("Unhandled modifier type");
			return RID();
	}
}

RID VoxelModifier::get_block_shader(const BaseGPUResources &base_resources, const Type type) {
	switch (type) {
		case TYPE_SPHERE:
			return base_resources.block_modifier_sphere_shader.rid;
		case TYPE_MESH:
			return base_resources.block_modifier_mesh_shader.rid;
		default:
			ZN_PRINT_ERROR("Unhandled modifier type");
			return RID();
	}
}

#endif

} // namespace zylann::voxel
