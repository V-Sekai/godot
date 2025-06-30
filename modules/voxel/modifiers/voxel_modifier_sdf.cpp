/**************************************************************************/
/*  voxel_modifier_sdf.cpp                                                */
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

#include "voxel_modifier_sdf.h"
#include "../util/godot/core/packed_arrays.h"
#include "../util/memory/memory.h"

namespace zylann::voxel {

void VoxelModifierSdf::set_operation(Operation op) {
	RWLockWrite wlock(_rwlock);
	_operation = op;
#ifdef VOXEL_ENABLE_GPU
	_shader_data_need_update = true;
#endif
}

void VoxelModifierSdf::set_smoothness(float p_smoothness) {
	RWLockWrite wlock(_rwlock);
	const float smoothness = math::max(p_smoothness, 0.f);
	if (smoothness == _smoothness) {
		return;
	}
	_smoothness = smoothness;
#ifdef VOXEL_ENABLE_GPU
	_shader_data_need_update = true;
#endif
	update_aabb();
}

inline float get_largest_coord(Vector3 v) {
	return math::max(math::max(v.x, v.y), v.z);
}

#ifdef VOXEL_ENABLE_GPU

void VoxelModifierSdf::update_base_shader_data_no_lock() {
	struct BaseModifierParams {
		FixedArray<float, 16> world_to_model;
		int32_t operation;
		float smoothness;
		float sd_scale;
	};

	BaseModifierParams base_params;
	transform3d_to_mat4(get_transform().affine_inverse(), to_span(base_params.world_to_model));
	base_params.operation = get_operation();
	base_params.smoothness = get_smoothness();
	base_params.sd_scale = get_largest_coord(get_transform().get_basis().get_scale());
	PackedByteArray pba0;
	godot::copy_bytes_to(pba0, base_params);

	if (_shader_data == nullptr) {
		_shader_data = make_shared_instance<ComputeShaderParameters>();

		std::shared_ptr<ComputeShaderResource> res0 = ComputeShaderResourceFactory::create_storage_buffer(pba0);
		_shader_data->params.push_back(ComputeShaderParameter{ 4, res0 });

	} else {
		ZN_ASSERT(_shader_data->params.size() >= 1);
		ComputeShaderResource::update_storage_buffer(_shader_data->params[0].resource, pba0);
	}
}

#endif

} // namespace zylann::voxel
