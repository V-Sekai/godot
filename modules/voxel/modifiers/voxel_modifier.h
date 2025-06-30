/**************************************************************************/
/*  voxel_modifier.h                                                      */
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

#include "../util/containers/fixed_array.h"
#include "../util/godot/core/rid.h"
#include "../util/godot/core/transform_3d.h"
#include "../util/math/vector3f.h"
#include "../util/thread/rw_lock.h"

#ifdef VOXEL_ENABLE_GPU
#include "../engine/gpu/compute_shader_parameters.h"
#endif

namespace zylann::voxel {

struct VoxelModifierContext {
	Span<float> sdf; // Signed distance values to modify
	Span<const Vector3f> positions; // Positions associated to each signed distance
};

struct BaseGPUResources;

class VoxelModifier {
public:
	enum Type { TYPE_SPHERE, TYPE_MESH };

	virtual ~VoxelModifier() {}

	virtual void apply(VoxelModifierContext ctx) const = 0;

	const Transform3D &get_transform() const {
		return _transform;
	}

	void set_transform(Transform3D t);

	const AABB &get_aabb() const {
		return _aabb;
	}

	virtual Type get_type() const = 0;
	virtual bool is_sdf() const = 0;

#ifdef VOXEL_ENABLE_GPU
	struct ShaderData {
		VoxelModifier::Type modifier_type;
		std::shared_ptr<ComputeShaderParameters> params;
	};

	virtual void get_shader_data(ShaderData &out_shader_data) = 0;

	static RID get_detail_shader(const BaseGPUResources &base_resources, const Type type);
	static RID get_block_shader(const BaseGPUResources &base_resources, const Type type);
#endif

protected:
	virtual void update_aabb() = 0;

	RWLock _rwlock;
	AABB _aabb;

#ifdef VOXEL_ENABLE_GPU
	std::shared_ptr<ComputeShaderParameters> _shader_data;
	bool _shader_data_need_update = false;
#endif

private:
	Transform3D _transform;
};

} // namespace zylann::voxel
