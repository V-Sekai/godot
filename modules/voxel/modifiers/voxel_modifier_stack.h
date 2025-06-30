/**************************************************************************/
/*  voxel_modifier_stack.h                                                */
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

#include "../util/containers/std_unordered_map.h"
#include "../util/containers/std_vector.h"
#include "../util/math/vector3f.h"
#include "../util/memory/memory.h"
#include "voxel_modifier.h"

namespace zylann::voxel {

class VoxelBuffer;

class VoxelModifierStack {
public:
	uint32_t allocate_id();

	VoxelModifierStack();
	VoxelModifierStack(VoxelModifierStack &&other);

	VoxelModifierStack &operator=(VoxelModifierStack &&other);

	template <typename T>
	T *add_modifier(uint32_t id) {
		ZN_ASSERT(!has_modifier(id));
		UniquePtr<VoxelModifier> &uptr = _modifiers[id];
		uptr = make_unique_instance<T>();
		VoxelModifier *ptr = uptr.get();
		RWLockWrite lock(_stack_lock);
		_stack.push_back(ptr);
		return static_cast<T *>(ptr);
	}

	void remove_modifier(uint32_t id);
	bool has_modifier(uint32_t id) const;
	VoxelModifier *get_modifier(uint32_t id) const;
	void apply(VoxelBuffer &voxels, AABB aabb) const;
	void apply(float &sdf, Vector3f position) const;

	void apply(
			Span<const float> x_buffer,
			Span<const float> y_buffer,
			Span<const float> z_buffer,
			Span<float> sdf_buffer,
			Vector3f min_pos,
			Vector3f max_pos
	) const;

#ifdef VOXEL_ENABLE_GPU
	void apply_for_gpu_rendering(StdVector<VoxelModifier::ShaderData> &out_data, const AABB aabb) const;
#endif

	void clear();

	template <typename F>
	void for_each_modifier(F f) const {
		RWLockRead rlock(_stack_lock);
		for (const VoxelModifier *modifier : _stack) {
			f(*modifier);
		}
	}

private:
	void move_from_noclear(VoxelModifierStack &other);

	StdUnorderedMap<uint32_t, UniquePtr<VoxelModifier>> _modifiers;
	uint32_t _next_id = 1;
	// TODO Later, replace this with a spatial acceleration structure based on AABBs, like BVH
	StdVector<VoxelModifier *> _stack;
	RWLock _stack_lock;
};

} // namespace zylann::voxel
