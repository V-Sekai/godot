/**************************************************************************/
/*  voxel_box_mover.h                                                     */
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

#include "../../util/godot/classes/ref_counted.h"
#include "../../util/godot/macros.h"

ZN_GODOT_FORWARD_DECLARE(class Node);

namespace zylann::voxel {

class VoxelTerrain;

// Helper to get simple AABB physics
class VoxelBoxMover : public RefCounted {
	GDCLASS(VoxelBoxMover, RefCounted)
public:
	Vector3 get_motion(Vector3 pos, Vector3 motion, AABB aabb, VoxelTerrain &terrain);

	void set_collision_mask(uint32_t mask);
	inline uint32_t get_collision_mask() const {
		return _collision_mask;
	}

	void set_step_climbing_enabled(bool enable);
	bool is_step_climbing_enabled() const;

	void set_max_step_height(float height);
	float get_max_step_height() const;

	bool has_stepped_up() const;

private:
#if defined(ZN_GODOT)
	Vector3 _b_get_motion(Vector3 p_pos, Vector3 p_motion, AABB p_aabb, Node *p_terrain_node);
#elif defined(ZN_GODOT_EXTENSION)
	// TODO GDX: it seems binding a method taking a `Node*` fails to compile. It is supposed to be working.
	Vector3 _b_get_motion(Vector3 p_pos, Vector3 p_motion, AABB p_aabb, Object *p_terrain_node_o);
#endif

	static void _bind_methods();

	// Config
	uint32_t _collision_mask = 0xffffffff; // Everything
	bool _step_climbing_enabled = false;
	real_t _max_step_height = 0.5;

	// States
	bool _has_stepped_up = false;
};

} // namespace zylann::voxel
