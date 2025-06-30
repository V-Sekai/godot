/**************************************************************************/
/*  raycast.h                                                             */
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

#include "../meshers/voxel_mesher.h"
#include "../util/godot/core/transform_3d.h"
#include "../util/godot/core/vector3.h"
#include "voxel_raycast_result.h"

namespace zylann::voxel {

class VoxelData;
class VoxelMesherBlocky;

Ref<VoxelRaycastResult> raycast_sdf(
		const VoxelData &voxel_data,
		const Vector3 ray_origin,
		const Vector3 ray_dir,
		const float max_distance,
		const uint8_t binary_search_iterations,
		const bool normal_enabled
);

Ref<VoxelRaycastResult> raycast_blocky(
		const VoxelData &voxel_data,
		const VoxelMesherBlocky &mesher,
		const Vector3 ray_origin,
		const Vector3 ray_dir,
		const float max_distance,
		const uint32_t p_collision_mask
);

Ref<VoxelRaycastResult> raycast_nonzero(
		const VoxelData &voxel_data,
		const Vector3 ray_origin,
		const Vector3 ray_dir,
		const float max_distance,
		const uint8_t p_channel
);

Ref<VoxelRaycastResult> raycast_generic(
		const VoxelData &voxel_data,
		const Ref<VoxelMesher> mesher,
		const Vector3 ray_origin,
		const Vector3 ray_dir,
		const float max_distance,
		const uint32_t p_collision_mask,
		const uint8_t binary_search_iterations,
		const bool normal_enabled
);

Ref<VoxelRaycastResult> raycast_generic_world(
		const VoxelData &voxel_data,
		const Ref<VoxelMesher> mesher,
		const Transform3D &to_world,
		const Vector3 ray_origin_world,
		const Vector3 ray_dir_world,
		const float max_distance_world,
		const uint32_t p_collision_mask,
		const uint8_t binary_search_iterations,
		const bool normal_enabled
);

} // namespace zylann::voxel
