/**************************************************************************/
/*  voxel_raycast_result.cpp                                              */
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

#include "voxel_raycast_result.h"

Vector3 VoxelRaycastResult::_b_get_position() const {
	return position.to_vec3();
}

Vector3 VoxelRaycastResult::_b_get_previous_position() const {
	return previous_position.to_vec3();
}

float VoxelRaycastResult::_b_get_distance() const {
	return distance_along_ray;
}

void VoxelRaycastResult::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_position"),
			&VoxelRaycastResult::_b_get_position);
	ClassDB::bind_method(D_METHOD("get_previous_position"),
			&VoxelRaycastResult::_b_get_previous_position);
	ClassDB::bind_method(D_METHOD("get_distance"),
			&VoxelRaycastResult::_b_get_distance);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "position"), "", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "previous_position"), "",
			"get_previous_position");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "distance"), "", "get_distance");
}
