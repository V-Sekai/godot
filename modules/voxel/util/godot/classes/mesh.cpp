/**************************************************************************/
/*  mesh.cpp                                                              */
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

#include "mesh.h"

namespace zylann::godot {

bool is_surface_triangulated(const Array &surface) {
	PackedVector3Array positions = surface[Mesh::ARRAY_VERTEX];
	PackedInt32Array indices = surface[Mesh::ARRAY_INDEX];
	return positions.size() >= 3 && indices.size() >= 3;
}

bool is_mesh_empty(Span<const Array> surfaces) {
	if (surfaces.size() == 0) {
		return true;
	}
	for (const Array &surface : surfaces) {
		if (is_surface_triangulated(surface)) {
			return false;
		}
	}
	return true;
}

void scale_vec3_array(PackedVector3Array &array, float scale) {
	// Getting raw pointer because between GDExtension and modules, syntax and performance of operator[] differs.
	Vector3 *array_data = array.ptrw();
	const int count = array.size();
	for (int i = 0; i < count; ++i) {
		array_data[i] *= scale;
	}
}

void offset_vec3_array(PackedVector3Array &array, Vector3 offset) {
	// Getting raw pointer because between GDExtension and modules, syntax and performance of operator[] differs.
	Vector3 *array_data = array.ptrw();
	const int count = array.size();
	for (int i = 0; i < count; ++i) {
		array_data[i] += offset;
	}
}

void scale_surface(Array &surface, float scale) {
	PackedVector3Array positions = surface[Mesh::ARRAY_VERTEX];
	// Avoiding stupid CoW, assuming this array holds the only instance of this vector
	surface[Mesh::ARRAY_VERTEX] = PackedVector3Array();
	scale_vec3_array(positions, scale);
	surface[Mesh::ARRAY_VERTEX] = positions;
}

void offset_surface(Array &surface, Vector3 offset) {
	PackedVector3Array positions = surface[Mesh::ARRAY_VERTEX];
	// Avoiding stupid CoW, assuming this array holds the only instance of this vector
	surface[Mesh::ARRAY_VERTEX] = PackedVector3Array();
	offset_vec3_array(positions, offset);
	surface[Mesh::ARRAY_VERTEX] = positions;
}

} // namespace zylann::godot
