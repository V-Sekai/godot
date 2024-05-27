/**************************************************************************/
/*  slicer.h                                                              */
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

#ifndef SLICER_H
#define SLICER_H

#include "scene/3d/node_3d.h"
#include "scene/resources/mesh.h"
#include "sliced_mesh.h"

/**
 * Helper for cutting a convex mesh along a plane and returning
 * two new meshes representing both sides of the cut
 */
class Slicer : public Node3D {
	GDCLASS(Slicer, Node3D);

protected:
	static void _bind_methods();

public:
	/**
	 * Slice the passed in mesh along the passed in plane, setting the interrior cut surface to the passed in material
	 */
	Ref<SlicedMesh> slice_by_plane(const Ref<Mesh> mesh, const Plane plane, const Ref<Material> cross_section_material);

	/**
	 * Generates a plane based on the given position and normal and perform a cut along that plane
	 */
	Ref<SlicedMesh> slice_mesh(const Ref<Mesh> mesh, const Vector3 position, const Vector3 normal, const Ref<Material> cross_section_material);

	/**
	 * Generates a plane based on the given position and normal and offsets it by the given Transform3D before applying the slice
	 */
	Ref<SlicedMesh> slice(const Ref<Mesh> mesh, const Transform3D mesh_transform, const Vector3 position, const Vector3 normal, const Ref<Material> cross_section_material);
	~Slicer() {}
	Slicer() {}
};

#endif // SLICER_H
