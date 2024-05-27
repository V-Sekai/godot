/**************************************************************************/
/*  sliced_mesh.h                                                         */
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

#ifndef SLICED_MESH_H
#define SLICED_MESH_H

#include "core/io/resource.h"

#include "utils/intersector.h"

/**
 * A simple container for the results of a mesh slice.
 * upper_mesh contains the part of the mesh that was above
 * the plane normal and lower_mesh contains the part that was
 * below
 */
class SlicedMesh : public Resource {
	GDCLASS(SlicedMesh, Resource);

protected:
	static void _bind_methods();

public:
	Ref<Mesh> upper_mesh;
	Ref<Mesh> lower_mesh;

	void set_upper_mesh(const Ref<Mesh> &_upper_mesh) {
		upper_mesh = _upper_mesh;
	}
	Ref<Mesh> get_upper_mesh() const {
		return upper_mesh;
	};

	void set_lower_mesh(const Ref<Mesh> &_lower_mesh) {
		lower_mesh = _lower_mesh;
	}
	Ref<Mesh> get_lower_mesh() const {
		return lower_mesh;
	};

	SlicedMesh(Ref<Mesh> _upper_mesh, Ref<Mesh> _lower_mesh) {
		upper_mesh = _upper_mesh;
		lower_mesh = _lower_mesh;
	}

	/**
	 * Transforms a vector of split results and a vector of faces representing
	 * the cross section of a slice and creates an upper and lower mesh from them
	 */
	SlicedMesh(const Vector<Intersector::SplitResult> &surface_splits = Vector<Intersector::SplitResult>(), const Vector<SlicerFace> &cross_section_faces = Vector<SlicerFace>(), Ref<Material> cross_section_material = Ref<Material>());
	~SlicedMesh() {}
};

#endif // SLICED_MESH_H
