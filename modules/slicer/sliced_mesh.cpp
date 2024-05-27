/**************************************************************************/
/*  sliced_mesh.cpp                                                       */
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

#include "sliced_mesh.h"
#include "core/error/error_macros.h"
#include "scene/resources/material.h"
#include "utils/surface_filler.h"

/*
 * Creates a new surface composed of the uncut faces that were above the plane and the new faces generated
 * from the cut faces that fell on the plane
 */
void create_surface(const Vector<SlicerFace> &faces, const Ref<Material> material, Ref<ArrayMesh> mesh) {
	ERR_FAIL_COND(mesh.is_null());
	if (faces.size() == 0) {
		return;
	}

	SurfaceFiller filler(faces);

	for (int i = 0; i < faces.size() * 3; i++) {
		filler.fill(i, i);
	}

	filler.add_to_mesh(mesh, material);
}

/**
 * Create a new surface of the cross section faces. This should be called twice: once for the upper_mesh
 * and again for the lower_mesh
 */
void create_cross_section_surface(const Vector<SlicerFace> &faces, const Ref<Material> material, Ref<ArrayMesh> mesh, bool is_upper) {
	ERR_FAIL_COND(mesh.is_null());
	ERR_FAIL_COND(material.is_null());
	if (faces.size() == 0) {
		return;
	}

	SurfaceFiller filler(faces);

	for (int i = 0; i < faces.size(); i++) {
		// The cross section faces have the same normal as the plane that cut
		// them. That means that, for the upper half of the cut, we want to add
		// the vertices counterclockwise so that the normal is facing outwards
		if (is_upper) {
			filler.fill(i * 3, i * 3);
			filler.fill(i * 3 + 1, i * 3 + 2);
			filler.fill(i * 3 + 2, i * 3 + 1);
		} else {
			filler.fill(i * 3, i * 3);
			filler.fill(i * 3 + 1, i * 3 + 1);
			filler.fill(i * 3 + 2, i * 3 + 2);
		}
	}

	filler.add_to_mesh(mesh, material);
}

/**
 * Creates either an upper or lower half of the sliced mesh
 */
Ref<Mesh> create_mesh_half(
		const Vector<Intersector::SplitResult> &surface_splits,
		const Vector<SlicerFace> &cross_section_faces,
		Ref<Material> cross_section_material,
		bool is_upper) {
	Ref<ArrayMesh> mesh = memnew(ArrayMesh);

	for (int i = 0; i < surface_splits.size(); i++) {
		if (is_upper) {
			create_surface(surface_splits[i].upper_faces, surface_splits[i].material, mesh);
		} else {
			create_surface(surface_splits[i].lower_faces, surface_splits[i].material, mesh);
		}
	}

	if (cross_section_material.is_null() && mesh->get_surface_count() > 0) {
		// I believe Ezy-Slice has a way of specifying the existing material to use,
		// we may want to add that as a TODO
		cross_section_material = mesh->surface_get_material(0);
	} else if (cross_section_material.is_null()) {
		cross_section_material = Ref<Material>(memnew(StandardMaterial3D));
	}

	create_cross_section_surface(cross_section_faces, cross_section_material, mesh, is_upper);
	return mesh;
}

void SlicedMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_upper_mesh", "mesh"), &SlicedMesh::set_upper_mesh);
	ClassDB::bind_method(D_METHOD("get_upper_mesh"), &SlicedMesh::get_upper_mesh);
	ClassDB::bind_method(D_METHOD("set_lower_mesh", "mesh"), &SlicedMesh::set_lower_mesh);
	ClassDB::bind_method(D_METHOD("get_lower_mesh"), &SlicedMesh::get_lower_mesh);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "upper_mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_upper_mesh", "get_upper_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "lower_mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_lower_mesh", "get_lower_mesh");
}

void SlicedMesh::create_mesh(const Vector<Intersector::SplitResult> &surface_splits, const Vector<SlicerFace> &cross_section_faces, const Ref<Material> cross_section_material) {
	upper_mesh = create_mesh_half(surface_splits, cross_section_faces, cross_section_material, true);
	lower_mesh = create_mesh_half(surface_splits, cross_section_faces, cross_section_material, false);
}
