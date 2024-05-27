/**************************************************************************/
/*  slicer.cpp                                                            */
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

#include "slicer.h"

#include "core/error/error_macros.h"
#include "utils/intersector.h"
#include "utils/slicer_face.h"
#include "utils/triangulator.h"

Ref<SlicedMesh> Slicer::slice_by_plane(const Ref<Mesh> mesh, const Plane plane, const Ref<Material> cross_section_material) {
	// TODO - This function is a little heavy. Maybe we should break it up
	ERR_FAIL_COND_V(mesh.is_null(), Ref<SlicedMesh>());

	Vector<Intersector::SplitResult> split_results;
	split_results.resize(mesh->get_surface_count());

	// The upper and lower meshes will share the same intersection points
	Vector<Vector3> intersection_points;

	for (int i = 0; i < mesh->get_surface_count(); i++) {
		Intersector::SplitResult results = split_results[i];

		results.material = mesh->surface_get_material(i);
		Vector<SlicerFace> faces = SlicerFace::faces_from_surface(mesh, i);

		for (int j = 0; j < faces.size(); j++) {
			Intersector::split_face_by_plane(plane, faces[j], results);
		}

		int ip_size = intersection_points.size();
		intersection_points.resize(ip_size + results.intersection_points.size());
		for (int j = 0; j < results.intersection_points.size(); j++) {
			intersection_points.write[ip_size + j] = results.intersection_points[j];
		}
		results.intersection_points.resize(0);

		split_results.write[i] = results;
	}

	// If no intersection has occurred then there's really nothing for us to do
	// but still, is this the expected behavior? Would it be better to return an
	// actual SliceMesh with either the upper_mesh or lower_mesh null?
	if (intersection_points.size() == 0) {
		return Ref<SlicedMesh>();
	}

	Vector<SlicerFace> cross_section_faces = Triangulator::monotone_chain(intersection_points, plane.normal);

	SlicedMesh *sliced_mesh = memnew(SlicedMesh(split_results, cross_section_faces, cross_section_material));
	return Ref<SlicedMesh>(sliced_mesh);
}

Ref<SlicedMesh> Slicer::slice_mesh(const Ref<Mesh> mesh, const Vector3 position, const Vector3 normal, const Ref<Material> cross_section_material) {
	Plane plane(normal, normal.dot(position));
	return slice_by_plane(mesh, plane, cross_section_material);
}

Ref<SlicedMesh> Slicer::slice(const Ref<Mesh> mesh, const Transform3D mesh_transform, const Vector3 position, const Vector3 normal, const Ref<Material> cross_section_material) {
	// We need to reorient the plane so that it will correctly slice the mesh whose vertices are based on the origin
	Vector3 origin = position - mesh_transform.origin;
	real_t dist = normal.dot(origin);
	Vector3 adjusted_normal = mesh_transform.basis.xform_inv(normal);

	return slice_by_plane(mesh, Plane(adjusted_normal, dist), cross_section_material);
}

void Slicer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("slice_by_plane", "mesh", "plane", "cross_section_material"), &Slicer::slice_by_plane, Variant::NIL);
	ClassDB::bind_method(D_METHOD("slice_mesh", "mesh", "position", "normal", "cross_section_material"), &Slicer::slice_mesh, Variant::NIL);
	ClassDB::bind_method(D_METHOD("slice", "mesh_instance", "mesh_transform", "position", "normal", "cross_section_material"), &Slicer::slice, Variant::NIL);
}
