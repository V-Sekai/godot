/**************************************************************************/
/*  kusudama_mesh_3d.cpp                                                  */
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

#include "kusudama_mesh_3d.h"
#include "core/math/math_funcs.h"

KusudamaMesh3D::KusudamaMesh3D() {
}

KusudamaMesh3D::~KusudamaMesh3D() {
	clear_surfaces();
}

void KusudamaMesh3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cones", "cones"), &KusudamaMesh3D::set_cones);
	ClassDB::bind_method(D_METHOD("get_cones"), &KusudamaMesh3D::get_cones);
	ClassDB::bind_method(D_METHOD("add_cone", "cone_data"), &KusudamaMesh3D::add_cone);
	ClassDB::bind_method(D_METHOD("remove_cone", "index"), &KusudamaMesh3D::remove_cone);
	ClassDB::bind_method(D_METHOD("clear_cones"), &KusudamaMesh3D::clear_cones);
	ClassDB::bind_method(D_METHOD("get_cone_count"), &KusudamaMesh3D::get_cone_count);
	ClassDB::bind_method(D_METHOD("set_sphere_radius", "radius"), &KusudamaMesh3D::set_sphere_radius);
	ClassDB::bind_method(D_METHOD("get_sphere_radius"), &KusudamaMesh3D::get_sphere_radius);
	ClassDB::bind_method(D_METHOD("set_segments", "segments"), &KusudamaMesh3D::set_segments);
	ClassDB::bind_method(D_METHOD("get_segments"), &KusudamaMesh3D::get_segments);

	// Array inspector for cones
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "cones", PROPERTY_HINT_ARRAY_TYPE, "Dictionary"), "set_cones", "get_cones");

	// Mesh parameters
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "sphere_radius", PROPERTY_HINT_RANGE, "0.01,1,0.01"), "set_sphere_radius", "get_sphere_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "segments", PROPERTY_HINT_RANGE, "3,64,1"), "set_segments", "get_segments");
}

void KusudamaMesh3D::set_cones(TypedArray<Dictionary> p_cones) {
	open_cones.clear();
	for (int i = 0; i < p_cones.size(); i++) {
		Dictionary cone_dict = p_cones[i];
		ConeData cone;

		if (cone_dict.has("control_point")) {
			cone.control_point = cone_dict["control_point"];
		}
		if (cone_dict.has("radius")) {
			cone.radius = (real_t)cone_dict["radius"];
		}
		// Tangent circle parameters are not serialized (precalculated)

		open_cones.push_back(cone);
	}
	dirty = true;
	emit_changed();
}

TypedArray<Dictionary> KusudamaMesh3D::get_cones() const {
	TypedArray<Dictionary> result;
	for (const ConeData &cone : open_cones) {
		Dictionary cone_dict;
		cone_dict["control_point"] = cone.control_point;
		cone_dict["radius"] = cone.radius;
		// Tangent circle parameters are not serialized (precalculated)
		result.push_back(cone_dict);
	}
	return result;
}

void KusudamaMesh3D::add_cone(Dictionary p_cone_data) {
	ConeData cone;

	if (p_cone_data.has("control_point")) {
		cone.control_point = p_cone_data["control_point"];
	}
	if (p_cone_data.has("radius")) {
		cone.radius = (real_t)p_cone_data["radius"];
	}
	// Tangent circle parameters are not accepted (precalculated)

	open_cones.push_back(cone);
	dirty = true;
	emit_changed();
}

void KusudamaMesh3D::remove_cone(int p_index) {
	if (p_index >= 0 && p_index < (int)open_cones.size()) {
		open_cones.remove_at(p_index);
		dirty = true;
		emit_changed();
	}
}

void KusudamaMesh3D::clear_cones() {
	open_cones.clear();
	dirty = true;
	emit_changed();
}

int KusudamaMesh3D::get_cone_count() const {
	return open_cones.size();
}

void KusudamaMesh3D::set_sphere_radius(real_t p_radius) {
	sphere_radius = p_radius;
	dirty = true;
	emit_changed();
}

real_t KusudamaMesh3D::get_sphere_radius() const {
	return sphere_radius;
}

void KusudamaMesh3D::set_segments(int p_segments) {
	segments = MAX(3, p_segments);
	dirty = true;
	emit_changed();
}

int KusudamaMesh3D::get_segments() const {
	return segments;
}

void KusudamaMesh3D::_update_mesh() const {
	if (!dirty) {
		return;
	}

	clear_surfaces();

	if (open_cones.is_empty()) {
		dirty = false;
		return;
	}

	LocalVector<Vector3> vertices;
	real_t dp = Math::TAU / (real_t)segments;

	// Build wireframe geometry for all cones
	for (const ConeData &cone : open_cones) {
		Vector3 cone_center = cone.control_point.normalized() * sphere_radius;
		real_t cone_radius = cone.radius * sphere_radius;

		// Draw cone boundary circle
		for (int i = 0; i < segments; i++) {
			real_t a0 = (real_t)i * dp;
			real_t a1 = (real_t)((i + 1) % segments) * dp;

			Vector3 perp1 = cone_center.get_any_perpendicular().normalized();
			Vector3 perp2 = cone_center.cross(perp1).normalized();

			Vector3 p0 = cone_center + perp1 * cone_radius * Math::cos(a0) + perp2 * cone_radius * Math::sin(a0);
			Vector3 p1 = cone_center + perp1 * cone_radius * Math::cos(a1) + perp2 * cone_radius * Math::sin(a1);

			vertices.push_back(p0);
			vertices.push_back(p1);
		}

		// Draw lines from origin to cone boundary
		for (int i = 0; i < segments; i += 4) {
			real_t angle = (real_t)i * dp;
			Vector3 perp1 = cone_center.get_any_perpendicular().normalized();
			Vector3 perp2 = cone_center.cross(perp1).normalized();
			Vector3 boundary = cone_center + perp1 * cone_radius * Math::cos(angle) + perp2 * cone_radius * Math::sin(angle);

			vertices.push_back(Vector3());
			vertices.push_back(boundary);
		}

		// Tangent circle parameters are precalculated and not stored in ConeData
	}

	// Create the mesh surface
	Array arrays;
	arrays.resize(Mesh::ARRAY_MAX);
	arrays[Mesh::ARRAY_VERTEX] = vertices;

	add_surface_from_arrays(Mesh::PRIMITIVE_LINES, arrays);
	dirty = false;
}

int KusudamaMesh3D::get_surface_count() const {
	const_cast<KusudamaMesh3D *>(this)->_update_mesh();
	return Mesh::get_surface_count();
}

int KusudamaMesh3D::surface_get_array_len(int p_surface) const {
	const_cast<KusudamaMesh3D *>(this)->_update_mesh();
	return Mesh::surface_get_array_len(p_surface);
}

int KusudamaMesh3D::surface_get_array_index_len(int p_surface) const {
	const_cast<KusudamaMesh3D *>(this)->_update_mesh();
	return Mesh::surface_get_array_index_len(p_surface);
}

Array KusudamaMesh3D::surface_get_arrays(int p_surface) const {
	const_cast<KusudamaMesh3D *>(this)->_update_mesh();
	return Mesh::surface_get_arrays(p_surface);
}

Array KusudamaMesh3D::surface_get_blend_shape_arrays(int p_surface) const {
	const_cast<KusudamaMesh3D *>(this)->_update_mesh();
	return Mesh::surface_get_blend_shape_arrays(p_surface);
}

void KusudamaMesh3D::surface_set_material(int p_surface, const Ref<Material> &p_material) {
	Mesh::surface_set_material(p_surface, p_material);
}

Ref<Material> KusudamaMesh3D::surface_get_material(int p_surface) const {
	return Mesh::surface_get_material(p_surface);
}

int KusudamaMesh3D::surface_get_primitive_type(int p_surface) const {
	const_cast<KusudamaMesh3D *>(this)->_update_mesh();
	return Mesh::surface_get_primitive_type(p_surface);
}

void KusudamaMesh3D::surface_set_name(int p_surface, const String &p_name) {
	Mesh::surface_set_name(p_surface, p_name);
}

String KusudamaMesh3D::surface_get_name(int p_surface) const {
	return Mesh::surface_get_name(p_surface);
}

void KusudamaMesh3D::clear_surfaces() {
	Mesh::clear_surfaces();
	mesh_rid = RID();
}

AABB KusudamaMesh3D::get_aabb() const {
	const_cast<KusudamaMesh3D *>(this)->_update_mesh();
	return Mesh::get_aabb();
}

RID KusudamaMesh3D::get_rid() const {
	const_cast<KusudamaMesh3D *>(this)->_update_mesh();
	if (mesh_rid.is_null()) {
		mesh_rid = Mesh::get_rid();
	}
	return mesh_rid;
}

void KusudamaMesh3D::rid_changed() {
	Mesh::rid_changed();
	mesh_rid = RID();
}
