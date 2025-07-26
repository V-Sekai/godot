/**************************************************************************/
/*  capsule_shape_3d.cpp                                                  */
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

#include "capsule_shape_3d.h"

#include "scene/resources/3d/primitive_meshes.h"
#include "servers/physics_3d/physics_server_3d.h"

Vector<Vector3> CapsuleShape3D::get_debug_mesh_lines() const {
	Vector<Vector3> points;

	// Use tapered logic when radii differ, otherwise classic capsule logic.
	real_t r_top = get_radius_top();
	real_t r_bot = get_radius_bottom();
	real_t h = get_mid_height();

	// Top hemisphere
	Vector3 top_center = Vector3(0, h * 0.5f, 0);
	for (int i = 0; i < 360; i++) {
		real_t ra = Math::deg_to_rad((real_t)i);
		real_t rb = Math::deg_to_rad((real_t)i + 1);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r_top;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r_top;

		points.push_back(Vector3(a.x, 0, a.y) + top_center);
		points.push_back(Vector3(b.x, 0, b.y) + top_center);

		if (i % 90 == 0) {
			points.push_back(Vector3(0, a.x, a.y) + top_center);
			points.push_back(Vector3(0, b.x, b.y) + top_center);
			points.push_back(Vector3(a.y, a.x, 0) + top_center);
			points.push_back(Vector3(b.y, b.x, 0) + top_center);
		}
	}

	// Bottom hemisphere
	Vector3 bottom_center = Vector3(0, -h * 0.5f, 0);
	for (int i = 0; i < 360; i++) {
		real_t ra = Math::deg_to_rad((real_t)i);
		real_t rb = Math::deg_to_rad((real_t)i + 1);
		Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * r_bot;
		Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * r_bot;

		points.push_back(Vector3(a.x, 0, a.y) + bottom_center);
		points.push_back(Vector3(b.x, 0, b.y) + bottom_center);

		if (i % 90 == 0) {
			points.push_back(Vector3(0, a.x, a.y) + bottom_center);
			points.push_back(Vector3(0, b.x, b.y) + bottom_center);
			points.push_back(Vector3(a.y, a.x, 0) + bottom_center);
			points.push_back(Vector3(b.y, b.x, 0) + bottom_center);
		}
	}

	// Connecting lines (cylinder/tapered part)
	points.push_back(Vector3(r_top, h * 0.5f, 0));
	points.push_back(Vector3(r_bot, -h * 0.5f, 0));
	points.push_back(Vector3(-r_top, h * 0.5f, 0));
	points.push_back(Vector3(-r_bot, -h * 0.5f, 0));
	points.push_back(Vector3(0, h * 0.5f, r_top));
	points.push_back(Vector3(0, -h * 0.5f, r_bot));
	points.push_back(Vector3(0, h * 0.5f, -r_top));
	points.push_back(Vector3(0, -h * 0.5f, -r_bot));

	return points;
}

Ref<ArrayMesh> CapsuleShape3D::get_debug_arraymesh_faces(const Color &p_modulate) const {
	Array capsule_array;
	capsule_array.resize(RS::ARRAY_MAX);

	// Choose tapered or legacy mesh generator depending on radii.
	if (Math::is_equal_approx(radius_top, radius_bottom)) {
		// legacy capsule
		CapsuleMesh::create_mesh_array(capsule_array, radius, height, 32, 8);
	} else {
		// tapered capsule
		CapsuleMesh::create_mesh_array(capsule_array, radius_top, radius_bottom, mid_height, 32, 8);
	}

	Vector<Color> colors;
	const PackedVector3Array &verts = capsule_array[RS::ARRAY_VERTEX];
	const int32_t verts_size = verts.size();
	for (int i = 0; i < verts_size; i++) {
		colors.append(p_modulate);
	}

	Ref<ArrayMesh> capsule_mesh = memnew(ArrayMesh);
	capsule_array[RS::ARRAY_COLOR] = colors;
	capsule_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, capsule_array);
	return capsule_mesh;
}

real_t CapsuleShape3D::get_enclosing_radius() const {
	if (Math::is_equal_approx(radius_top, radius_bottom)) {
		return height * 0.5f;
	} else {
		return MAX(radius_top, radius_bottom) + mid_height * 0.5f;
	}
}

void CapsuleShape3D::_update_shape() {
	Dictionary d;
	// Preserve legacy behavior:
	// - For classic capsule, send radius & full height.
	// - For tapered capsule, send average radius and mid_height (keeps previous tapered behavior).
	if (Math::is_equal_approx(radius_top, radius_bottom)) {
		d["radius"] = radius;
		d["height"] = height;
	} else {
		d["radius"] = (radius_top + radius_bottom) / 2.0;
		d["height"] = mid_height;
	}
	PhysicsServer3D::get_singleton()->shape_set_data(get_shape(), d);
	Shape3D::_update_shape();
}

void CapsuleShape3D::set_radius(float p_radius) {
	ERR_FAIL_COND_MSG(p_radius < 0.0f, "CapsuleShape3D radius cannot be negative.");
	// Legacy API: set both tapered radii and keep legacy radius field.
	radius = p_radius;
	radius_top = p_radius;
	radius_bottom = p_radius;
	if (height < radius * 2.0f) {
		height = radius * 2.0f;
	}
	// Update mid_height to reflect full height and new radii.
	mid_height = height - (radius_top + radius_bottom);
	_update_shape();
	emit_changed();
}

float CapsuleShape3D::get_radius() const {
	return radius;
}

void CapsuleShape3D::set_height(float p_height) {
	ERR_FAIL_COND_MSG(p_height < 0.0f, "CapsuleShape3D height cannot be negative.");
	// Legacy API: treat as classic capsule height.
	height = p_height;
	// Ensure radius is not larger than half height.
	if (radius > height * 0.5f) {
		radius = height * 0.5f;
		radius_top = radius_bottom = radius;
	}
	// Update mid_height from legacy full height.
	mid_height = height - (radius_top + radius_bottom);
	_update_shape();
	emit_changed();
}

float CapsuleShape3D::get_height() const {
	// Return full height for legacy API.
	return mid_height + radius_top + radius_bottom;
}

void CapsuleShape3D::set_mid_height(real_t p_mid_height) {
	ERR_FAIL_COND_MSG(p_mid_height < 0.0f, "CapsuleShape3D mid-height cannot be negative.");
	// Legacy API uses radius for hemispheres; update full height accordingly,
	// and keep tapered radii as they are.
	mid_height = p_mid_height;
	height = mid_height + (radius_top + radius_bottom);
	_update_shape();
	emit_changed();
}

real_t CapsuleShape3D::get_mid_height() const {
	return mid_height;
}

void CapsuleShape3D::set_radius_top(real_t p_radius_top) {
	ERR_FAIL_COND_MSG(p_radius_top < 0.0f, "CapsuleShape3D radius_top cannot be negative.");
	radius_top = p_radius_top;
	// Keep legacy fields in sync as averages where appropriate.
	radius = (radius_top + radius_bottom) * 0.5f;
	// Update legacy full height to match tapered full height.
	height = mid_height + radius_top + radius_bottom;
	_update_shape();
	emit_changed();
}

real_t CapsuleShape3D::get_radius_top() const {
	return radius_top;
}

void CapsuleShape3D::set_radius_bottom(real_t p_radius_bottom) {
	ERR_FAIL_COND_MSG(p_radius_bottom < 0.0f, "CapsuleShape3D radius_bottom cannot be negative.");
	radius_bottom = p_radius_bottom;
	radius = (radius_top + radius_bottom) * 0.5f;
	height = mid_height + radius_top + radius_bottom;
	_update_shape();
	emit_changed();
}

real_t CapsuleShape3D::get_radius_bottom() const {
	return radius_bottom;
}

void CapsuleShape3D::_bind_methods() {
	// Legacy bindings
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CapsuleShape3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CapsuleShape3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &CapsuleShape3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CapsuleShape3D::get_height);
	ClassDB::bind_method(D_METHOD("set_mid_height", "mid_height"), &CapsuleShape3D::set_mid_height);
	ClassDB::bind_method(D_METHOD("get_mid_height"), &CapsuleShape3D::get_mid_height);

	// Tapered bindings
	ClassDB::bind_method(D_METHOD("set_radius_top", "radius_top"), &CapsuleShape3D::set_radius_top);
	ClassDB::bind_method(D_METHOD("get_radius_top"), &CapsuleShape3D::get_radius_top);
	ClassDB::bind_method(D_METHOD("set_radius_bottom", "radius_bottom"), &CapsuleShape3D::set_radius_bottom);
	ClassDB::bind_method(D_METHOD("get_radius_bottom"), &CapsuleShape3D::get_radius_bottom);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_height", "get_height");

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_top", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_radius_top", "get_radius_top");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_bottom", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_radius_bottom", "get_radius_bottom");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mid_height", PROPERTY_HINT_RANGE, "0.001,100,0.001,or_greater,suffix:m"), "set_mid_height", "get_mid_height");

	// Link properties similar to previous implementations
	ADD_LINKED_PROPERTY("radius", "height");
	ADD_LINKED_PROPERTY("height", "radius");
	ADD_LINKED_PROPERTY("radius_top", "height");
	ADD_LINKED_PROPERTY("radius_bottom", "height");
	ADD_LINKED_PROPERTY("mid_height", "height");
	ADD_LINKED_PROPERTY("height", "radius_top");
	ADD_LINKED_PROPERTY("height", "radius_bottom");
	ADD_LINKED_PROPERTY("height", "mid_height");
}

CapsuleShape3D::CapsuleShape3D() :
		Shape3D(PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_CAPSULE)) {
	// Initialize legacy-consistent values.
	radius = 0.5f;
	height = 2.0f;
	radius_top = 0.5f;
	radius_bottom = 0.5f;
	mid_height = 1.0f;
	_update_shape();
}
