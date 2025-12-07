/**************************************************************************/
/*  csg_sculpted_primitive.cpp                                            */
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

#include "csg_sculpted_primitive.h"

#include "core/math/geometry_3d.h"
#include "scene/resources/image_texture.h"

// Helper function to generate profile points based on curve type
static void generate_profile_points(CSGSculptedPrimitive3D::ProfileCurve p_curve, real_t p_begin, real_t p_end, real_t p_hollow, CSGSculptedPrimitive3D::HollowShape p_hollow_shape, int p_segments, Vector<Vector2> &r_profile, Vector<Vector2> &r_hollow_profile) {
	r_profile.clear();
	r_hollow_profile.clear();

	real_t begin_angle = p_begin * Math::TAU;
	real_t end_angle = p_end * Math::TAU;
	real_t angle_range = end_angle - begin_angle;

	if (angle_range <= 0.0) {
		angle_range = Math::TAU;
	}

	switch (p_curve) {
		case CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE:
		case CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE_HALF: {
			int segments = p_segments;
			if (p_curve == CSGSculptedPrimitive3D::PROFILE_CURVE_CIRCLE_HALF) {
				segments = segments / 2;
			}

			for (int i = 0; i <= segments; i++) {
				real_t angle = begin_angle + (angle_range * i / segments);
				r_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)));
			}

			if (p_hollow > 0.0) {
				real_t hollow_radius = 1.0 - p_hollow;
				for (int i = 0; i <= segments; i++) {
					real_t angle = begin_angle + (angle_range * i / segments);
					r_hollow_profile.push_back(Vector2(Math::cos(angle) * hollow_radius, Math::sin(angle) * hollow_radius));
				}
			}
		} break;

		case CSGSculptedPrimitive3D::PROFILE_CURVE_SQUARE: {
			// Square profile
			real_t step = angle_range / 4.0;
			for (int i = 0; i < 4; i++) {
				real_t angle = begin_angle + step * i;
				r_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)).normalized());
			}
			r_profile.push_back(r_profile[0]); // Close the loop

			if (p_hollow > 0.0) {
				real_t hollow_radius = 1.0 - p_hollow;
				for (int i = 0; i < 4; i++) {
					real_t angle = begin_angle + step * i;
					r_hollow_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)).normalized() * hollow_radius);
				}
				r_hollow_profile.push_back(r_hollow_profile[0]);
			}
		} break;

		case CSGSculptedPrimitive3D::PROFILE_CURVE_ISOTRI:
		case CSGSculptedPrimitive3D::PROFILE_CURVE_EQUALTRI:
		case CSGSculptedPrimitive3D::PROFILE_CURVE_RIGHTTRI: {
			// Triangle profiles
			int sides = 3;
			real_t step = angle_range / sides;
			for (int i = 0; i < sides; i++) {
				real_t angle = begin_angle + step * i;
				r_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)).normalized());
			}
			r_profile.push_back(r_profile[0]);

			if (p_hollow > 0.0) {
				real_t hollow_radius = 1.0 - p_hollow;
				for (int i = 0; i < sides; i++) {
					real_t angle = begin_angle + step * i;
					r_hollow_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)).normalized() * hollow_radius);
				}
				r_hollow_profile.push_back(r_hollow_profile[0]);
			}
		} break;
	}
}

// Helper function to apply path transformations
static Vector3 apply_path_transform(const Vector2 &p_profile_point, real_t p_path_pos, const CSGSculptedPrimitive3D::PathCurve p_path_curve, real_t p_twist, const Vector2 &p_taper, const Vector2 &p_shear, real_t p_radius_offset, real_t p_revolutions, real_t p_skew) {
	Vector3 result;

	// Normalize path_pos to 0.0-1.0 range for calculations
	real_t normalized_path = CLAMP(p_path_pos, 0.0, 1.0);

	// Apply taper (taper.x is at begin, taper.y is at end)
	real_t taper_factor = 1.0 - (p_taper.x * (1.0 - normalized_path) + p_taper.y * normalized_path);
	Vector2 scaled_profile = p_profile_point * taper_factor;

	// Apply radius offset
	real_t radius = scaled_profile.length() + p_radius_offset;
	if (radius < 0.0) {
		radius = 0.0;
	}

	// Apply twist
	real_t twist_angle = p_twist * normalized_path * Math::TAU;
	Vector2 twisted_profile = Vector2(
			scaled_profile.x * Math::cos(twist_angle) - scaled_profile.y * Math::sin(twist_angle),
			scaled_profile.x * Math::sin(twist_angle) + scaled_profile.y * Math::cos(twist_angle));

	// Apply path curve
	switch (p_path_curve) {
		case CSGSculptedPrimitive3D::PATH_CURVE_LINE: {
			result = Vector3(twisted_profile.x, twisted_profile.y, normalized_path - 0.5);
		} break;

		case CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE:
		case CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE_33:
		case CSGSculptedPrimitive3D::PATH_CURVE_CIRCLE2: {
			real_t path_angle = normalized_path * p_revolutions * Math::TAU;
			real_t path_radius = radius;
			result = Vector3(
					path_radius * Math::cos(path_angle),
					twisted_profile.y,
					path_radius * Math::sin(path_angle));
		} break;
	}

	// Apply shear
	result.x += p_shear.x * normalized_path;
	result.y += p_shear.y * normalized_path;

	// Apply skew
	result.z += p_skew * normalized_path;

	return result;
}

void CSGSculptedPrimitive3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_profile_curve", "curve"), &CSGSculptedPrimitive3D::set_profile_curve);
	ClassDB::bind_method(D_METHOD("get_profile_curve"), &CSGSculptedPrimitive3D::get_profile_curve);

	ClassDB::bind_method(D_METHOD("set_profile_begin", "begin"), &CSGSculptedPrimitive3D::set_profile_begin);
	ClassDB::bind_method(D_METHOD("get_profile_begin"), &CSGSculptedPrimitive3D::get_profile_begin);

	ClassDB::bind_method(D_METHOD("set_profile_end", "end"), &CSGSculptedPrimitive3D::set_profile_end);
	ClassDB::bind_method(D_METHOD("get_profile_end"), &CSGSculptedPrimitive3D::get_profile_end);

	ClassDB::bind_method(D_METHOD("set_hollow", "hollow"), &CSGSculptedPrimitive3D::set_hollow);
	ClassDB::bind_method(D_METHOD("get_hollow"), &CSGSculptedPrimitive3D::get_hollow);

	ClassDB::bind_method(D_METHOD("set_hollow_shape", "shape"), &CSGSculptedPrimitive3D::set_hollow_shape);
	ClassDB::bind_method(D_METHOD("get_hollow_shape"), &CSGSculptedPrimitive3D::get_hollow_shape);

	ClassDB::bind_method(D_METHOD("set_path_curve", "curve"), &CSGSculptedPrimitive3D::set_path_curve);
	ClassDB::bind_method(D_METHOD("get_path_curve"), &CSGSculptedPrimitive3D::get_path_curve);

	ClassDB::bind_method(D_METHOD("set_path_begin", "begin"), &CSGSculptedPrimitive3D::set_path_begin);
	ClassDB::bind_method(D_METHOD("get_path_begin"), &CSGSculptedPrimitive3D::get_path_begin);

	ClassDB::bind_method(D_METHOD("set_path_end", "end"), &CSGSculptedPrimitive3D::set_path_end);
	ClassDB::bind_method(D_METHOD("get_path_end"), &CSGSculptedPrimitive3D::get_path_end);

	ClassDB::bind_method(D_METHOD("set_profile_scale", "scale"), &CSGSculptedPrimitive3D::set_profile_scale);
	ClassDB::bind_method(D_METHOD("get_profile_scale"), &CSGSculptedPrimitive3D::get_profile_scale);

	ClassDB::bind_method(D_METHOD("set_shear", "shear"), &CSGSculptedPrimitive3D::set_shear);
	ClassDB::bind_method(D_METHOD("get_shear"), &CSGSculptedPrimitive3D::get_shear);

	ClassDB::bind_method(D_METHOD("set_twist_begin", "twist"), &CSGSculptedPrimitive3D::set_twist_begin);
	ClassDB::bind_method(D_METHOD("get_twist_begin"), &CSGSculptedPrimitive3D::get_twist_begin);

	ClassDB::bind_method(D_METHOD("set_twist_end", "twist"), &CSGSculptedPrimitive3D::set_twist_end);
	ClassDB::bind_method(D_METHOD("get_twist_end"), &CSGSculptedPrimitive3D::get_twist_end);

	ClassDB::bind_method(D_METHOD("set_radius_offset", "offset"), &CSGSculptedPrimitive3D::set_radius_offset);
	ClassDB::bind_method(D_METHOD("get_radius_offset"), &CSGSculptedPrimitive3D::get_radius_offset);

	ClassDB::bind_method(D_METHOD("set_taper", "taper"), &CSGSculptedPrimitive3D::set_taper);
	ClassDB::bind_method(D_METHOD("get_taper"), &CSGSculptedPrimitive3D::get_taper);

	ClassDB::bind_method(D_METHOD("set_revolutions", "revolutions"), &CSGSculptedPrimitive3D::set_revolutions);
	ClassDB::bind_method(D_METHOD("get_revolutions"), &CSGSculptedPrimitive3D::get_revolutions);

	ClassDB::bind_method(D_METHOD("set_skew", "skew"), &CSGSculptedPrimitive3D::set_skew);
	ClassDB::bind_method(D_METHOD("get_skew"), &CSGSculptedPrimitive3D::get_skew);

	ClassDB::bind_method(D_METHOD("set_material", "material"), &CSGSculptedPrimitive3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &CSGSculptedPrimitive3D::get_material);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "profile_curve", PROPERTY_HINT_ENUM, "Circle,Square,Isosceles Triangle,Equal Triangle,Right Triangle,Circle Half"), "set_profile_curve", "get_profile_curve");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "profile_begin", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_profile_begin", "get_profile_begin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "profile_end", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_profile_end", "get_profile_end");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "hollow", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_hollow", "get_hollow");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "hollow_shape", PROPERTY_HINT_ENUM, "Same,Circle,Square,Triangle"), "set_hollow_shape", "get_hollow_shape");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_curve", PROPERTY_HINT_ENUM, "Line,Circle,Circle 33,Circle 2"), "set_path_curve", "get_path_curve");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_begin", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_path_begin", "get_path_begin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_end", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_path_end", "get_path_end");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "profile_scale"), "set_profile_scale", "get_profile_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "shear"), "set_shear", "get_shear");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "twist_begin", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_twist_begin", "get_twist_begin");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "twist_end", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_twist_end", "get_twist_end");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_offset", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_radius_offset", "get_radius_offset");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "taper", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_taper", "get_taper");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "revolutions", PROPERTY_HINT_RANGE, "0.1,4,0.1"), "set_revolutions", "get_revolutions");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "skew", PROPERTY_HINT_RANGE, "-1,1,0.01"), "set_skew", "get_skew");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "Material"), "set_material", "get_material");

	BIND_ENUM_CONSTANT(PROFILE_CURVE_CIRCLE);
	BIND_ENUM_CONSTANT(PROFILE_CURVE_SQUARE);
	BIND_ENUM_CONSTANT(PROFILE_CURVE_ISOTRI);
	BIND_ENUM_CONSTANT(PROFILE_CURVE_EQUALTRI);
	BIND_ENUM_CONSTANT(PROFILE_CURVE_RIGHTTRI);
	BIND_ENUM_CONSTANT(PROFILE_CURVE_CIRCLE_HALF);

	BIND_ENUM_CONSTANT(PATH_CURVE_LINE);
	BIND_ENUM_CONSTANT(PATH_CURVE_CIRCLE);
	BIND_ENUM_CONSTANT(PATH_CURVE_CIRCLE_33);
	BIND_ENUM_CONSTANT(PATH_CURVE_CIRCLE2);

	BIND_ENUM_CONSTANT(HOLLOW_SAME);
	BIND_ENUM_CONSTANT(HOLLOW_CIRCLE);
	BIND_ENUM_CONSTANT(HOLLOW_SQUARE);
	BIND_ENUM_CONSTANT(HOLLOW_TRIANGLE);
}

CSGSculptedPrimitive3D::CSGSculptedPrimitive3D() {
}

// Profile parameter setters/getters
void CSGSculptedPrimitive3D::set_profile_curve(int p_curve) {
	profile_curve = (ProfileCurve)p_curve;
	_make_dirty();
}

int CSGSculptedPrimitive3D::get_profile_curve() const {
	return (int)profile_curve;
}

void CSGSculptedPrimitive3D::set_profile_begin(real_t p_begin) {
	profile_begin = CLAMP(p_begin, 0.0, 1.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_profile_begin() const {
	return profile_begin;
}

void CSGSculptedPrimitive3D::set_profile_end(real_t p_end) {
	profile_end = CLAMP(p_end, 0.0, 1.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_profile_end() const {
	return profile_end;
}

void CSGSculptedPrimitive3D::set_hollow(real_t p_hollow) {
	hollow = CLAMP(p_hollow, 0.0, 1.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_hollow() const {
	return hollow;
}

void CSGSculptedPrimitive3D::set_hollow_shape(int p_shape) {
	hollow_shape = (HollowShape)p_shape;
	_make_dirty();
}

int CSGSculptedPrimitive3D::get_hollow_shape() const {
	return (int)hollow_shape;
}

// Path parameter setters/getters
void CSGSculptedPrimitive3D::set_path_curve(int p_curve) {
	path_curve = (PathCurve)p_curve;
	_make_dirty();
}

int CSGSculptedPrimitive3D::get_path_curve() const {
	return (int)path_curve;
}

void CSGSculptedPrimitive3D::set_path_begin(real_t p_begin) {
	path_begin = CLAMP(p_begin, 0.0, 1.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_path_begin() const {
	return path_begin;
}

void CSGSculptedPrimitive3D::set_path_end(real_t p_end) {
	path_end = CLAMP(p_end, 0.0, 1.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_path_end() const {
	return path_end;
}

void CSGSculptedPrimitive3D::set_profile_scale(const Vector2 &p_scale) {
	scale = p_scale;
	_make_dirty();
}

Vector2 CSGSculptedPrimitive3D::get_profile_scale() const {
	return scale;
}

void CSGSculptedPrimitive3D::set_shear(const Vector2 &p_shear) {
	shear = p_shear;
	_make_dirty();
}

Vector2 CSGSculptedPrimitive3D::get_shear() const {
	return shear;
}

void CSGSculptedPrimitive3D::set_twist_begin(real_t p_twist) {
	twist_begin = CLAMP(p_twist, -1.0, 1.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_twist_begin() const {
	return twist_begin;
}

void CSGSculptedPrimitive3D::set_twist_end(real_t p_twist) {
	twist_end = CLAMP(p_twist, -1.0, 1.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_twist_end() const {
	return twist_end;
}

void CSGSculptedPrimitive3D::set_radius_offset(real_t p_offset) {
	radius_offset = CLAMP(p_offset, -1.0, 1.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_radius_offset() const {
	return radius_offset;
}

void CSGSculptedPrimitive3D::set_taper(const Vector2 &p_taper) {
	taper = Vector2(CLAMP(p_taper.x, -1.0, 1.0), CLAMP(p_taper.y, -1.0, 1.0));
	_make_dirty();
}

Vector2 CSGSculptedPrimitive3D::get_taper() const {
	return taper;
}

void CSGSculptedPrimitive3D::set_revolutions(real_t p_revolutions) {
	revolutions = CLAMP(p_revolutions, 0.1, 4.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_revolutions() const {
	return revolutions;
}

void CSGSculptedPrimitive3D::set_skew(real_t p_skew) {
	skew = CLAMP(p_skew, -1.0, 1.0);
	_make_dirty();
}

real_t CSGSculptedPrimitive3D::get_skew() const {
	return skew;
}

void CSGSculptedPrimitive3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	_make_dirty();
}

Ref<Material> CSGSculptedPrimitive3D::get_material() const {
	return material;
}

// CSGSculptedBox3D implementation
void CSGSculptedBox3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &CSGSculptedBox3D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &CSGSculptedBox3D::get_size);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size"), "set_size", "get_size");
}

CSGSculptedBox3D::CSGSculptedBox3D() {
	profile_curve = PROFILE_CURVE_SQUARE;
}

void CSGSculptedBox3D::set_size(const Vector3 &p_size) {
	size = p_size;
	_make_dirty();
}

Vector3 CSGSculptedBox3D::get_size() const {
	return size;
}

CSGBrush *CSGSculptedBox3D::_build_brush() {
	// For now, use a simplified implementation that extends CSGBox3D behavior
	// Full sculpted prim mesh generation with all parameters
	CSGBrush *brush = memnew(CSGBrush);

	// Generate profile points
	Vector<Vector2> profile;
	Vector<Vector2> hollow_profile;
	int segments = 8; // Default segments for box
	generate_profile_points(profile_curve, profile_begin, profile_end, hollow, hollow_shape, segments, profile, hollow_profile);

	// Generate path points
	int path_segments = 8;
	real_t path_range = path_end - path_begin;
	if (path_range <= 0.0) {
		path_range = 1.0;
	}

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	// Generate vertices along the path
	for (int p = 0; p <= path_segments; p++) {
		real_t path_pos = path_begin + (path_range * p / path_segments);
		real_t normalized_path = (path_pos - path_begin) / path_range;
		real_t twist = Math::lerp(twist_begin, twist_end, normalized_path);

		for (int i = 0; i < profile.size(); i++) {
			Vector3 vertex = apply_path_transform(profile[i] * scale, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
			vertex *= size;
			vertices.push_back(vertex);
			uvs.push_back(Vector2((real_t)i / profile.size(), path_pos));
		}

		if (hollow > 0.0 && hollow_profile.size() > 0) {
			for (int i = 0; i < hollow_profile.size(); i++) {
				Vector3 vertex = apply_path_transform(hollow_profile[i] * scale, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
				vertex *= size;
				vertices.push_back(vertex);
				uvs.push_back(Vector2((real_t)i / hollow_profile.size(), path_pos));
			}
		}
	}

	// Generate faces
	int profile_count = profile.size();
	int hollow_count = hollow_profile.size();
	int total_profile = profile_count + (hollow > 0.0 ? hollow_count : 0);

	for (int p = 0; p < path_segments; p++) {
		int base1 = p * total_profile;
		int base2 = (p + 1) * total_profile;

		// Outer faces
		for (int i = 0; i < profile_count - 1; i++) {
			indices.push_back(base1 + i);
			indices.push_back(base2 + i);
			indices.push_back(base1 + i + 1);

			indices.push_back(base1 + i + 1);
			indices.push_back(base2 + i);
			indices.push_back(base2 + i + 1);
		}

		// Hollow faces if applicable
		if (hollow > 0.0 && hollow_count > 0) {
			int hollow_base1 = base1 + profile_count;
			int hollow_base2 = base2 + profile_count;

			for (int i = 0; i < hollow_count - 1; i++) {
				indices.push_back(hollow_base1 + i);
				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i);

				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i + 1);
				indices.push_back(hollow_base2 + i);
			}
		}
	}

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		bool flip = get_flip_faces();
		for (int i = 0; i < face_count; i++) {
			int idx = i * 3;
			if (flip) {
				facesw[idx] = vertices[indices[idx + 2]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx]];
				face_uvsw[idx] = uvs[indices[idx + 2]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx]];
			} else {
				facesw[idx] = vertices[indices[idx]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx + 2]];
				face_uvsw[idx] = uvs[indices[idx]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx + 2]];
			}
			smoothw[i] = false;
			materialsw[i] = material;
			invertw[i] = flip;
		}
	}

	brush->build_from_faces(faces, face_uvs, smooth, materials, invert);
	return brush;
}

// CSGSculptedCylinder3D implementation
void CSGSculptedCylinder3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CSGSculptedCylinder3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CSGSculptedCylinder3D::get_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGSculptedCylinder3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGSculptedCylinder3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
}

CSGSculptedCylinder3D::CSGSculptedCylinder3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_LINE;
}

void CSGSculptedCylinder3D::set_radius(const real_t p_radius) {
	radius = p_radius;
	_make_dirty();
}

real_t CSGSculptedCylinder3D::get_radius() const {
	return radius;
}

void CSGSculptedCylinder3D::set_height(const real_t p_height) {
	height = p_height;
	_make_dirty();
}

real_t CSGSculptedCylinder3D::get_height() const {
	return height;
}

CSGBrush *CSGSculptedCylinder3D::_build_brush() {
	// Similar to box but with circular profile and linear path
	CSGBrush *brush = memnew(CSGBrush);

	Vector<Vector2> profile;
	Vector<Vector2> hollow_profile;
	int segments = 16;
	generate_profile_points(profile_curve, profile_begin, profile_end, hollow, hollow_shape, segments, profile, hollow_profile);

	int path_segments = 8;
	real_t path_range = path_end - path_begin;
	if (path_range <= 0.0) {
		path_range = 1.0;
	}

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	for (int p = 0; p <= path_segments; p++) {
		real_t path_pos = path_begin + (path_range * p / path_segments);
		real_t normalized_path = (path_pos - path_begin) / path_range;
		real_t twist = Math::lerp(twist_begin, twist_end, normalized_path);
		real_t z_pos = (path_pos - 0.5) * height;

		for (int i = 0; i < profile.size(); i++) {
			Vector3 vertex = apply_path_transform(profile[i] * scale * radius, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
			vertex.z = z_pos;
			vertices.push_back(vertex);
			uvs.push_back(Vector2((real_t)i / profile.size(), path_pos));
		}

		if (hollow > 0.0 && hollow_profile.size() > 0) {
			for (int i = 0; i < hollow_profile.size(); i++) {
				Vector3 vertex = apply_path_transform(hollow_profile[i] * scale * radius, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
				vertex.z = z_pos;
				vertices.push_back(vertex);
				uvs.push_back(Vector2((real_t)i / hollow_profile.size(), path_pos));
			}
		}
	}

	// Generate faces (similar to box)
	int profile_count = profile.size();
	int hollow_count = hollow_profile.size();
	int total_profile = profile_count + (hollow > 0.0 ? hollow_count : 0);

	for (int p = 0; p < path_segments; p++) {
		int base1 = p * total_profile;
		int base2 = (p + 1) * total_profile;

		for (int i = 0; i < profile_count - 1; i++) {
			indices.push_back(base1 + i);
			indices.push_back(base2 + i);
			indices.push_back(base1 + i + 1);

			indices.push_back(base1 + i + 1);
			indices.push_back(base2 + i);
			indices.push_back(base2 + i + 1);
		}

		if (hollow > 0.0 && hollow_count > 0) {
			int hollow_base1 = base1 + profile_count;
			int hollow_base2 = base2 + profile_count;

			for (int i = 0; i < hollow_count - 1; i++) {
				indices.push_back(hollow_base1 + i);
				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i);

				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i + 1);
				indices.push_back(hollow_base2 + i);
			}
		}
	}

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		bool flip = get_flip_faces();
		for (int i = 0; i < face_count; i++) {
			int idx = i * 3;
			if (flip) {
				facesw[idx] = vertices[indices[idx + 2]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx]];
				face_uvsw[idx] = uvs[indices[idx + 2]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx]];
			} else {
				facesw[idx] = vertices[indices[idx]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx + 2]];
				face_uvsw[idx] = uvs[indices[idx]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx + 2]];
			}
			smoothw[i] = true;
			materialsw[i] = material;
			invertw[i] = flip;
		}
	}

	brush->build_from_faces(faces, face_uvs, smooth, materials, invert);
	return brush;
}

// CSGSculptedSphere3D implementation
void CSGSculptedSphere3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &CSGSculptedSphere3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &CSGSculptedSphere3D::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
}

CSGSculptedSphere3D::CSGSculptedSphere3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_CIRCLE;
	revolutions = 0.5; // Half sphere by default
}

void CSGSculptedSphere3D::set_radius(const real_t p_radius) {
	radius = p_radius;
	_make_dirty();
}

real_t CSGSculptedSphere3D::get_radius() const {
	return radius;
}

CSGBrush *CSGSculptedSphere3D::_build_brush() {
	// Sphere uses circular profile swept along circular path
	CSGBrush *brush = memnew(CSGBrush);

	Vector<Vector2> profile;
	Vector<Vector2> hollow_profile;
	int segments = 16;
	generate_profile_points(profile_curve, profile_begin, profile_end, hollow, hollow_shape, segments, profile, hollow_profile);

	int path_segments = 16;
	real_t path_range = path_end - path_begin;
	if (path_range <= 0.0) {
		path_range = 1.0;
	}

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	for (int p = 0; p <= path_segments; p++) {
		real_t path_pos = path_begin + (path_range * p / path_segments);
		real_t normalized_path = (path_pos - path_begin) / path_range;
		real_t twist = Math::lerp(twist_begin, twist_end, normalized_path);

		for (int i = 0; i < profile.size(); i++) {
			Vector3 vertex = apply_path_transform(profile[i] * scale * radius, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
			vertices.push_back(vertex);
			uvs.push_back(Vector2((real_t)i / profile.size(), path_pos));
		}

		if (hollow > 0.0 && hollow_profile.size() > 0) {
			for (int i = 0; i < hollow_profile.size(); i++) {
				Vector3 vertex = apply_path_transform(hollow_profile[i] * scale * radius, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
				vertices.push_back(vertex);
				uvs.push_back(Vector2((real_t)i / hollow_profile.size(), path_pos));
			}
		}
	}

	// Generate faces
	int profile_count = profile.size();
	int hollow_count = hollow_profile.size();
	int total_profile = profile_count + (hollow > 0.0 ? hollow_count : 0);

	for (int p = 0; p < path_segments; p++) {
		int base1 = p * total_profile;
		int base2 = (p + 1) * total_profile;

		for (int i = 0; i < profile_count - 1; i++) {
			indices.push_back(base1 + i);
			indices.push_back(base2 + i);
			indices.push_back(base1 + i + 1);

			indices.push_back(base1 + i + 1);
			indices.push_back(base2 + i);
			indices.push_back(base2 + i + 1);
		}

		if (hollow > 0.0 && hollow_count > 0) {
			int hollow_base1 = base1 + profile_count;
			int hollow_base2 = base2 + profile_count;

			for (int i = 0; i < hollow_count - 1; i++) {
				indices.push_back(hollow_base1 + i);
				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i);

				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i + 1);
				indices.push_back(hollow_base2 + i);
			}
		}
	}

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		bool flip = get_flip_faces();
		for (int i = 0; i < face_count; i++) {
			int idx = i * 3;
			if (flip) {
				facesw[idx] = vertices[indices[idx + 2]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx]];
				face_uvsw[idx] = uvs[indices[idx + 2]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx]];
			} else {
				facesw[idx] = vertices[indices[idx]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx + 2]];
				face_uvsw[idx] = uvs[indices[idx]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx + 2]];
			}
			smoothw[i] = true;
			materialsw[i] = material;
			invertw[i] = flip;
		}
	}

	brush->build_from_faces(faces, face_uvs, smooth, materials, invert);
	return brush;
}

// CSGSculptedTorus3D implementation
void CSGSculptedTorus3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "inner_radius"), &CSGSculptedTorus3D::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &CSGSculptedTorus3D::get_inner_radius);

	ClassDB::bind_method(D_METHOD("set_outer_radius", "outer_radius"), &CSGSculptedTorus3D::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &CSGSculptedTorus3D::get_outer_radius);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_radius"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_radius"), "set_outer_radius", "get_outer_radius");
}

CSGSculptedTorus3D::CSGSculptedTorus3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_CIRCLE;
}

void CSGSculptedTorus3D::set_inner_radius(const real_t p_inner_radius) {
	inner_radius = p_inner_radius;
	_make_dirty();
}

real_t CSGSculptedTorus3D::get_inner_radius() const {
	return inner_radius;
}

void CSGSculptedTorus3D::set_outer_radius(const real_t p_outer_radius) {
	outer_radius = p_outer_radius;
	_make_dirty();
}

real_t CSGSculptedTorus3D::get_outer_radius() const {
	return outer_radius;
}

CSGBrush *CSGSculptedTorus3D::_build_brush() {
	// Torus uses circular profile swept along circular path with inner/outer radius
	CSGBrush *brush = memnew(CSGBrush);

	Vector<Vector2> profile;
	Vector<Vector2> hollow_profile;
	int segments = 16;
	generate_profile_points(profile_curve, profile_begin, profile_end, hollow, hollow_shape, segments, profile, hollow_profile);

	int path_segments = 16;
	real_t path_range = path_end - path_begin;
	if (path_range <= 0.0) {
		path_range = 1.0;
	}

	real_t major_radius = (inner_radius + outer_radius) / 2.0;
	real_t minor_radius = (outer_radius - inner_radius) / 2.0;

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	for (int p = 0; p <= path_segments; p++) {
		real_t path_pos = path_begin + (path_range * p / path_segments);
		real_t normalized_path = (path_pos - path_begin) / path_range;
		real_t twist = Math::lerp(twist_begin, twist_end, normalized_path);
		real_t path_angle = path_pos * revolutions * Math::TAU;

		for (int i = 0; i < profile.size(); i++) {
			Vector2 prof = profile[i] * scale * minor_radius;
			// Apply path transformations (twist, taper, shear, radius_offset, skew)
			Vector3 transformed = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
			// Transform to torus coordinates using the transformed profile
			// Use transformed.x and transformed.y as the profile coordinates
			Vector3 torus_pos = Vector3(
					(major_radius + transformed.x) * Math::cos(path_angle),
					transformed.y,
					(major_radius + transformed.x) * Math::sin(path_angle));
			// Apply skew offset in Z
			torus_pos.z += transformed.z - (normalized_path - 0.5);
			vertices.push_back(torus_pos);
			uvs.push_back(Vector2((real_t)i / profile.size(), path_pos));
		}

		if (hollow > 0.0 && hollow_profile.size() > 0) {
			for (int i = 0; i < hollow_profile.size(); i++) {
				Vector2 prof = hollow_profile[i] * scale * minor_radius;
				// Apply path transformations
				Vector3 transformed = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
				Vector3 torus_pos = Vector3(
						(major_radius + transformed.x) * Math::cos(path_angle),
						transformed.y,
						(major_radius + transformed.x) * Math::sin(path_angle));
				torus_pos.z += transformed.z - (normalized_path - 0.5);
				vertices.push_back(torus_pos);
				uvs.push_back(Vector2((real_t)i / hollow_profile.size(), path_pos));
			}
		}
	}

	// Generate faces (similar to sphere)
	int profile_count = profile.size();
	int hollow_count = hollow_profile.size();
	int total_profile = profile_count + (hollow > 0.0 ? hollow_count : 0);

	for (int p = 0; p < path_segments; p++) {
		int base1 = p * total_profile;
		int base2 = (p + 1) * total_profile;

		for (int i = 0; i < profile_count - 1; i++) {
			indices.push_back(base1 + i);
			indices.push_back(base2 + i);
			indices.push_back(base1 + i + 1);

			indices.push_back(base1 + i + 1);
			indices.push_back(base2 + i);
			indices.push_back(base2 + i + 1);
		}

		if (hollow > 0.0 && hollow_count > 0) {
			int hollow_base1 = base1 + profile_count;
			int hollow_base2 = base2 + profile_count;

			for (int i = 0; i < hollow_count - 1; i++) {
				indices.push_back(hollow_base1 + i);
				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i);

				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i + 1);
				indices.push_back(hollow_base2 + i);
			}
		}
	}

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert_faces;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert_faces.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert_faces.ptrw();

		bool flip = get_flip_faces();
		for (int i = 0; i < face_count; i++) {
			int idx = i * 3;
			if (flip) {
				facesw[idx] = vertices[indices[idx + 2]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx]];
				face_uvsw[idx] = uvs[indices[idx + 2]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx]];
			} else {
				facesw[idx] = vertices[indices[idx]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx + 2]];
				face_uvsw[idx] = uvs[indices[idx]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx + 2]];
			}
			smoothw[i] = true;
			materialsw[i] = material;
			invertw[i] = flip;
		}
	}

	brush->build_from_faces(faces, face_uvs, smooth, materials, invert_faces);
	return brush;
}

// CSGSculptedPrism3D implementation
void CSGSculptedPrism3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &CSGSculptedPrism3D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &CSGSculptedPrism3D::get_size);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size"), "set_size", "get_size");
}

CSGSculptedPrism3D::CSGSculptedPrism3D() {
	profile_curve = PROFILE_CURVE_SQUARE;
	path_curve = PATH_CURVE_LINE;
}

void CSGSculptedPrism3D::set_size(const Vector3 &p_size) {
	size = p_size;
	_make_dirty();
}

Vector3 CSGSculptedPrism3D::get_size() const {
	return size;
}

CSGBrush *CSGSculptedPrism3D::_build_brush() {
	// Prism is similar to box but with a 5-sided profile
	// Use box implementation as base with profile modifications
	CSGBrush *brush = memnew(CSGBrush);

	Vector<Vector2> profile;
	Vector<Vector2> hollow_profile;
	int segments = 16;
	generate_profile_points(profile_curve, profile_begin, profile_end, hollow, hollow_shape, segments, profile, hollow_profile);

	int path_segments = 16;
	real_t path_range = path_end - path_begin;
	if (path_range <= 0.0) {
		path_range = 1.0;
	}

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	for (int p = 0; p <= path_segments; p++) {
		real_t path_pos = path_begin + (path_range * p / path_segments);
		real_t normalized_path = (path_pos - path_begin) / path_range;
		real_t twist = Math::lerp(twist_begin, twist_end, normalized_path);

		for (int i = 0; i < profile.size(); i++) {
			Vector2 prof = profile[i] * scale;
			prof.x *= size.x;
			prof.y *= size.y;
			Vector3 vertex = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
			vertex.z *= size.z;
			vertices.push_back(vertex);
			uvs.push_back(Vector2((real_t)i / profile.size(), path_pos));
		}

		if (hollow > 0.0 && hollow_profile.size() > 0) {
			for (int i = 0; i < hollow_profile.size(); i++) {
				Vector2 prof = hollow_profile[i] * scale;
				prof.x *= size.x;
				prof.y *= size.y;
				Vector3 vertex = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
				vertex.z *= size.z;
				vertices.push_back(vertex);
				uvs.push_back(Vector2((real_t)i / hollow_profile.size(), path_pos));
			}
		}
	}

	// Generate faces
	int profile_count = profile.size();
	int hollow_count = hollow_profile.size();
	int total_profile = profile_count + (hollow > 0.0 ? hollow_count : 0);

	for (int p = 0; p < path_segments; p++) {
		int base1 = p * total_profile;
		int base2 = (p + 1) * total_profile;

		for (int i = 0; i < profile_count - 1; i++) {
			indices.push_back(base1 + i);
			indices.push_back(base2 + i);
			indices.push_back(base1 + i + 1);

			indices.push_back(base1 + i + 1);
			indices.push_back(base2 + i);
			indices.push_back(base2 + i + 1);
		}

		if (hollow > 0.0 && hollow_count > 0) {
			int hollow_base1 = base1 + profile_count;
			int hollow_base2 = base2 + profile_count;

			for (int i = 0; i < hollow_count - 1; i++) {
				indices.push_back(hollow_base1 + i);
				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i);

				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i + 1);
				indices.push_back(hollow_base2 + i);
			}
		}
	}

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		bool flip = get_flip_faces();
		for (int i = 0; i < face_count; i++) {
			int idx = i * 3;
			if (flip) {
				facesw[idx] = vertices[indices[idx + 2]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx]];
				face_uvsw[idx] = uvs[indices[idx + 2]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx]];
			} else {
				facesw[idx] = vertices[indices[idx]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx + 2]];
				face_uvsw[idx] = uvs[indices[idx]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx + 2]];
			}
			smoothw[i] = false;
			materialsw[i] = material;
			invertw[i] = flip;
		}
	}

	brush->build_from_faces(faces, face_uvs, smooth, materials, invert);
	return brush;
}

// CSGSculptedTube3D implementation
void CSGSculptedTube3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "inner_radius"), &CSGSculptedTube3D::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &CSGSculptedTube3D::get_inner_radius);

	ClassDB::bind_method(D_METHOD("set_outer_radius", "outer_radius"), &CSGSculptedTube3D::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &CSGSculptedTube3D::get_outer_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGSculptedTube3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGSculptedTube3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_radius"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_radius"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
}

CSGSculptedTube3D::CSGSculptedTube3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_LINE;
}

void CSGSculptedTube3D::set_inner_radius(const real_t p_inner_radius) {
	inner_radius = p_inner_radius;
	_make_dirty();
}

real_t CSGSculptedTube3D::get_inner_radius() const {
	return inner_radius;
}

void CSGSculptedTube3D::set_outer_radius(const real_t p_outer_radius) {
	outer_radius = p_outer_radius;
	_make_dirty();
}

real_t CSGSculptedTube3D::get_outer_radius() const {
	return outer_radius;
}

void CSGSculptedTube3D::set_height(const real_t p_height) {
	height = p_height;
	_make_dirty();
}

real_t CSGSculptedTube3D::get_height() const {
	return height;
}

CSGBrush *CSGSculptedTube3D::_build_brush() {
	// Tube is a hollow cylinder - similar to cylinder but with explicit inner/outer radius
	CSGBrush *brush = memnew(CSGBrush);

	Vector<Vector2> profile;
	Vector<Vector2> hollow_profile;
	int segments = 16;
	// For tube, we always have a hollow, so set hollow to represent the inner radius
	real_t effective_hollow = inner_radius / outer_radius;
	generate_profile_points(profile_curve, profile_begin, profile_end, effective_hollow, hollow_shape, segments, profile, hollow_profile);

	int path_segments = 16;
	real_t path_range = path_end - path_begin;
	if (path_range <= 0.0) {
		path_range = 1.0;
	}

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	for (int p = 0; p <= path_segments; p++) {
		real_t path_pos = path_begin + (path_range * p / path_segments);
		real_t normalized_path = (path_pos - path_begin) / path_range;
		real_t twist = Math::lerp(twist_begin, twist_end, normalized_path);

		for (int i = 0; i < profile.size(); i++) {
			Vector2 prof = profile[i] * scale * outer_radius;
			Vector3 vertex = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
			vertex.y *= height;
			vertices.push_back(vertex);
			uvs.push_back(Vector2((real_t)i / profile.size(), path_pos));
		}

		if (hollow_profile.size() > 0) {
			for (int i = 0; i < hollow_profile.size(); i++) {
				Vector2 prof = hollow_profile[i] * scale * outer_radius;
				Vector3 vertex = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
				vertex.y *= height;
				vertices.push_back(vertex);
				uvs.push_back(Vector2((real_t)i / hollow_profile.size(), path_pos));
			}
		}
	}

	// Generate faces
	int profile_count = profile.size();
	int hollow_count = hollow_profile.size();
	int total_profile = profile_count + hollow_count;

	for (int p = 0; p < path_segments; p++) {
		int base1 = p * total_profile;
		int base2 = (p + 1) * total_profile;

		for (int i = 0; i < profile_count - 1; i++) {
			indices.push_back(base1 + i);
			indices.push_back(base2 + i);
			indices.push_back(base1 + i + 1);

			indices.push_back(base1 + i + 1);
			indices.push_back(base2 + i);
			indices.push_back(base2 + i + 1);
		}

		if (hollow_count > 0) {
			int hollow_base1 = base1 + profile_count;
			int hollow_base2 = base2 + profile_count;

			for (int i = 0; i < hollow_count - 1; i++) {
				indices.push_back(hollow_base1 + i);
				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i);

				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i + 1);
				indices.push_back(hollow_base2 + i);
			}
		}
	}

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		bool flip = get_flip_faces();
		for (int i = 0; i < face_count; i++) {
			int idx = i * 3;
			if (flip) {
				facesw[idx] = vertices[indices[idx + 2]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx]];
				face_uvsw[idx] = uvs[indices[idx + 2]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx]];
			} else {
				facesw[idx] = vertices[indices[idx]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx + 2]];
				face_uvsw[idx] = uvs[indices[idx]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx + 2]];
			}
			smoothw[i] = true;
			materialsw[i] = material;
			invertw[i] = flip;
		}
	}

	brush->build_from_faces(faces, face_uvs, smooth, materials, invert);
	return brush;
}

// CSGSculptedRing3D implementation
void CSGSculptedRing3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_inner_radius", "inner_radius"), &CSGSculptedRing3D::set_inner_radius);
	ClassDB::bind_method(D_METHOD("get_inner_radius"), &CSGSculptedRing3D::get_inner_radius);

	ClassDB::bind_method(D_METHOD("set_outer_radius", "outer_radius"), &CSGSculptedRing3D::set_outer_radius);
	ClassDB::bind_method(D_METHOD("get_outer_radius"), &CSGSculptedRing3D::get_outer_radius);

	ClassDB::bind_method(D_METHOD("set_height", "height"), &CSGSculptedRing3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &CSGSculptedRing3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inner_radius"), "set_inner_radius", "get_inner_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "outer_radius"), "set_outer_radius", "get_outer_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
}

CSGSculptedRing3D::CSGSculptedRing3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_CIRCLE;
}

void CSGSculptedRing3D::set_inner_radius(const real_t p_inner_radius) {
	inner_radius = p_inner_radius;
	_make_dirty();
}

real_t CSGSculptedRing3D::get_inner_radius() const {
	return inner_radius;
}

void CSGSculptedRing3D::set_outer_radius(const real_t p_outer_radius) {
	outer_radius = p_outer_radius;
	_make_dirty();
}

real_t CSGSculptedRing3D::get_outer_radius() const {
	return outer_radius;
}

void CSGSculptedRing3D::set_height(const real_t p_height) {
	height = p_height;
	_make_dirty();
}

real_t CSGSculptedRing3D::get_height() const {
	return height;
}

CSGBrush *CSGSculptedRing3D::_build_brush() {
	// Ring is similar to torus but with different proportions (thinner, more ring-like)
	CSGBrush *brush = memnew(CSGBrush);

	Vector<Vector2> profile;
	Vector<Vector2> hollow_profile;
	int segments = 16;
	generate_profile_points(profile_curve, profile_begin, profile_end, hollow, hollow_shape, segments, profile, hollow_profile);

	int path_segments = 16;
	real_t path_range = path_end - path_begin;
	if (path_range <= 0.0) {
		path_range = 1.0;
	}

	real_t major_radius = (inner_radius + outer_radius) / 2.0;
	real_t minor_radius = height / 2.0; // Ring uses height as the minor radius

	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	for (int p = 0; p <= path_segments; p++) {
		real_t path_pos = path_begin + (path_range * p / path_segments);
		real_t normalized_path = (path_pos - path_begin) / path_range;
		real_t twist = Math::lerp(twist_begin, twist_end, normalized_path);
		real_t path_angle = path_pos * revolutions * Math::TAU;

		for (int i = 0; i < profile.size(); i++) {
			Vector2 prof = profile[i] * scale * minor_radius;
			// Apply path transformations (twist, taper, shear, radius_offset, skew)
			Vector3 transformed = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
			// Transform to ring coordinates using the transformed profile
			real_t radius_at_angle = major_radius + transformed.x;
			Vector3 ring_pos = Vector3(
					radius_at_angle * Math::cos(path_angle),
					transformed.y,
					radius_at_angle * Math::sin(path_angle));
			// Apply skew offset in Z
			ring_pos.z += transformed.z - (normalized_path - 0.5);
			vertices.push_back(ring_pos);
			uvs.push_back(Vector2((real_t)i / profile.size(), path_pos));
		}

		if (hollow > 0.0 && hollow_profile.size() > 0) {
			for (int i = 0; i < hollow_profile.size(); i++) {
				Vector2 prof = hollow_profile[i] * scale * minor_radius;
				// Apply path transformations
				Vector3 transformed = apply_path_transform(prof, path_pos, path_curve, twist, taper, shear, radius_offset, revolutions, skew);
				real_t radius_at_angle = major_radius + transformed.x;
				Vector3 ring_pos = Vector3(
						radius_at_angle * Math::cos(path_angle),
						transformed.y,
						radius_at_angle * Math::sin(path_angle));
				ring_pos.z += transformed.z - (normalized_path - 0.5);
				vertices.push_back(ring_pos);
				uvs.push_back(Vector2((real_t)i / hollow_profile.size(), path_pos));
			}
		}
	}

	// Generate faces
	int profile_count = profile.size();
	int hollow_count = hollow_profile.size();
	int total_profile = profile_count + (hollow > 0.0 ? hollow_count : 0);

	for (int p = 0; p < path_segments; p++) {
		int base1 = p * total_profile;
		int base2 = (p + 1) * total_profile;

		for (int i = 0; i < profile_count - 1; i++) {
			indices.push_back(base1 + i);
			indices.push_back(base2 + i);
			indices.push_back(base1 + i + 1);

			indices.push_back(base1 + i + 1);
			indices.push_back(base2 + i);
			indices.push_back(base2 + i + 1);
		}

		if (hollow > 0.0 && hollow_count > 0) {
			int hollow_base1 = base1 + profile_count;
			int hollow_base2 = base2 + profile_count;

			for (int i = 0; i < hollow_count - 1; i++) {
				indices.push_back(hollow_base1 + i);
				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i);

				indices.push_back(hollow_base1 + i + 1);
				indices.push_back(hollow_base2 + i + 1);
				indices.push_back(hollow_base2 + i);
			}
		}
	}

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert.ptrw();

		bool flip = get_flip_faces();
		for (int i = 0; i < face_count; i++) {
			int idx = i * 3;
			if (flip) {
				facesw[idx] = vertices[indices[idx + 2]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx]];
				face_uvsw[idx] = uvs[indices[idx + 2]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx]];
			} else {
				facesw[idx] = vertices[indices[idx]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx + 2]];
				face_uvsw[idx] = uvs[indices[idx]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx + 2]];
			}
			smoothw[i] = true;
			materialsw[i] = material;
			invertw[i] = flip;
		}
	}

	brush->build_from_faces(faces, face_uvs, smooth, materials, invert);
	return brush;
}

// CSGSculptedTexture3D implementation
void CSGSculptedTexture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_sculpt_texture", "texture"), &CSGSculptedTexture3D::set_sculpt_texture);
	ClassDB::bind_method(D_METHOD("get_sculpt_texture"), &CSGSculptedTexture3D::get_sculpt_texture);

	ClassDB::bind_method(D_METHOD("set_mirror", "mirror"), &CSGSculptedTexture3D::set_mirror);
	ClassDB::bind_method(D_METHOD("get_mirror"), &CSGSculptedTexture3D::get_mirror);

	ClassDB::bind_method(D_METHOD("set_invert", "invert"), &CSGSculptedTexture3D::set_invert);
	ClassDB::bind_method(D_METHOD("get_invert"), &CSGSculptedTexture3D::get_invert);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "sculpt_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_sculpt_texture", "get_sculpt_texture");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "mirror"), "set_mirror", "get_mirror");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "invert"), "set_invert", "get_invert");
}

void CSGSculptedTexture3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (sculpt_texture.is_valid()) {
				sculpt_texture->connect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (sculpt_texture.is_valid()) {
				sculpt_texture->disconnect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
			}
		} break;
	}
}

CSGSculptedTexture3D::CSGSculptedTexture3D() {
	profile_curve = PROFILE_CURVE_CIRCLE;
	path_curve = PATH_CURVE_LINE;
}

void CSGSculptedTexture3D::_texture_changed() {
	_make_dirty();
}

void CSGSculptedTexture3D::set_sculpt_texture(const Ref<Texture2D> &p_texture) {
	if (sculpt_texture.is_valid()) {
		if (is_inside_tree()) {
			sculpt_texture->disconnect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
		}
	}
	sculpt_texture = p_texture;
	if (sculpt_texture.is_valid()) {
		if (is_inside_tree()) {
			sculpt_texture->connect("changed", callable_mp(this, &CSGSculptedTexture3D::_texture_changed));
		}
	}
	_make_dirty();
}

Ref<Texture2D> CSGSculptedTexture3D::get_sculpt_texture() const {
	return sculpt_texture;
}

void CSGSculptedTexture3D::set_mirror(bool p_mirror) {
	mirror = p_mirror;
	_make_dirty();
}

bool CSGSculptedTexture3D::get_mirror() const {
	return mirror;
}

void CSGSculptedTexture3D::set_invert(bool p_invert) {
	invert = p_invert;
	_make_dirty();
}

bool CSGSculptedTexture3D::get_invert() const {
	return invert;
}

CSGBrush *CSGSculptedTexture3D::_build_brush() {
	CSGBrush *brush = memnew(CSGBrush);

	if (!sculpt_texture.is_valid()) {
		// Return empty brush if no texture
		return brush;
	}

	Ref<Image> image = sculpt_texture->get_image();
	if (!image.is_valid() || image->is_empty()) {
		return brush;
	}

	int width = image->get_width();
	int height = image->get_height();

	if (width < 2 || height < 2) {
		return brush;
	}

	// Convert texture RGB values to 3D coordinates
	// Texture format: R=X, G=Y, B=Z, each mapped from 0-255 to -1 to 1
	Vector<Vector3> vertices;
	Vector<Vector2> uvs;
	Vector<int> indices;

	// Generate vertices from texture
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			Color pixel = image->get_pixel(x, y);

			// Convert RGB (0-1) to XYZ (-1 to 1)
			// Use base scale parameter for scaling
			real_t x_coord = (pixel.r * 2.0 - 1.0) * scale.x;
			real_t y_coord = (pixel.g * 2.0 - 1.0) * scale.y;
			real_t z_coord = (pixel.b * 2.0 - 1.0);

			// Apply invert flag
			if (invert) {
				z_coord = -z_coord;
			}

			// Apply mirror flag
			if (mirror) {
				x_coord = -x_coord;
			}

			vertices.push_back(Vector3(x_coord, y_coord, z_coord));
			uvs.push_back(Vector2((real_t)x / (width - 1), (real_t)y / (height - 1)));
		}
	}

	// Generate triangle indices (grid triangulation)
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int i0 = y * width + x;
			int i1 = y * width + (x + 1);
			int i2 = (y + 1) * width + x;
			int i3 = (y + 1) * width + (x + 1);

			// First triangle
			indices.push_back(i0);
			indices.push_back(i2);
			indices.push_back(i1);

			// Second triangle
			indices.push_back(i1);
			indices.push_back(i2);
			indices.push_back(i3);
		}
	}

	// Convert to CSGBrush format
	Vector<Vector3> faces;
	Vector<Vector2> face_uvs;
	Vector<bool> smooth;
	Vector<Ref<Material>> materials;
	Vector<bool> invert_faces;

	int face_count = indices.size() / 3;
	faces.resize(face_count * 3);
	face_uvs.resize(face_count * 3);
	smooth.resize(face_count);
	materials.resize(face_count);
	invert_faces.resize(face_count);

	{
		Vector3 *facesw = faces.ptrw();
		Vector2 *face_uvsw = face_uvs.ptrw();
		bool *smoothw = smooth.ptrw();
		Ref<Material> *materialsw = materials.ptrw();
		bool *invertw = invert_faces.ptrw();

		bool flip = get_flip_faces();
		for (int i = 0; i < face_count; i++) {
			int idx = i * 3;
			if (flip) {
				facesw[idx] = vertices[indices[idx + 2]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx]];
				face_uvsw[idx] = uvs[indices[idx + 2]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx]];
			} else {
				facesw[idx] = vertices[indices[idx]];
				facesw[idx + 1] = vertices[indices[idx + 1]];
				facesw[idx + 2] = vertices[indices[idx + 2]];
				face_uvsw[idx] = uvs[indices[idx]];
				face_uvsw[idx + 1] = uvs[indices[idx + 1]];
				face_uvsw[idx + 2] = uvs[indices[idx + 2]];
			}
			smoothw[i] = true;
			materialsw[i] = material;
			invertw[i] = flip;
		}
	}

	brush->build_from_faces(faces, face_uvs, smooth, materials, invert_faces);
	return brush;
}
