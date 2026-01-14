/**************************************************************************/
/*  csg_sculpted_primitive_base.cpp                                       */
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

#include "csg_sculpted_primitive_base.h"

#include "core/math/geometry_3d.h"

// Helper function to generate profile points based on curve type
void generate_profile_points(CSGSculptedPrimitive3D::ProfileCurve p_curve, real_t p_begin, real_t p_end, real_t p_hollow, CSGSculptedPrimitive3D::HollowShape p_hollow_shape, int p_segments, Vector<Vector2> &r_profile, Vector<Vector2> &r_hollow_profile) {
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

			// Generate points, ensuring the profile closes if it's a full circle
			bool is_full_circle = (p_begin == 0.0 && p_end == 1.0) || angle_range >= Math::TAU - 0.001;
			int point_count = is_full_circle ? segments : segments + 1;

			for (int i = 0; i < point_count; i++) {
				real_t angle = begin_angle + (angle_range * i / segments);
				r_profile.push_back(Vector2(Math::cos(angle), Math::sin(angle)));
			}

			// Close the loop if it's a full circle
			if (is_full_circle && r_profile.size() > 0) {
				r_profile.push_back(r_profile[0]);
			}

			if (p_hollow > 0.0) {
				real_t hollow_radius = 1.0 - p_hollow;
				for (int i = 0; i < point_count; i++) {
					real_t angle = begin_angle + (angle_range * i / segments);
					r_hollow_profile.push_back(Vector2(Math::cos(angle) * hollow_radius, Math::sin(angle) * hollow_radius));
				}
				// Close the hollow loop if it's a full circle
				if (is_full_circle && r_hollow_profile.size() > 0) {
					r_hollow_profile.push_back(r_hollow_profile[0]);
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
Vector3 apply_path_transform(const Vector2 &p_profile_point, real_t p_path_pos, const CSGSculptedPrimitive3D::PathCurve p_path_curve, real_t p_twist, const Vector2 &p_taper, const Vector2 &p_shear, real_t p_radius_offset, real_t p_revolutions, real_t p_skew) {
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
