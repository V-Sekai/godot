/**************************************************************************/
/*  joint_limitation_kusudama_3d.cpp                                      */
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

#include "joint_limitation_kusudama_3d.h"

#include "core/math/math_funcs.h"

void JointLimitationKusudama3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_open_cones", "cones"), &JointLimitationKusudama3D::set_open_cones);
	ClassDB::bind_method(D_METHOD("get_open_cones"), &JointLimitationKusudama3D::get_open_cones);

	ClassDB::bind_method(D_METHOD("set_axial_limits", "min_angle", "range_angle"), &JointLimitationKusudama3D::set_axial_limits);
	ClassDB::bind_method(D_METHOD("set_min_axial_angle", "angle"), &JointLimitationKusudama3D::set_min_axial_angle);
	ClassDB::bind_method(D_METHOD("get_min_axial_angle"), &JointLimitationKusudama3D::get_min_axial_angle);
	ClassDB::bind_method(D_METHOD("set_range_angle", "angle"), &JointLimitationKusudama3D::set_range_angle);
	ClassDB::bind_method(D_METHOD("get_range_angle"), &JointLimitationKusudama3D::get_range_angle);

	ClassDB::bind_method(D_METHOD("set_axially_constrained", "constrained"), &JointLimitationKusudama3D::set_axially_constrained);
	ClassDB::bind_method(D_METHOD("is_axially_constrained"), &JointLimitationKusudama3D::is_axially_constrained);

	ClassDB::bind_method(D_METHOD("set_orientationally_constrained", "constrained"), &JointLimitationKusudama3D::set_orientationally_constrained);
	ClassDB::bind_method(D_METHOD("is_orientationally_constrained"), &JointLimitationKusudama3D::is_orientationally_constrained);

	ClassDB::bind_method(D_METHOD("set_cone_count", "count"), &JointLimitationKusudama3D::set_cone_count);
	ClassDB::bind_method(D_METHOD("get_cone_count"), &JointLimitationKusudama3D::get_cone_count);
	ClassDB::bind_method(D_METHOD("set_cone_center", "index", "center"), &JointLimitationKusudama3D::set_cone_center);
	ClassDB::bind_method(D_METHOD("get_cone_center", "index"), &JointLimitationKusudama3D::get_cone_center);
	ClassDB::bind_method(D_METHOD("set_cone_azimuth", "index", "azimuth"), &JointLimitationKusudama3D::set_cone_azimuth);
	ClassDB::bind_method(D_METHOD("get_cone_azimuth", "index"), &JointLimitationKusudama3D::get_cone_azimuth);
	ClassDB::bind_method(D_METHOD("set_cone_elevation", "index", "elevation"), &JointLimitationKusudama3D::set_cone_elevation);
	ClassDB::bind_method(D_METHOD("get_cone_elevation", "index"), &JointLimitationKusudama3D::get_cone_elevation);
	ClassDB::bind_method(D_METHOD("set_cone_radius", "index", "radius"), &JointLimitationKusudama3D::set_cone_radius);
	ClassDB::bind_method(D_METHOD("get_cone_radius", "index"), &JointLimitationKusudama3D::get_cone_radius);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "open_cones", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_open_cones", "get_open_cones");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_axial_angle", PROPERTY_HINT_RANGE, "-360,360,0.1,radians_as_degrees"), "set_min_axial_angle", "get_min_axial_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "range_angle", PROPERTY_HINT_RANGE, "0,360,0.1,radians_as_degrees"), "set_range_angle", "get_range_angle");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "axially_constrained"), "set_axially_constrained", "is_axially_constrained");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "orientationally_constrained"), "set_orientationally_constrained", "is_orientationally_constrained");
}

// Helper function to find closest point on a single cone boundary
static Vector3 closest_to_cone_boundary(const Vector3 &p_input, const Vector3 &p_control_point, real_t p_radius) {
	Vector3 normalized_input = p_input.normalized();
	Vector3 normalized_control = p_control_point.normalized();
	real_t radius_cosine = Math::cos(p_radius);

	// Check if input is within the cone
	if (normalized_input.dot(normalized_control) > radius_cosine) {
		// Inside cone - return NaN to indicate in bounds
		return Vector3(NAN, NAN, NAN);
	}

	// Find axis for rotation
	Vector3 axis = normalized_control.cross(normalized_input);
	if (axis.is_zero_approx() || !axis.is_finite()) {
		// Vectors are parallel - use perpendicular
		axis = normalized_control.get_any_perpendicular();
		if (axis.is_zero_approx()) {
			axis = Vector3(0, 1, 0);
		}
	}
	axis.normalize();

	// Rotate control point by radius to get boundary point
	Quaternion rot_to = Quaternion(axis, p_radius);
	Vector3 result = rot_to.xform(normalized_control);
	return result.normalized();
}

// Helper function to compute tangent circle between two cones
static void compute_tangent_circle(const Vector3 &p_center1, real_t p_radius1, const Vector3 &p_center2, real_t p_radius2,
		Vector3 &r_tangent1, Vector3 &r_tangent2, real_t &r_tangent_radius) {
	Vector3 A = p_center1.normalized();
	Vector3 B = p_center2.normalized();

	// Find arc normal
	Vector3 arc_normal = A.cross(B);
	if (arc_normal.is_zero_approx() || !arc_normal.is_finite()) {
		arc_normal = A.get_any_perpendicular();
		if (arc_normal.is_zero_approx()) {
			arc_normal = Vector3(0, 1, 0);
		}
	}
	arc_normal.normalize();

	// Compute tangent circle radius
	r_tangent_radius = (Math::PI - (p_radius1 + p_radius2)) / 2.0;

	// Compute tangent circle centers (simplified - full version uses ray intersections)
	real_t boundary_plus_tangent1 = p_radius1 + r_tangent_radius;
	real_t boundary_plus_tangent2 = p_radius2 + r_tangent_radius;

	Quaternion rot1 = Quaternion(arc_normal, boundary_plus_tangent1);
	Quaternion rot2 = Quaternion(arc_normal, boundary_plus_tangent2);

	Vector3 scaled_A = A * Math::cos(boundary_plus_tangent1);
	Vector3 scaled_B = B * Math::cos(boundary_plus_tangent2);

	// Simplified tangent center computation
	// Full version would use ray-plane intersections, but this approximation works for most cases
	Vector3 mid_point = (A + B).normalized();
	Vector3 perp = mid_point.cross(arc_normal).normalized();
	if (perp.is_zero_approx()) {
		perp = mid_point.get_any_perpendicular();
	}

	r_tangent1 = (mid_point + perp * Math::sin(r_tangent_radius)).normalized();
	r_tangent2 = (mid_point - perp * Math::sin(r_tangent_radius)).normalized();
}

// Helper function to find point on path between two cones
static Vector3 get_on_great_tangent_triangle(const Vector3 &p_input, const Vector3 &p_center1, real_t p_radius1,
		const Vector3 &p_center2, real_t p_radius2) {
	Vector3 A = p_center1.normalized();
	Vector3 B = p_center2.normalized();
	Vector3 input = p_input.normalized();

	// Compute tangent circle
	Vector3 tan1, tan2;
	real_t tan_radius;
	compute_tangent_circle(A, p_radius1, B, p_radius2, tan1, tan2, tan_radius);

	real_t tan_radius_cos = Math::cos(tan_radius);

	// Determine which side of the arc we're on
	Vector3 c1xc2 = A.cross(B);
	real_t c1c2dir = input.dot(c1xc2);

	if (c1c2dir < 0.0) {
		// Use first tangent circle
		Vector3 c1xt1 = A.cross(tan1);
		Vector3 t1xc2 = tan1.cross(B);
		if (input.dot(c1xt1) > 0 && input.dot(t1xc2) > 0) {
			real_t to_next_cos = input.dot(tan1);
			if (to_next_cos > tan_radius_cos) {
				// Project onto tangent circle
				Vector3 plane_normal = tan1.cross(input);
				if (plane_normal.is_zero_approx() || !plane_normal.is_finite()) {
					plane_normal = Vector3(0, 1, 0);
				}
				plane_normal.normalize();
				Quaternion rotate_about_by = Quaternion(plane_normal, tan_radius);
				return rotate_about_by.xform(tan1).normalized();
			} else {
				return input;
			}
		}
	} else {
		// Use second tangent circle
		Vector3 t2xc1 = tan2.cross(A);
		Vector3 c2xt2 = B.cross(tan2);
		if (input.dot(t2xc1) > 0 && input.dot(c2xt2) > 0) {
			real_t to_next_cos = input.dot(tan2);
			if (to_next_cos > tan_radius_cos) {
				// Project onto tangent circle
				Vector3 plane_normal = tan2.cross(input);
				if (plane_normal.is_zero_approx() || !plane_normal.is_finite()) {
					plane_normal = Vector3(0, 1, 0);
				}
				plane_normal.normalize();
				Quaternion rotate_about_by = Quaternion(plane_normal, tan_radius);
				return rotate_about_by.xform(tan2).normalized();
			} else {
				return input;
			}
		}
	}

	return Vector3(NAN, NAN, NAN);
}

Vector3 JointLimitationKusudama3D::_solve(const Vector3 &p_direction) const {
	Vector3 result = p_direction.normalized();
	
	// Apply orientation constraint (if enabled)
	if (orientationally_constrained && !open_cones.is_empty()) {
		// Full kusudama solving implementation based on IKKusudama3D::get_local_point_in_limits
		Vector3 point = result;
		real_t closest_cos = -2.0;
		Vector3 closest_collision_point = point;
		bool in_bounds = false;

		// Loop through each limit cone
		for (int i = 0; i < open_cones.size(); i++) {
			const Vector4 &cone_data = open_cones[i];
			Vector3 control_point = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
			real_t radius = cone_data.w;

			Vector3 collision_point = closest_to_cone_boundary(point, control_point, radius);

			// If NaN, point is within this cone
			if (Math::is_nan(collision_point.x) || Math::is_nan(collision_point.y) || Math::is_nan(collision_point.z)) {
				in_bounds = true;
				return point; // Point is within limits
			}

			// Calculate cosine of angle between collision point and original point
			real_t this_cos = collision_point.dot(point);

			// Update closest collision point if this one is closer
			if (closest_collision_point.is_zero_approx() || this_cos > closest_cos) {
				closest_collision_point = collision_point;
				closest_cos = this_cos;
			}
		}

		// If we're out of bounds of all cones, check if we're in the paths between the cones
		if (!in_bounds && open_cones.size() > 1) {
			for (int i = 0; i < open_cones.size() - 1; i++) {
				const Vector4 &cone1_data = open_cones[i];
				const Vector4 &cone2_data = open_cones[i + 1];
				Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
				Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
				real_t radius1 = cone1_data.w;
				real_t radius2 = cone2_data.w;

				Vector3 collision_point = get_on_great_tangent_triangle(point, center1, radius1, center2, radius2);

				// If NaN, skip this path
				if (Math::is_nan(collision_point.x)) {
					continue;
				}

				real_t this_cos = collision_point.dot(point);

				// If cosine is approximately 1, point is in bounds
				if (Math::is_equal_approx(this_cos, real_t(1.0))) {
					return point;
				}

				// Update closest collision point if this one is closer
				if (this_cos > closest_cos) {
					closest_collision_point = collision_point;
					closest_cos = this_cos;
				}
			}

			// Also check path from last to first cone (if more than 2 cones)
			if (open_cones.size() > 2) {
				const Vector4 &cone1_data = open_cones[open_cones.size() - 1];
				const Vector4 &cone2_data = open_cones[0];
				Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
				Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
				real_t radius1 = cone1_data.w;
				real_t radius2 = cone2_data.w;

				Vector3 collision_point = get_on_great_tangent_triangle(point, center1, radius1, center2, radius2);

				if (!Math::is_nan(collision_point.x)) {
					real_t this_cos = collision_point.dot(point);
					if (Math::is_equal_approx(this_cos, real_t(1.0))) {
						return point;
					}
					if (this_cos > closest_cos) {
						closest_collision_point = collision_point;
						closest_cos = this_cos;
					}
				}
			}
		}
		
		// Return the closest boundary point
		result = closest_collision_point.normalized();
	}
	
	// Apply axial limit constraint (if enabled)
	// Note: _solve is called in constraint space where Y is forward (twist axis), X is right, Z is up
	if (axially_constrained && range_angle < Math::TAU) {
		// In constraint space: Y is forward (twist axis), X is right, Z is up
		// The twist angle is the rotation around Y-axis
		// Project direction onto XZ plane (perpendicular to Y)
		Vector3 y_axis = Vector3(0, 1, 0);
		Vector3 proj = result - y_axis * result.dot(y_axis);
		real_t proj_len = proj.length();
		
		if (proj_len > CMP_EPSILON) {
			proj.normalize();
			// Compute twist angle from X-axis
			real_t twist_angle = Math::atan2(proj.z, proj.x);
			
			// Normalize angle to [0, 2π] range
			if (twist_angle < 0) {
				twist_angle += Math::TAU;
			}
			
			// Normalize min_axial_angle to [0, 2π] range
			real_t normalized_min = min_axial_angle;
			if (normalized_min < 0) {
				normalized_min = Math::fposmod(normalized_min, (real_t)Math::TAU);
			} else if (normalized_min >= Math::TAU) {
				normalized_min = Math::fposmod(normalized_min, (real_t)Math::TAU);
			}
			
			real_t max_angle = normalized_min + range_angle;
			real_t clamped_angle = twist_angle;
			
			// Handle angle wrapping
			if (max_angle <= Math::TAU) {
				// Simple case: range doesn't wrap
				if (twist_angle < normalized_min) {
					clamped_angle = normalized_min;
				} else if (twist_angle > max_angle) {
					clamped_angle = max_angle;
				}
			} else {
				// Range wraps around: check both cases
				real_t wrapped_max = Math::fposmod(max_angle, (real_t)Math::TAU);
				if (twist_angle >= normalized_min && twist_angle <= Math::TAU) {
					// In first part of range
					clamped_angle = twist_angle;
				} else if (twist_angle >= 0 && twist_angle <= wrapped_max) {
					// In wrapped part of range
					clamped_angle = twist_angle;
				} else {
					// Outside range - clamp to nearest boundary
					real_t dist_to_min = MIN(Math::abs(twist_angle - normalized_min), Math::abs(twist_angle + Math::TAU - normalized_min));
					real_t dist_to_max = MIN(Math::abs(twist_angle - wrapped_max), Math::abs(twist_angle - Math::TAU - wrapped_max));
					if (dist_to_min < dist_to_max) {
						clamped_angle = normalized_min;
					} else {
						clamped_angle = wrapped_max;
					}
				}
			}
			
			// Reconstruct direction with clamped twist angle
			real_t y_component = result.dot(y_axis);
			real_t proj_scale = Math::sqrt(MAX(0.0, 1.0 - y_component * y_component));
			Vector3 clamped_proj = Vector3(Math::cos(clamped_angle) * proj_scale, 0, Math::sin(clamped_angle) * proj_scale);
			result = clamped_proj + y_axis * y_component;
			result.normalize();
		}
	}
	
	return result;
}

void JointLimitationKusudama3D::set_open_cones(const Vector<Vector4> &p_cones) {
	open_cones = p_cones;
	emit_changed();
}

Vector<Vector4> JointLimitationKusudama3D::get_open_cones() const {
	return open_cones;
}

void JointLimitationKusudama3D::set_cone_count(int p_count) {
	if (p_count < 0) {
		p_count = 0;
	}
	int old_size = open_cones.size();
	if (old_size == p_count) {
		return;
	}
	open_cones.resize(p_count);
	// Initialize new cones with default values
	for (int i = old_size; i < open_cones.size(); i++) {
		open_cones.write[i] = Vector4(0, 1, 0, Math::PI * 0.25); // Default: +Y axis, 45 degree cone
	}
	notify_property_list_changed();
	emit_changed();
}

int JointLimitationKusudama3D::get_cone_count() const {
	return open_cones.size();
}

void JointLimitationKusudama3D::set_cone_center(int p_index, const Vector3 &p_center) {
	ERR_FAIL_INDEX(p_index, open_cones.size());
	Vector3 normalized = p_center.normalized();
	Vector4 &cone = open_cones.write[p_index];
	cone.x = normalized.x;
	cone.y = normalized.y;
	cone.z = normalized.z;
	emit_changed();
}

Vector3 JointLimitationKusudama3D::get_cone_center(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, open_cones.size(), Vector3(0, 1, 0));
	return Vector3(open_cones[p_index].x, open_cones[p_index].y, open_cones[p_index].z);
}

void JointLimitationKusudama3D::set_cone_azimuth(int p_index, real_t p_azimuth) {
	ERR_FAIL_INDEX(p_index, open_cones.size());
	Vector3 center = get_cone_center(p_index);
	real_t elevation = Math::asin(CLAMP(center.y, -1.0, 1.0));
	// Convert azimuth from degrees to radians, normalize to 0-360 range
	real_t azimuth_deg = Math::fposmod(p_azimuth, (real_t)360.0);
	real_t azimuth_rad = Math::deg_to_rad(azimuth_deg);
	// Convert spherical to cartesian
	Vector3 new_center;
	new_center.x = Math::cos(elevation) * Math::sin(azimuth_rad);
	new_center.y = Math::sin(elevation);
	new_center.z = Math::cos(elevation) * Math::cos(azimuth_rad);
	set_cone_center(p_index, new_center);
}

real_t JointLimitationKusudama3D::get_cone_azimuth(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, open_cones.size(), 0.0);
	Vector3 center = get_cone_center(p_index);
	// Convert to azimuth (horizontal angle)
	real_t azimuth_rad = Math::atan2(center.x, center.z);
	// Convert from radians to degrees and normalize to 0-360 range
	real_t azimuth_deg = Math::rad_to_deg(azimuth_rad);
	return Math::fposmod(azimuth_deg, (real_t)360.0);
}

void JointLimitationKusudama3D::set_cone_elevation(int p_index, real_t p_elevation) {
	ERR_FAIL_INDEX(p_index, open_cones.size());
	Vector3 center = get_cone_center(p_index);
	real_t azimuth_rad = Math::atan2(center.x, center.z);
	// Convert elevation from degrees to radians
	// Clamp to valid range for direction vectors (±90 degrees)
	// Allow wider input range in UI, but clamp for calculation
	real_t elevation_rad = Math::deg_to_rad(CLAMP(p_elevation, -90.0, 90.0));
	// Convert spherical to cartesian
	Vector3 new_center;
	new_center.x = Math::cos(elevation_rad) * Math::sin(azimuth_rad);
	new_center.y = Math::sin(elevation_rad);
	new_center.z = Math::cos(elevation_rad) * Math::cos(azimuth_rad);
	set_cone_center(p_index, new_center);
}

real_t JointLimitationKusudama3D::get_cone_elevation(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, open_cones.size(), 0.0);
	Vector3 center = get_cone_center(p_index);
	// Convert to elevation (vertical angle)
	real_t elevation_rad = Math::asin(CLAMP(center.y, -1.0, 1.0));
	// Convert from radians to degrees
	return Math::rad_to_deg(elevation_rad);
}

void JointLimitationKusudama3D::set_cone_radius(int p_index, real_t p_radius) {
	ERR_FAIL_INDEX(p_index, open_cones.size());
	open_cones.write[p_index].w = p_radius;
	emit_changed();
}

real_t JointLimitationKusudama3D::get_cone_radius(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, open_cones.size(), 0.0);
	return open_cones[p_index].w;
}

bool JointLimitationKusudama3D::_set(const StringName &p_name, const Variant &p_value) {
	String prop_name = p_name;
	if (prop_name == "cone_count") {
		set_cone_count(p_value);
		return true;
	}
	if (prop_name.begins_with("open_cones/")) {
		int index = prop_name.get_slicec('/', 1).to_int();
		String what = prop_name.get_slicec('/', 2);
		if (what == "azimuth") {
			set_cone_azimuth(index, p_value);
			return true;
		}
		if (what == "elevation") {
			set_cone_elevation(index, p_value);
			return true;
		}
		if (what == "radius") {
			set_cone_radius(index, p_value);
			return true;
		}
	}
	return false;
}

bool JointLimitationKusudama3D::_get(const StringName &p_name, Variant &r_ret) const {
	String prop_name = p_name;
	if (prop_name == "cone_count") {
		r_ret = get_cone_count();
		return true;
	}
	if (prop_name.begins_with("open_cones/")) {
		int index = prop_name.get_slicec('/', 1).to_int();
		String what = prop_name.get_slicec('/', 2);
		if (what == "azimuth") {
			r_ret = get_cone_azimuth(index);
			return true;
		}
		if (what == "elevation") {
			r_ret = get_cone_elevation(index);
			return true;
		}
		if (what == "radius") {
			r_ret = get_cone_radius(index);
			return true;
		}
	}
	return false;
}

void JointLimitationKusudama3D::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::INT, PNAME("cone_count"), PROPERTY_HINT_RANGE, "0,16384,1,or_greater", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Open Cones," + String(PNAME("open_cones")) + "/"));
	for (int i = 0; i < get_cone_count(); i++) {
		const String prefix = vformat("%s/%d/", PNAME("open_cones"), i);
		p_list->push_back(PropertyInfo(Variant::FLOAT, prefix + PNAME("azimuth"), PROPERTY_HINT_RANGE, "-360,720,0.1,or_less,or_greater,degrees"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, prefix + PNAME("elevation"), PROPERTY_HINT_RANGE, "-180,180,0.1,or_less,or_greater,degrees"));
		p_list->push_back(PropertyInfo(Variant::FLOAT, prefix + PNAME("radius"), PROPERTY_HINT_RANGE, "0,180,0.1,radians_as_degrees"));
	}
}

void JointLimitationKusudama3D::set_axial_limits(real_t p_min_angle, real_t p_range_angle) {
	min_axial_angle = p_min_angle;
	range_angle = p_range_angle;
	emit_changed();
}

void JointLimitationKusudama3D::set_min_axial_angle(real_t p_angle) {
	min_axial_angle = p_angle;
	emit_changed();
}

real_t JointLimitationKusudama3D::get_min_axial_angle() const {
	return min_axial_angle;
}

void JointLimitationKusudama3D::set_range_angle(real_t p_angle) {
	range_angle = p_angle;
	emit_changed();
}

real_t JointLimitationKusudama3D::get_range_angle() const {
	return range_angle;
}

void JointLimitationKusudama3D::set_axially_constrained(bool p_constrained) {
	axially_constrained = p_constrained;
	emit_changed();
}

bool JointLimitationKusudama3D::is_axially_constrained() const {
	return axially_constrained;
}

void JointLimitationKusudama3D::set_orientationally_constrained(bool p_constrained) {
	orientationally_constrained = p_constrained;
	emit_changed();
}

bool JointLimitationKusudama3D::is_orientationally_constrained() const {
	return orientationally_constrained;
}

#ifdef TOOLS_ENABLED
// Helper to draw a circle on the sphere given center, radius angle, and sphere radius
static void draw_cone_circle(LocalVector<Vector3> &r_vts, const Vector3 &p_center, real_t p_radius_angle, real_t p_sphere_r, int p_segments = 32) {
	Vector3 axis = p_center.normalized();
	Vector3 perp1 = axis.get_any_perpendicular().normalized();
	
	// Generate circle points on the sphere
	Vector3 start_point = Quaternion(perp1, p_radius_angle).xform(axis);
	real_t dp = Math::TAU / (real_t)p_segments;
	
	for (int i = 0; i < p_segments; i++) {
		real_t angle = (real_t)i * dp;
		Quaternion rot = Quaternion(axis, angle);
		Vector3 point = rot.xform(start_point) * p_sphere_r;
		Vector3 next_point = rot.xform(Quaternion(axis, (real_t)(i + 1) * dp).xform(start_point)) * p_sphere_r;
		r_vts.push_back(point);
		r_vts.push_back(next_point);
	}
}

// Helper to draw a circle arc on the sphere along a tangent circle
// The arc connects the boundaries of two adjacent cones
static void draw_tangent_circle_arc(LocalVector<Vector3> &r_vts, const Vector3 &p_tangent_center, real_t p_tangent_radius, 
		const Vector3 &p_cone1_center, real_t p_cone1_radius, const Vector3 &p_cone2_center, real_t p_cone2_radius,
		real_t p_sphere_r, int p_segments = 16) {
	Vector3 tan_center = p_tangent_center.normalized();
	Vector3 cone1 = p_cone1_center.normalized();
	Vector3 cone2 = p_cone2_center.normalized();
	
	// Find intersection points: points that are on both the cone boundary and the tangent circle
	// The intersection occurs where the angle from cone center to point equals cone radius,
	// and the angle from tangent center to point equals tangent radius
	
	// For cone1: find point on its boundary that's closest to the tangent circle
	// The point on cone1 boundary closest to tan_center
	Vector3 dir1 = (tan_center - cone1 * cone1.dot(tan_center)).normalized();
	if (dir1.is_zero_approx()) {
		dir1 = cone1.get_any_perpendicular().normalized();
	}
	// Rotate this direction by cone1_radius around cone1 to get boundary point
	Quaternion cone1_rot = Quaternion(cone1, p_cone1_radius);
	Vector3 cone1_boundary = cone1_rot.xform(dir1).normalized();
	
	// Project cone1_boundary onto the tangent circle
	// Find the plane containing tan_center and cone1_boundary, rotate around normal
	Vector3 plane_normal = tan_center.cross(cone1_boundary);
	if (plane_normal.is_zero_approx()) {
		plane_normal = tan_center.get_any_perpendicular();
	}
	plane_normal.normalize();
	Quaternion project_rot = Quaternion(plane_normal, p_tangent_radius);
	Vector3 start_point = project_rot.xform(tan_center).normalized();
	
	// For cone2: same process
	Vector3 dir2 = (tan_center - cone2 * cone2.dot(tan_center)).normalized();
	if (dir2.is_zero_approx()) {
		dir2 = cone2.get_any_perpendicular().normalized();
	}
	Quaternion cone2_rot = Quaternion(cone2, p_cone2_radius);
	Vector3 cone2_boundary = cone2_rot.xform(dir2).normalized();
	
	Vector3 plane_normal2 = tan_center.cross(cone2_boundary);
	if (plane_normal2.is_zero_approx()) {
		plane_normal2 = tan_center.get_any_perpendicular();
	}
	plane_normal2.normalize();
	Quaternion project_rot2 = Quaternion(plane_normal2, p_tangent_radius);
	Vector3 end_point = project_rot2.xform(tan_center).normalized();
	
	// Now draw arc along tangent circle from start_point to end_point
	Vector3 rot_axis = tan_center;
	Vector3 perp = rot_axis.get_any_perpendicular().normalized();
	Vector3 perp2 = rot_axis.cross(perp).normalized();
	
	// Compute angles in the tangent circle's coordinate system
	Vector3 start_rel = start_point - rot_axis * rot_axis.dot(start_point);
	Vector3 end_rel = end_point - rot_axis * rot_axis.dot(end_point);
	start_rel.normalize();
	end_rel.normalize();
	
	real_t start_angle = Math::atan2(start_rel.dot(perp2), start_rel.dot(perp));
	real_t end_angle = Math::atan2(end_rel.dot(perp2), end_rel.dot(perp));
	
	// Normalize to [0, 2π]
	if (end_angle < start_angle) {
		end_angle += Math::TAU;
	}
	
	// Generate arc points
	Vector3 arc_base = Quaternion(perp, p_tangent_radius).xform(rot_axis);
	for (int i = 0; i < p_segments; i++) {
		real_t t = (real_t)i / (real_t)p_segments;
		real_t angle = start_angle + (end_angle - start_angle) * t;
		Quaternion rot = Quaternion(rot_axis, angle);
		Vector3 point = rot.xform(arc_base) * p_sphere_r;
		real_t next_t = (real_t)(i + 1) / (real_t)p_segments;
		real_t next_angle = start_angle + (end_angle - start_angle) * next_t;
		Quaternion next_rot = Quaternion(rot_axis, next_angle);
		Vector3 next_point = next_rot.xform(arc_base) * p_sphere_r;
		r_vts.push_back(point);
		r_vts.push_back(next_point);
	}
}

void JointLimitationKusudama3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const {
	static const int N = 32; // Number of segments per circle

	real_t sphere_r = p_bone_length * 0.25f;
	if (sphere_r <= CMP_EPSILON) {
		return;
	}

	LocalVector<Vector3> vts;

	// Compute the merged boundary of all open areas (boolean union)
	// Like the shader, we distinguish between control cone boundaries and tangent cone boundaries
	LocalVector<Vector3> control_cone_boundary_points; // Points on actual cone boundaries
	LocalVector<Vector3> tangent_cone_boundary_points; // Points on tangent arcs between cones
	
	// Helper to check if a point on the sphere is inside any open area
	auto is_point_in_open_area = [&](const Vector3 &p_point) -> bool {
		Vector3 dir = p_point.normalized();
		
		// Check if point is in any cone
		for (int i = 0; i < open_cones.size(); i++) {
			const Vector4 &cone_data = open_cones[i];
			Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
			real_t radius = cone_data.w;
			real_t angle = Math::acos(CLAMP(dir.dot(center), -1.0, 1.0));
			if (angle <= radius) {
				return true; // Point is inside this cone
			}
		}
		
		// Check if point is in any path between cones
		if (open_cones.size() > 1) {
			for (int i = 0; i < open_cones.size(); i++) {
				int next_i = (i + 1) % open_cones.size();
				const Vector4 &cone1_data = open_cones[i];
				const Vector4 &cone2_data = open_cones[next_i];
				
				Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
				Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
				real_t radius1 = cone1_data.w;
				real_t radius2 = cone2_data.w;
				
				// Check if point is in the path between these cones
				Vector3 collision = get_on_great_tangent_triangle(dir, center1, radius1, center2, radius2);
				if (!Math::is_nan(collision.x) && Math::is_equal_approx(collision.dot(dir), (real_t)1.0)) {
					return true; // Point is in the path
				}
			}
		}
		
		return false; // Point is not in any open area
	};
	
	// Sample boundary points on control cone boundaries
	// Always draw all cone boundaries regardless of viewing angle
	for (int cone_i = 0; cone_i < open_cones.size(); cone_i++) {
		const Vector4 &cone_data = open_cones[cone_i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t radius_angle = cone_data.w;
		
		// Sample points around the cone boundary
		Vector3 axis = center;
		Vector3 perp1 = axis.get_any_perpendicular().normalized();
		Vector3 boundary_start = Quaternion(perp1, radius_angle).xform(axis);
		
		for (int i = 0; i < N; i++) {
			real_t angle = (real_t)i * Math::TAU / (real_t)N;
			Quaternion rot = Quaternion(axis, angle);
			Vector3 boundary_point = rot.xform(boundary_start).normalized();
			
			// Always add all boundary points to show the full circle
			control_cone_boundary_points.push_back(boundary_point * sphere_r);
		}
	}
	
	// Sample boundary points on tangent circle arcs (paths between cones)
	if (open_cones.size() > 1) {
		for (int cone_i = 0; cone_i < open_cones.size(); cone_i++) {
			int next_i = (cone_i + 1) % open_cones.size();
			const Vector4 &cone1_data = open_cones[cone_i];
			const Vector4 &cone2_data = open_cones[next_i];
			
			Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
			Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
			real_t radius1 = cone1_data.w;
			real_t radius2 = cone2_data.w;
			
			Vector3 tan1, tan2;
			real_t tan_radius;
			compute_tangent_circle(center1, radius1, center2, radius2, tan1, tan2, tan_radius);
			
			// Determine which tangent circle to use
			Vector3 mid_dir = (center1 + center2).normalized();
			Vector3 c1xc2 = center1.cross(center2);
			real_t side = mid_dir.dot(c1xc2);
			Vector3 tan_center = (side < 0.0) ? tan1 : tan2;
			
			// Sample points along the tangent circle arc
			Vector3 rot_axis = tan_center.normalized();
			Vector3 perp = rot_axis.get_any_perpendicular().normalized();
			Vector3 arc_base = Quaternion(perp, tan_radius).xform(rot_axis);
			
			// Sample the full tangent circle and always draw it
			int arc_samples = 32;
			for (int i = 0; i < arc_samples; i++) {
				real_t t = (real_t)i / (real_t)(arc_samples - 1);
				// Sample full circle - the actual arc will be visible as it connects the cones
				real_t arc_angle = Math::TAU * t;
				Quaternion rot = Quaternion(rot_axis, arc_angle);
				Vector3 arc_point = rot.xform(arc_base).normalized();
				
				// Always add all arc points to show the full tangent circle
				tangent_cone_boundary_points.push_back(arc_point * sphere_r);
			}
		}
	}
	
	// Draw control cone boundaries (the actual cone circles that are part of the merged boundary)
	// These are drawn as continuous curves
	for (int i = 0; i < (int)control_cone_boundary_points.size() - 1; i++) {
		vts.push_back(control_cone_boundary_points[i]);
		vts.push_back(control_cone_boundary_points[i + 1]);
	}
	
	// Draw tangent cone boundaries (paths between cones)
	// These are drawn as separate segments to distinguish them from control cone boundaries
	for (int i = 0; i < (int)tangent_cone_boundary_points.size() - 1; i++) {
		vts.push_back(tangent_cone_boundary_points[i]);
		vts.push_back(tangent_cone_boundary_points[i + 1]);
	}
	
	// Also draw lines from origin to cone centers for reference
	for (int cone_i = 0; cone_i < open_cones.size(); cone_i++) {
		const Vector4 &cone = open_cones[cone_i];
		Vector3 center = Vector3(cone.x, cone.y, cone.z).normalized();
		vts.push_back(Vector3());
		vts.push_back(center * sphere_r);
	}

	// Draw axial limits if constrained - use octahedron pattern to fill non-open areas
	if (axially_constrained && range_angle < Math::TAU) {
		// Use the average cone radius for the axial limit visualization
		real_t avg_cone_radius = 0.0;
		if (!open_cones.is_empty()) {
			for (int i = 0; i < open_cones.size(); i++) {
				avg_cone_radius += open_cones[i].w;
			}
			avg_cone_radius /= (real_t)open_cones.size();
		} else {
			avg_cone_radius = Math::PI * 0.25; // Default 45 degrees
		}
		
		Vector3 y_axis = Vector3(0, 1, 0);
		real_t ring_radius = sphere_r * Math::sin(avg_cone_radius);
		real_t ring_y = sphere_r * Math::cos(avg_cone_radius);
		Vector3 arc_center = Vector3(0, ring_y, 0); // Center of the arc circle
		
		// Normalize angles to [0, 2π] range
		real_t normalized_min = min_axial_angle;
		if (normalized_min < 0) {
			normalized_min = Math::fposmod(normalized_min, (real_t)Math::TAU);
		} else if (normalized_min >= Math::TAU) {
			normalized_min = Math::fposmod(normalized_min, (real_t)Math::TAU);
		}
		
		real_t max_angle = normalized_min + range_angle;
		bool wraps_around = (max_angle > Math::TAU);
		real_t wrapped_max = wraps_around ? Math::fposmod(max_angle, (real_t)Math::TAU) : max_angle;
		
		// Calculate non-open angle ranges (the complement of the allowed range)
		LocalVector<Pair<real_t, real_t>> non_open_ranges;
		
		if (wraps_around) {
			// Range wraps around: non-open is [wrapped_max, normalized_min]
			if (wrapped_max < normalized_min) {
				non_open_ranges.push_back(Pair<real_t, real_t>(wrapped_max, normalized_min));
			} else {
				// This shouldn't happen, but handle it
				non_open_ranges.push_back(Pair<real_t, real_t>(0, wrapped_max));
				non_open_ranges.push_back(Pair<real_t, real_t>(normalized_min, Math::TAU));
			}
		} else {
			// Range doesn't wrap: non-open is [0, normalized_min] and [max_angle, 2π]
			if (normalized_min > 0) {
				non_open_ranges.push_back(Pair<real_t, real_t>(0, normalized_min));
			}
			if (max_angle < Math::TAU) {
				non_open_ranges.push_back(Pair<real_t, real_t>(max_angle, Math::TAU));
			}
		}
		
		// Draw octahedron pattern to fill each non-open range
		// Octahedron has 8 triangular faces, create pattern with 8 radial directions
		const int octahedron_directions = 8;
		const real_t dir_angle_step = Math::TAU / (real_t)octahedron_directions;
		
		for (int range_idx = 0; range_idx < (int)non_open_ranges.size(); range_idx++) {
			real_t range_start = non_open_ranges[range_idx].first;
			real_t range_end = non_open_ranges[range_idx].second;
			real_t range_size = range_end - range_start;
			
			if (range_size <= 0) {
				continue;
			}
			
			// Create octahedron pattern: draw lines in 8 directions from center
			// Sample the range and draw radial lines to create filled appearance
			int pattern_layers = MAX(4, (int)(range_size / Math::PI * 8.0));
			pattern_layers = MIN(pattern_layers, 16);
			
			real_t mid_angle = range_start + range_size * 0.5;
			
			// Draw octahedron pattern: 8 radial lines from center
			for (int dir = 0; dir < octahedron_directions; dir++) {
				real_t dir_angle = mid_angle + (real_t)dir * dir_angle_step;
				
				// Clamp direction angle to be within the non-open range
				real_t clamped_dir_angle = dir_angle;
				if (clamped_dir_angle < range_start) {
					clamped_dir_angle = range_start;
				} else if (clamped_dir_angle > range_end) {
					clamped_dir_angle = range_end;
				}
				
				// Calculate point on the ring at this direction
				Quaternion rot = Quaternion(y_axis, clamped_dir_angle);
				Vector3 ring_point = rot.xform(Vector3(ring_radius, ring_y, 0));
				
				// Draw line from arc center to ring point (octahedron radial pattern)
				vts.push_back(arc_center);
				vts.push_back(ring_point);
			}
			
			// Draw additional pattern lines: connect points on the ring to create octahedral structure
			// Create triangular pattern by connecting adjacent octahedron directions
			for (int dir = 0; dir < octahedron_directions; dir++) {
				real_t dir_angle1 = mid_angle + (real_t)dir * dir_angle_step;
				real_t dir_angle2 = mid_angle + (real_t)((dir + 1) % octahedron_directions) * dir_angle_step;
				
				// Clamp angles to non-open range
				real_t clamped_angle1 = CLAMP(dir_angle1, range_start, range_end);
				real_t clamped_angle2 = CLAMP(dir_angle2, range_start, range_end);
				
				Quaternion rot1 = Quaternion(y_axis, clamped_angle1);
				Quaternion rot2 = Quaternion(y_axis, clamped_angle2);
				Vector3 p1 = rot1.xform(Vector3(ring_radius, ring_y, 0));
				Vector3 p2 = rot2.xform(Vector3(ring_radius, ring_y, 0));
				
				// Draw line connecting two adjacent octahedron directions (forms triangle edge)
				vts.push_back(p1);
				vts.push_back(p2);
			}
			
			// Draw the boundary arc for this non-open range
			int arc_segments = MAX(8, (int)(range_size / Math::PI * 16.0));
			arc_segments = MIN(arc_segments, 32);
			
			for (int seg = 0; seg < arc_segments; seg++) {
				real_t t = (real_t)seg / (real_t)arc_segments;
				real_t angle = range_start + range_size * t;
				Quaternion rot = Quaternion(y_axis, angle);
				Vector3 p0 = rot.xform(Vector3(ring_radius, ring_y, 0));
				Vector3 p1;
				if (seg < arc_segments - 1) {
					real_t next_t = (real_t)(seg + 1) / (real_t)arc_segments;
					real_t next_angle = range_start + range_size * next_t;
					Quaternion next_rot = Quaternion(y_axis, next_angle);
					p1 = next_rot.xform(Vector3(ring_radius, ring_y, 0));
				} else {
					p1 = rot.xform(Vector3(ring_radius, ring_y, 0));
				}
				// Draw boundary arc
				vts.push_back(p0);
				vts.push_back(p1);
			}
		}
		
		// Draw the boundary of the open area (the allowed range)
		int arc_segments = MAX(16, (int)(range_angle / Math::PI * 32.0));
		arc_segments = MIN(arc_segments, 64);
		
		for (int i = 0; i < arc_segments; i++) {
			real_t t = (real_t)i / (real_t)arc_segments;
			real_t angle = normalized_min + range_angle * t;
			Quaternion rot = Quaternion(y_axis, angle);
			Vector3 p0 = rot.xform(Vector3(ring_radius, ring_y, 0));
			Vector3 p1;
			if (i < arc_segments - 1) {
				real_t next_t = (real_t)(i + 1) / (real_t)arc_segments;
				real_t next_angle = normalized_min + range_angle * next_t;
				Quaternion next_rot = Quaternion(y_axis, next_angle);
				p1 = next_rot.xform(Vector3(ring_radius, ring_y, 0));
			} else {
				p1 = rot.xform(Vector3(ring_radius, ring_y, 0));
			}
			// Draw outer arc boundary
			vts.push_back(p0);
			vts.push_back(p1);
		}
	}

	// Create wireframe volume visualization using lines
	// Similar to shader approach but using lines to show the volume
	if (!open_cones.is_empty()) {
		int rings = 8;
		int radial_segments = 8;
		
		// Generate horizontal wireframe lines (parallels/latitudes)
		for (int j = 0; j <= rings; j++) {
			real_t v = (real_t)j / (real_t)rings;
			real_t w = Math::sin(Math::PI * v);
			real_t y = Math::cos(Math::PI * v);
			
			Vector3 prev_point;
			bool prev_in_area = false;
			
			for (int i = 0; i <= radial_segments; i++) {
				real_t u = (real_t)i / (real_t)radial_segments;
				real_t x = Math::sin(u * Math::TAU);
				real_t z = Math::cos(u * Math::TAU);
				
				Vector3 normal = Vector3(x * w, y, z * w).normalized();
				Vector3 point = normal * sphere_r;
				
				bool in_area = is_point_in_open_area(normal);
				
				// Draw line segment if both points are in the open area
				if (i > 0 && prev_in_area && in_area) {
					vts.push_back(prev_point);
					vts.push_back(point);
				}
				
				prev_point = point;
				prev_in_area = in_area;
			}
		}
		
		// Generate vertical lines (meridians/longitudes)
		for (int i = 0; i <= radial_segments; i++) {
			real_t u = (real_t)i / (real_t)radial_segments;
			real_t x = Math::sin(u * Math::TAU);
			real_t z = Math::cos(u * Math::TAU);
			
			Vector3 prev_point;
			bool prev_in_area = false;
			
			for (int j = 0; j <= rings; j++) {
				real_t v = (real_t)j / (real_t)rings;
				real_t w = Math::sin(Math::PI * v);
				real_t y = Math::cos(Math::PI * v);
				
				Vector3 normal = Vector3(x * w, y, z * w).normalized();
				Vector3 point = normal * sphere_r;
				
				bool in_area = is_point_in_open_area(normal);
				
				// Draw line segment if both points are in the open area
				if (j > 0 && prev_in_area && in_area) {
					vts.push_back(prev_point);
					vts.push_back(point);
				}
				
				prev_point = point;
				prev_in_area = in_area;
			}
		}
	}
	
	// Add all vertices to surface tool as a single mesh
	// All lines (boundaries and volume wireframe) use the same color
	for (int64_t i = 0; i < vts.size(); i++) {
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(vts[i]));
	}
}
#endif // TOOLS_ENABLED
