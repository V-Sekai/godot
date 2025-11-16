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

// Helper function matching shader's is_in_inter_cone_path logic
static bool is_in_inter_cone_path(const Vector3 &p_normal_dir, const Vector3 &p_tangent_1, const Vector3 &p_cone_1,
		const Vector3 &p_tangent_2, const Vector3 &p_cone_2) {
	Vector3 c1xc2 = p_cone_1.cross(p_cone_2);
	real_t c1c2dir = p_normal_dir.dot(c1xc2);
	
	if (c1c2dir < 0.0) {
		Vector3 c1xt1 = p_cone_1.cross(p_tangent_1);
		Vector3 t1xc2 = p_tangent_1.cross(p_cone_2);
		real_t c1t1dir = p_normal_dir.dot(c1xt1);
		real_t t1c2dir = p_normal_dir.dot(t1xc2);
		
		return (c1t1dir > 0.0 && t1c2dir > 0.0);
	} else {
		Vector3 t2xc1 = p_tangent_2.cross(p_cone_1);
		Vector3 c2xt2 = p_cone_2.cross(p_tangent_2);
		real_t t2c1dir = p_normal_dir.dot(t2xc1);
		real_t c2t2dir = p_normal_dir.dot(c2xt2);
		
		return (c2t2dir > 0.0 && t2c1dir > 0.0);
	}
}

// Helper function to check if a point on the sphere is in the union of all open areas
// Matches the shader's color_allowed logic
static bool is_point_in_union(const Vector3 &p_point, const Vector<Vector4> &p_open_cones) {
	Vector3 dir = p_point.normalized();
	
	// Check if point is in any cone (inside, not on boundary - matches shader)
	for (int i = 0; i < p_open_cones.size(); i++) {
		const Vector4 &cone_data = p_open_cones[i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t radius = cone_data.w;
		real_t angle = Math::acos(CLAMP(dir.dot(center), -1.0, 1.0));
		if (angle < radius) {
			return true; // Point is inside this cone (strictly inside, not on boundary)
		}
	}
	
	// Check if point is in any path between cones (matching shader logic)
	if (p_open_cones.size() > 1) {
		for (int i = 0; i < p_open_cones.size(); i++) {
			int next_i = (i + 1) % p_open_cones.size();
			const Vector4 &cone1_data = p_open_cones[i];
			const Vector4 &cone2_data = p_open_cones[next_i];
			
			Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
			Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
			real_t radius1 = cone1_data.w;
			real_t radius2 = cone2_data.w;
			
			// Compute tangent circles
			Vector3 tan1, tan2;
			real_t tan_radius;
			compute_tangent_circle(center1, radius1, center2, radius2, tan1, tan2, tan_radius);
			
			// Check if point is in the inter-cone path using shader logic
			if (is_in_inter_cone_path(dir, tan1, center1, tan2, center2)) {
				return true; // Point is in the path
			}
		}
	}
	
	return false; // Point is not in any open area
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
	// Store raw value (non-normalized) to allow editor to accept values outside [-1, 1]
	// Normalization happens lazily when values are used
	Vector4 &cone = open_cones.write[p_index];
	cone.x = p_center.x;
	cone.y = p_center.y;
	cone.z = p_center.z;
	emit_changed();
}

Vector3 JointLimitationKusudama3D::get_cone_center(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, open_cones.size(), Vector3(0, 1, 0));
	Vector3 center = Vector3(open_cones[p_index].x, open_cones[p_index].y, open_cones[p_index].z);
	// Normalize when reading to ensure we always return a normalized value
	if (center.length_squared() > CMP_EPSILON) {
		return center.normalized();
	}
	return Vector3(0, 1, 0);
}

void JointLimitationKusudama3D::set_cone_center_quaternion(int p_index, const Quaternion &p_quaternion) {
	ERR_FAIL_INDEX(p_index, open_cones.size());
	// Convert quaternion to direction vector by rotating the default direction (0, 1, 0)
	Vector3 default_dir = Vector3(0, 1, 0);
	Vector3 center = p_quaternion.normalized().xform(default_dir);
	// Store raw value (non-normalized) to allow editor to accept values outside [-1, 1]
	Vector4 &cone = open_cones.write[p_index];
	cone.x = center.x;
	cone.y = center.y;
	cone.z = center.z;
	emit_changed();
}

Quaternion JointLimitationKusudama3D::get_cone_center_quaternion(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, open_cones.size(), Quaternion());
	Vector3 center = get_cone_center(p_index); // This already normalizes
	Vector3 default_dir = Vector3(0, 1, 0);
	// Create quaternion representing rotation from default_dir to center
	return Quaternion(default_dir, center);
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
		if (what == "center") {
			// Handle quaternion input from inspector
			if (p_value.get_type() == Variant::QUATERNION) {
				set_cone_center_quaternion(index, p_value);
			} else {
				set_cone_center(index, p_value);
			}
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
		if (what == "center") {
			// Return as quaternion for inspector display with degrees
			r_ret = get_cone_center_quaternion(index);
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
		// Use quaternion for inspector display with Euler angles in degrees
		p_list->push_back(PropertyInfo(Variant::QUATERNION, prefix + PNAME("center"), PROPERTY_HINT_NONE, ""));
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
// Uses spherical interpolation for smooth curve fitting
static void draw_cone_circle(LocalVector<Vector3> &r_vts, const Vector3 &p_center, real_t p_radius_angle, real_t p_sphere_r, int p_segments = 64) {
	Vector3 axis = p_center.normalized();
	Vector3 perp1 = axis.get_any_perpendicular().normalized();
	
	// Generate circle points on the sphere using spherical interpolation
	Vector3 start_point = Quaternion(perp1, p_radius_angle).xform(axis).normalized();
	real_t dp = Math::TAU / (real_t)p_segments;
	
	Vector3 prev_point = start_point * p_sphere_r;
	for (int i = 1; i <= p_segments; i++) {
		real_t angle = (real_t)i * dp;
		Quaternion rot = Quaternion(axis, angle);
		Vector3 current_point = rot.xform(start_point).normalized() * p_sphere_r;
		
		// Use spherical linear interpolation for smoother curve
		// Reduced subdivision for radius rings
		int subdiv = 2; // Reduced subdivision for radius rings
		for (int j = 1; j <= subdiv; j++) {
			real_t t = (real_t)j / (real_t)subdiv;
			Vector3 p0 = prev_point.normalized();
			Vector3 p1 = current_point.normalized();
			Vector3 interp = p0.slerp(p1, t).normalized() * p_sphere_r;
			
			if (j == 1) {
				r_vts.push_back(prev_point);
			}
			r_vts.push_back(interp);
		}
		
		prev_point = current_point;
	}
}

// Helper to draw a circle arc on the sphere along a tangent circle
// The arc connects the boundaries of two adjacent cones
// Uses spherical interpolation for smooth curve fitting with fragment shader level detail
static void draw_tangent_circle_arc(LocalVector<Vector3> &r_vts, const Vector3 &p_tangent_center, real_t p_tangent_radius, 
		const Vector3 &p_cone1_center, real_t p_cone1_radius, const Vector3 &p_cone2_center, real_t p_cone2_radius,
		real_t p_sphere_r, int p_segments = 256) {
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
	
	// Generate arc points using spherical interpolation for smooth curve fitting
	Vector3 arc_base = Quaternion(perp, p_tangent_radius).xform(rot_axis).normalized();
	Vector3 start_arc_point = start_point.normalized();
	Vector3 end_arc_point = end_point.normalized();
	
	// Use spherical linear interpolation (slerp) for smooth curve along the arc
	int subdiv = 16; // High subdivision for fragment shader level detail
	Vector3 prev_arc_point = start_arc_point * p_sphere_r;
	
	for (int i = 1; i <= p_segments; i++) {
		real_t t = (real_t)i / (real_t)p_segments;
		real_t angle = start_angle + (end_angle - start_angle) * t;
		Quaternion rot = Quaternion(rot_axis, angle);
		Vector3 current_arc_point = rot.xform(arc_base).normalized() * p_sphere_r;
		
		// Subdivide using spherical interpolation for smoother curve
		for (int j = 1; j <= subdiv; j++) {
			real_t subdiv_t = (real_t)j / (real_t)subdiv;
			Vector3 p0 = prev_arc_point.normalized();
			Vector3 p1 = current_arc_point.normalized();
			Vector3 interp = p0.slerp(p1, subdiv_t).normalized() * p_sphere_r;
			
			if (j == 1) {
				r_vts.push_back(prev_arc_point);
			}
			r_vts.push_back(interp);
		}
		
		prev_arc_point = current_arc_point;
	}
}

void JointLimitationKusudama3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const {
	static const int N = 32; // Number of segments per circle

	real_t socket_r = p_bone_length * 0.25f;
	if (socket_r <= CMP_EPSILON) {
		return;
	}

	LocalVector<Vector3> vts;
	
	// Draw rotation freedom indicators at the joint origin
	// Show how much the bone can still rotate around its axis
	if (axially_constrained && range_angle < Math::TAU) {
		real_t indicator_r = socket_r * 1.2f; // Extend outside the unit sphere for better visibility
		
		// Normalize angles
		real_t normalized_min = min_axial_angle;
		if (normalized_min < 0) {
			normalized_min = Math::fposmod(normalized_min, (real_t)Math::TAU);
		} else if (normalized_min >= Math::TAU) {
			normalized_min = Math::fposmod(normalized_min, (real_t)Math::TAU);
		}
		
		real_t max_angle = normalized_min + range_angle;
		real_t wrapped_max = (max_angle > Math::TAU) ? Math::fposmod(max_angle, (real_t)Math::TAU) : max_angle;
		
		// Draw indicator arcs showing disallowed rotation areas at the origin (inverted)
		Vector3 indicator_pos = Vector3(0, 0, 0);
		Vector3 y_axis = Vector3(0, 1, 0); // Bone axis (forward direction)
		Vector3 x_axis = Vector3(1, 0, 0);
		Vector3 z_axis = Vector3(0, 0, 1);
		
		// Draw disallowed rotation arcs (inverse of allowed range)
		bool wraps_around = (max_angle > Math::TAU);
		
		if (wraps_around) {
			// Range wraps: draw arc from wrapped_max to normalized_min
			real_t disallowed_range = normalized_min - wrapped_max;
			if (disallowed_range < 0) {
				disallowed_range += Math::TAU;
			}
			int arc_segments = MAX(16, (int)(disallowed_range / Math::PI * 32.0));
			arc_segments = MIN(arc_segments, 64);
			for (int i = 0; i < arc_segments; i++) {
				real_t t = (real_t)i / (real_t)arc_segments;
				real_t angle = wrapped_max + disallowed_range * t;
				if (angle >= Math::TAU) {
					angle -= Math::TAU;
				}
				Vector3 dir = (x_axis * Math::cos(angle) + z_axis * Math::sin(angle)) * indicator_r;
				Vector3 p0 = indicator_pos + dir;
				Vector3 p1;
				if (i < arc_segments - 1) {
					real_t next_t = (real_t)(i + 1) / (real_t)arc_segments;
					real_t next_angle = wrapped_max + disallowed_range * next_t;
					if (next_angle >= Math::TAU) {
						next_angle -= Math::TAU;
					}
					Vector3 next_dir = (x_axis * Math::cos(next_angle) + z_axis * Math::sin(next_angle)) * indicator_r;
					p1 = indicator_pos + next_dir;
				} else {
					p1 = indicator_pos + dir;
				}
				vts.push_back(p0);
				vts.push_back(p1);
			}
		} else {
			// Range doesn't wrap: draw arcs from 0 to normalized_min and from max_angle to TAU
			if (normalized_min > 0) {
				int arc_segments = MAX(16, (int)(normalized_min / Math::PI * 32.0));
				arc_segments = MIN(arc_segments, 64);
				for (int i = 0; i < arc_segments; i++) {
					real_t t = (real_t)i / (real_t)arc_segments;
					real_t angle = normalized_min * t;
					Vector3 dir = (x_axis * Math::cos(angle) + z_axis * Math::sin(angle)) * indicator_r;
					Vector3 p0 = indicator_pos + dir;
					Vector3 p1;
					if (i < arc_segments - 1) {
						real_t next_t = (real_t)(i + 1) / (real_t)arc_segments;
						real_t next_angle = normalized_min * next_t;
						Vector3 next_dir = (x_axis * Math::cos(next_angle) + z_axis * Math::sin(next_angle)) * indicator_r;
						p1 = indicator_pos + next_dir;
					} else {
						p1 = indicator_pos + dir;
					}
					vts.push_back(p0);
					vts.push_back(p1);
				}
			}
			if (max_angle < Math::TAU) {
				real_t disallowed_range = Math::TAU - max_angle;
				int arc_segments = MAX(16, (int)(disallowed_range / Math::PI * 32.0));
				arc_segments = MIN(arc_segments, 64);
				for (int i = 0; i < arc_segments; i++) {
					real_t t = (real_t)i / (real_t)arc_segments;
					real_t angle = max_angle + disallowed_range * t;
					Vector3 dir = (x_axis * Math::cos(angle) + z_axis * Math::sin(angle)) * indicator_r;
					Vector3 p0 = indicator_pos + dir;
					Vector3 p1;
					if (i < arc_segments - 1) {
						real_t next_t = (real_t)(i + 1) / (real_t)arc_segments;
						real_t next_angle = max_angle + disallowed_range * next_t;
						Vector3 next_dir = (x_axis * Math::cos(next_angle) + z_axis * Math::sin(next_angle)) * indicator_r;
						p1 = indicator_pos + next_dir;
					} else {
						p1 = indicator_pos + dir;
					}
					vts.push_back(p0);
					vts.push_back(p1);
				}
			}
		}
		
		// Draw lines from origin to arc endpoints to show limits
		Vector3 limit1_dir = (x_axis * Math::cos(normalized_min) + z_axis * Math::sin(normalized_min)) * indicator_r;
		Vector3 limit2_dir = (x_axis * Math::cos(wrapped_max) + z_axis * Math::sin(wrapped_max)) * indicator_r;
		vts.push_back(indicator_pos);
		vts.push_back(indicator_pos + limit1_dir);
		vts.push_back(indicator_pos);
		vts.push_back(indicator_pos + limit2_dir);
	}
	
	// Create wireframe visualization: sphere with cones and tangents subtracted
	// Draw sphere wireframe, but skip (subtract) areas inside cones and tangent paths
	if (!open_cones.is_empty()) {
		int rings = 32;
		int radial_segments = 32;
		
		// Generate horizontal wireframe lines (parallels/latitudes)
		// Draw sphere lines, but subtract areas in cones and tangent paths
		// Draw smooth boundary curves where transitions occur
		for (int j = 0; j <= rings; j++) {
			real_t v = (real_t)j / (real_t)rings;
			real_t w = Math::sin(Math::PI * v);
			real_t y = Math::cos(Math::PI * v);
			
			Vector3 prev_point;
			bool prev_in_allowed = false;
			
			for (int i = 0; i <= radial_segments; i++) {
				real_t u = (real_t)i / (real_t)radial_segments;
				real_t x = Math::sin(u * Math::TAU);
				real_t z = Math::cos(u * Math::TAU);
				
				Vector3 normal = Vector3(x * w, y, z * w).normalized();
				Vector3 point = normal * socket_r;
				
				// Check if point is in any cone or tangent path (allowed area to subtract)
				bool in_allowed = is_point_in_union(normal, open_cones);
				
				// Draw line segment only if both points are NOT in allowed areas (sphere minus cones/tangents)
				if (i > 0 && !prev_in_allowed && !in_allowed) {
					// Both points are in disallowed area - draw the sphere edge
					vts.push_back(prev_point);
					vts.push_back(point);
				}
				
				// Draw smooth boundary curve at transition from disallowed to allowed (cut boundary)
				if (i > 0 && prev_in_allowed != in_allowed) {
					// Transition point - draw smooth spline boundary curve with fragment shader level detail
					// Use spherical interpolation to create very fine smooth boundary
					int boundary_subdiv = 64; // Very high subdivision for fragment shader level detail
					for (int k = 0; k <= boundary_subdiv; k++) {
						real_t t = (real_t)k / (real_t)boundary_subdiv;
						Vector3 p0 = prev_point.normalized();
						Vector3 p1 = point.normalized();
						Vector3 boundary_point = p0.slerp(p1, t).normalized() * socket_r;
						
						if (k > 0) {
							Vector3 prev_boundary = p0.slerp(p1, (real_t)(k - 1) / (real_t)boundary_subdiv).normalized() * socket_r;
							vts.push_back(prev_boundary);
							vts.push_back(boundary_point);
						}
					}
				}
				
				prev_point = point;
				prev_in_allowed = in_allowed;
			}
		}
		
		// Generate vertical lines (meridians/longitudes)
		// Draw sphere lines, but subtract areas in cones and tangent paths
		// Draw smooth boundary curves where transitions occur
		for (int i = 0; i <= radial_segments; i++) {
			real_t u = (real_t)i / (real_t)radial_segments;
			real_t x = Math::sin(u * Math::TAU);
			real_t z = Math::cos(u * Math::TAU);
			
			Vector3 prev_point;
			bool prev_in_allowed = false;
			
			for (int j = 0; j <= rings; j++) {
				real_t v = (real_t)j / (real_t)rings;
				real_t w = Math::sin(Math::PI * v);
				real_t y = Math::cos(Math::PI * v);
				
				Vector3 normal = Vector3(x * w, y, z * w).normalized();
				Vector3 point = normal * socket_r;
				
				// Check if point is in any cone or tangent path (allowed area to subtract)
				bool in_allowed = is_point_in_union(normal, open_cones);
				
				// Draw line segment only if both points are NOT in allowed areas (sphere minus cones/tangents)
				if (j > 0 && !prev_in_allowed && !in_allowed) {
					// Both points are in disallowed area - draw the sphere edge
					vts.push_back(prev_point);
					vts.push_back(point);
				}
				
				// Draw smooth boundary curve at transition from disallowed to allowed (cut boundary)
				if (j > 0 && prev_in_allowed != in_allowed) {
					// Transition point - draw smooth spline boundary curve with fragment shader level detail
					// Use spherical interpolation to create very fine smooth boundary
					int boundary_subdiv = 64; // Very high subdivision for fragment shader level detail
					for (int k = 0; k <= boundary_subdiv; k++) {
						real_t t = (real_t)k / (real_t)boundary_subdiv;
						Vector3 p0 = prev_point.normalized();
						Vector3 p1 = point.normalized();
						Vector3 boundary_point = p0.slerp(p1, t).normalized() * socket_r;
						
						if (k > 0) {
							Vector3 prev_boundary = p0.slerp(p1, (real_t)(k - 1) / (real_t)boundary_subdiv).normalized() * socket_r;
							vts.push_back(prev_boundary);
							vts.push_back(boundary_point);
						}
					}
				}
				
				prev_point = point;
				prev_in_allowed = in_allowed;
			}
		}
	}
	
	// Draw exact cone boundaries (circle at cone radius)
	for (int cone_i = 0; cone_i < open_cones.size(); cone_i++) {
		const Vector4 &cone_data = open_cones[cone_i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t cone_radius = cone_data.w; // Cone radius in radians
		
		// Draw the exact boundary circle of the cone (using spline interpolation with reduced detail)
		draw_cone_circle(vts, center, cone_radius, socket_r, 64);
	}
	
	// Tangent path boundaries are shown in the wireframe visualization where the sphere is cut
	// The is_point_in_union function includes tangent paths, so boundaries are automatically visible
	
	// Draw simple markers at exact cone center locations
	for (int cone_i = 0; cone_i < open_cones.size(); cone_i++) {
		const Vector4 &cone_data = open_cones[cone_i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		Vector3 center_point = center * socket_r;
		
		// Draw a small cross at the cone center
		Vector3 perp1 = center.get_any_perpendicular().normalized();
		Vector3 perp2 = center.cross(perp1).normalized();
		real_t marker_size = socket_r * 0.02f; // Small marker size
		
		// Draw two orthogonal lines forming a cross
		vts.push_back(center_point - perp1 * marker_size);
		vts.push_back(center_point + perp1 * marker_size);
		vts.push_back(center_point - perp2 * marker_size);
		vts.push_back(center_point + perp2 * marker_size);
	}
	
	// Add all vertices to surface tool as a single mesh
	// All lines (boundaries) use the same color
	for (int64_t i = 0; i < vts.size(); i++) {
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(vts[i]));
	}
	
}
#endif // TOOLS_ENABLED
