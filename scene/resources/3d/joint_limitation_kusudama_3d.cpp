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

#ifdef TOOLS_ENABLED
#include "scene/resources/surface_tool.h"
#endif // TOOLS_ENABLED

void JointLimitationKusudama3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cones", "cones"), &JointLimitationKusudama3D::set_cones);
	ClassDB::bind_method(D_METHOD("get_cones"), &JointLimitationKusudama3D::get_cones);

	ClassDB::bind_method(D_METHOD("set_orientationally_constrained", "constrained"), &JointLimitationKusudama3D::set_orientationally_constrained);
	ClassDB::bind_method(D_METHOD("is_orientationally_constrained"), &JointLimitationKusudama3D::is_orientationally_constrained);

	ClassDB::bind_method(D_METHOD("set_cone_count", "count"), &JointLimitationKusudama3D::set_cone_count);
	ClassDB::bind_method(D_METHOD("get_cone_count"), &JointLimitationKusudama3D::get_cone_count);
	ClassDB::bind_method(D_METHOD("set_cone_center", "index", "center"), &JointLimitationKusudama3D::set_cone_center);
	ClassDB::bind_method(D_METHOD("get_cone_center", "index"), &JointLimitationKusudama3D::get_cone_center);
	ClassDB::bind_method(D_METHOD("set_cone_radius", "index", "radius"), &JointLimitationKusudama3D::set_cone_radius);
	ClassDB::bind_method(D_METHOD("get_cone_radius", "index"), &JointLimitationKusudama3D::get_cone_radius);

	ClassDB::bind_method(D_METHOD("set_axial_limits", "min_angle", "range_angle"), &JointLimitationKusudama3D::set_axial_limits);
	ClassDB::bind_method(D_METHOD("set_min_axial_angle", "angle"), &JointLimitationKusudama3D::set_min_axial_angle);
	ClassDB::bind_method(D_METHOD("get_min_axial_angle"), &JointLimitationKusudama3D::get_min_axial_angle);
	ClassDB::bind_method(D_METHOD("set_range_angle", "angle"), &JointLimitationKusudama3D::set_range_angle);
	ClassDB::bind_method(D_METHOD("get_range_angle"), &JointLimitationKusudama3D::get_range_angle);
	ClassDB::bind_method(D_METHOD("set_axially_constrained", "constrained"), &JointLimitationKusudama3D::set_axially_constrained);
	ClassDB::bind_method(D_METHOD("is_axially_constrained"), &JointLimitationKusudama3D::is_axially_constrained);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "cones", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_cones", "get_cones");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "orientationally_constrained"), "set_orientationally_constrained", "is_orientationally_constrained");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "axially_constrained"), "set_axially_constrained", "is_axially_constrained");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_axial_angle", PROPERTY_HINT_RANGE, "-360,360,0.1,or_less,or_greater,radians_as_degrees"), "set_min_axial_angle", "get_min_axial_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "range_angle", PROPERTY_HINT_RANGE, "0,360,0.1,radians_as_degrees"), "set_range_angle", "get_range_angle");
}

// Helper function to find closest point on a single cone boundary
// Helper function to get orthogonal vector (from IKLimitCone3D::get_orthogonal)
static Vector3 get_orthogonal(const Vector3 &p_in) {
	Vector3 result;
	real_t threshold = p_in.length() * 0.6f;
	if (threshold > 0.f) {
		if (Math::abs(p_in.x) <= threshold) {
			real_t inverse = 1.f / Math::sqrt(p_in.y * p_in.y + p_in.z * p_in.z);
			return Vector3(0.f, inverse * p_in.z, -inverse * p_in.y);
		} else if (Math::abs(p_in.y) <= threshold) {
			real_t inverse = 1.f / Math::sqrt(p_in.x * p_in.x + p_in.z * p_in.z);
			return Vector3(-inverse * p_in.z, 0.f, inverse * p_in.x);
		}
		real_t inverse = 1.f / Math::sqrt(p_in.x * p_in.x + p_in.y * p_in.y);
		return Vector3(inverse * p_in.y, -inverse * p_in.x, 0.f);
	}
	return result;
}

// Helper function to create quaternion from axis and angle (from IKKusudama3D::get_quaternion_axis_angle)
static Quaternion get_quaternion_axis_angle(const Vector3 &p_axis, real_t p_angle) {
	// Handle zero-length axis case
	if (p_axis.length_squared() < CMP_EPSILON2) {
		return Quaternion(); // Return identity quaternion
	}

	// Handle very small angle case
	if (Math::abs(p_angle) < CMP_EPSILON) {
		return Quaternion(); // Return identity quaternion
	}

	// Use standard quaternion creation (interval arithmetic not available, but this should work)
	Vector3 normalized_axis = p_axis.normalized();
	real_t half_angle = p_angle * 0.5;
	real_t sin_half = Math::sin(half_angle);
	return Quaternion(normalized_axis.x * sin_half, normalized_axis.y * sin_half, normalized_axis.z * sin_half, Math::cos(half_angle));
}

// Helper function to find closest point on cone boundary (from IKLimitCone3D::closest_to_cone)
// Returns NaN if point is inside cone, otherwise returns closest boundary point
static Vector3 closest_to_cone_boundary(const Vector3 &p_input, const Vector3 &p_control_point, real_t p_radius) {
	Vector3 normalized_input = p_input.normalized();
	Vector3 normalized_control_point = p_control_point.normalized();
	real_t radius_cosine = Math::cos(p_radius);

	// If point is inside cone, return NaN (matches many_bone_ik behavior)
	if (normalized_input.dot(normalized_control_point) > radius_cosine) {
		return Vector3(NAN, NAN, NAN);
	}

	// Find axis for rotation using cross product (robust handling without interval arithmetic)
	Vector3 axis = normalized_control_point.cross(normalized_input);

	// Additional validation for the axis
	if (!axis.is_finite() || Math::is_zero_approx(axis.length_squared())) {
		// Fallback: use the most orthogonal axis to the control point
		axis = get_orthogonal(normalized_control_point);
		if (Math::is_zero_approx(axis.length_squared())) {
			axis = Vector3(0, 1, 0);
		}
		axis.normalize();
	} else {
		axis.normalize();
	}

	// Rotate control point by radius around axis to get boundary point
	Quaternion rot_to = get_quaternion_axis_angle(axis, p_radius);
	Vector3 axis_control_point = normalized_control_point;
	if (Math::is_zero_approx(axis_control_point.length_squared())) {
		axis_control_point = Vector3(0, 1, 0);
	}
	Vector3 result = rot_to.xform(axis_control_point);
	return result.normalized();
}

// Helper function to compute plane-ray intersection
static Vector3 ray_plane_intersection(const Vector3 &p_ray_start, const Vector3 &p_ray_end, const Vector3 &p_plane_a, const Vector3 &p_plane_b, const Vector3 &p_plane_c) {
	Vector3 ray_dir = (p_ray_end - p_ray_start).normalized();
	Vector3 plane_edge1 = p_plane_b - p_plane_a;
	Vector3 plane_edge2 = p_plane_c - p_plane_a;
	Vector3 plane_normal = plane_edge1.cross(plane_edge2).normalized();

	Vector3 ray_to_plane = p_ray_start - p_plane_a;
	real_t plane_distance = -plane_normal.dot(ray_to_plane);
	real_t ray_dot_normal = plane_normal.dot(ray_dir);

	if (Math::abs(ray_dot_normal) < CMP_EPSILON) {
		return Vector3(NAN, NAN, NAN); // Ray is parallel to plane
	}

	real_t intersection_param = plane_distance / ray_dot_normal;
	return p_ray_start + ray_dir * intersection_param;
}

// Helper function to extend a ray in both directions
static void extend_ray(Vector3 &r_start, Vector3 &r_end, real_t p_amount) {
	Vector3 mid_point = (r_start + r_end) * 0.5;
	Vector3 start_heading = r_start - mid_point;
	Vector3 end_heading = r_end - mid_point;
	Vector3 start_extension = start_heading.normalized() * p_amount;
	Vector3 end_extension = end_heading.normalized() * p_amount;
	r_start = start_heading + start_extension + mid_point;
	r_end = end_heading + end_extension + mid_point;
}

// Helper function to compute ray-sphere intersection
static int ray_sphere_intersection(const Vector3 &p_ray_start, const Vector3 &p_ray_end, const Vector3 &p_sphere_center, real_t p_radius, Vector3 *r_intersection1, Vector3 *r_intersection2) {
	Vector3 ray_start_rel = p_ray_start - p_sphere_center;
	Vector3 ray_end_rel = p_ray_end - p_sphere_center;
	Vector3 direction = ray_end_rel - ray_start_rel;
	Vector3 ray_dir_normalized = direction.normalized();
	Vector3 ray_to_center = -ray_start_rel;
	real_t ray_dot_center = ray_dir_normalized.dot(ray_to_center);
	real_t radius_squared = p_radius * p_radius;
	real_t center_dist_squared = ray_to_center.length_squared();
	real_t ray_dot_squared = ray_dot_center * ray_dot_center;
	real_t discriminant = radius_squared - center_dist_squared + ray_dot_squared;

	if (discriminant < 0.0) {
		return 0; // No intersection
	}
	discriminant = Math::sqrt(discriminant);

	int result = 0;
	if (ray_dot_center < discriminant) {
		if (ray_dot_center + discriminant >= 0) {
			discriminant = -discriminant;
			result = 1;
		}
	} else {
		result = 2;
	}

	*r_intersection1 = ray_dir_normalized * (ray_dot_center - discriminant) + p_sphere_center;
	*r_intersection2 = ray_dir_normalized * (ray_dot_center + discriminant) + p_sphere_center;
	return result;
}

// Helper function to compute tangent circle between two cones
// Uses plane intersection method matching the previous implementation in ik_open_cone_3d.cpp
// This ensures the tangent circles are computed on the correct side (the open side)
static void compute_tangent_circle(const Vector3 &p_center1, real_t p_radius1, const Vector3 &p_center2, real_t p_radius2,
		Vector3 &r_tangent1, Vector3 &r_tangent2, real_t &r_tangent_radius) {
	Vector3 center1 = p_center1.normalized();
	Vector3 center2 = p_center2.normalized();

	// Compute tangent circle radius (matches IKKusudama3D)
	r_tangent_radius = (Math::PI - (p_radius1 + p_radius2)) / 2.0;

	// Find arc normal (axis perpendicular to both cone centers)
	Vector3 arc_normal = center1.cross(center2);
	real_t arc_normal_len = arc_normal.length();

	// Handle parallel/opposite cones (matches many_bone_ik logic)
	if (arc_normal_len < CMP_EPSILON) {
		// Handle singularity case - use get_orthogonal for robust handling
		Vector3 reference_axis = get_orthogonal(center1);
		if (reference_axis.is_zero_approx()) {
			reference_axis = Vector3(0, 1, 0); // Ultimate fallback
		}
		arc_normal = reference_axis.normalized();

		// For opposite cones, tangent circles are at 90 degrees from the cone centers
		// Use a perpendicular vector in the plane perpendicular to center1
		Vector3 perp1 = get_orthogonal(center1);
		if (perp1.is_zero_approx()) {
			perp1 = Vector3(0, 1, 0);
		}
		perp1.normalize();

		// Rotate around center1 by the tangent radius to get tangent centers
		Quaternion rot1 = get_quaternion_axis_angle(center1, r_tangent_radius);
		Quaternion rot2 = get_quaternion_axis_angle(center1, -r_tangent_radius);
		r_tangent1 = rot1.xform(perp1).normalized();
		r_tangent2 = rot2.xform(perp1).normalized();
		return;
	}
	arc_normal.normalize();

	// Use plane intersection method matching ik_open_cone_3d.cpp exactly
	real_t boundary_plus_tangent_radius_a = p_radius1 + r_tangent_radius;
	real_t boundary_plus_tangent_radius_b = p_radius2 + r_tangent_radius;

	// The axis of this cone, scaled to minimize its distance to the tangent contact points
	Vector3 scaled_axis_a = center1 * Math::cos(boundary_plus_tangent_radius_a);
	// A point on the plane running through the tangent contact points
	Vector3 safe_arc_normal = arc_normal;
	if (Math::is_zero_approx(safe_arc_normal.length_squared())) {
		safe_arc_normal = Vector3(0, 1, 0);
	}
	Quaternion temp_var = get_quaternion_axis_angle(safe_arc_normal.normalized(), boundary_plus_tangent_radius_a);
	Vector3 plane_dir1_a = temp_var.xform(center1);
	// Another point on the same plane
	Vector3 safe_center1 = center1;
	if (Math::is_zero_approx(safe_center1.length_squared())) {
		safe_center1 = Vector3(0, 0, 1);
	}
	Quaternion temp_var2 = get_quaternion_axis_angle(safe_center1.normalized(), Math::PI / 2);
	Vector3 plane_dir2_a = temp_var2.xform(plane_dir1_a);

	Vector3 scaled_axis_b = center2 * Math::cos(boundary_plus_tangent_radius_b);
	// A point on the plane running through the tangent contact points
	Quaternion temp_var3 = get_quaternion_axis_angle(safe_arc_normal.normalized(), boundary_plus_tangent_radius_b);
	Vector3 plane_dir1_b = temp_var3.xform(center2);
	// Another point on the same plane
	Vector3 safe_center2 = center2;
	if (Math::is_zero_approx(safe_center2.length_squared())) {
		safe_center2 = Vector3(0, 0, 1);
	}
	Quaternion temp_var4 = get_quaternion_axis_angle(safe_center2.normalized(), Math::PI / 2);
	Vector3 plane_dir2_b = temp_var4.xform(plane_dir1_b);

	// Ray from scaled center of next cone to half way point between the circumference of this cone and the next cone
	Vector3 ray1_b_start = plane_dir1_b;
	Vector3 ray1_b_end = scaled_axis_b;
	Vector3 ray2_b_start = plane_dir1_b;
	Vector3 ray2_b_end = plane_dir2_b;

	extend_ray(ray1_b_start, ray1_b_end, 99.0);
	extend_ray(ray2_b_start, ray2_b_end, 99.0);

	Vector3 intersection1 = ray_plane_intersection(ray1_b_start, ray1_b_end, scaled_axis_a, plane_dir1_a, plane_dir2_a);
	Vector3 intersection2 = ray_plane_intersection(ray2_b_start, ray2_b_end, scaled_axis_a, plane_dir1_a, plane_dir2_a);

	Vector3 intersection_ray_start = intersection1;
	Vector3 intersection_ray_end = intersection2;
	extend_ray(intersection_ray_start, intersection_ray_end, 99.0);

	Vector3 sphere_intersect1;
	Vector3 sphere_intersect2;
	Vector3 sphere_center(0, 0, 0);
	ray_sphere_intersection(intersection_ray_start, intersection_ray_end, sphere_center, 1.0, &sphere_intersect1, &sphere_intersect2);

	r_tangent1 = sphere_intersect1.normalized();
	r_tangent2 = sphere_intersect2.normalized();

	// Handle degenerate tangent centers (NaN or zero) - matches many_bone_ik exactly
	if (!r_tangent1.is_finite() || Math::is_zero_approx(r_tangent1.length_squared())) {
		r_tangent1 = get_orthogonal(center1);
		if (Math::is_zero_approx(r_tangent1.length_squared())) {
			r_tangent1 = Vector3(0, 1, 0);
		}
		r_tangent1.normalize();
	}
	if (!r_tangent2.is_finite() || Math::is_zero_approx(r_tangent2.length_squared())) {
		Vector3 orthogonal_base = r_tangent1.is_finite() ? r_tangent1 : center1;
		r_tangent2 = get_orthogonal(orthogonal_base);
		if (Math::is_zero_approx(r_tangent2.length_squared())) {
			r_tangent2 = Vector3(1, 0, 0);
		}
		r_tangent2.normalize();
	}
}

// Helper function to find point on path between two cones (from IKLimitCone3D::get_on_great_tangent_triangle)
// Matches many_bone_ik algorithm exactly
static Vector3 get_on_great_tangent_triangle(const Vector3 &p_input, const Vector3 &p_center1, real_t p_radius1,
		const Vector3 &p_center2, real_t p_radius2) {
	Vector3 center1 = p_center1.normalized();
	Vector3 center2 = p_center2.normalized();
	Vector3 input = p_input.normalized();

	// Compute tangent circle
	Vector3 tan1, tan2;
	real_t tan_radius;
	compute_tangent_circle(center1, p_radius1, center2, p_radius2, tan1, tan2, tan_radius);

	real_t tan_radius_cos = Math::cos(tan_radius);

	// Use cross products to determine which side (matches many_bone_ik exactly)
	Vector3 c1xc2 = center1.cross(center2);
	real_t c1c2dir = input.dot(c1xc2);

	if (c1c2dir < 0.0) {
		// Use first tangent circle (tan1)
		Vector3 c1xt1 = center1.cross(tan1);
		Vector3 t1xc2 = tan1.cross(center2);
		if (input.dot(c1xt1) > 0 && input.dot(t1xc2) > 0) {
			real_t to_next_cos = input.dot(tan1);
			if (to_next_cos > tan_radius_cos) {
				// Project onto tangent circle
				Vector3 plane_normal = tan1.cross(input);
				// Ensure plane_normal is valid before creating quaternion
				if (!plane_normal.is_finite() || Math::is_zero_approx(plane_normal.length_squared())) {
					plane_normal = Vector3(0, 1, 0);
				}
				plane_normal.normalize();
				Quaternion rotate_about_by = get_quaternion_axis_angle(plane_normal, tan_radius);
				return rotate_about_by.xform(tan1).normalized();
			} else {
				return input;
			}
		} else {
			return Vector3(NAN, NAN, NAN);
		}
	} else {
		// Use second tangent circle (tan2)
		Vector3 t2xc1 = tan2.cross(center1);
		Vector3 c2xt2 = center2.cross(tan2);
		if (input.dot(t2xc1) > 0 && input.dot(c2xt2) > 0) {
			if (input.dot(tan2) > tan_radius_cos) {
				Vector3 plane_normal = tan2.cross(input);
				// Ensure plane_normal is valid before creating quaternion
				if (!plane_normal.is_finite() || Math::is_zero_approx(plane_normal.length_squared())) {
					plane_normal = Vector3(0, 1, 0);
				}
				plane_normal.normalize();
				Quaternion rotate_about_by = get_quaternion_axis_angle(plane_normal, tan_radius);
				return rotate_about_by.xform(tan2).normalized();
			} else {
				return input;
			}
		} else {
			return Vector3(NAN, NAN, NAN);
		}
	}
}

Vector3 JointLimitationKusudama3D::_solve(const Vector3 &p_direction) const {
	// If constraints are disabled, return the original direction
	if (!orientationally_constrained || cones.is_empty()) {
		return p_direction.normalized();
	}

	// Use the many_bone_ik algorithm (from IKKusudama3D::get_local_point_in_limits)
	Vector3 point = p_direction.normalized();
	real_t closest_cos = -2.0;
	Vector3 closest_collision_point = point;

	// Loop through each limit cone
	for (int i = 0; i < cones.size(); i++) {
		const Vector4 &cone_data = cones[i];
		Vector3 control_point = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t radius = cone_data.w;

		Vector3 collision_point = closest_to_cone_boundary(point, control_point, radius);

		// If the collision point is NaN, return the original point (point is in bounds)
		if (Math::is_nan(collision_point.x) || Math::is_nan(collision_point.y) || Math::is_nan(collision_point.z)) {
			return point;
		}

		// Calculate the cosine of the angle between the collision point and the original point
		real_t this_cos = collision_point.dot(point);

		// If the closest collision point is not set or the cosine is greater than the current closest cosine, update the closest collision point and cosine
		if (closest_collision_point.is_zero_approx() || this_cos > closest_cos) {
			closest_collision_point = collision_point;
			closest_cos = this_cos;
		}
	}

	// If we're out of bounds of all cones, check if we're in the paths between the cones
	// IMPORTANT: We explicitly do NOT check the pair (last_cone, first_cone) to prevent wrap-around
	if (cones.size() > 1) {
		for (int i = 0; i < cones.size() - 1; i++) {
			int next_i = i + 1; // Only connect to next adjacent cone, no wrap-around
			const Vector4 &cone1_data = cones[i];
			const Vector4 &cone2_data = cones[next_i];
			Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
			Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
			real_t radius1 = cone1_data.w;
			real_t radius2 = cone2_data.w;

			Vector3 collision_point = get_on_great_tangent_triangle(point, center1, radius1, center2, radius2);

			// If the collision point is NaN, skip to the next iteration
			if (Math::is_nan(collision_point.x)) {
				continue;
			}

			real_t this_cos = collision_point.dot(point);

			// If the cosine is approximately 1, return the original point (matches many_bone_ik)
			if (Math::is_equal_approx(this_cos, real_t(1.0))) {
				return point;
			}

			// If the cosine is greater than the current closest cosine, update the closest collision point and cosine
			if (this_cos > closest_cos) {
				closest_collision_point = collision_point;
				closest_cos = this_cos;
			}
		}
	}

	// Return the closest boundary point between cones
	return closest_collision_point.normalized();
}

void JointLimitationKusudama3D::set_cones(const Vector<Vector4> &p_cones) {
	cones = p_cones;
	emit_changed();
}

Vector<Vector4> JointLimitationKusudama3D::get_cones() const {
	return cones;
}

void JointLimitationKusudama3D::set_cone_count(int p_count) {
	if (p_count < 0) {
		p_count = 0;
	}
	int old_size = cones.size();
	if (old_size == p_count) {
		return;
	}
	cones.resize(p_count);
	// Initialize new cones with default values
	for (int i = old_size; i < cones.size(); i++) {
		cones.write[i] = Vector4(0, 1, 0, Math::PI * 0.25); // Default: +Y axis, 45 degree cone
	}
	notify_property_list_changed();
	emit_changed();
}

int JointLimitationKusudama3D::get_cone_count() const {
	return cones.size();
}

void JointLimitationKusudama3D::set_cone_center(int p_index, const Vector3 &p_center) {
	ERR_FAIL_INDEX(p_index, cones.size());
	// Store raw value (non-normalized) to allow editor to accept values outside [-1, 1]
	// Normalization happens lazily when values are used
	Vector4 &cone = cones.write[p_index];
	cone.x = p_center.x;
	cone.y = p_center.y;
	cone.z = p_center.z;
	emit_changed();
}

Vector3 JointLimitationKusudama3D::get_cone_center(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, cones.size(), Vector3(0, 1, 0));
	Vector3 center = Vector3(cones[p_index].x, cones[p_index].y, cones[p_index].z);
	// Normalize when reading to ensure we always return a normalized value
	if (center.length_squared() > CMP_EPSILON) {
		return center.normalized();
	}
	return Vector3(0, 1, 0);
}

void JointLimitationKusudama3D::set_cone_center_quaternion(int p_index, const Quaternion &p_quaternion) {
	ERR_FAIL_INDEX(p_index, cones.size());
	// Convert quaternion to direction vector by rotating the default direction (0, 1, 0)
	Vector3 default_dir = Vector3(0, 1, 0);
	Vector3 center = p_quaternion.normalized().xform(default_dir);
	// Store raw value (non-normalized) to allow editor to accept values outside [-1, 1]
	Vector4 &cone = cones.write[p_index];
	cone.x = center.x;
	cone.y = center.y;
	cone.z = center.z;
	emit_changed();
}

Quaternion JointLimitationKusudama3D::get_cone_center_quaternion(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, cones.size(), Quaternion());
	Vector3 center = get_cone_center(p_index); // This already normalizes
	Vector3 default_dir = Vector3(0, 1, 0);
	// Create quaternion representing rotation from default_dir to center
	return Quaternion(default_dir, center);
}

void JointLimitationKusudama3D::set_cone_radius(int p_index, real_t p_radius) {
	ERR_FAIL_INDEX(p_index, cones.size());
	cones.write[p_index].w = p_radius;
	emit_changed();
}

real_t JointLimitationKusudama3D::get_cone_radius(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, cones.size(), 0.0);
	return cones[p_index].w;
}

bool JointLimitationKusudama3D::_set(const StringName &p_name, const Variant &p_value) {
	String prop_name = p_name;
	if (prop_name == "cone_count") {
		set_cone_count(p_value);
		return true;
	}
	if (prop_name.begins_with("cones/")) {
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
	if (prop_name.begins_with("cones/")) {
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
	p_list->push_back(PropertyInfo(Variant::INT, PNAME("cone_count"), PROPERTY_HINT_RANGE, "0,16384,1,or_greater", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Cones," + String(PNAME("cones")) + "/"));
	for (int i = 0; i < get_cone_count(); i++) {
		const String prefix = vformat("%s/%d/", PNAME("cones"), i);
		// Use quaternion for inspector display with Euler angles in degrees
		p_list->push_back(PropertyInfo(Variant::QUATERNION, prefix + PNAME("center"), PROPERTY_HINT_NONE, ""));
		p_list->push_back(PropertyInfo(Variant::FLOAT, prefix + PNAME("radius"), PROPERTY_HINT_RANGE, "0,180,0.1,radians_as_degrees"));
	}
}

void JointLimitationKusudama3D::set_orientationally_constrained(bool p_constrained) {
	orientationally_constrained = p_constrained;
	emit_changed();
}

bool JointLimitationKusudama3D::is_orientationally_constrained() const {
	return orientationally_constrained;
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

Vector3 JointLimitationKusudama3D::solve(const Vector3 &p_local_forward_vector, const Vector3 &p_local_right_vector, const Quaternion &p_rotation_offset, const Vector3 &p_local_current_vector, const Quaternion &p_rotation, Quaternion *r_constrained_rotation) const {
	// First solve direction constraint using base implementation
	Vector3 constrained_dir = JointLimitation3D::solve(p_local_forward_vector, p_local_right_vector, p_rotation_offset, p_local_current_vector, p_rotation, r_constrained_rotation);

	// Apply twist constraints if enabled and output requested
	if (!r_constrained_rotation || !axially_constrained || range_angle >= Math::TAU) {
		return constrained_dir;
	}

	// Use make_space to transform rotation into constraint space
	// In constraint space, Y is forward (twist axis), X is right, Z is up
	Quaternion space = make_space(p_local_forward_vector, p_local_right_vector, p_rotation_offset);
	Quaternion rot = p_rotation.normalized();

	// Transform rotation into constraint space
	Quaternion constraint_space_rot = space.inverse() * rot * space;
	constraint_space_rot.normalize();

	// In constraint space, twist axis is Y (0, 1, 0)
	Vector3 twist_axis = Vector3(0, 1, 0);

	// Simple swing-twist decomposition in constraint space
	const Vector3 v(constraint_space_rot.x, constraint_space_rot.y, constraint_space_rot.z);
	const real_t proj_len = v.dot(twist_axis);
	const Vector3 twist_vec = twist_axis * proj_len;
	Quaternion twist(twist_vec.x, twist_vec.y, twist_vec.z, constraint_space_rot.w);

	if (!twist.is_normalized()) {
		if (Math::is_zero_approx(twist.length_squared())) {
			// No twist component, return as-is
			*r_constrained_rotation = p_rotation;
			return constrained_dir;
		}
		twist.normalize();
	}

	// Calculate swing as remaining rotation
	Quaternion swing = constraint_space_rot * twist.inverse();
	swing.normalize();

	// Use clamp_to_cos_half_angle approach similar to IKKusudama3D
	real_t normalized_min = Math::fposmod(min_axial_angle, (real_t)Math::TAU);
	real_t center_angle = normalized_min + range_angle * 0.5;
	real_t half_range = range_angle * 0.5;
	real_t twist_half_range_half_cos = Math::cos(half_range * 0.5);

	// Compute twist angle from the twist quaternion
	real_t twist_angle = 2.0 * Math::acos(CLAMP(Math::abs(twist.w), 0.0, 1.0));

	// Determine twist direction
	Vector3 twist_axis_from_quat = Vector3(twist.x, twist.y, twist.z);
	if (twist_axis_from_quat.length_squared() > CMP_EPSILON) {
		twist_axis_from_quat.normalize();
		if (twist_axis_from_quat.dot(twist_axis) < 0) {
			twist_angle = -twist_angle;
		}
	}

	// Normalize twist angle relative to center
	real_t twist_relative_to_center = Math::fposmod((real_t)(twist_angle - center_angle + Math::PI), (real_t)Math::TAU) - (real_t)Math::PI;

	// Clamp twist using cos(half_angle) method
	Quaternion twist_relative = Quaternion(twist_axis, twist_relative_to_center).normalized();

	Quaternion clamped_twist_relative = twist_relative;
	if (clamped_twist_relative.w < 0.0) {
		clamped_twist_relative = clamped_twist_relative * -1;
	}
	real_t previous_coefficient = (1.0 - (clamped_twist_relative.w * clamped_twist_relative.w));
	if (twist_half_range_half_cos > clamped_twist_relative.w && previous_coefficient > CMP_EPSILON) {
		real_t composite_coefficient = Math::sqrt((1.0 - (twist_half_range_half_cos * twist_half_range_half_cos)) / previous_coefficient);
		clamped_twist_relative.w = twist_half_range_half_cos;
		clamped_twist_relative.x *= composite_coefficient;
		clamped_twist_relative.y *= composite_coefficient;
		clamped_twist_relative.z *= composite_coefficient;
	}
	clamped_twist_relative.normalize();

	// Convert back to absolute twist
	Quaternion center_twist = Quaternion(twist_axis, center_angle).normalized();
	Quaternion clamped_twist = center_twist * clamped_twist_relative;
	clamped_twist.normalize();

	// Recompose rotation in constraint space: swing * clamped_twist
	Quaternion constrained_rot = swing * clamped_twist;
	constrained_rot.normalize();

	// Transform back from constraint space to original space
	*r_constrained_rotation = space * constrained_rot * space.inverse();
	r_constrained_rotation->normalize();

	return constrained_dir;
}

#ifdef TOOLS_ENABLED

void JointLimitationKusudama3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const {
	if (!is_orientationally_constrained() || cones.is_empty()) {
		return;
	}

	real_t sphere_r = p_bone_length * (real_t)0.25;
	if (sphere_r <= CMP_EPSILON) {
		return;
	}

	static const int N = 32; // Number of segments for rings
	static const real_t DP = Math::TAU / (real_t)N;

	// Draw forbidden area for each cone (similar to JointLimitationCone3D)
	for (int i = 0; i < cones.size(); i++) {
		const Vector4 &cone_data = cones[i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t radius = cone_data.w;

		// Find two perpendicular vectors to the cone center
		Vector3 perp1, perp2;
		if (Math::abs(center.y) < 0.9f) {
			perp1 = Vector3(0, 1, 0).cross(center).normalized();
		} else {
			perp1 = Vector3(1, 0, 0).cross(center).normalized();
		}
		perp2 = center.cross(perp1).normalized();

		// Calculate boundary ring position (forbidden area edge)
		real_t boundary_y = sphere_r * Math::cos(radius);
		real_t boundary_r = sphere_r * Math::sin(radius);

		// Draw center ring (small ring at the center of the cone)
		// This is a small ring near the center to show the cone center direction
		// Use a fixed small angle instead of varying with cone radius to prevent shifting
		real_t center_ring_radius = (real_t)0.05; // Fixed 0.05 radians (~2.86 degrees)
		real_t center_ring_y = sphere_r * Math::cos(center_ring_radius);
		real_t center_ring_r = sphere_r * Math::sin(center_ring_radius);
		if (center_ring_r > CMP_EPSILON) {
			for (int j = 0; j < N; j++) {
				real_t angle0 = (real_t)j * DP;
				real_t angle1 = (real_t)((j + 1) % N) * DP;

				Vector3 dir0 = (perp1 * Math::cos(angle0) + perp2 * Math::sin(angle0)).normalized();
				Vector3 dir1 = (perp1 * Math::cos(angle1) + perp2 * Math::sin(angle1)).normalized();

				Vector3 p0 = center * center_ring_y + dir0 * center_ring_r;
				Vector3 p1 = center * center_ring_y + dir1 * center_ring_r;

				p_surface_tool->set_color(p_color);
				p_surface_tool->add_vertex(p_transform.xform(p0));
				p_surface_tool->set_color(p_color);
				p_surface_tool->add_vertex(p_transform.xform(p1));
			}
		}

		// Draw boundary ring (edge of allowed area - marks where forbidden area begins)
		// This shows the boundary of the open cone area
		if (boundary_r > CMP_EPSILON) {
			for (int j = 0; j < N; j++) {
				real_t angle0 = (real_t)j * DP;
				real_t angle1 = (real_t)((j + 1) % N) * DP;

				Vector3 dir0 = (perp1 * Math::cos(angle0) + perp2 * Math::sin(angle0)).normalized();
				Vector3 dir1 = (perp1 * Math::cos(angle1) + perp2 * Math::sin(angle1)).normalized();

				Vector3 p0 = center * boundary_y + dir0 * boundary_r;
				Vector3 p1 = center * boundary_y + dir1 * boundary_r;

				p_surface_tool->set_color(p_color);
				p_surface_tool->add_vertex(p_transform.xform(p0));
				p_surface_tool->set_color(p_color);
				p_surface_tool->add_vertex(p_transform.xform(p1));
			}
		}
	}
}

#endif // TOOLS_ENABLED
