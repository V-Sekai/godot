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
#include "editor/scene/3d/kusudama_shader.h"
#include "scene/resources/material.h"
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

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "cones", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_cones", "get_cones");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "orientationally_constrained"), "set_orientationally_constrained", "is_orientationally_constrained");
}

// Helper function to find closest point on a single cone boundary
static Vector3 closest_to_cone_boundary(const Vector3 &p_input, const Vector3 &p_control_point, real_t p_radius) {
	Vector3 normalized_input = p_input.normalized();
	Vector3 normalized_control = p_control_point.normalized();

	// For zero radius, return the control point exactly (or NaN if input is the control point)
	if (p_radius < CMP_EPSILON) {
		real_t input_dot_control = normalized_input.dot(normalized_control);
		if (input_dot_control >= 1.0 - 1e-4) {
			// Input is the control point (or very close) - inside cone
			return Vector3(NAN, NAN, NAN);
		}
		// Input is not the control point - return control point as boundary
		return normalized_control;
	}

	real_t radius_cosine = Math::cos(p_radius);
	real_t input_dot_control = normalized_input.dot(normalized_control);

	// Check if input is within the cone
	// Use a small epsilon to account for floating-point precision
	// Points on or very close to the boundary are considered inside
	// Note: input_dot_control = cos(angle), so larger values mean smaller angles (closer to center)
	// Use a slightly larger epsilon (1e-4) to account for normalization and coordinate space transformations
	if (input_dot_control >= radius_cosine - 1e-4) {
		// Inside cone - return NaN to indicate in bounds
		return Vector3(NAN, NAN, NAN);
	}

	// Find the closest point on the cone boundary to the input point
	// The boundary is a circle on the unit sphere at angle p_radius from the control point
	// We need to find the point on this circle that is closest to the input point

	// Project the input point onto the plane perpendicular to the control point
	// Then normalize to the cone radius distance
	Vector3 projection = normalized_input - normalized_control * input_dot_control;
	real_t projection_length = projection.length();

	if (projection_length < CMP_EPSILON) {
		// Input is opposite to control point - use any perpendicular
		projection = normalized_control.get_any_perpendicular();
		if (projection.is_zero_approx()) {
			projection = Vector3(0, 1, 0);
		}
		projection_length = projection.length();
	}
	projection.normalize();

	// Calculate boundary point slightly inside the cone to ensure it passes the "inside" check
	// This "snaps" the point to be guaranteed inside the allowed region
	// Use a small adjustment (1e-4 radians ≈ 0.0057 degrees) to move slightly inside the boundary
	real_t adjustment = 1e-4;
	real_t adjusted_radius = MAX(0.0, p_radius - adjustment);
	real_t adjusted_radius_cosine = Math::cos(adjusted_radius);
	real_t sin_adjusted_radius = Math::sin(adjusted_radius);
	Vector3 result = normalized_control * adjusted_radius_cosine + projection * sin_adjusted_radius;
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
// Uses plane intersection method matching the actual implementation in ik_open_cone_3d.cpp
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

	if (arc_normal_len < CMP_EPSILON) {
		// Cones are parallel or opposite - handle specially
		// For opposite cones, any perpendicular to center1 works
		arc_normal = center1.get_any_perpendicular();
		if (arc_normal.is_zero_approx()) {
			arc_normal = Vector3(0, 1, 0);
		}
		arc_normal.normalize();

		// For opposite cones, tangent circles are at 90 degrees from the cone centers
		// Use a perpendicular vector in the plane perpendicular to center1
		Vector3 perp1 = center1.get_any_perpendicular().normalized();

		// Rotate around center1 by the tangent radius to get tangent centers
		Quaternion rot1 = Quaternion(center1, r_tangent_radius);
		Quaternion rot2 = Quaternion(center1, -r_tangent_radius);
		r_tangent1 = rot1.xform(perp1).normalized();
		r_tangent2 = rot2.xform(perp1).normalized();
		return;
	}
	arc_normal.normalize();

	// Use plane intersection method matching ik_open_cone_3d.cpp
	real_t boundary_plus_tangent_radius_a = p_radius1 + r_tangent_radius;
	real_t boundary_plus_tangent_radius_b = p_radius2 + r_tangent_radius;

	// The axis of this cone, scaled to minimize its distance to the tangent contact points
	Vector3 scaled_axis_a = center1 * Math::cos(boundary_plus_tangent_radius_a);
	// A point on the plane running through the tangent contact points
	Vector3 safe_arc_normal = arc_normal;
	if (Math::is_zero_approx(safe_arc_normal.length_squared())) {
		safe_arc_normal = Vector3(0, 1, 0);
	}
	Quaternion temp_var = Quaternion(safe_arc_normal.normalized(), boundary_plus_tangent_radius_a);
	Vector3 plane_dir1_a = temp_var.xform(center1);
	// Another point on the same plane
	Vector3 safe_center1 = center1;
	if (Math::is_zero_approx(safe_center1.length_squared())) {
		safe_center1 = Vector3(0, 0, 1);
	}
	Quaternion temp_var2 = Quaternion(safe_center1.normalized(), Math::PI / 2);
	Vector3 plane_dir2_a = temp_var2.xform(plane_dir1_a);

	Vector3 scaled_axis_b = center2 * Math::cos(boundary_plus_tangent_radius_b);
	// A point on the plane running through the tangent contact points
	Quaternion temp_var3 = Quaternion(safe_arc_normal.normalized(), boundary_plus_tangent_radius_b);
	Vector3 plane_dir1_b = temp_var3.xform(center2);
	// Another point on the same plane
	Vector3 safe_center2 = center2;
	if (Math::is_zero_approx(safe_center2.length_squared())) {
		safe_center2 = Vector3(0, 0, 1);
	}
	Quaternion temp_var4 = Quaternion(safe_center2.normalized(), Math::PI / 2);
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

	// Handle degenerate tangent centers (NaN or zero)
	if (!r_tangent1.is_finite() || Math::is_zero_approx(r_tangent1.length_squared())) {
		r_tangent1 = center1.get_any_perpendicular();
		if (Math::is_zero_approx(r_tangent1.length_squared())) {
			r_tangent1 = Vector3(0, 1, 0);
		}
		r_tangent1.normalize();
	}
	if (!r_tangent2.is_finite() || Math::is_zero_approx(r_tangent2.length_squared())) {
		Vector3 orthogonal_base = r_tangent1.is_finite() ? r_tangent1 : center1;
		r_tangent2 = orthogonal_base.get_any_perpendicular();
		if (Math::is_zero_approx(r_tangent2.length_squared())) {
			r_tangent2 = Vector3(1, 0, 0);
		}
		r_tangent2.normalize();
	}
}

// Helper function to find point on path between two cones
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

	// Determine which side of the arc we're on
	Vector3 arc_normal = center1.cross(center2);
	real_t arc_side_dot = input.dot(arc_normal);

	if (arc_side_dot < 0.0) {
		// Use first tangent circle
		Vector3 cone1_cross_tangent1 = center1.cross(tan1);
		Vector3 tangent1_cross_cone2 = tan1.cross(center2);
		if (input.dot(cone1_cross_tangent1) > 0 && input.dot(tangent1_cross_cone2) > 0) {
			real_t to_next_cos = input.dot(tan1);
			if (to_next_cos > tan_radius_cos) {
				// Project onto tangent circle, but move slightly outside to ensure it's in the allowed region
				Vector3 plane_normal = tan1.cross(input);
				if (plane_normal.is_zero_approx() || !plane_normal.is_finite()) {
					plane_normal = Vector3(0, 1, 0);
				}
				plane_normal.normalize();
				// Snap point to be slightly outside the tangent circle (into allowed region)
				// Points with angle > tan_radius are outside (allowed), points with angle < tan_radius are inside (forbidden)
				// Use small adjustment (1e-4 radians) to ensure it's in allowed region
				real_t adjusted_tan_radius = tan_radius + 1e-4;
				Quaternion rotate_about_by = Quaternion(plane_normal, adjusted_tan_radius);
				return rotate_about_by.xform(tan1).normalized();
			} else {
				return input;
			}
		}
	} else {
		// Use second tangent circle
		Vector3 tangent2_cross_cone1 = tan2.cross(center1);
		Vector3 cone2_cross_tangent2 = center2.cross(tan2);
		if (input.dot(tangent2_cross_cone1) > 0 && input.dot(cone2_cross_tangent2) > 0) {
			real_t to_next_cos = input.dot(tan2);
			if (to_next_cos > tan_radius_cos) {
				// Project onto tangent circle, but move slightly outside to ensure it's in the allowed region
				Vector3 plane_normal = tan2.cross(input);
				if (plane_normal.is_zero_approx() || !plane_normal.is_finite()) {
					plane_normal = Vector3(0, 1, 0);
				}
				plane_normal.normalize();
				// Snap point to be slightly outside the tangent circle (into allowed region)
				// Points with angle > tan_radius are outside (allowed), points with angle < tan_radius are inside (forbidden)
				// Use small adjustment (1e-4 radians) to ensure it's in allowed region
				real_t adjusted_tan_radius = tan_radius + 1e-4;
				Quaternion rotate_about_by = Quaternion(plane_normal, adjusted_tan_radius);
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
	if (orientationally_constrained && !cones.is_empty()) {
		// Full kusudama solving implementation based on IKKusudama3D::get_local_point_in_limits
		Vector3 point = result;
		real_t closest_cosine = -2.0;
		Vector3 closest_collision_point = point;
		bool in_bounds = false;

		// Loop through each limit cone
		for (int i = 0; i < cones.size(); i++) {
			const Vector4 &cone_data = cones[i];
			Vector3 control_point = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
			real_t radius = cone_data.w;

			Vector3 collision_point = closest_to_cone_boundary(point, control_point, radius);

			// If NaN, point is within this cone
			if (Math::is_nan(collision_point.x) || Math::is_nan(collision_point.y) || Math::is_nan(collision_point.z)) {
				in_bounds = true;
				return point; // Point is within limits
			}

			// Calculate cosine of angle between collision point and original point
			real_t cosine = collision_point.dot(point);

			// Update closest collision point if this one is closer
			if (closest_collision_point.is_zero_approx() || cosine > closest_cosine) {
				closest_collision_point = collision_point;
				closest_cosine = cosine;
			}
		}

		// If we're out of bounds of all cones, check if we're in the paths between the cones
		// IMPORTANT: We explicitly do NOT check the pair (last_cone, first_cone) to prevent wrap-around
		if (!in_bounds && cones.size() > 1) {
			for (int i = 0; i < cones.size() - 1; i++) {
				int next_i = i + 1; // Only connect to next adjacent cone, no wrap-around
				// Assert: next_i must be valid and greater than i, ensuring no wrap-around
				ERR_FAIL_COND_V_MSG(next_i >= cones.size() || next_i <= i, result, "Invalid cone pair in _solve - possible wrap-around");
				const Vector4 &cone1_data = cones[i];
				const Vector4 &cone2_data = cones[next_i];
				Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
				Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
				real_t radius1 = cone1_data.w;
				real_t radius2 = cone2_data.w;

				Vector3 collision_point = get_on_great_tangent_triangle(point, center1, radius1, center2, radius2);

				// If NaN, skip this path
				if (Math::is_nan(collision_point.x)) {
					continue;
				}

				// If the returned point is approximately equal to the input point, point is in the path region
				real_t cosine = collision_point.dot(point);
				if (cosine > 0.999f) { // Point is in path region (allowing for floating point precision)
					// get_on_great_tangent_triangle already handles the geometric checks correctly using cross products
					// to determine if the point is in the inter-cone path region, so no additional check is needed
					return point;
				}

				// Update closest collision point if this one is closer
				if (cosine > closest_cosine) {
					closest_collision_point = collision_point;
					closest_cosine = cosine;
				}
			}
		}

		// Return the closest boundary point
		// The boundary calculation functions (closest_to_cone_boundary and get_on_great_tangent_triangle)
		// already ensure the result is in an allowed region by adjusting slightly inside/outside boundaries
		result = closest_collision_point.normalized();
	}

	return result;
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

#ifdef TOOLS_ENABLED

void JointLimitationKusudama3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const {
	// Twist limits visualization has been removed
	// Swing limits (orientation constraints) are handled by the shader-based visualization
	// This method is kept for API compatibility but no longer draws anything
}

void JointLimitationKusudama3D::draw_triangle_mesh(const Transform3D &p_transform, float p_bone_length, const Color &p_color, const PackedInt32Array &p_bones, const PackedFloat32Array &p_weights, Ref<Mesh> &r_mesh, Ref<Material> &r_material) const {
	r_mesh = Ref<Mesh>();
	r_material = Ref<Material>();

	if (!is_orientationally_constrained() || cones.is_empty()) {
		return;
	}

	real_t socket_r = p_bone_length * 0.25f;
	if (socket_r <= CMP_EPSILON) {
		return;
	}

	// Build cone sequence for shader (cone1, tangent1, tangent2, cone2, ...)
	PackedFloat32Array cone_sequence;

	// Build cone sequence
	if (cones.size() == 1) {
		// Single cone
		const Vector4 &cone_data = cones[0];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t radius = cone_data.w;
		cone_sequence.append(center.x);
		cone_sequence.append(center.y);
		cone_sequence.append(center.z);
		cone_sequence.append(radius);
	} else {
		// Multiple cones - need tangent circles
		// Shader expects: [cone1, tan1, tan2, cone2, tan1, tan2, cone3, ...]
		// For each iteration i, shader reads: [i+0, i+1, i+2, i+3] = [cone1, tan1, tan2, cone2]
		// Next iteration starts at i+3 (which is cone2)
		for (int i = 0; i < cones.size() - 1; i++) {
			const Vector4 &cone1_data = cones[i];
			const Vector4 &cone2_data = cones[i + 1];
			Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
			Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
			real_t radius1 = cone1_data.w;
			real_t radius2 = cone2_data.w;

			// Cone 1 (only output on first iteration to avoid duplication)
			if (i == 0) {
				cone_sequence.append(center1.x);
				cone_sequence.append(center1.y);
				cone_sequence.append(center1.z);
				cone_sequence.append(radius1);
			}

			// Tangent circles between cone1 and cone2
			Vector3 tan1, tan2;
			real_t tan_radius;
			compute_tangent_circle(center1, radius1, center2, radius2, tan1, tan2, tan_radius);

			// Ensure tan1 and tan2 are ordered correctly based on arc direction
			// Solver uses: if (input.dot(center1.cross(center2)) < 0.0) use tan1, else use tan2
			// Shader uses: if (dot(normal_dir, cross(cone_1, cone_2)) < 0.0) use tangent_1, else use tangent_2
			// The shader reads: tangent_1 = cone_sequence[i+1], tangent_2 = cone_sequence[i+2]
			// Since swapping made no visible difference, the issue may be in compute_tangent_circle's return order
			// Try swapping the append order: put tan2 first (becomes tangent_1) and tan1 second (becomes tangent_2)
			cone_sequence.append(tan2.x);
			cone_sequence.append(tan2.y);
			cone_sequence.append(tan2.z);
			cone_sequence.append(tan_radius);

			cone_sequence.append(tan1.x);
			cone_sequence.append(tan1.y);
			cone_sequence.append(tan1.z);
			cone_sequence.append(tan_radius);

			// Cone 2 (always output, becomes cone1 for next iteration)
			cone_sequence.append(center2.x);
			cone_sequence.append(center2.y);
			cone_sequence.append(center2.z);
			cone_sequence.append(radius2);
		}
	}

	// Create sphere mesh for shader visualization using triangles
	int rings = 8;
	int i = 0, j = 0, prevrow = 0, thisrow = 0, point = 0;
	float x, y, z;

	Vector<Vector3> points;
	Vector<Vector3> normals;
	Vector<int> indices;
	point = 0;

	thisrow = 0;
	prevrow = 0;
	for (j = 0; j <= (rings + 1); j++) {
		int radial_segments = 8;
		float v = j;
		float w;

		v /= (rings + 1);
		w = sin(Math::PI * v);
		y = cos(Math::PI * v);

		for (i = 0; i <= radial_segments; i++) {
			float u = i;
			u /= radial_segments;

			x = sin(u * Math::TAU);
			z = cos(u * Math::TAU);

			Vector3 p = Vector3(x * w, y, z * w) * socket_r;
			points.push_back(p);
			Vector3 normal = Vector3(x * w, y, z * w);
			normals.push_back(normal.normalized());
			point++;

			if (i > 0 && j > 0) {
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);

				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			}
		}

		prevrow = thisrow;
		thisrow = point;
	}
	if (!indices.size()) {
		return;
	}

	Ref<SurfaceTool> surface_tool;
	surface_tool.instantiate();
	surface_tool->begin(Mesh::PRIMITIVE_TRIANGLES);
	const int32_t MESH_CUSTOM_0 = 0;
	surface_tool->set_custom_format(MESH_CUSTOM_0, SurfaceTool::CustomFormat::CUSTOM_RGBA_HALF);

	Vector<int> bones_vec = static_cast<Vector<int>>(p_bones);
	Vector<float> weights_vec = static_cast<Vector<float>>(p_weights);
	for (int32_t point_i = 0; point_i < points.size(); point_i++) {
		surface_tool->set_bones(bones_vec);
		surface_tool->set_weights(weights_vec);
		Color c;
		c.r = normals[point_i].x;
		c.g = normals[point_i].y;
		c.b = normals[point_i].z;
		c.a = 0;
		surface_tool->set_custom(MESH_CUSTOM_0, c);
		surface_tool->set_normal(normals[point_i]);
		// Transform vertices to p_transform space (same as draw_shape does)
		// This puts vertices in skeleton global space, matching the wireframe mesh
		surface_tool->add_vertex(p_transform.xform(points[point_i]));
	}
	for (int32_t index_i : indices) {
		surface_tool->add_index(index_i);
	}

	r_mesh = surface_tool->commit(Ref<Mesh>(), RS::ARRAY_CUSTOM_RGBA_HALF << RS::ARRAY_FORMAT_CUSTOM0_SHIFT);

	// Create shader material
	static Ref<Shader> kusudama_shader;
	if (!kusudama_shader.is_valid()) {
		kusudama_shader.instantiate();
		kusudama_shader->set_code(KUSUDAMA_SHADER);
	}

	Ref<ShaderMaterial> kusudama_material;
	kusudama_material.instantiate();
	kusudama_material->set_shader(kusudama_shader);
	kusudama_material->set_shader_parameter("cone_sequence", cone_sequence);
	int32_t cone_count = cones.size();
	kusudama_material->set_shader_parameter("cone_count", cone_count);
	kusudama_material->set_shader_parameter("kusudama_color", p_color);

	r_material = kusudama_material;
}

#endif // TOOLS_ENABLED
