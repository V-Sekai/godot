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
#include "core/io/logger.h"
#include "core/variant/variant.h"

void JointLimitationKusudama3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cones", "cones"), &JointLimitationKusudama3D::set_cones);
	ClassDB::bind_method(D_METHOD("get_cones"), &JointLimitationKusudama3D::get_cones);

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

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "cones", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_cones", "get_cones");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_axial_angle", PROPERTY_HINT_RANGE, "-360,360,0.1,radians_as_degrees"), "set_min_axial_angle", "get_min_axial_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "range_angle", PROPERTY_HINT_RANGE, "0,360,0.1,radians_as_degrees"), "set_range_angle", "get_range_angle");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "axially_constrained"), "set_axially_constrained", "is_axially_constrained");
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
	int intersection_count = ray_sphere_intersection(intersection_ray_start, intersection_ray_end, sphere_center, 1.0, &sphere_intersect1, &sphere_intersect2);

	// Ensure intersections are on the unit sphere (normalize them)
	// The ray-sphere intersection may return points that are not exactly on the unit sphere due to numerical precision
	sphere_intersect1 = sphere_intersect1.normalized();
	sphere_intersect2 = sphere_intersect2.normalized();

	// Check if the two intersections are too close (degenerate case - ray passes through or very close to origin)
	// If they're nearly identical after normalization, compute the second tangent as the point on the opposite side of the arc
	real_t dot_between = sphere_intersect1.dot(sphere_intersect2);
	if (dot_between > 0.999f) {
		// The two intersections are nearly identical - this happens when the ray passes very close to the origin
		// The two tangent circles should be on opposite sides of the arc between the two cones
		// Compute the second tangent by reflecting the first across the plane defined by the arc normal
		Vector3 arc_normal = center1.cross(center2);
		if (arc_normal.length_squared() < CMP_EPSILON) {
			// Cones are parallel/opposite - use a perpendicular to center1
			arc_normal = center1.get_any_perpendicular();
			if (arc_normal.is_zero_approx()) {
				arc_normal = Vector3(0, 1, 0);
			}
		}
		arc_normal.normalize();
		
		// Reflect sphere_intersect1 across the plane perpendicular to arc_normal
		// This gives us the point on the opposite side of the arc
		real_t dot_with_normal = sphere_intersect1.dot(arc_normal);
		sphere_intersect2 = (sphere_intersect1 - 2.0 * dot_with_normal * arc_normal).normalized();
		
		// Ensure they're actually different (should have dot product close to -1 if arc_normal is correct)
		real_t new_dot = sphere_intersect1.dot(sphere_intersect2);
		if (new_dot > 0.999f) {
			// Still too close - try the opposite approach: rotate 180 degrees around arc_normal
			Quaternion rot = Quaternion(arc_normal, Math::PI);
			sphere_intersect2 = rot.xform(sphere_intersect1).normalized();
		}
	}

	// Debug: Check if intersections are too close (degenerate case)
	static int debug_count = 0;
	debug_count++;
	if (debug_count % 1000 == 0) {
		real_t dist = sphere_intersect1.distance_to(sphere_intersect2);
		real_t dot = sphere_intersect1.dot(sphere_intersect2);
		print_verbose(vformat("compute_tangent_circle: intersection_count=%d dist_between_tangents=%.6f dot=%.6f intersect1=(%.6f,%.6f,%.6f) intersect2=(%.6f,%.6f,%.6f) ray_start=(%.3f,%.3f,%.3f) ray_end=(%.3f,%.3f,%.3f)",
			intersection_count, dist, dot,
			sphere_intersect1.x, sphere_intersect1.y, sphere_intersect1.z,
			sphere_intersect2.x, sphere_intersect2.y, sphere_intersect2.z,
			intersection_ray_start.x, intersection_ray_start.y, intersection_ray_start.z,
			intersection_ray_end.x, intersection_ray_end.y, intersection_ray_end.z));
	}

	r_tangent1 = sphere_intersect1;
	r_tangent2 = sphere_intersect2;

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

// Helper function to check a specific tangent side for path detection
// The cross product order depends on which side of the arc we're on (coordinate system convention)
static Vector3 check_tangent_side(const Vector3 &p_input, const Vector3 &p_tangent, const Vector3 &p_cone1, const Vector3 &p_cone2, real_t p_tan_radius_cos, real_t p_tan_radius, bool p_use_tan1_order) {
	Vector3 cross1, cross2;
	if (p_use_tan1_order) {
		// For tan1 side: center1.cross(tan1) and tan1.cross(center2)
		cross1 = p_cone1.cross(p_tangent);
		cross2 = p_tangent.cross(p_cone2);
	} else {
		// For tan2 side: tan2.cross(center1) and center2.cross(tan2) - order is reversed
		cross1 = p_tangent.cross(p_cone1);
		cross2 = p_cone2.cross(p_tangent);
	}
	if (p_input.dot(cross1) > 0 && p_input.dot(cross2) > 0) {
		real_t to_next_cos = p_input.dot(p_tangent);
		if (to_next_cos > p_tan_radius_cos) {
			// Project onto tangent circle, but move slightly outside to ensure it's in the allowed region
			Vector3 plane_normal = p_tangent.cross(p_input);
			if (plane_normal.is_zero_approx() || !plane_normal.is_finite()) {
				plane_normal = Vector3(0, 1, 0);
			}
			plane_normal.normalize();
			// Use slightly larger angle to move point outside the tangent circle (into allowed region)
			// Points with angle > tan_radius are outside (allowed), points with angle < tan_radius are inside (forbidden)
			// Use minimal adjustment (5e-5 radians) to ensure it's in allowed region without moving too far
			real_t adjusted_tan_radius = p_tan_radius + 5e-5;
			Quaternion rotate_about_by = Quaternion(plane_normal, adjusted_tan_radius);
			return rotate_about_by.xform(p_tangent).normalized();
		} else {
			return p_input;
		}
	}
	return Vector3(NAN, NAN, NAN);
}

// Helper function to find point on path between two cones
// Uses the simpler test version that avoids origin-crossing issues
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

	// Use the same logic as the test version - check only the side the point is on
	// This matches the test implementation which doesn't have special opposite-tangent handling
	// IMPORTANT: Cross product order depends on coordinate system convention
	if (arc_side_dot < 0.0) {
		// Use first tangent circle - use tan1 cross product order
		Vector3 result = check_tangent_side(input, tan1, center1, center2, tan_radius_cos, tan_radius, true);
		if (!Math::is_nan(result.x)) {
			return result;
		}
	} else {
		// Use second tangent circle - use tan2 cross product order (reversed)
		Vector3 result = check_tangent_side(input, tan2, center1, center2, tan_radius_cos, tan_radius, false);
		if (!Math::is_nan(result.x)) {
			return result;
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
// CSG boolean approach: Start with closed sphere, subtract cones and inter-cone paths
// 1. Generate full closed sphere mesh
// 2. For each cone: subtract spherical cap (center + radius)
// 3. For each inter-cone path: subtract path region (bounded by tangent circles)
// Result: forbidden region mesh
static void generate_forbidden_region_mesh(const Vector<Vector4> &p_cones, real_t p_sphere_r,
		int p_radial_segments, int p_rings, LocalVector<Vector3> &r_vertices, LocalVector<int> &r_indices) {
	// Step 1: Generate a full closed sphere mesh
	LocalVector<Vector3> sphere_vertices;
	LocalVector<int> sphere_indices;
	
	// Generate sphere vertices
	sphere_vertices.reserve((p_rings + 1) * (p_radial_segments + 1));
	for (int ring = 0; ring <= p_rings; ring++) {
		real_t v = (real_t)ring / (real_t)p_rings;
		real_t w = Math::sin(Math::PI * v);
		real_t y = Math::cos(Math::PI * v);
		
		for (int seg = 0; seg <= p_radial_segments; seg++) {
			real_t u = (real_t)seg / (real_t)p_radial_segments;
			real_t x = Math::sin(u * Math::TAU) * w;
			real_t z = Math::cos(u * Math::TAU) * w;
			sphere_vertices.push_back(Vector3(x, y, z) * p_sphere_r);
		}
	}
	
	// Generate sphere triangle indices
	for (int ring = 0; ring < p_rings; ring++) {
		int ring_start = ring * (p_radial_segments + 1);
		int next_ring_start = (ring + 1) * (p_radial_segments + 1);
		
		for (int seg = 0; seg < p_radial_segments; seg++) {
			int i0 = ring_start + seg;
			int i1 = ring_start + ((seg + 1) % (p_radial_segments + 1));
			int i2 = next_ring_start + seg;
			int i3 = next_ring_start + ((seg + 1) % (p_radial_segments + 1));
			
			// Two triangles per quad (clockwise winding)
			sphere_indices.push_back(i0);
			sphere_indices.push_back(i2);
			sphere_indices.push_back(i1);
			
			sphere_indices.push_back(i1);
			sphere_indices.push_back(i2);
			sphere_indices.push_back(i3);
		}
	}
	
	// Step 2 & 3: CSG subtraction - remove triangles that are in allowed regions
	// For each triangle, check if it's inside any cone or inter-cone path
	// If inside any allowed region, subtract it (don't add to forbidden mesh)
	for (int tri = 0; tri < sphere_indices.size(); tri += 3) {
		int i0 = sphere_indices[tri];
		int i1 = sphere_indices[tri + 1];
		int i2 = sphere_indices[tri + 2];
		
		Vector3 v0 = sphere_vertices[i0];
		Vector3 v1 = sphere_vertices[i1];
		Vector3 v2 = sphere_vertices[i2];
		
		// Normalize vertices to sphere surface
		Vector3 n0 = v0.normalized();
		Vector3 n1 = v1.normalized();
		Vector3 n2 = v2.normalized();
		
		// Check triangle center and all three vertices - if ANY is in allowed region, subtract the triangle
		// Use the solver's _solve method directly to ensure exact match with solver behavior
		// Create a temporary limitation object to use _solve
		// Note: We need to check if _solve returns the input unchanged, which means point is in allowed region
		Vector3 center = (v0 + v1 + v2) / 3.0;
		Vector3 center_normalized = center.normalized();
		
		bool triangle_in_allowed = false;
		
		// Check center and all vertices using the exact same logic as test_is_point_allowed
		// This ensures the CSG mesh matches the solver exactly
		Vector3 points_to_check[4] = { center_normalized, n0, n1, n2 };
		for (int p_idx = 0; p_idx < 4; p_idx++) {
			Vector3 vertex = points_to_check[p_idx];
			
			// Use the exact same logic as test_is_point_allowed (which matches _solve)
			// Check cones first - use direct dot product check to match test exactly
			bool in_bounds = false;
			for (int i = 0; i < p_cones.size(); i++) {
				const Vector4 &cone_data = p_cones[i];
				Vector3 control_point = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
				real_t radius = cone_data.w;
				
				// Direct check matching test_is_point_in_cone exactly
				real_t radius_cosine = Math::cos(radius);
				real_t input_dot_control = vertex.dot(control_point);
				// Use same epsilon as solver (1e-4) to ensure visualization matches solver exactly
				// This matches closest_to_cone_boundary and test_is_point_in_cone
				if (input_dot_control >= radius_cosine - 1e-4) {
					in_bounds = true;
					break;
				}
			}
			
			// If not in any cone, check paths (exact same as _solve)
			if (!in_bounds && p_cones.size() > 1) {
				for (int i = 0; i < p_cones.size() - 1; i++) {
					const Vector4 &cone1_data = p_cones[i];
					const Vector4 &cone2_data = p_cones[i + 1];
					Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
					Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
					real_t radius1 = cone1_data.w;
					real_t radius2 = cone2_data.w;
					
					Vector3 collision_point = get_on_great_tangent_triangle(vertex, center1, radius1, center2, radius2);
					if (!Math::is_nan(collision_point.x)) {
						real_t cosine = collision_point.dot(vertex);
						if (cosine > 0.999f) {
							in_bounds = true;
							break;
						}
					}
				}
			}
			
			// If point is in allowed region, mark triangle as in allowed region
			if (in_bounds) {
				triangle_in_allowed = true;
				break;
			}
		}
		
		// If triangle is NOT in any allowed region (cone or path), keep it (it's in forbidden region)
		if (!triangle_in_allowed) {
			// Add triangle vertices
			int idx0 = r_vertices.size();
			r_vertices.push_back(v0);
			int idx1 = r_vertices.size();
			r_vertices.push_back(v1);
			int idx2 = r_vertices.size();
			r_vertices.push_back(v2);
			
			// Add triangle indices
			r_indices.push_back(idx0);
			r_indices.push_back(idx1);
			r_indices.push_back(idx2);
		}
	}
}

// Helper to draw a circle on the sphere given center, radius angle, and sphere radius
// Uses spherical interpolation for smooth curve fitting
static void draw_cone_circle(LocalVector<Vector3> &r_vertices, const Vector3 &p_center, real_t p_radius_angle, real_t p_sphere_r, int p_segments = 16) {
	Vector3 axis = p_center.normalized();
	Vector3 perp1 = axis.get_any_perpendicular().normalized();

	// Generate circle points on the sphere using spherical interpolation
	Vector3 start_point = Quaternion(perp1, p_radius_angle).xform(axis).normalized();
	real_t angle_step = Math::TAU / (real_t)p_segments;

	Vector3 prev_point = start_point * p_sphere_r;
	for (int i = 1; i <= p_segments; i++) {
		real_t angle = (real_t)i * angle_step;
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
				r_vertices.push_back(prev_point);
			}
			r_vertices.push_back(interp);
		}

		prev_point = current_point;
	}
}

// Helper function to draw an arc along a circle on the sphere between two points
// The circle is defined by its center (axis) and the points lie on the circle
static void draw_arc_between_points(LocalVector<Vector3> &r_vertices, const Vector3 &p_start, const Vector3 &p_end, const Vector3 &p_circle_center, real_t p_sphere_r, int p_segments = 32) {
	// Normalize inputs
	Vector3 start = p_start.normalized();
	Vector3 end = p_end.normalized();
	Vector3 circle_center = p_circle_center.normalized();

	// Project start and end onto the plane perpendicular to circle center
	Vector3 start_proj = (start - circle_center * start.dot(circle_center));
	Vector3 end_proj = (end - circle_center * end.dot(circle_center));
	
	real_t start_proj_len = start_proj.length();
	real_t end_proj_len = end_proj.length();
	
	if (start_proj_len < CMP_EPSILON || end_proj_len < CMP_EPSILON) {
		// Points are aligned with circle center, can't draw arc
		return;
	}

	start_proj.normalize();
	end_proj.normalize();

	// Find the angle between the projections
	real_t angle = Math::acos(CLAMP(start_proj.dot(end_proj), -1.0, 1.0));
	
	// Determine direction (use cross product to check orientation relative to circle center)
	Vector3 cross = start_proj.cross(end_proj);
	if (circle_center.dot(cross) < 0.0) {
		angle = Math::TAU - angle;
	}

	// Draw arc by rotating around the circle center axis
	// Add vertices in pairs (p0, p1) for each line segment
	Vector3 axis = circle_center;
	Vector3 prev_point = start * p_sphere_r;
	
	for (int i = 1; i <= p_segments; i++) {
		real_t t = (real_t)i / (real_t)p_segments;
		real_t current_angle = angle * t;
		
		// Rotate start_proj around axis by current_angle
		Quaternion rot = Quaternion(axis, current_angle);
		Vector3 rotated_proj = rot.xform(start_proj);
		
		// Reconstruct point on circle (maintain same distance from center as start)
		real_t dist_from_center = start.dot(circle_center);
		Vector3 current_point = (circle_center * dist_from_center + rotated_proj * Math::sqrt(1.0 - dist_from_center * dist_from_center)).normalized() * p_sphere_r;
		
		// Add line segment as pair of vertices (prev_point, current_point)
		r_vertices.push_back(prev_point);
		r_vertices.push_back(current_point);
		prev_point = current_point;
	}
}

// Draw the exact path boundary between two cones
static void draw_path_boundary(LocalVector<Vector3> &r_vertices, const Vector3 &p_center1, real_t p_radius1,
		const Vector3 &p_center2, real_t p_radius2, real_t p_sphere_r, int p_segments = 64) {
	Vector3 center1 = p_center1.normalized();
	Vector3 center2 = p_center2.normalized();

	// Compute tangent circles
	Vector3 tan1, tan2;
	real_t tan_radius;
	compute_tangent_circle(center1, p_radius1, center2, p_radius2, tan1, tan2, tan_radius);

	// Find intersection points between cone boundaries and tangent circles
	// The path boundary consists of:
	// 1. Arc along cone1 boundary from intersection with tan1 to intersection with tan2
	// 2. Arc along tan1 (or tan2) connecting the cone boundaries
	// 3. Arc along cone2 boundary from intersection with tan1 to intersection with tan2

	// Find points on cone1 boundary closest to tangent circles
	Vector3 cone1_to_tan1 = (tan1 - center1 * tan1.dot(center1)).normalized();
	Vector3 cone1_to_tan2 = (tan2 - center1 * tan2.dot(center1)).normalized();
	Vector3 cone1_boundary_tan1 = (center1 * Math::cos(p_radius1) + cone1_to_tan1 * Math::sin(p_radius1)).normalized();
	Vector3 cone1_boundary_tan2 = (center1 * Math::cos(p_radius1) + cone1_to_tan2 * Math::sin(p_radius1)).normalized();

	// Find points on cone2 boundary closest to tangent circles
	Vector3 cone2_to_tan1 = (tan1 - center2 * tan1.dot(center2)).normalized();
	Vector3 cone2_to_tan2 = (tan2 - center2 * tan2.dot(center2)).normalized();
	Vector3 cone2_boundary_tan1 = (center2 * Math::cos(p_radius2) + cone2_to_tan1 * Math::sin(p_radius2)).normalized();
	Vector3 cone2_boundary_tan2 = (center2 * Math::cos(p_radius2) + cone2_to_tan2 * Math::sin(p_radius2)).normalized();

	// Determine which tangent circle forms the path boundary
	// The path is the region outside both tangent circles
	// Use the solver's get_on_great_tangent_triangle to determine which tangent is used for a point in the path region
	// This ensures we use the exact same logic as the solver
	Vector3 arc_normal = center1.cross(center2);
	
	// Use a point between the cone centers as representative of the path region
	Vector3 path_midpoint = (center1 + center2).normalized();
	
	// Use the solver's logic to determine which tangent circle this point would use
	// get_on_great_tangent_triangle checks arc_side_dot = input.dot(arc_normal)
	// If < 0, it uses tan1; if >= 0, it uses tan2
	real_t path_side_dot = path_midpoint.dot(arc_normal);
	
	// However, we need to verify the point is actually in the path region
	// If get_on_great_tangent_triangle returns the point unchanged (cosine > 0.999), it's in the path
	// Otherwise, we need to find a point that is in the path region
	Vector3 test_result = get_on_great_tangent_triangle(path_midpoint, center1, p_radius1, center2, p_radius2);
	real_t cosine = test_result.dot(path_midpoint);
	if (cosine < 0.999f) {
		// Point is not in path region, try a point further from both cones
		// Use a point that's clearly between the cones but outside both
		Vector3 perp = center1.cross(center2).normalized();
		if (perp.is_zero_approx()) {
			perp = center1.get_any_perpendicular().normalized();
		}
		// Create a point in the plane between the cones, rotated around the perpendicular
		path_midpoint = (center1 + center2 + perp * 0.5).normalized();
		path_side_dot = path_midpoint.dot(arc_normal);
	}
	
	// Draw arc along cone1 boundary
	draw_arc_between_points(r_vertices, cone1_boundary_tan1, cone1_boundary_tan2, center1, p_sphere_r, p_segments);
	
	// Draw arc along the tangent circle that forms the boundary on the path side
	// The solver uses: if arc_side_dot < 0, it checks tan1; if >= 0, it checks tan2
	// But for visualization, we need to draw the tangent circle that forms the boundary of the allowed path region
	// The path is outside both tangent circles, so we need to determine which one is on the boundary
	// Check which tangent circle the path midpoint is closer to being outside of
	real_t tan_radius_cos = Math::cos(tan_radius);
	real_t dist_to_tan1 = path_midpoint.dot(tan1);
	real_t dist_to_tan2 = path_midpoint.dot(tan2);
	
	// The path is outside both tangent circles (dist < tan_radius_cos means outside)
	// We want to draw the tangent circle that's on the boundary of the path region
	// This should be the one that the path midpoint is closer to the boundary of
	// But actually, we should use the solver's logic: if path_side_dot < 0, the solver checks tan1 side
	// So the boundary on that side is formed by tan1
	// However, we might need to check which tangent is actually on the correct side
	// Try both and see which one makes sense geometrically
	bool use_tan1 = (path_side_dot < 0.0);
	
	// Verify: the path midpoint should be outside the tangent circle we're using
	// If not, try the other one
	if (use_tan1 && dist_to_tan1 > tan_radius_cos) {
		// Path midpoint is inside tan1 circle, but we're trying to use tan1 - try tan2 instead
		use_tan1 = false;
	} else if (!use_tan1 && dist_to_tan2 > tan_radius_cos) {
		// Path midpoint is inside tan2 circle, but we're trying to use tan2 - try tan1 instead
		use_tan1 = true;
	}
	
	if (use_tan1) {
		// Path region is on tan1 side, draw arc along tan1
		draw_arc_between_points(r_vertices, cone1_boundary_tan1, cone2_boundary_tan1, tan1, p_sphere_r, p_segments);
	} else {
		// Path region is on tan2 side, draw arc along tan2
		draw_arc_between_points(r_vertices, cone1_boundary_tan2, cone2_boundary_tan2, tan2, p_sphere_r, p_segments);
	}
	
	// Draw arc along cone2 boundary
	draw_arc_between_points(r_vertices, cone2_boundary_tan1, cone2_boundary_tan2, center2, p_sphere_r, p_segments);
}

void JointLimitationKusudama3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const {
	// Don't draw visualization if orientation constraints are disabled
	if (!orientationally_constrained) {
		return;
	}
	real_t socket_r = p_bone_length * 0.25f;
	if (socket_r <= CMP_EPSILON) {
		return;
	}

	LocalVector<Vector3> vertices;

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
				vertices.push_back(p0);
				vertices.push_back(p1);
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
					vertices.push_back(p0);
					vertices.push_back(p1);
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
					vertices.push_back(p0);
					vertices.push_back(p1);
				}
			}
		}

		// Draw lines from origin to arc endpoints to show limits
		Vector3 limit1_dir = (x_axis * Math::cos(normalized_min) + z_axis * Math::sin(normalized_min)) * indicator_r;
		Vector3 limit2_dir = (x_axis * Math::cos(wrapped_max) + z_axis * Math::sin(wrapped_max)) * indicator_r;
		vertices.push_back(indicator_pos);
		vertices.push_back(indicator_pos + limit1_dir);
		vertices.push_back(indicator_pos);
		vertices.push_back(indicator_pos + limit2_dir);
	}

	// CSG boolean visualization: Generate forbidden region mesh by subtracting allowed regions from sphere
	if (!cones.is_empty()) {
		// Generate triangle mesh for forbidden region using CSG subtraction
		LocalVector<Vector3> forbidden_vertices;
		LocalVector<int> forbidden_indices;
		
		// Use high density for accurate CSG result
		int rings = 128;
		int radial_segments = 128;
		
		generate_forbidden_region_mesh(cones, socket_r, radial_segments, rings, forbidden_vertices, forbidden_indices);
		
		// Draw wireframe edges of the forbidden region mesh
		// For each triangle, draw its three edges
		for (int tri = 0; tri < forbidden_indices.size(); tri += 3) {
			int i0 = forbidden_indices[tri];
			int i1 = forbidden_indices[tri + 1];
			int i2 = forbidden_indices[tri + 2];
			
			// Draw triangle edges as wireframe
			vertices.push_back(forbidden_vertices[i0]);
			vertices.push_back(forbidden_vertices[i1]);
			
			vertices.push_back(forbidden_vertices[i1]);
			vertices.push_back(forbidden_vertices[i2]);
			
			vertices.push_back(forbidden_vertices[i2]);
			vertices.push_back(forbidden_vertices[i0]);
		}
	}

	// Draw exact cone boundary rings on the unit sphere
	// These show the exact boundary of each cone
	for (int cone_i = 0; cone_i < cones.size(); cone_i++) {
		const Vector4 &cone_data = cones[cone_i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t cone_radius = cone_data.w; // Cone radius in radians

		// Draw the exact boundary circle of the cone (using spline interpolation with reduced detail)
		draw_cone_circle(vertices, center, cone_radius, socket_r, 64);
	}

	// Draw exact tangent boundary rings on the unit sphere
	// These show the exact boundary of each tangent circle connecting adjacent cones
	// Only connect adjacent cones in sequence (no wrap-around from last to first)
	for (int cone_i = 0; cone_i < cones.size() - 1; cone_i++) {
		const Vector4 &cone_data1 = cones[cone_i];
		Vector3 center1 = Vector3(cone_data1.x, cone_data1.y, cone_data1.z).normalized();
		real_t radius1 = cone_data1.w; // Cone radius in radians

		// Connect only to the next adjacent cone (no wrap-around)
		const Vector4 &cone_data2 = cones[cone_i + 1];
		Vector3 center2 = Vector3(cone_data2.x, cone_data2.y, cone_data2.z).normalized();
		real_t radius2 = cone_data2.w; // Cone radius in radians

		// Compute tangent circles
		Vector3 tan1, tan2;
		real_t tan_radius;
		compute_tangent_circle(center1, radius1, center2, radius2, tan1, tan2, tan_radius);

		// Draw both tangent circle boundaries
		draw_cone_circle(vertices, tan1, tan_radius, socket_r, 64);
		draw_cone_circle(vertices, tan2, tan_radius, socket_r, 64);

		// Draw the exact path boundary as a curved line
		// This shows the precise boundary of the allowed region between the two cones
		draw_path_boundary(vertices, center1, radius1, center2, radius2, socket_r, 128);
	}

	// Add all vertices to surface tool as a single mesh
	// Bone weights are set by the gizmo before calling draw_shape, so they apply to all vertices
	// For bone weights to work, vertices should be in parent bone's local space (rest pose)
	// p_transform = parent_global_rest * limitation_space
	// To get vertices in parent bone local space, we need: parent_global_rest.affine_inverse() * p_transform = limitation_space
	// However, we don't have parent_global_rest separately, so we use p_transform which puts vertices in global space
	// The bone weights should still transform them correctly if set properly by the gizmo
	for (int64_t i = 0; i < vertices.size(); i++) {
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(vertices[i]));
	}
}
#endif // TOOLS_ENABLED
