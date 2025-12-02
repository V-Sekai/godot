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
	ClassDB::bind_method(D_METHOD("set_orientationally_constrained", "constrained"), &JointLimitationKusudama3D::set_orientationally_constrained);
	ClassDB::bind_method(D_METHOD("is_orientationally_constrained"), &JointLimitationKusudama3D::is_orientationally_constrained);

	ClassDB::bind_method(D_METHOD("set_cone_count", "count"), &JointLimitationKusudama3D::set_cone_count);
	ClassDB::bind_method(D_METHOD("get_cone_count"), &JointLimitationKusudama3D::get_cone_count);
	ClassDB::bind_method(D_METHOD("set_cone_center", "index", "center"), &JointLimitationKusudama3D::set_cone_center);
	ClassDB::bind_method(D_METHOD("get_cone_center", "index"), &JointLimitationKusudama3D::get_cone_center);
	ClassDB::bind_method(D_METHOD("set_cone_radius", "index", "radius"), &JointLimitationKusudama3D::set_cone_radius);
	ClassDB::bind_method(D_METHOD("get_cone_radius", "index"), &JointLimitationKusudama3D::get_cone_radius);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "orientationally_constrained"), "set_orientationally_constrained", "is_orientationally_constrained");
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
	// Extract all unique cones from the quads: cone0 from quad[0][0], cone1 from quad[0][3], cone2 from quad[1][3], etc.
	for (int i = 0; i < cones.size(); i++) {
		const Projection &quad = cones[i];
		// Check cone1 (column 0) of this quad
		Vector4 cone1_vec = quad[0];
		Vector3 control_point = Vector3(cone1_vec.x, cone1_vec.y, cone1_vec.z).normalized();
		real_t radius = cone1_vec.w;

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
	// Also check the last cone (column 3 of the last quad)
	if (cones.size() > 0) {
		const Projection &last_quad = cones[cones.size() - 1];
		Vector4 cone2_vec = last_quad[3];
		Vector3 control_point = Vector3(cone2_vec.x, cone2_vec.y, cone2_vec.z).normalized();
		real_t radius = cone2_vec.w;

		Vector3 collision_point = closest_to_cone_boundary(point, control_point, radius);
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

	// If we're out of bounds of all cones, check if we're in the paths between the cones
	// IMPORTANT: We explicitly do NOT check the pair (last_cone, first_cone) to prevent wrap-around
	// For each cone pair, get the quad as a 4x4 Projection: [cone1, tan1, tan2, cone2] and check all 4 regions
	if (cones.size() > 0) {
		for (int i = 0; i < cones.size(); i++) {
			// Get quad as 4x4 Projection matrix: columns are [cone1, tan1, tan2, cone2]
			// Each column is Vector4(x, y, z, radius)
			const Projection &quad = cones[i];
			
			// Extract elements from quad
			Vector4 cone1_vec = quad[0]; // Column 0 = cone1
			Vector4 tan1_vec = quad[1];  // Column 1 = tan1
			Vector4 tan2_vec = quad[2];  // Column 2 = tan2
			Vector4 cone2_vec = quad[3];  // Column 3 = cone2
			
			Vector3 center1 = Vector3(cone1_vec.x, cone1_vec.y, cone1_vec.z).normalized();
			Vector3 center2 = Vector3(cone2_vec.x, cone2_vec.y, cone2_vec.z).normalized();
			real_t radius1 = cone1_vec.w;
			real_t radius2 = cone2_vec.w;
			
			Vector3 tan1 = Vector3(tan1_vec.x, tan1_vec.y, tan1_vec.z).normalized();
			Vector3 tan2 = Vector3(tan2_vec.x, tan2_vec.y, tan2_vec.z).normalized();
			real_t tan_radius = tan1_vec.w; // tan1 and tan2 have same radius

			// Check all 4 regions of the quad: iterate through [cone1, tan1, tan2, cone2]
			real_t tan_radius_cos = Math::cos(tan_radius);
			for (int quad_idx = 0; quad_idx < 4; quad_idx++) {
				Vector4 elem_vec = quad[quad_idx];
				Vector3 elem_center = Vector3(elem_vec.x, elem_vec.y, elem_vec.z).normalized();
				real_t elem_radius = elem_vec.w;
				bool is_tangent = (quad_idx == 1 || quad_idx == 2);
				
				Vector3 collision_point;
				
				if (!is_tangent) {
					// Check cone region
					collision_point = closest_to_cone_boundary(point, elem_center, elem_radius);
				} else {
					// Check tangent region
					// For tan1 (quad_idx == 1): check region between center1 and center2
					// For tan2 (quad_idx == 2): check region between center1 and center2
					Vector3 c1xt = center1.cross(elem_center);
					Vector3 txc2 = elem_center.cross(center2);
					if (quad_idx == 2) {
						// tan2: reverse the cross products
						c1xt = elem_center.cross(center1);
						txc2 = center2.cross(elem_center);
					}
					
					if (point.dot(c1xt) > 0 && point.dot(txc2) > 0) {
						real_t to_tan_cos = point.dot(elem_center);
						if (to_tan_cos > tan_radius_cos) {
							// Project onto tangent circle
							Vector3 plane_normal = elem_center.cross(point);
							if (!plane_normal.is_finite() || Math::is_zero_approx(plane_normal.length_squared())) {
								plane_normal = Vector3(0, 1, 0);
							}
							plane_normal.normalize();
							Quaternion rotate_about_by = get_quaternion_axis_angle(plane_normal, elem_radius);
							collision_point = rotate_about_by.xform(elem_center).normalized();
						} else {
							// Point is inside tangent circle, so it's valid
							collision_point = point;
						}
					} else {
						collision_point = Vector3(NAN, NAN, NAN);
					}
				}

				// Process collision point
				if (Math::is_nan(collision_point.x)) {
					continue;
				}

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

	// Return the closest boundary point between cones
	return closest_collision_point.normalized();
}

void JointLimitationKusudama3D::set_cone_count(int p_count) {
	if (p_count < 0) {
		p_count = 0;
	}
	// Number of quads = number of individual cones - 1
	int quad_count = p_count > 0 ? p_count - 1 : 0;
	int old_size = cones.size();
	if (old_size == quad_count) {
		return;
	}
	cones.resize(quad_count);
	// Initialize new quads with default values
	for (int i = old_size; i < cones.size(); i++) {
		Projection &quad = cones.write[i];
		Vector4 default_cone = Vector4(0, 1, 0, Math::PI * 0.25); // Default: +Y axis, 45 degree cone
		quad[0] = default_cone; // cone1
		quad[3] = default_cone; // cone2 (same as next quad's cone1)
		// tan1 and tan2 will be computed when needed
	}
	notify_property_list_changed();
	emit_changed();
}

int JointLimitationKusudama3D::get_cone_count() const {
	// Number of unique cones = number of quads + 1
	// Each quad has cone1 (col 0) and cone2 (col 3), and cone2 of quad i = cone1 of quad i+1
	return cones.size() + 1;
}

void JointLimitationKusudama3D::set_cone_center(int p_index, const Vector3 &p_center) {
	int quad_count = cones.size();
	ERR_FAIL_INDEX(p_index, quad_count + 1);
	// Store raw value (non-normalized) to allow editor to accept values outside [-1, 1]
	// Normalization happens lazily when values are used
	if (p_index < quad_count) {
		// Access column 0 of quad at p_index
		Vector4 &cone = cones.write[p_index][0];
		cone.x = p_center.x;
		cone.y = p_center.y;
		cone.z = p_center.z;
	} else {
		// Access column 3 of last quad
		Vector4 &cone = cones.write[quad_count - 1][3];
		cone.x = p_center.x;
		cone.y = p_center.y;
		cone.z = p_center.z;
	}
	emit_changed();
}

Vector3 JointLimitationKusudama3D::get_cone_center(int p_index) const {
	int quad_count = cones.size();
	ERR_FAIL_INDEX_V(p_index, quad_count + 1, Vector3(0, 1, 0));
	Vector4 cone_vec;
	if (p_index < quad_count) {
		// Access column 0 of quad at p_index
		cone_vec = cones[p_index][0];
	} else {
		// Access column 3 of last quad
		cone_vec = cones[quad_count - 1][3];
	}
	Vector3 center = Vector3(cone_vec.x, cone_vec.y, cone_vec.z);
	// Normalize when reading to ensure we always return a normalized value
	if (center.length_squared() > CMP_EPSILON) {
		return center.normalized();
	}
	return Vector3(0, 1, 0);
}


void JointLimitationKusudama3D::set_cone_radius(int p_index, real_t p_radius) {
	int quad_count = cones.size();
	ERR_FAIL_INDEX(p_index, quad_count + 1);
	if (p_index < quad_count) {
		// Access column 0 of quad at p_index
		cones.write[p_index][0].w = p_radius;
	} else {
		// Access column 3 of last quad
		cones.write[quad_count - 1][3].w = p_radius;
	}
	emit_changed();
}

real_t JointLimitationKusudama3D::get_cone_radius(int p_index) const {
	int quad_count = cones.size();
	ERR_FAIL_INDEX_V(p_index, quad_count + 1, 0.0);
	if (p_index < quad_count) {
		// Access column 0 of quad at p_index
		return cones[p_index][0].w;
	} else {
		// Access column 3 of last quad
		return cones[quad_count - 1][3].w;
	}
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
				int quad_count = cones.size();
				ERR_FAIL_INDEX_V(index, quad_count + 1, false);
				// Convert quaternion to direction vector by rotating the default direction (0, 1, 0)
				Vector3 default_dir = Vector3(0, 1, 0);
				Vector3 center = Quaternion(p_value).normalized().xform(default_dir);
				set_cone_center(index, center);
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
			int quad_count = cones.size();
			ERR_FAIL_INDEX_V(index, quad_count + 1, false);
			Vector3 center = get_cone_center(index); // This already normalizes
			Vector3 default_dir = Vector3(0, 1, 0);
			// Create quaternion representing rotation from default_dir to center
			r_ret = Quaternion(default_dir, center);
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
// Helper function to draw a great circle arc on the sphere (non-dashed)
static void draw_great_circle_arc(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, const Vector3 &p_start_dir, const Vector3 &p_end_dir, real_t p_sphere_r, const Color &p_color, int p_segments = 32, bool p_long_way = false) {
	Vector3 start = p_start_dir.normalized();
	Vector3 end = p_end_dir.normalized();
	Vector3 axis = start.cross(end);
	if (axis.length_squared() < CMP_EPSILON2) {
		// Parallel vectors, draw straight line
		Vector3 p0 = start * p_sphere_r;
		Vector3 p1 = end * p_sphere_r;
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(p0));
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(p1));
		return;
	}
	axis.normalize();
	
	real_t angle = start.angle_to(end);
	if (p_long_way) {
		angle = Math::TAU - angle;
	}
	real_t d_angle = angle / (real_t)p_segments;
	
	Vector3 prev = start * p_sphere_r;
	for (int i = 1; i <= p_segments; i++) {
		Quaternion rot = Quaternion(axis, d_angle * (real_t)i);
		Vector3 cur = rot.xform(start) * p_sphere_r;
		
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(prev));
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(cur));
		
		prev = cur;
	}
}

// Helper function to draw a dashed great circle arc on the sphere
static void draw_dashed_great_circle_arc(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, const Vector3 &p_start_dir, const Vector3 &p_end_dir, real_t p_sphere_r, const Color &p_color, int p_segments = 32) {
	Vector3 start = p_start_dir.normalized();
	Vector3 end = p_end_dir.normalized();
	Vector3 axis = start.cross(end);
	if (axis.length_squared() < CMP_EPSILON2) {
		// Parallel vectors, draw straight line
		Vector3 p0 = start * p_sphere_r;
		Vector3 p1 = end * p_sphere_r;
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(p0));
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(p1));
		return;
	}
	axis.normalize();
	
	real_t angle = start.angle_to(end);
	
	// Draw dashed pattern: 60% dash, 40% gap
	int num_dashes = p_segments / 5;
	if (num_dashes < 2) {
		num_dashes = 2;
	}
	real_t dash_ratio = 0.6;
	real_t total_angle_per_dash = angle / (real_t)num_dashes;
	real_t dash_angle = total_angle_per_dash * dash_ratio;
	
	real_t current_angle = 0.0;
	int dash_segments = (int)(dash_angle / (angle / (real_t)p_segments));
	if (dash_segments < 2) {
		dash_segments = 2;
	}
	
	for (int dash = 0; dash < num_dashes; dash++) {
		real_t dash_start_angle = current_angle;
		real_t dash_end_angle = current_angle + dash_angle;
		if (dash_end_angle > angle) {
			dash_end_angle = angle;
		}
		
		real_t d_angle = (dash_end_angle - dash_start_angle) / (real_t)dash_segments;
		Vector3 prev = get_quaternion_axis_angle(axis, dash_start_angle).xform(start) * p_sphere_r;
		
		for (int i = 1; i <= dash_segments; i++) {
			real_t cur_angle = dash_start_angle + d_angle * (real_t)i;
			if (cur_angle > angle) {
				cur_angle = angle;
			}
			Vector3 cur = get_quaternion_axis_angle(axis, cur_angle).xform(start) * p_sphere_r;
			
			p_surface_tool->set_color(p_color);
			p_surface_tool->add_vertex(p_transform.xform(prev));
			p_surface_tool->set_color(p_color);
			p_surface_tool->add_vertex(p_transform.xform(cur));
			
			prev = cur;
		}
		
		current_angle += total_angle_per_dash;
		if (current_angle >= angle) {
			break;
		}
	}
}

// Helper function to draw a circle on the sphere (not necessarily a great circle)
static void draw_sphere_circle(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, const Vector3 &p_center_dir, real_t p_angle, real_t p_sphere_r, const Color &p_color, int p_segments = 32) {
	Vector3 center = p_center_dir.normalized();
	
	// Find two perpendicular vectors to the center
	Vector3 perp1, perp2;
	if (Math::abs(center.y) < 0.9f) {
		perp1 = Vector3(0, 1, 0).cross(center).normalized();
	} else {
		perp1 = Vector3(1, 0, 0).cross(center).normalized();
	}
	perp2 = center.cross(perp1).normalized();
	
	real_t y_offset = p_sphere_r * Math::cos(p_angle);
	real_t circle_r = p_sphere_r * Math::sin(p_angle);
	
	if (circle_r <= CMP_EPSILON) {
		return;
	}
	
	static const real_t DP = Math::TAU / (real_t)p_segments;
	Vector3 prev = center * y_offset + (perp1 * Math::cos(0.0) + perp2 * Math::sin(0.0)) * circle_r;
	
	for (int i = 1; i <= p_segments; i++) {
		real_t angle = (real_t)i * DP;
		Vector3 dir = (perp1 * Math::cos(angle) + perp2 * Math::sin(angle)).normalized();
		Vector3 cur = center * y_offset + dir * circle_r;
		
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(prev));
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(cur));
		
		prev = cur;
	}
}

// Helper function to find the contact point where a cone and tangent circle meet
// Uses the solver's closest_to_cone_boundary logic to find the point on the cone boundary
// in the direction of the tangent center, then projects it onto the tangent circle
static Vector3 get_cone_tangent_contact_point(const Vector3 &p_cone_center, real_t p_cone_radius, const Vector3 &p_tan_center, real_t p_tan_radius) {
	Vector3 cone_center = p_cone_center.normalized();
	Vector3 tan_center = p_tan_center.normalized();
	
	// Use the solver's closest_to_cone_boundary to find a point on the cone boundary
	// in the direction of the tangent center
	Vector3 boundary_point = closest_to_cone_boundary(tan_center, cone_center, p_cone_radius);
	
	// If the point is inside the cone (shouldn't happen for tangent centers), 
	// fall back to rotation method
	if (Math::is_nan(boundary_point.x)) {
		// Fallback: rotate cone center by radius around axis perpendicular to both
		Vector3 axis = cone_center.cross(tan_center);
		if (axis.length_squared() < CMP_EPSILON2) {
			axis = get_orthogonal(cone_center);
			if (axis.length_squared() < CMP_EPSILON2) {
				axis = Vector3(0, 1, 0);
			}
		}
		axis.normalize();
		Quaternion rot = get_quaternion_axis_angle(axis, p_cone_radius);
		boundary_point = rot.xform(cone_center).normalized();
	}
	
	// Now project this boundary point onto the tangent circle using the solver's logic
	// The contact point should be on the tangent circle at distance tan_radius from tan_center
	real_t tan_radius_cos = Math::cos(p_tan_radius);
	real_t to_tan_cos = boundary_point.dot(tan_center);
	
	if (to_tan_cos > tan_radius_cos) {
		// Project onto tangent circle (same logic as get_on_great_tangent_triangle)
		Vector3 plane_normal = tan_center.cross(boundary_point);
		if (!plane_normal.is_finite() || Math::is_zero_approx(plane_normal.length_squared())) {
			plane_normal = Vector3(0, 1, 0);
		}
		plane_normal.normalize();
		Quaternion rotate_about_by = get_quaternion_axis_angle(plane_normal, p_tan_radius);
		return rotate_about_by.xform(tan_center).normalized();
	} else {
		// Already on or inside tangent circle, return the boundary point
		return boundary_point.normalized();
	}
}

void JointLimitationKusudama3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const {
	if (!is_orientationally_constrained() || cones.is_empty()) {
		return;
	}

	real_t sphere_r = p_bone_length * (real_t)0.25;
	if (sphere_r <= CMP_EPSILON) {
		return;
	}

	static const int N = 32; // Number of segments for rings and arcs

	// 1. Draw kusudama cone boundaries and center indicators
	int cone_count = get_cone_count();
	for (int i = 0; i < cone_count; i++) {
		Vector3 center = get_cone_center(i);
		real_t radius = get_cone_radius(i);

		// Draw cone boundary ring (for visibility, not filled)
		draw_sphere_circle(p_surface_tool, p_transform, center, radius, sphere_r, p_color, N);
		
		// Draw center indicator (small ring at cone center)
		real_t center_ring_radius = (real_t)0.05; // Fixed 0.05 radians (~2.86 degrees)
		draw_sphere_circle(p_surface_tool, p_transform, center, center_ring_radius, sphere_r, p_color, N);
	}


	// 3. Draw fish bone structure (dashed lines connecting cone centers in order)
	if (cone_count > 1) {
		for (int i = 0; i < cone_count - 1; i++) {
			Vector3 center1 = get_cone_center(i);
			Vector3 center2 = get_cone_center(i + 1);
			
			// Draw dashed great circle arc from cone1 center to cone2 center on sphere surface
			draw_dashed_great_circle_arc(p_surface_tool, p_transform, center1, center2, sphere_r, p_color, N);
		}
	}
}

#endif // TOOLS_ENABLED
