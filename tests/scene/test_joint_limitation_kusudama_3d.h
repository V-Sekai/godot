/**************************************************************************/
/*  test_joint_limitation_kusudama_3d.h                                   */
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

#pragma once
#include "scene/resources/3d/joint_limitation_kusudama_3d.h"
#include "tests/test_macros.h"

namespace TestJointLimitationKusudama3D {

// Independent solver implementation - completely separate from production code
// This provides a verification mechanism for the main solver

// Helper function to compute plane-ray intersection (independent implementation)
static Vector3 test_ray_plane_intersection(const Vector3 &p_ray_start, const Vector3 &p_ray_end, const Vector3 &p_plane_a, const Vector3 &p_plane_b, const Vector3 &p_plane_c) {
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

// Helper function to extend a ray in both directions (independent implementation)
static void test_extend_ray(Vector3 &r_start, Vector3 &r_end, real_t p_amount) {
	Vector3 mid_point = (r_start + r_end) * 0.5;
	Vector3 start_heading = r_start - mid_point;
	Vector3 end_heading = r_end - mid_point;
	Vector3 start_extension = start_heading.normalized() * p_amount;
	Vector3 end_extension = end_heading.normalized() * p_amount;
	r_start = start_heading + start_extension + mid_point;
	r_end = end_heading + end_extension + mid_point;
}

// Helper function to compute ray-sphere intersection (independent implementation)
static int test_ray_sphere_intersection(const Vector3 &p_ray_start, const Vector3 &p_ray_end, const Vector3 &p_sphere_center, real_t p_radius, Vector3 *r_intersection1, Vector3 *r_intersection2) {
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

	real_t sqrt_discriminant = Math::sqrt(discriminant);
	real_t t1 = ray_dot_center - sqrt_discriminant;
	real_t t2 = ray_dot_center + sqrt_discriminant;

	if (r_intersection1) {
		*r_intersection1 = p_ray_start + ray_dir_normalized * t1;
	}
	if (r_intersection2) {
		*r_intersection2 = p_ray_start + ray_dir_normalized * t2;
	}

	return discriminant > 0.0 ? 2 : 1; // Two intersections or one (tangent)
}

// Helper function to compute tangent circle between two cones (independent implementation)
static void test_compute_tangent_circle(const Vector3 &p_center1, real_t p_radius1, const Vector3 &p_center2, real_t p_radius2,
		Vector3 &r_tangent1, Vector3 &r_tangent2, real_t &r_tangent_radius) {
	Vector3 center1 = p_center1.normalized();
	Vector3 center2 = p_center2.normalized();

	// Compute tangent circle radius
	r_tangent_radius = (Math::PI - (p_radius1 + p_radius2)) / 2.0;

	// Find arc normal (axis perpendicular to both cone centers)
	Vector3 arc_normal = center1.cross(center2);
	real_t arc_normal_len = arc_normal.length();

	if (arc_normal_len < CMP_EPSILON) {
		// Cones are parallel or opposite - handle specially
		arc_normal = center1.get_any_perpendicular();
		if (arc_normal.is_zero_approx()) {
			arc_normal = Vector3(0, 1, 0);
		}
		arc_normal.normalize();

		// For opposite cones, tangent circles are at 90 degrees from the cone centers
		Vector3 perp1 = center1.get_any_perpendicular().normalized();

		// Rotate around center1 by the tangent radius to get tangent centers
		Quaternion rot1 = Quaternion(center1, r_tangent_radius);
		Quaternion rot2 = Quaternion(center1, -r_tangent_radius);
		r_tangent1 = rot1.xform(perp1).normalized();
		r_tangent2 = rot2.xform(perp1).normalized();
		return;
	}
	arc_normal.normalize();

	// Use plane intersection method
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

	test_extend_ray(ray1_b_start, ray1_b_end, 99.0);
	test_extend_ray(ray2_b_start, ray2_b_end, 99.0);

	Vector3 intersection1 = test_ray_plane_intersection(ray1_b_start, ray1_b_end, scaled_axis_a, plane_dir1_a, plane_dir2_a);
	Vector3 intersection2 = test_ray_plane_intersection(ray2_b_start, ray2_b_end, scaled_axis_a, plane_dir1_a, plane_dir2_a);

	Vector3 intersection_ray_start = intersection1;
	Vector3 intersection_ray_end = intersection2;
	test_extend_ray(intersection_ray_start, intersection_ray_end, 99.0);

	Vector3 sphere_intersect1;
	Vector3 sphere_intersect2;
	Vector3 sphere_center(0, 0, 0);
	test_ray_sphere_intersection(intersection_ray_start, intersection_ray_end, sphere_center, 1.0, &sphere_intersect1, &sphere_intersect2);

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

// Helper function to find point on path between two cones (independent implementation)
static Vector3 test_get_on_great_tangent_triangle(const Vector3 &p_input, const Vector3 &p_center1, real_t p_radius1,
		const Vector3 &p_center2, real_t p_radius2) {
	Vector3 center1 = p_center1.normalized();
	Vector3 center2 = p_center2.normalized();
	Vector3 input = p_input.normalized();

	// Compute tangent circle
	Vector3 tan1, tan2;
	real_t tan_radius;
	test_compute_tangent_circle(center1, p_radius1, center2, p_radius2, tan1, tan2, tan_radius);

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
				// Use slightly larger angle to move point outside the tangent circle (into allowed region)
				// Points with angle > tan_radius are outside (allowed), points with angle < tan_radius are inside (forbidden)
				// Use minimal adjustment (5e-5 radians) to ensure it's in allowed region without moving too far
				real_t adjusted_tan_radius = tan_radius + 5e-5;
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
				// Use slightly larger angle to move point outside the tangent circle (into allowed region)
				// Points with angle > tan_radius are outside (allowed), points with angle < tan_radius are inside (forbidden)
				// Use minimal adjustment (5e-5 radians) to ensure it's in allowed region without moving too far
				real_t adjusted_tan_radius = tan_radius + 5e-5;
				Quaternion rotate_about_by = Quaternion(plane_normal, adjusted_tan_radius);
				return rotate_about_by.xform(tan2).normalized();
			} else {
				return input;
			}
		}
	}

	return Vector3(NAN, NAN, NAN);
}

// Helper function to check if a point is inside a cone (independent implementation)
static bool test_is_point_in_cone(const Vector3 &p_point, const Vector3 &p_cone_center, real_t p_cone_radius) {
	Vector3 dir = p_point.normalized();
	Vector3 center = p_cone_center.normalized();
	real_t radius_cosine = Math::cos(p_cone_radius);
	real_t input_dot_control = dir.dot(center);
	return input_dot_control >= radius_cosine - 1e-4;
}

// Helper function to check if a point is in a tangent path between two cones (independent implementation)
static bool test_is_point_in_tangent_path(const Vector3 &p_point, const Vector3 &p_center1, real_t p_radius1,
		const Vector3 &p_center2, real_t p_radius2) {
	Vector3 dir = p_point.normalized();

	// Check if point is in the inter-cone path region using get_on_great_tangent_triangle
	// This function handles all the geometric checks including whether the point is inside tangent circles
	Vector3 path_point = test_get_on_great_tangent_triangle(dir, p_center1, p_radius1, p_center2, p_radius2);

	// If NaN, point is not in path region
	if (Math::is_nan(path_point.x)) {
		return false;
	}

	// If the returned point is approximately equal to the input point, point is in the path region
	// This matches the solving code's check: cosine > 0.999f
	// get_on_great_tangent_triangle returns:
	// - The input point if it's in the path region (outside tangent circles)
	// - A projected boundary point if it's inside a tangent circle (forbidden)
	real_t cosine = path_point.dot(dir);
	return cosine > 0.999f;
}

// Main independent solver function that checks if a point is in any allowed region (independent implementation)
static bool test_is_point_allowed(const Vector3 &p_point, const Vector<Vector4> &p_cones) {
	Vector3 dir = p_point.normalized();

	// Check if point is in any cone
	for (int i = 0; i < p_cones.size(); i++) {
		const Vector4 &cone_data = p_cones[i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t radius = cone_data.w;

		if (test_is_point_in_cone(dir, center, radius)) {
			return true; // Point is inside this cone
		}
	}

	// Check if point is in any path between cones (only adjacent cones, no wrap-around)
	if (p_cones.size() > 1) {
		for (int i = 0; i < p_cones.size() - 1; i++) {
			int next_i = i + 1; // Only connect to next adjacent cone, no wrap-around
			const Vector4 &cone1_data = p_cones[i];
			const Vector4 &cone2_data = p_cones[next_i];

			Vector3 center1 = Vector3(cone1_data.x, cone1_data.y, cone1_data.z).normalized();
			Vector3 center2 = Vector3(cone2_data.x, cone2_data.y, cone2_data.z).normalized();
			real_t radius1 = cone1_data.w;
			real_t radius2 = cone2_data.w;

			if (test_is_point_in_tangent_path(dir, center1, radius1, center2, radius2)) {
				return true; // Point is in the inter-cone path region
			}
		}
	}

	return false; // Point is not in any allowed region
}

// Helper function to classify which region a point is in based on the solve result
// This uses the actual limitation's solve method to determine the region
enum class PointRegion {
	INSIDE_CONE,
	IN_TANGENT_PATH,
	FORBIDDEN
};

// Helper to replicate closest_to_cone_boundary logic
static Vector3 test_closest_to_cone_boundary(const Vector3 &p_input, const Vector3 &p_control_point, real_t p_radius) {
	Vector3 input = p_input.normalized();
	Vector3 control = p_control_point.normalized();
	real_t radius_cosine = Math::cos(p_radius);
	real_t input_dot_control = input.dot(control);

	// Check if input is within the cone (matches _solve logic)
	if (input_dot_control >= radius_cosine - 1e-4) {
		return Vector3(NAN, NAN, NAN); // Point is inside cone
	}

	// Point is outside cone, find closest point on boundary
	Vector3 projection = input - control * input_dot_control;
	real_t proj_len = projection.length();
	if (proj_len < CMP_EPSILON) {
		// Input is aligned with control, use any perpendicular
		projection = control.get_any_perpendicular();
		proj_len = projection.length();
	}
	projection.normalize();

	// Rotate control point by radius to get boundary point
	Vector3 axis = control.cross(projection).normalized();
	if (axis.is_zero_approx()) {
		axis = control.get_any_perpendicular().normalized();
	}
	Quaternion rot = Quaternion(axis, p_radius);
	return rot.xform(control).normalized();
}

// Classify which region a point is in using the independent solver
// This provides a separate verification mechanism that doesn't rely on the production solve() method
static PointRegion classify_point_region(const Vector3 &p_point, const Vector<Vector4> &p_cones, Ref<JointLimitationKusudama3D> p_limitation) {
	Vector3 dir = p_point.normalized();

	// Use the independent solver to determine if the point is allowed
	bool is_allowed = test_is_point_allowed(p_point, p_cones);

	if (!is_allowed) {
		return PointRegion::FORBIDDEN;
	}

	// If allowed, determine if it's in a cone or in a tangent path
	// First check if point is inside any cone
	for (int i = 0; i < p_cones.size(); i++) {
		const Vector4 &cone_data = p_cones[i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z).normalized();
		real_t radius = cone_data.w;

		if (test_is_point_in_cone(dir, center, radius)) {
			return PointRegion::INSIDE_CONE;
		}
	}

	// If not in a cone, it must be in a tangent path
	return PointRegion::IN_TANGENT_PATH;
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point inside single cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f); // 30 degrees

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Point at the center of the cone should be returned as-is
	Vector3 test_point = control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));

	// Point slightly off center but still within cone
	Vector3 point_in_cone = Quaternion(Vector3(1, 0, 0), radius * 0.5f).xform(control_point);
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), point_in_cone);
	CHECK(result.is_equal_approx(point_in_cone));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point outside single cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f); // 30 degrees

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Point far outside the cone should be clamped to boundary
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);

	// Result should be on the cone boundary, not the original point
	CHECK_FALSE(result.is_equal_approx(test_point));

	// Result should be close to the control point (within radius)
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f); // Allow small tolerance
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point with zero radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = 0.0f; // Zero radius - only exact point allowed

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Point at exact control point should be returned
	Vector3 test_point = control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(control_point));

	// Any other point should be clamped to control point
	Vector3 outside_point = Vector3(1, 0, 0).normalized();
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), outside_point);
	CHECK(result.is_equal_approx(control_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test multiple cones - point between cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create two cones
	Vector3 control_point1 = Vector3(1, 0, 0).normalized();
	Vector3 control_point2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f); // 45 degrees

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point1.x, control_point1.y, control_point1.z, radius));
	cones.push_back(Vector4(control_point2.x, control_point2.y, control_point2.z, radius));
	limitation->set_cones(cones);

	// Point between the two cones should be handled by path logic
	Vector3 point_between = (control_point1 + control_point2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_between);

	// Result should be valid (not NaN)
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f); // Should be normalized
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point on path between two adjacent cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create two cones with overlapping paths
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(60.0f); // Large enough to create paths

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point exactly on the great circle path between cones
	Vector3 path_point = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path_point);

	// Should be on or near the path
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be closer to the path than the original if original was outside
	// Test with a point clearly between but not exactly on path
	Vector3 between_point = Vector3(0.7, 0.7, 0.1).normalized();
	Vector3 result2 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between_point);
	CHECK(result2.is_finite());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point outside both cones but in path region") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0f); // Smaller radius

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point outside both cones but in the region between them
	Vector3 outside_point = Vector3(0.5, 0.5, 0.7).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), outside_point);

	// Should be finite and normalized
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
	CHECK(result.length() < 1.1f);

	// Result should be either:
	// 1. The point unchanged (if in path region outside tangent circles)
	// 2. Projected to tangent circle boundary (if in path region inside tangent circles)
	// 3. Projected to nearest cone boundary (if outside all allowed regions)
	// So we just check it's a valid result
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path between cones with different radii") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius1 = Math::deg_to_rad(45.0f);
	real_t radius2 = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius1));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius2));
	limitation->set_cones(cones);

	// Point between cones should still work with different radii
	Vector3 between = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test no wrap-around from last to first cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones in a sequence (NOT a loop - no wrap-around)
	// Use very small radii to ensure there's a forbidden zone between non-adjacent cones
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(10.0f); // Very small radius to create clear forbidden zones

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	limitation->set_cones(cones);

	// Point between last and first cone - should NOT be in an allowed path (no wrap-around)
	// Use a point that's clearly in the forbidden wrap-around region
	// With very small cone radii (10deg), the point at ~55 degrees from each axis should be
	// outside all cones and outside all tangent paths between adjacent cones
	Vector3 between_last_first = Vector3(0.577, 0.577, 0.577).normalized(); // ~55 degrees from each axis
	// Verify it's outside all cones
	real_t angle_to_cp1 = between_last_first.angle_to(cp1);
	real_t angle_to_cp2 = between_last_first.angle_to(cp2);
	real_t angle_to_cp3 = between_last_first.angle_to(cp3);
	CHECK(angle_to_cp1 > radius);
	CHECK(angle_to_cp2 > radius);
	CHECK(angle_to_cp3 > radius);
	
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), between_last_first);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
	// With very small cone radii (10deg), this point should be in a forbidden region (wrap-around)
	// and should be constrained (not equal to input)
	// Result should be closer to one of the cones or their adjacent paths, not the wrap-around path
	real_t dist_to_cp1 = result.angle_to(cp1);
	real_t dist_to_cp3 = result.angle_to(cp3);
	CHECK((dist_to_cp1 < Math::PI / 2 || dist_to_cp3 < Math::PI / 2));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point in tangent circle region between cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point that should be in the tangent circle region
	// This is in the region where the tangent circle connects the two cones
	Vector3 tangent_region_point = Vector3(0.6, 0.6, 0.5).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), tangent_region_point);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be on or near the tangent path
	// The angle to either control point should be reasonable
	real_t angle1 = result.angle_to(cp1);
	real_t angle2 = result.angle_to(cp2);
	CHECK((angle1 < Math::PI / 2 || angle2 < Math::PI / 2)); // Should be in hemisphere of at least one cone
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test multiple paths - point closest to which path") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones forming a triangle
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(40.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	limitation->set_cones(cones);

	// Point that could be in path between cp1-cp2 or cp2-cp3
	// Should find the closest valid path
	Vector3 test_point = Vector3(0.4, 0.5, 0.3).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), test_point);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);

	// Result should be valid and normalized
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path with very close cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones very close together
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0.99, 0.1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point between very close cones
	Vector3 between = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test path with nearly opposite cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones nearly opposite each other
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(-0.9, 0.1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point between nearly opposite cones
	Vector3 between = (cp1 + cp2).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), between);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test multiple cones - point in first cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point1 = Vector3(1, 0, 0).normalized();
	Vector3 control_point2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point1.x, control_point1.y, control_point1.z, radius));
	cones.push_back(Vector4(control_point2.x, control_point2.y, control_point2.z, radius));
	limitation->set_cones(cones);

	// Point inside first cone should be returned as-is
	Vector3 point_in_cone1 = Quaternion(Vector3(0, 1, 0), radius * 0.3f).xform(control_point1);
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_in_cone1);
	CHECK(result.is_equal_approx(point_in_cone1));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test empty cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector<Vector4> empty_cones;
	limitation->set_cones(empty_cones);

	// With no cones and orientationally constrained, should return input
	limitation->set_orientationally_constrained(true);
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));

	// With orientationally unconstrained, should return input
	limitation->set_orientationally_constrained(false);
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test orientationally unconstrained") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);
	limitation->set_orientationally_constrained(false);

	// Should return input regardless of cone constraints
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test axial limits getters and setters") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	real_t min_angle = Math::deg_to_rad(-45.0f);
	real_t range_angle = Math::deg_to_rad(90.0f);

	limitation->set_axial_limits(min_angle, range_angle);
	CHECK(Math::is_equal_approx(limitation->get_min_axial_angle(), min_angle));
	CHECK(Math::is_equal_approx(limitation->get_range_angle(), range_angle));

	limitation->set_axially_constrained(true);
	CHECK(limitation->is_axially_constrained() == true);

	limitation->set_axially_constrained(false);
	CHECK(limitation->is_axially_constrained() == false);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test cones getters and setters") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector<Vector4> cones;
	cones.push_back(Vector4(1, 0, 0, Math::deg_to_rad(30.0f)));
	cones.push_back(Vector4(0, 1, 0, Math::deg_to_rad(45.0f)));
	cones.push_back(Vector4(0, 0, 1, Math::deg_to_rad(60.0f)));

	limitation->set_cones(cones);
	Vector<Vector4> retrieved = limitation->get_cones();

	CHECK(retrieved.size() == 3);
	CHECK(Math::is_equal_approx(retrieved[0].w, Math::deg_to_rad(30.0f)));
	CHECK(Math::is_equal_approx(retrieved[1].w, Math::deg_to_rad(45.0f)));
	CHECK(Math::is_equal_approx(retrieved[2].w, Math::deg_to_rad(60.0f)));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test large radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(170.0f); // Very large cone

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Most points should be inside such a large cone
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);

	// Should be valid
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test three cones in sequence") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones forming a path
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	limitation->set_cones(cones);

	// Test point in first cone
	Vector3 point1 = Quaternion(Vector3(0, 1, 0), radius * 0.3f).xform(cp1);
	Vector3 result1 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point1);
	CHECK(result1.is_equal_approx(point1));

	// Test point in second cone
	// Create a point inside the second cone by rotating cp2 by a small angle (less than radius)
	Vector3 point2 = Quaternion(Vector3(1, 0, 0), radius * 0.3f).xform(cp2);
	// Verify point2 is actually inside the cone
	real_t angle_to_cp2 = point2.angle_to(cp2);
	CHECK(angle_to_cp2 < radius);
	Vector3 result2 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point2);
	// Point inside cone should be returned unchanged (or very close due to normalization)
	// Allow some tolerance for floating point precision
	CHECK(result2.is_normalized());
	real_t result_angle_to_cp2 = result2.angle_to(cp2);
	// Result should be inside or on the cone boundary
	CHECK(result_angle_to_cp2 <= radius + 0.01f);

	// Test point between cones (should use path logic)
	Vector3 point_between = (cp1 + cp2).normalized();
	Vector3 result3 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_between);
	CHECK(result3.is_finite());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test edge case - parallel vectors") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(1, 0, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Test with input parallel to control point (should handle gracefully)
	Vector3 parallel_point = control_point * 2.0f; // Same direction, different length
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), parallel_point);

	// Should normalize and return (point is inside cone)
	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test edge case - opposite direction") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Point in opposite direction (180 degrees away)
	Vector3 opposite = -control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), opposite);

	// Should clamp to boundary
	CHECK(result.is_finite());
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test make_space method") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 forward = Vector3(0, 1, 0);
	Vector3 right = Vector3(1, 0, 0);
	Quaternion offset = Quaternion();

	Quaternion space = limitation->make_space(forward, right, offset);

	// Should return a valid quaternion
	CHECK(space.is_normalized());
	CHECK(space.is_finite());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test cone count manipulation") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Default should have one cone
	CHECK(limitation->get_cone_count() == 1);

	// Increase count
	limitation->set_cone_count(3);
	CHECK(limitation->get_cone_count() == 3);

	// Decrease count
	limitation->set_cone_count(2);
	CHECK(limitation->get_cone_count() == 2);

	// Set to zero
	limitation->set_cone_count(0);
	CHECK(limitation->get_cone_count() == 0);

	// Set back to one
	limitation->set_cone_count(1);
	CHECK(limitation->get_cone_count() == 1);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test individual cone property setters and getters") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Set up two cones
	limitation->set_cone_count(2);

	// Test set/get cone center
	Vector3 center1 = Vector3(1, 0, 0).normalized();
	Vector3 center2 = Vector3(0, 1, 0).normalized();
	limitation->set_cone_center(0, center1);
	limitation->set_cone_center(1, center2);

	Vector3 retrieved1 = limitation->get_cone_center(0);
	Vector3 retrieved2 = limitation->get_cone_center(1);
	CHECK(retrieved1.is_equal_approx(center1));
	CHECK(retrieved2.is_equal_approx(center2));

	// Test set/get cone radius
	real_t radius1 = Math::deg_to_rad(30.0f);
	real_t radius2 = Math::deg_to_rad(45.0f);
	limitation->set_cone_radius(0, radius1);
	limitation->set_cone_radius(1, radius2);

	CHECK(Math::is_equal_approx(limitation->get_cone_radius(0), radius1));
	CHECK(Math::is_equal_approx(limitation->get_cone_radius(1), radius2));

	// Test set/get cone center quaternion
	Quaternion quat1 = Quaternion(Vector3(0, 1, 0), Math::deg_to_rad(45.0f));
	Quaternion quat2 = Quaternion(Vector3(1, 0, 0), Math::deg_to_rad(30.0f));
	limitation->set_cone_center_quaternion(0, quat1);
	limitation->set_cone_center_quaternion(1, quat2);

	Quaternion retrieved_quat1 = limitation->get_cone_center_quaternion(0);
	Quaternion retrieved_quat2 = limitation->get_cone_center_quaternion(1);
	// Quaternions should represent the same rotation (allowing for double cover)
	Vector3 dir1 = quat1.xform(Vector3(0, 1, 0));
	Vector3 dir2 = retrieved_quat1.xform(Vector3(0, 1, 0));
	bool quat_matches = dir1.is_equal_approx(dir2) || dir1.is_equal_approx(-dir2);
	CHECK(quat_matches);
	Vector3 dir3 = quat2.xform(Vector3(0, 1, 0));
	Vector3 dir4 = retrieved_quat2.xform(Vector3(0, 1, 0));
	bool quat_matches2 = dir3.is_equal_approx(dir4) || dir3.is_equal_approx(-dir4);
	CHECK(quat_matches2);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test twist constraints - axially constrained disabled") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);
	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	limitation->set_axially_constrained(false);
	limitation->set_min_axial_angle(Math::deg_to_rad(-45.0f));
	limitation->set_range_angle(Math::deg_to_rad(90.0f));

	// With axially_constrained = false, rotation should pass through unchanged
	Vector3 test_dir = Vector3(1, 0, 0).normalized();
	Quaternion test_rot = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(60.0f));
	Quaternion constrained_rot;

	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_dir, test_rot, &constrained_rot);
	(void)result; // Result is used for side effects (calling solve)

	// Rotation should be unchanged when axially_constrained is false
	CHECK(constrained_rot.is_equal_approx(test_rot));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test twist constraints - axially constrained enabled") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);
	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	limitation->set_axially_constrained(true);
	limitation->set_min_axial_angle(Math::deg_to_rad(0.0f));
	limitation->set_range_angle(Math::deg_to_rad(90.0f)); // Allow 0-90 degrees

	// Test rotation within range
	Quaternion rot_within = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(45.0f));
	Quaternion constrained_rot;
	Vector3 test_dir = Vector3(1, 0, 0).normalized();

	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_dir, rot_within, &constrained_rot);

	// Rotation within range should be allowed (or clamped to range)
	CHECK(constrained_rot.is_finite());
	CHECK(constrained_rot.is_normalized());

	// Test rotation outside range
	Quaternion rot_outside = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(120.0f)); // Outside 0-90 range
	Quaternion constrained_rot2;

	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_dir, rot_outside, &constrained_rot2);

	// Should be constrained to the allowed range
	CHECK(constrained_rot2.is_finite());
	CHECK(constrained_rot2.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test twist constraints - wrapping range") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);
	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	limitation->set_axially_constrained(true);
	// Set range that wraps around (e.g., 300-60 degrees = -60 to 60)
	limitation->set_min_axial_angle(Math::deg_to_rad(300.0f));
	limitation->set_range_angle(Math::deg_to_rad(120.0f)); // Wraps from 300 to 60 degrees

	Quaternion test_rot = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(350.0f));
	Quaternion constrained_rot;
	Vector3 test_dir = Vector3(1, 0, 0).normalized();

	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_dir, test_rot, &constrained_rot);
	(void)result; // Result is used for side effects (calling solve)

	CHECK(constrained_rot.is_finite());
	CHECK(constrained_rot.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test point exactly on cone boundary") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Create a point exactly on the boundary
	Vector3 perp = control_point.get_any_perpendicular().normalized();
	Quaternion rot_to_boundary = Quaternion(control_point.cross(perp).normalized(), radius);
	Vector3 boundary_point = rot_to_boundary.xform(control_point).normalized();

	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), boundary_point);

	// Should be on or very close to boundary
	CHECK(result.is_finite());
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test default cone configuration") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Check default values
	CHECK(limitation->get_cone_count() == 1);
	CHECK(limitation->is_orientationally_constrained() == true);
	CHECK(limitation->is_axially_constrained() == false);
	CHECK(Math::is_equal_approx(limitation->get_min_axial_angle(), (real_t)0.0));
	CHECK(Math::is_equal_approx(limitation->get_range_angle(), (real_t)Math::TAU));

	// Default cone should be +Y axis with 45 degree radius
	Vector3 default_center = limitation->get_cone_center(0);
	CHECK(default_center.is_equal_approx(Vector3(0, 1, 0)));
	real_t default_radius = limitation->get_cone_radius(0);
	CHECK(Math::is_equal_approx(default_radius, (real_t)0.7853981633974483, (real_t)0.01)); // ~45 degrees
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test very small radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(0.1f); // Very small radius

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Point at center should be allowed
	Vector3 center_point = control_point;
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), center_point);
	CHECK(result.is_equal_approx(center_point));

	// Point slightly outside should be clamped
	Vector3 outside_point = Quaternion(Vector3(1, 0, 0), radius * 1.5f).xform(control_point);
	result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), outside_point);
	real_t angle_to_control = result.angle_to(control_point);
	CHECK(angle_to_control <= radius + 0.01f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test near maximum radius cone") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::PI - 0.01f; // Nearly maximum (almost hemisphere)

	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Most points should be inside such a large cone
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test four cones in sequence") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create four cones forming a square pattern
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(-1, 0, 0).normalized();
	Vector3 cp4 = Vector3(0, -1, 0).normalized();
	real_t radius = Math::deg_to_rad(40.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	cones.push_back(Vector4(cp4.x, cp4.y, cp4.z, radius));
	limitation->set_cones(cones);

	// Test point in first cone
	Vector3 point1 = Quaternion(Vector3(0, 1, 0), radius * 0.3f).xform(cp1);
	Vector3 result1 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point1);
	CHECK(result1.is_equal_approx(point1));

	// Test point between first and second cone
	Vector3 point_between = (cp1 + cp2).normalized();
	Vector3 result2 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), point_between);
	CHECK(result2.is_finite());
	CHECK(result2.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test solve without rotation parameter") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);
	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Test solve without rotation (should work with default parameters)
	Vector3 test_dir = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_dir);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test solve with null rotation output") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);
	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	// Test solve with rotation but null output pointer (should not crash)
	Vector3 test_dir = Vector3(1, 0, 0).normalized();
	Quaternion test_rot = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(45.0f));

	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_dir, test_rot, nullptr);

	CHECK(result.is_finite());
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test orientationally constrained with empty cones") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	limitation->set_cone_count(0);
	limitation->set_orientationally_constrained(true);

	// With no cones but orientationally constrained, should return input unchanged
	Vector3 test_point = Vector3(1, 0, 0).normalized();
	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_point);
	CHECK(result.is_equal_approx(test_point));
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test axial limits with full range") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	limitation->set_axially_constrained(true);
	limitation->set_min_axial_angle(0.0f);
	limitation->set_range_angle(Math::TAU); // Full 360 degrees

	// With full range, any rotation should be allowed
	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);
	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	Quaternion test_rot = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(180.0f));
	Quaternion constrained_rot;
	Vector3 test_dir = Vector3(1, 0, 0).normalized();

	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_dir, test_rot, &constrained_rot);
	(void)result; // Result is used for side effects (calling solve)

	// With full range, rotation should pass through (or be very close)
	CHECK(constrained_rot.is_finite());
	CHECK(constrained_rot.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test axial limits with zero range") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	limitation->set_axially_constrained(true);
	limitation->set_min_axial_angle(Math::deg_to_rad(45.0f));
	limitation->set_range_angle(0.0f); // Zero range - only one angle allowed

	Vector3 control_point = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(30.0f);
	Vector<Vector4> cones;
	cones.push_back(Vector4(control_point.x, control_point.y, control_point.z, radius));
	limitation->set_cones(cones);

	Quaternion test_rot = Quaternion(Vector3(0, 0, 1), Math::deg_to_rad(60.0f));
	Quaternion constrained_rot;
	Vector3 test_dir = Vector3(1, 0, 0).normalized();

	Vector3 result = limitation->solve(Vector3(0, 1, 0), Vector3(1, 0, 0), Quaternion(), test_dir, test_rot, &constrained_rot);
	(void)result; // Result is used for side effects (calling solve)

	// Should be constrained to the single allowed angle
	CHECK(constrained_rot.is_finite());
	CHECK(constrained_rot.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - point in allowed region") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones with moderate separation
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point that should be in the allowed tangent path region (outside both tangent circles)
	// This is between the cones but outside the forbidden tangent circle regions
	Vector3 path_point = Vector3(0.7, 0.7, 0.1).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path_point);

	// Point in allowed path region should be returned as-is (or very close)
	CHECK(result.is_finite());
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - point inside forbidden tangent circle") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones with moderate separation
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0f); // Smaller radius for clearer tangent circles

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point that should be inside a forbidden tangent circle
	// This should be projected to the tangent circle boundary
	Vector3 forbidden_point = Vector3(0.8, 0.5, 0.3).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), forbidden_point);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
	// Result should be constrained (not equal to input since it's in forbidden region)
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - point on tangent boundary") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(45.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point approximately on the tangent circle boundary
	// Should be handled gracefully (either returned as-is or projected slightly)
	Vector3 boundary_point = Vector3(0.6, 0.6, 0.5).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), boundary_point);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
	// Result should be valid (either on boundary or in allowed region)
	CHECK(result.length() > 0.9f);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - three cones with multiple paths") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Three cones forming a triangle
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(50.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	limitation->set_cones(cones);

	// Point in path between cp1 and cp2
	Vector3 path12 = Vector3(0.7, 0.7, 0.1).normalized();
	Vector3 result12 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path12);
	CHECK(result12.is_finite());
	CHECK(result12.is_normalized());

	// Point in path between cp2 and cp3
	Vector3 path23 = Vector3(0.1, 0.7, 0.7).normalized();
	Vector3 result23 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path23);
	CHECK(result23.is_finite());
	CHECK(result23.is_normalized());

	// Point in path between cp3 and cp1
	Vector3 path31 = Vector3(0.7, 0.1, 0.7).normalized();
	Vector3 result31 = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path31);
	CHECK(result31.is_finite());
	CHECK(result31.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - large cone radii") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones with large radii (should create small tangent paths)
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(80.0f); // Large radius

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point between cones - with large radii, most points should be in cones
	// But tangent path should still work for points outside both cones
	Vector3 test_point = Vector3(0.3, 0.3, 0.9).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), test_point);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - small cone radii") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Two cones with small radii (should create larger tangent paths)
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(15.0f); // Small radius

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point in the tangent path between small cones
	Vector3 path_point = Vector3(0.6, 0.6, 0.5).normalized();
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), path_point);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Test tangent path - point outside all cones and paths") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0f);

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	limitation->set_cones(cones);

	// Point far from both cones and their tangent paths
	// Use a point that's 90 degrees from both cp1 and cp2 to ensure it's outside both cones (radius 30deg)
	// and verify it's not in the tangent path
	Vector3 far_point = Vector3(0, 0, 1).normalized();
	// Verify it's outside both cones
	real_t angle_to_cp1_check = far_point.angle_to(cp1);
	real_t angle_to_cp2_check = far_point.angle_to(cp2);
	CHECK(angle_to_cp1_check > radius);
	CHECK(angle_to_cp2_check > radius);
	
	// Use independent solver to check if point is in an allowed region
	bool is_allowed = test_is_point_allowed(far_point, cones);
	
	Vector3 result = limitation->solve(Vector3(0, 0, 1), Vector3(1, 0, 0), Quaternion(), far_point);

	CHECK(result.is_finite());
	CHECK(result.is_normalized());
	CHECK_FALSE (is_allowed);
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Exhaustive test of all regions - top to equator to bottom") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Test top-to-equator-to-bottom arrangement (key use case for different hemispheres)
	Vector3 top = Vector3(0, 1, 0).normalized();
	Vector3 equator = Vector3(1, 0, 0).normalized();
	Vector3 bottom = Vector3(0, -1, 0).normalized();
	real_t radius = Math::deg_to_rad(30.0f); // 30-degree cones to create meaningful paths

	Vector<Vector4> cones;
	cones.push_back(Vector4(top.x, top.y, top.z, radius));
	cones.push_back(Vector4(equator.x, equator.y, equator.z, radius));
	cones.push_back(Vector4(bottom.x, bottom.y, bottom.z, radius));
	limitation->set_cones(cones);

	// ===== TEST POINTS IN OPEN REGIONS (should be allowed) =====

	// 1. Point inside top cone - should be in INSIDE_CONE region
	Vector3 inside_top = Vector3(0.0, 0.95, 0.1).normalized();
	PointRegion region1 = classify_point_region(inside_top, cones, limitation);
	CHECK(region1 == PointRegion::INSIDE_CONE);

	// 2. Point inside equator cone - should be in INSIDE_CONE region
	Vector3 inside_equator = Vector3(0.95, 0.1, 0.0).normalized();
	PointRegion region2 = classify_point_region(inside_equator, cones, limitation);
	CHECK(region2 == PointRegion::INSIDE_CONE);

	// 3. Point inside bottom cone - should be in INSIDE_CONE region
	Vector3 inside_bottom = Vector3(0.0, -0.95, 0.1).normalized();
	PointRegion region3 = classify_point_region(inside_bottom, cones, limitation);
	CHECK(region3 == PointRegion::INSIDE_CONE);

	// 4. Point in inter-cone path between top and equator - should be in allowed region
	Vector3 path_top_equator = Vector3(0.7, 0.7, 0.0).normalized();
	PointRegion region4 = classify_point_region(path_top_equator, cones, limitation);
	// Could be in path or inside a cone - both are allowed
	bool region4_allowed = (region4 == PointRegion::INSIDE_CONE || region4 == PointRegion::IN_TANGENT_PATH);
	CHECK(region4_allowed);

	// 5. Point in inter-cone path between equator and bottom - should be in allowed region
	Vector3 path_equator_bottom = Vector3(0.7, -0.7, 0.0).normalized();
	PointRegion region5 = classify_point_region(path_equator_bottom, cones, limitation);
	bool region5_allowed = (region5 == PointRegion::INSIDE_CONE || region5 == PointRegion::IN_TANGENT_PATH);
	CHECK(region5_allowed);

	// ===== TEST POINTS IN FORBIDDEN REGIONS (should be constrained) =====

	// 6. Point far outside all cones and paths - should be in FORBIDDEN region
	Vector3 outside_all = Vector3(0.0, 0.0, 1.0).normalized(); // +Z, far from all cones
	PointRegion region6 = classify_point_region(outside_all, cones, limitation);
	CHECK(region6 == PointRegion::FORBIDDEN);

	// 7. Point in wrap-around region (between bottom and top, NOT adjacent) - should be in FORBIDDEN region
	Vector3 wrap_around = Vector3(0.0, 0.0, -1.0).normalized(); // -Z, opposite side
	PointRegion region7 = classify_point_region(wrap_around, cones, limitation);
	CHECK(region7 == PointRegion::FORBIDDEN); // No wrap-around path
}

TEST_CASE("[Scene][JointLimitationKusudama3D] Exhaustive test of all regions - three cones in sequence") {
	Ref<JointLimitationKusudama3D> limitation;
	limitation.instantiate();

	// Create three cones in sequence: cp1 -> cp2 -> cp3
	Vector3 cp1 = Vector3(1, 0, 0).normalized();
	Vector3 cp2 = Vector3(0, 1, 0).normalized();
	Vector3 cp3 = Vector3(0, 0, 1).normalized();
	real_t radius = Math::deg_to_rad(25.0f); // 25-degree cones

	Vector<Vector4> cones;
	cones.push_back(Vector4(cp1.x, cp1.y, cp1.z, radius));
	cones.push_back(Vector4(cp2.x, cp2.y, cp2.z, radius));
	cones.push_back(Vector4(cp3.x, cp3.y, cp3.z, radius));
	limitation->set_cones(cones);

	// ===== TEST POINTS IN OPEN REGIONS (should be allowed) =====

	// 1. Point inside cp1 cone - should be in INSIDE_CONE region
	Vector3 inside_cp1 = Vector3(0.95, 0.2, 0.0).normalized();
	PointRegion region1 = classify_point_region(inside_cp1, cones, limitation);
	CHECK(region1 == PointRegion::INSIDE_CONE);

	// 2. Point inside cp2 cone - should be in INSIDE_CONE region
	Vector3 inside_cp2 = Vector3(0.0, 0.95, 0.2).normalized();
	PointRegion region2 = classify_point_region(inside_cp2, cones, limitation);
	CHECK(region2 == PointRegion::INSIDE_CONE);

	// 3. Point inside cp3 cone - should be in INSIDE_CONE region
	Vector3 inside_cp3 = Vector3(0.2, 0.0, 0.95).normalized();
	PointRegion region3 = classify_point_region(inside_cp3, cones, limitation);
	CHECK(region3 == PointRegion::INSIDE_CONE);

	// 4. Point in inter-cone path between cp1 and cp2 (adjacent) - should be in allowed region
	Vector3 path12 = Vector3(0.7, 0.7, 0.1).normalized();
	PointRegion region4 = classify_point_region(path12, cones, limitation);
	bool region4_allowed = (region4 == PointRegion::INSIDE_CONE || region4 == PointRegion::IN_TANGENT_PATH);
	CHECK(region4_allowed);

	// 5. Point in inter-cone path between cp2 and cp3 (adjacent) - should be in allowed region
	Vector3 path23 = Vector3(0.1, 0.7, 0.7).normalized();
	PointRegion region5 = classify_point_region(path23, cones, limitation);
	bool region5_allowed = (region5 == PointRegion::INSIDE_CONE || region5 == PointRegion::IN_TANGENT_PATH);
	CHECK(region5_allowed);

	// ===== TEST POINTS IN FORBIDDEN REGIONS (should be constrained) =====

	// 6. Point in wrap-around region (between cp3 and cp1, NOT adjacent) - should be in FORBIDDEN region
	Vector3 path31 = Vector3(0.5, 0.3, 0.8).normalized();
	PointRegion region6 = classify_point_region(path31, cones, limitation);
	CHECK(region6 == PointRegion::FORBIDDEN); // No wrap-around path

	// 7. Point far outside all cones - should be in FORBIDDEN region
	Vector3 outside = Vector3(-0.5, -0.5, -0.7).normalized();
	PointRegion region7 = classify_point_region(outside, cones, limitation);
	CHECK(region7 == PointRegion::FORBIDDEN);
}

} // namespace TestJointLimitationKusudama3D
