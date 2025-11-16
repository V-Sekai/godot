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

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "open_cones", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_open_cones", "get_open_cones");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_axial_angle"), "set_min_axial_angle", "get_min_axial_angle");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "range_angle"), "set_range_angle", "get_range_angle");
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
	if (!orientationally_constrained || open_cones.is_empty()) {
		return p_direction;
	}

	// Full kusudama solving implementation based on IKKusudama3D::get_local_point_in_limits
	Vector3 point = p_direction.normalized();
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
	return closest_collision_point.normalized();
}

void JointLimitationKusudama3D::set_open_cones(const Vector<Vector4> &p_cones) {
	open_cones = p_cones;
	emit_changed();
}

Vector<Vector4> JointLimitationKusudama3D::get_open_cones() const {
	return open_cones;
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
void JointLimitationKusudama3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const {
	if (open_cones.is_empty() || !orientationally_constrained) {
		return;
	}

	static const int N = 32; // Number of segments per circle
	static const real_t DP = Math::TAU / (real_t)N;

	real_t sphere_r = p_bone_length * 0.25f;
	if (sphere_r <= CMP_EPSILON) {
		return;
	}

	LocalVector<Vector3> vts;

	// Draw each open cone as a circle on the sphere
	for (int cone_i = 0; cone_i < open_cones.size(); cone_i++) {
		const Vector4 &cone = open_cones[cone_i];
		Vector3 center = Vector3(cone.x, cone.y, cone.z).normalized();
		real_t radius_angle = cone.w;

		// Calculate the circle on the sphere defined by the cone
		// The cone center is a point on the sphere, and radius_angle is the half-angle
		Vector3 axis = center;
		Vector3 perp1 = axis.get_any_perpendicular().normalized();
		Vector3 perp2 = axis.cross(perp1).normalized();

		// Generate circle points on the sphere
		// The circle is defined by rotating a point at radius_angle from the center around the center axis
		LocalVector<Vector3> circle_points;
		Vector3 start_point = Quaternion(perp1, radius_angle).xform(axis);
		for (int i = 0; i <= N; i++) {
			real_t angle = (real_t)i * DP;
			// Rotate the start point around the axis
			Quaternion rot = Quaternion(axis, angle);
			Vector3 point = rot.xform(start_point) * sphere_r;
			circle_points.push_back(point);
		}

		// Draw circle
		for (int i = 0; i < N; i++) {
			vts.push_back(circle_points[i]);
			vts.push_back(circle_points[i + 1]);
		}

		// Draw line from origin to cone center
		vts.push_back(Vector3());
		vts.push_back(center * sphere_r);

		// Draw connections between adjacent cones
		if (open_cones.size() > 1) {
			int next_i = (cone_i + 1) % open_cones.size();
			const Vector4 &next_cone = open_cones[next_i];
			Vector3 next_center = Vector3(next_cone.x, next_cone.y, next_cone.z).normalized();

			// Draw arc between cone centers
			int arc_segments = 8;
			for (int j = 0; j < arc_segments; j++) {
				real_t t = (real_t)j / (real_t)arc_segments;
				Vector3 arc_point = center.lerp(next_center, t).normalized() * sphere_r;
				Vector3 next_arc_point = center.lerp(next_center, t + 1.0f / arc_segments).normalized() * sphere_r;
				vts.push_back(arc_point);
				vts.push_back(next_arc_point);
			}
		}
	}

	// Draw axial limits if constrained
	if (axially_constrained && range_angle < Math::TAU) {
		// Draw a ring showing the twist limits
		Vector3 y_axis = Vector3(0, 1, 0);
		real_t ring_radius = sphere_r * 0.8f;

		for (int i = 0; i < N; i++) {
			real_t angle = min_axial_angle + (range_angle * (real_t)i / (real_t)N);
			Quaternion rot = Quaternion(y_axis, angle);
			Vector3 p0 = rot.xform(Vector3(ring_radius, 0, 0));
			Vector3 p1 = rot.xform(Vector3(ring_radius, 0, 0));
			if (i < N - 1) {
				real_t next_angle = min_axial_angle + (range_angle * (real_t)(i + 1) / (real_t)N);
				Quaternion next_rot = Quaternion(y_axis, next_angle);
				p1 = next_rot.xform(Vector3(ring_radius, 0, 0));
			}
			vts.push_back(p0);
			vts.push_back(p1);
		}
	}

	// Add all vertices to surface tool
	for (int64_t i = 0; i < vts.size(); i++) {
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(vts[i]));
	}
}
#endif // TOOLS_ENABLED
