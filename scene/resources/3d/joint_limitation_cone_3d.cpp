/**************************************************************************/
/*  joint_limitation_cone_3d.cpp                                          */
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

#include "joint_limitation_cone_3d.h"
#include "core/math/ik_kusudama_3d.h"

void JointLimitationCone3D::set_radius_range(real_t p_radius_range) {
	radius_range = p_radius_range;
	emit_changed();
}

real_t JointLimitationCone3D::get_radius_range() const {
	return radius_range;
}

void JointLimitationCone3D::add_open_cone(Ref<IKLimitCone3D> p_cone) {
	if (p_cone.is_valid()) {
		open_cones.push_back(p_cone);
		emit_changed();
	}
}

void JointLimitationCone3D::remove_open_cone(Ref<IKLimitCone3D> p_cone) {
	open_cones.erase(p_cone);
	emit_changed();
}

void JointLimitationCone3D::clear_open_cones() {
	open_cones.clear();
	emit_changed();
}

TypedArray<IKLimitCone3D> JointLimitationCone3D::get_open_cones() const {
	TypedArray<IKLimitCone3D> result;
	for (const Ref<IKLimitCone3D> &cone : open_cones) {
		result.push_back(cone);
	}
	return result;
}

void JointLimitationCone3D::set_open_cones(TypedArray<IKLimitCone3D> p_cones) {
	open_cones.clear();
	for (int i = 0; i < p_cones.size(); i++) {
		Ref<IKLimitCone3D> cone = p_cones[i];
		if (cone.is_valid()) {
			open_cones.push_back(cone);
		}
	}
	emit_changed();
}

void JointLimitationCone3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius_range", "radius_range"), &JointLimitationCone3D::set_radius_range);
	ClassDB::bind_method(D_METHOD("get_radius_range"), &JointLimitationCone3D::get_radius_range);
	ClassDB::bind_method(D_METHOD("add_open_cone", "cone"), &JointLimitationCone3D::add_open_cone);
	ClassDB::bind_method(D_METHOD("remove_open_cone", "cone"), &JointLimitationCone3D::remove_open_cone);
	ClassDB::bind_method(D_METHOD("clear_open_cones"), &JointLimitationCone3D::clear_open_cones);
	ClassDB::bind_method(D_METHOD("get_open_cones"), &JointLimitationCone3D::get_open_cones);
	ClassDB::bind_method(D_METHOD("set_open_cones", "cones"), &JointLimitationCone3D::set_open_cones);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_range", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_radius_range", "get_radius_range");
}

Vector3 JointLimitationCone3D::_solve(const Vector3 &p_direction) const {
	// If no open cones are defined, fall back to simple cone behavior
	if (open_cones.is_empty()) {
		Vector3 center_axis = Vector3(0, 1, 0);
		real_t angle = p_direction.angle_to(center_axis);
		real_t max_angle = radius_range * Math::PI;

		if (angle <= max_angle) {
			return p_direction;
		}

		Vector3 plane_normal;
		if (Math::is_equal_approx((double)angle, Math::PI)) {
			plane_normal = center_axis.get_any_perpendicular();
		} else {
			plane_normal = center_axis.cross(p_direction).normalized();
		}

		Quaternion rotation = Quaternion(plane_normal, max_angle);
		Vector3 limited_dir = rotation.xform(center_axis);

		Vector3 projection = p_direction - center_axis * p_direction.dot(center_axis);
		if (projection.length_squared() > CMP_EPSILON) {
			Vector3 side_dir = projection.normalized();
			Quaternion side_rotation = Quaternion(center_axis.cross(side_dir).normalized(), max_angle);
			limited_dir = side_rotation.xform(center_axis);
		}

		return limited_dir.normalized();
	}

	// Multi-cone kuwsudama constraint solving
	// Check if the direction is within bounds using the kusudama multi-cone system
	Vector3 result = p_direction.normalized();
	Vector<double> in_bounds;

	// Try to constrain using the first open cone
	if (open_cones[0].is_valid()) {
		// Use the kusudama constraint from the first cone
		result = open_cones[0]->closest_to_cone(p_direction.normalized(), &in_bounds);

		// If the point is already in bounds, return as is
		if (!in_bounds.is_empty() && in_bounds[0] >= 0.0) {
			return result;
		}

		// Otherwise, check multi-cone path sequences for better constraint
		for (int i = 0; i < (int)open_cones.size() - 1; i++) {
			if (open_cones[i].is_valid() && open_cones[i + 1].is_valid()) {
				Vector3 path_result = open_cones[i]->get_closest_path_point(open_cones[i + 1], p_direction.normalized());

				// Use the result that requires least rotation
				real_t orig_angle = p_direction.normalized().angle_to(path_result);
				real_t prev_angle = result.angle_to(path_result);

				if (orig_angle < prev_angle) {
					result = path_result;
				}
			}
		}
	}

	return result.normalized();
}

#ifdef TOOLS_ENABLED
void JointLimitationCone3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color) const {
	static const int N = 16;
	static const real_t DP = Math::TAU / (real_t)N;

	real_t sphere_r = p_bone_length * (real_t)0.25;
	if (sphere_r <= CMP_EPSILON) {
		return;
	}

	LocalVector<Vector3> vts;

	// If multi-cone constraints exist, visualize each open cone
	if (!open_cones.is_empty()) {
		for (int cone_idx = 0; cone_idx < (int)open_cones.size(); cone_idx++) {
			Ref<IKLimitCone3D> cone = open_cones[cone_idx];
			if (!cone.is_valid()) {
				continue;
			}

			// Get cone parameters
			Vector3 control_point = cone->get_control_point();
			real_t radius = cone->get_radius();

			// Normalize control point for sphere visualization
			Vector3 cone_center = control_point.normalized() * sphere_r;
			real_t cone_radius = radius * sphere_r;

			// Draw cone boundary circle
			for (int i = 0; i < N; i++) {
				real_t a0 = (real_t)i * DP;
				real_t a1 = (real_t)((i + 1) % N) * DP;

				// Create perpendicular vectors for cone circle
				Vector3 perp1 = cone_center.get_any_perpendicular().normalized();
				Vector3 perp2 = cone_center.cross(perp1).normalized();

				Vector3 p0 = cone_center + perp1 * cone_radius * Math::cos(a0) + perp2 * cone_radius * Math::sin(a0);
				Vector3 p1 = cone_center + perp1 * cone_radius * Math::cos(a1) + perp2 * cone_radius * Math::sin(a1);

				vts.push_back(p0);
				vts.push_back(p1);
			}

			// Draw lines from origin to cone boundary
			for (int i = 0; i < N; i += 4) {
				real_t angle = (real_t)i * DP;
				Vector3 perp1 = cone_center.get_any_perpendicular().normalized();
				Vector3 perp2 = cone_center.cross(perp1).normalized();
				Vector3 boundary = cone_center + perp1 * cone_radius * Math::cos(angle) + perp2 * cone_radius * Math::sin(angle);

				vts.push_back(Vector3());
				vts.push_back(boundary);
			}

			// Draw tangent circle handles if available
			Vector3 tc1 = cone->get_tangent_circle_center_next_1();
			Vector3 tc2 = cone->get_tangent_circle_center_next_2();
			real_t tc_radius = cone->get_tangent_circle_radius_next();

			if (tc_radius > CMP_EPSILON) {
				// Draw tangent circle 1
				if (tc1.length() > CMP_EPSILON) {
					Vector3 tc1_norm = tc1.normalized() * sphere_r;
					real_t tc1_r = tc_radius * sphere_r;

					for (int i = 0; i < N; i++) {
						real_t a0 = (real_t)i * DP;
						real_t a1 = (real_t)((i + 1) % N) * DP;

						Vector3 tc_perp1 = tc1_norm.get_any_perpendicular().normalized();
						Vector3 tc_perp2 = tc1_norm.cross(tc_perp1).normalized();

						Vector3 p0 = tc1_norm + tc_perp1 * tc1_r * Math::cos(a0) + tc_perp2 * tc1_r * Math::sin(a0);
						Vector3 p1 = tc1_norm + tc_perp1 * tc1_r * Math::cos(a1) + tc_perp2 * tc1_r * Math::sin(a1);

						vts.push_back(p0);
						vts.push_back(p1);
					}
				}

				// Draw tangent circle 2
				if (tc2.length() > CMP_EPSILON) {
					Vector3 tc2_norm = tc2.normalized() * sphere_r;
					real_t tc2_r = tc_radius * sphere_r;

					for (int i = 0; i < N; i++) {
						real_t a0 = (real_t)i * DP;
						real_t a1 = (real_t)((i + 1) % N) * DP;

						Vector3 tc_perp1 = tc2_norm.get_any_perpendicular().normalized();
						Vector3 tc_perp2 = tc2_norm.cross(tc_perp1).normalized();

						Vector3 p0 = tc2_norm + tc_perp1 * tc2_r * Math::cos(a0) + tc_perp2 * tc2_r * Math::sin(a0);
						Vector3 p1 = tc2_norm + tc_perp1 * tc2_r * Math::cos(a1) + tc_perp2 * tc2_r * Math::sin(a1);

						vts.push_back(p0);
						vts.push_back(p1);
					}
				}
			}
		}
	} else {
		// Fallback to single cone visualization if no open cones
		real_t alpha = CLAMP((real_t)radius_range, (real_t)0.0, (real_t)1.0) * Math::PI;
		real_t y_cap = sphere_r * Math::cos(alpha);
		real_t r_cap = sphere_r * Math::sin(alpha);

		// Cone bottom.
		if (r_cap > CMP_EPSILON) {
			for (int i = 0; i < N; i++) {
				real_t a0 = (real_t)i * DP;
				real_t a1 = (real_t)((i + 1) % N) * DP;
				Vector3 p0 = Vector3(r_cap * Math::cos(a0), y_cap, r_cap * Math::sin(a0));
				Vector3 p1 = Vector3(r_cap * Math::cos(a1), y_cap, r_cap * Math::sin(a1));
				vts.push_back(p0);
				vts.push_back(p1);
			}
		}

		// Rotate arcs around Y-axis.
		real_t t_start;
		real_t arc_len;
		if (alpha <= (real_t)1e-6) {
			t_start = (real_t)0.5 * Math::PI;
			arc_len = Math::PI;
		} else {
			t_start = (real_t)0.5 * Math::PI + alpha;
			arc_len = Math::PI - alpha;
		}
		real_t dt = arc_len / (real_t)N;

		for (int k = 0; k < N; k++) {
			Basis ry(Vector3(0, 1, 0), (real_t)k * DP);

			Vector3 prev = ry.xform(Vector3(sphere_r * Math::cos(t_start), sphere_r * Math::sin(t_start), 0));

			for (int s = 1; s <= N; s++) {
				real_t t = t_start + dt * (real_t)s;
				Vector3 cur = ry.xform(Vector3(sphere_r * Math::cos(t), sphere_r * Math::sin(t), 0));

				vts.push_back(prev);
				vts.push_back(cur);

				prev = cur;
			}

			Vector3 mouth = ry.xform(Vector3(sphere_r * Math::cos(t_start), sphere_r * Math::sin(t_start), 0));
			Vector3 center = Vector3();

			vts.push_back(center);
			vts.push_back(mouth);
		}

		// Stack rings.
		for (int i = 1; i <= 3; i++) {
			for (int sgn = -1; sgn <= 1; sgn += 2) {
				real_t y = (real_t)sgn * sphere_r * ((real_t)i / (real_t)4.0);
				if (y >= y_cap - CMP_EPSILON) {
					continue;
				}
				real_t ring_r2 = sphere_r * sphere_r - y * y;
				if (ring_r2 <= (real_t)0.0) {
					continue;
				}
				real_t ring_r = Math::sqrt(ring_r2);

				for (int j = 0; j < N; j++) {
					real_t a0 = (real_t)j * DP;
					real_t a1 = (real_t)((j + 1) % N) * DP;
					Vector3 p0 = Vector3(ring_r * Math::cos(a0), y, ring_r * Math::sin(a0));
					Vector3 p1 = Vector3(ring_r * Math::cos(a1), y, ring_r * Math::sin(a1));

					vts.push_back(p0);
					vts.push_back(p1);
				}
			}
		}
	}

	for (int64_t i = 0; i < vts.size(); i++) {
		p_surface_tool->set_color(p_color);
		p_surface_tool->add_vertex(p_transform.xform(vts[i]));
	}
}
#endif // TOOLS_ENABLED
