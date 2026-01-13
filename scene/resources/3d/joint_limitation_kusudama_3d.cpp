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
#include "core/variant/variant.h"

void JointLimitationKusudama3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cones", "cones"), &JointLimitationKusudama3D::set_cones);
	ClassDB::bind_method(D_METHOD("get_cones"), &JointLimitationKusudama3D::get_cones);

	ClassDB::bind_method(D_METHOD("set_cone_count", "count"), &JointLimitationKusudama3D::set_cone_count);
	ClassDB::bind_method(D_METHOD("get_cone_count"), &JointLimitationKusudama3D::get_cone_count);
	ClassDB::bind_method(D_METHOD("set_cone_center", "index", "center"), &JointLimitationKusudama3D::set_cone_center);
	ClassDB::bind_method(D_METHOD("get_cone_center", "index"), &JointLimitationKusudama3D::get_cone_center);
	ClassDB::bind_method(D_METHOD("set_cone_radius", "index", "radius"), &JointLimitationKusudama3D::set_cone_radius);
	ClassDB::bind_method(D_METHOD("get_cone_radius", "index"), &JointLimitationKusudama3D::get_cone_radius);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "cones", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_cones", "get_cones");
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
	// Initialize new cones with default normalized values
	for (int i = old_size; i < cones.size(); i++) {
		cones.write[i] = Vector4(0, 1, 0, Math::PI * 0.25); // Default: +Y axis (normalized), 45 degree cone
	}
	notify_property_list_changed();
	emit_changed();
}

int JointLimitationKusudama3D::get_cone_count() const {
	return cones.size();
}

void JointLimitationKusudama3D::set_cone_center(int p_index, const Vector3 &p_center) {
	ERR_FAIL_INDEX(p_index, cones.size());
	// Normalize and store the center direction
	Vector3 normalized_center = p_center;
	if (normalized_center.length_squared() > CMP_EPSILON) {
		normalized_center.normalize();
	} else {
		normalized_center = Vector3(0, 1, 0); // Default fallback
	}
	Vector4 &cone = cones.write[p_index];
	cone.x = normalized_center.x;
	cone.y = normalized_center.y;
	cone.z = normalized_center.z;
	emit_changed();
}

Vector3 JointLimitationKusudama3D::get_cone_center(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, cones.size(), Vector3(0, 1, 0));
	// Return the stored normalized center
	const Vector4 &cone_data = cones[p_index];
	return Vector3(cone_data.x, cone_data.y, cone_data.z);
}

void JointLimitationKusudama3D::set_cone_center_quaternion(int p_index, const Quaternion &p_quaternion) {
	ERR_FAIL_INDEX(p_index, cones.size());
	// Convert quaternion to direction vector by rotating the default direction (0, 1, 0)
	Vector3 default_dir = Vector3(0, 1, 0);
	Vector3 center = p_quaternion.normalized().xform(default_dir);
	// Normalize and store
	if (center.length_squared() > CMP_EPSILON) {
		center.normalize();
	} else {
		center = Vector3(0, 1, 0); // Default fallback
	}
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

Vector3 JointLimitationKusudama3D::_solve(const Vector3 &p_direction) const {
	Vector3 result = p_direction.normalized();

	// Guard: No constraints applied
	if (cones.is_empty()) {
		return result;
	}

	// Guard: Check if point is within any cone
	for (int i = 0; i < cones.size(); i++) {
		const Vector4 &cone_data = cones[i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z);
		real_t radius = cone_data.w;

		if (is_point_in_cone(result, center, radius)) {
			return result;
		}
	}

	// Guard: Check if point is in any path between adjacent cones
	if (cones.size() > 1) {
		for (int i = 0; i < cones.size() - 1; i++) {
			int next_i = i + 1;
			const Vector4 &cone_data1 = cones[i];
			const Vector4 &cone_data2 = cones[next_i];

			Vector3 center1 = Vector3(cone_data1.x, cone_data1.y, cone_data1.z);
			real_t radius1 = cone_data1.w;

			Vector3 center2 = Vector3(cone_data2.x, cone_data2.y, cone_data2.z);
			real_t radius2 = cone_data2.w;

			if (is_point_in_tangent_path(result, center1, radius1, center2, radius2)) {
				return result;
			}
		}
	}

	// Point is outside all allowed regions, find closest boundary point
	real_t closest_distance = INFINITY;
	Vector3 closest_point = result;

	// Check distance to cone boundaries
	for (int i = 0; i < cones.size(); i++) {
		const Vector4 &cone_data = cones[i];
		Vector3 center = Vector3(cone_data.x, cone_data.y, cone_data.z);
		real_t radius = cone_data.w;

		// Find closest point on this cone's boundary
		Vector3 projected = result - center * result.dot(center);
		if (projected.length_squared() < CMP_EPSILON) {
			// Point is along the control point axis
			projected = center.cross(Vector3(0, 1, 0));
			if (projected.length_squared() < CMP_EPSILON) {
				projected = center.cross(Vector3(1, 0, 0));
			}
			projected.normalize();
		} else {
			projected.normalize();
		}

		// Point on boundary
		Vector3 boundary_point = center * Math::cos(radius) + projected * Math::sin(radius);
		boundary_point.normalize();

		real_t distance = result.distance_to(boundary_point);
		if (distance < closest_distance) {
			closest_distance = distance;
			closest_point = boundary_point;
		}
	}

	// Check distance to path boundaries
	if (cones.size() > 1) {
		for (int i = 0; i < cones.size() - 1; i++) {
			int next_i = i + 1;
			const Vector4 &cone_data1 = cones[i];
			const Vector4 &cone_data2 = cones[next_i];

			Vector3 center1 = Vector3(cone_data1.x, cone_data1.y, cone_data1.z);
			real_t radius1 = cone_data1.w;

			Vector3 center2 = Vector3(cone_data2.x, cone_data2.y, cone_data2.z);
			real_t radius2 = cone_data2.w;

			// Find closest point on the tangent path boundary
			Vector3 path_boundary = get_on_great_tangent_triangle(result, center1, radius1, center2, radius2);
			if (!Math::is_nan(path_boundary.x)) {
				real_t distance = result.distance_to(path_boundary);
				if (distance < closest_distance) {
					closest_distance = distance;
					closest_point = path_boundary;
				}
			}
		}
	}

	result = closest_point;

	return result;
}

// Helper functions for kusudama solving

#ifdef TOOLS_ENABLED
void JointLimitationKusudama3D::draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color, int p_bone_index) const {
	real_t sphere_r = p_bone_length * (real_t)0.25;
	if (sphere_r <= CMP_EPSILON) {
		return;
	}

	// Draw subdivided icosahedron sphere.
	LocalVector<Segment> icosahedron_lines = get_icosahedron_sphere(3);
	LocalVector<Vector3> crossed_points;

	// Only cull lines if we have cones to constrain
	if (!cones.is_empty()) {
		icosahedron_lines = cull_lines_by_boundary(icosahedron_lines, crossed_points);
		crossed_points = sort_by_nearest_point(crossed_points);
	}

	p_surface_tool->set_color(p_color);
	for (const Segment &seg : icosahedron_lines) {
		p_surface_tool->add_vertex(p_transform.xform(seg.first * sphere_r));
		p_surface_tool->add_vertex(p_transform.xform(seg.second * sphere_r));
	}
	p_surface_tool->set_color(Color(1.0, 0.0, 0.0, 1.0));
	if (!crossed_points.is_empty()) {
		for (uint32_t i = 0; i < crossed_points.size(); i++) {
			p_surface_tool->add_vertex(p_transform.xform(crossed_points[i] * sphere_r));
			p_surface_tool->add_vertex(p_transform.xform(crossed_points[(i + 1) % crossed_points.size()] * sphere_r));
		}
	}
}

LocalVector<JointLimitationKusudama3D::Segment> JointLimitationKusudama3D::get_icosahedron_sphere(int p_subdiv) const {
	// TODO: Define icosahedron statically in the header.
	// Make subdivided icosahedron sphere.
	// All points' length are 1.0 from 0.0.
	LocalVector<Segment> ret;

	if (p_subdiv < 0) {
		p_subdiv = 0;
	}

	// Base icosahedron (unit sphere).
	// Vertex set: (±1, ±φ, 0), (0, ±1, ±φ), (±φ, 0, ±1)
	const real_t phi = ((real_t)1.0 + Math::sqrt((real_t)5.0)) * (real_t)0.5;
	Vector3 v[12] = {
		Vector3(-1, phi, 0),
		Vector3(1, phi, 0),
		Vector3(-1, -phi, 0),
		Vector3(1, -phi, 0),
		Vector3(0, -1, phi),
		Vector3(0, 1, phi),
		Vector3(0, -1, -phi),
		Vector3(0, 1, -phi),
		Vector3(phi, 0, -1),
		Vector3(phi, 0, 1),
		Vector3(-phi, 0, -1),
		Vector3(-phi, 0, 1)
	};
	for (int i = 0; i < 12; i++) {
		v[i].normalize();
	}

	// Faces (20 triangles).
	static const int faces[20][3] = {
		{ 0, 11, 5 },
		{ 0, 5, 1 },
		{ 0, 1, 7 },
		{ 0, 7, 10 },
		{ 0, 10, 11 },
		{ 1, 5, 9 },
		{ 5, 11, 4 },
		{ 11, 10, 2 },
		{ 10, 7, 6 },
		{ 7, 1, 8 },
		{ 3, 9, 4 },
		{ 3, 4, 2 },
		{ 3, 2, 6 },
		{ 3, 6, 8 },
		{ 3, 8, 9 },
		{ 4, 9, 5 },
		{ 2, 4, 11 },
		{ 6, 2, 10 },
		{ 8, 6, 7 },
		{ 9, 8, 1 }
	};

	// Helper: subdivide a triangle and push its edges as line segments.
	// NOTE: We intentionally allow duplicated edges; this is a wireframe gizmo.
	auto subdivide = [&](const Vector3 &a, const Vector3 &b, const Vector3 &c, int depth, auto &&subdivide_ref) -> void {
		if (depth <= 0) {
			ret.push_back(Segment{ a, b });
			ret.push_back(Segment{ b, c });
			ret.push_back(Segment{ c, a });
			return;
		}
		Vector3 ab = (a + b) * (real_t)0.5;
		Vector3 bc = (b + c) * (real_t)0.5;
		Vector3 ca = (c + a) * (real_t)0.5;
		ab.normalize();
		bc.normalize();
		ca.normalize();
		subdivide_ref(a, ab, ca, depth - 1, subdivide_ref);
		subdivide_ref(b, bc, ab, depth - 1, subdivide_ref);
		subdivide_ref(c, ca, bc, depth - 1, subdivide_ref);
		subdivide_ref(ab, bc, ca, depth - 1, subdivide_ref);
	};

	for (int f = 0; f < 20; f++) {
		const Vector3 &a = v[faces[f][0]];
		const Vector3 &b = v[faces[f][1]];
		const Vector3 &c = v[faces[f][2]];
		subdivide(a, b, c, p_subdiv, subdivide);
	}

	return ret;
}

LocalVector<JointLimitationKusudama3D::Segment> JointLimitationKusudama3D::cull_lines_by_boundary(const LocalVector<Segment> &p_segments, LocalVector<Vector3> &r_crossed_points) const {
	LocalVector<Segment> ret;
	for (const Segment &seg : p_segments) {
		Vector3 from_solved;
		bool from_is_in_boundary = is_in_boundary(seg.first, from_solved);
		Vector3 to_solved;
		bool to_is_in_boundary = is_in_boundary(seg.second, to_solved);
		if (from_is_in_boundary && to_is_in_boundary) {
			continue;
		} else if (!from_is_in_boundary && !to_is_in_boundary) {
			ret.push_back(seg);
		} else {
			Segment new_seg;
			if (from_is_in_boundary) {
				new_seg.first = seg.second;
				new_seg.second = to_solved;
				r_crossed_points.push_back(to_solved);
			} else {
				new_seg.first = from_solved;
				new_seg.second = seg.first;
				r_crossed_points.push_back(from_solved);
			}
			ret.push_back(new_seg);
		}
	}
	return ret;
}

bool JointLimitationKusudama3D::is_in_boundary(const Vector3 &p_point, Vector3 &r_solved) const {
	// Return whether p_point is in boundary.
	r_solved = _solve(p_point);
	return r_solved.is_equal_approx(p_point);
}

LocalVector<Vector3> JointLimitationKusudama3D::sort_by_nearest_point(const LocalVector<Vector3> &p_points) const {
	LocalVector<Vector3> ret;
	LocalVector<Vector3> points = p_points;
	if (points.size() > 0) {
		ret.push_back(points[0]);
		points.remove_at(0);
		while (points.size() > 0) {
			uint32_t current = ret.size() - 1;
			int nearest_index = -1;
			double nearest = INFINITY;
			for (uint32_t i = 0; i < points.size(); i++) {
				double dist = ret[current].distance_squared_to(points[i]);
				if (dist < nearest) {
					nearest = dist;
					nearest_index = i;
				}
			}
			if (nearest_index >= 0) {
				ret.push_back(points[nearest_index]);
				points.remove_at(nearest_index);
			}
		}
	}
	return ret;
}

#endif // TOOLS_ENABLED

// Helper function implementations
bool JointLimitationKusudama3D::is_point_in_cone(const Vector3 &p_point, const Vector3 &p_cone_center, real_t p_cone_radius) const {
	real_t cos_angle = p_point.dot(p_cone_center);
	real_t cos_radius = Math::cos(p_cone_radius);
	return cos_angle >= cos_radius - CMP_EPSILON;
}

bool JointLimitationKusudama3D::is_point_in_tangent_path(const Vector3 &p_point, const Vector3 &p_center1, real_t p_radius1, const Vector3 &p_center2, real_t p_radius2) const {
	Vector3 dir = p_point.normalized();

	// Check if point is in the inter-cone path region using get_on_great_tangent_triangle
	// This function handles all the geometric checks including whether the point is inside tangent circles
	Vector3 path_point = get_on_great_tangent_triangle(dir, p_center1, p_radius1, p_center2, p_radius2);

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

Vector3 JointLimitationKusudama3D::get_on_great_tangent_triangle(const Vector3 &p_point, const Vector3 &p_center1, real_t p_radius1, const Vector3 &p_center2, real_t p_radius2) const {
	Vector3 center1 = p_center1.normalized();
	Vector3 center2 = p_center2.normalized();
	Vector3 input = p_point.normalized();

	// Compute tangent circles
	Vector3 tan1, tan2;
	real_t tan_radius;
	compute_tangent_circles(center1, p_radius1, center2, p_radius2, tan1, tan2, tan_radius);

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

Vector3 JointLimitationKusudama3D::ray_plane_intersection(const Vector3 &p_ray_start, const Vector3 &p_ray_end, const Vector3 &p_plane_a, const Vector3 &p_plane_b, const Vector3 &p_plane_c) const {
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

void JointLimitationKusudama3D::extend_ray(Vector3 &r_start, Vector3 &r_end, real_t p_amount) const {
	Vector3 mid_point = (r_start + r_end) * 0.5;
	Vector3 start_heading = r_start - mid_point;
	Vector3 end_heading = r_end - mid_point;
	Vector3 start_extension = start_heading.normalized() * p_amount;
	Vector3 end_extension = end_heading.normalized() * p_amount;
	r_start = start_heading + start_extension + mid_point;
	r_end = end_heading + end_extension + mid_point;
}

int JointLimitationKusudama3D::ray_sphere_intersection_full(const Vector3 &p_ray_start, const Vector3 &p_ray_end, const Vector3 &p_sphere_center, real_t p_radius, Vector3 *r_intersection1, Vector3 *r_intersection2) const {
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

void JointLimitationKusudama3D::compute_tangent_circles(const Vector3 &p_center1, real_t p_radius1, const Vector3 &p_center2, real_t p_radius2, Vector3 &r_tangent1, Vector3 &r_tangent2, real_t &r_tangent_radius) const {
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
	ray_sphere_intersection_full(intersection_ray_start, intersection_ray_end, sphere_center, 1.0, &sphere_intersect1, &sphere_intersect2);

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
