#include "ik_kusudama_3d.h"
#include "core/math/quaternion.h"

IKRay3D::IKRay3D() {
}

IKRay3D::IKRay3D(Vector3 p_p1, Vector3 p_p2) {
	point_1 = p_p1;
	point_2 = p_p2;
}

Vector3 IKRay3D::get_heading() {
	return point_2 - point_1;
}

void IKRay3D::set_heading(const Vector3 &p_new_head) {
	point_2 = point_1 + p_new_head;
}

real_t IKRay3D::get_scaled_projection(const Vector3 p_input) {
	Vector3 working_vector = p_input - point_1;
	Vector3 heading = get_heading();
	real_t headingMag = heading.length();
	real_t workingVectorMag = working_vector.length();
	if (workingVectorMag == 0 || headingMag == 0) {
		return 0;
	}
	return (working_vector.dot(heading) / (headingMag * workingVectorMag)) * (workingVectorMag / headingMag);
}

void IKRay3D::elongate(real_t amt) {
	Vector3 midPoint = (point_1 + point_2) * 0.5f;
	Vector3 p1Heading = point_1 - midPoint;
	Vector3 p2Heading = point_2 - midPoint;
	Vector3 p1Add = p1Heading.normalized() * amt;
	Vector3 p2Add = p2Heading.normalized() * amt;
	point_1 = p1Heading + p1Add + midPoint;
	point_2 = p2Heading + p2Add + midPoint;
}

Vector3 IKRay3D::get_intersects_plane(Vector3 ta, Vector3 tb, Vector3 tc) {
	Vector3 u = tb - ta;
	Vector3 v = tc - ta;
	Vector3 n = u.cross(v).normalized();
	Vector3 dir = get_heading();
	Vector3 w0 = -ta;
	real_t a = -(n.dot(w0));
	real_t b = n.dot(dir);
	real_t r = a / b;
	return point_1 + dir * r;
}

int IKRay3D::intersects_sphere(Vector3 sphereCenter, real_t radius, Vector3 *S1, Vector3 *S2) {
	Vector3 tp1 = point_1 - sphereCenter;
	Vector3 tp2 = point_2 - sphereCenter;
	int result = intersects_sphere(tp1, tp2, radius, S1, S2);
	*S1 += sphereCenter;
	*S2 += sphereCenter;
	return result;
}

void IKRay3D::set_point_1(Vector3 in) {
	point_1 = in;
}

void IKRay3D::set_point_2(Vector3 in) {
	point_2 = in;
}

Vector3 IKRay3D::get_point_2() {
	return point_2;
}

Vector3 IKRay3D::get_point_1() {
	return point_1;
}

int IKRay3D::intersects_sphere(Vector3 rp1, Vector3 rp2, real_t radius, Vector3 *S1, Vector3 *S2) {
	Vector3 direction = rp2 - rp1;
	Vector3 e = direction.normalized();
	Vector3 h = -rp1;
	real_t lf = e.dot(h);
	real_t radpow = radius * radius;
	real_t hdh = h.length_squared();
	real_t lfpow = lf * lf;
	real_t s = radpow - hdh + lfpow;
	if (s < 0.0f) {
		return 0;
	}
	s = Math::sqrt(s);

	int result = 0;
	if (lf < s) {
		if (lf + s >= 0) {
			s = -s;
			result = 1;
		}
	} else {
		result = 2;
	}

	*S1 = e * (lf - s) + rp1;
	*S2 = e * (lf + s) + rp1;
	return result;
}

void IKRay3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_heading"), &IKRay3D::get_heading);
	ClassDB::bind_method(D_METHOD("get_scaled_projection", "input"), &IKRay3D::get_scaled_projection);
	ClassDB::bind_method(D_METHOD("get_intersects_plane", "a", "b", "c"), &IKRay3D::get_intersects_plane);
}

// ===== IKLimitCone3D Implementation =====

void IKLimitCone3D::set_attached_to(Ref<IKKusudama3D> p_attached_to) {
	parent_kusudama.set_ref(p_attached_to);
}

Ref<IKKusudama3D> IKLimitCone3D::get_attached_to() {
	return parent_kusudama.get_ref();
}

void IKLimitCone3D::set_control_point(Vector3 p_control_point) {
	if (Math::is_zero_approx(p_control_point.length_squared())) {
		control_point = Vector3(0, 1, 0);
	} else {
		control_point = p_control_point.normalized();
	}
}

Vector3 IKLimitCone3D::get_control_point() const {
	return control_point;
}

double IKLimitCone3D::get_radius() const {
	return radius;
}

double IKLimitCone3D::get_radius_cosine() const {
	return radius_cosine;
}

void IKLimitCone3D::set_radius(double p_radius) {
	radius = p_radius;
	radius_cosine = cos(p_radius);
}

void IKLimitCone3D::set_tangent_circle_center_next_1(Vector3 point) {
	tangent_circle_center_next_1 = point.normalized();
}

void IKLimitCone3D::set_tangent_circle_center_next_2(Vector3 point) {
	tangent_circle_center_next_2 = point.normalized();
}

void IKLimitCone3D::set_tangent_circle_radius_next(double rad) {
	tangent_circle_radius_next = rad;
	tangent_circle_radius_next_cos = cos(rad);
}

Vector3 IKLimitCone3D::get_tangent_circle_center_next_1() {
	return tangent_circle_center_next_1;
}

Vector3 IKLimitCone3D::get_tangent_circle_center_next_2() {
	return tangent_circle_center_next_2;
}

double IKLimitCone3D::get_tangent_circle_radius_next() {
	return tangent_circle_radius_next;
}

double IKLimitCone3D::_get_tangent_circle_radius_next_cos() {
	return tangent_circle_radius_next_cos;
}

Vector3 IKLimitCone3D::get_orthogonal(Vector3 p_in) {
	Vector3 result;
	float threshold = p_in.length() * 0.6f;
	if (threshold > 0.f) {
		if (Math::abs(p_in.x) <= threshold) {
			float inverse = 1.f / Math::sqrt(p_in.y * p_in.y + p_in.z * p_in.z);
			return result = Vector3(0.f, inverse * p_in.z, -inverse * p_in.y);
		} else if (Math::abs(p_in.y) <= threshold) {
			float inverse = 1.f / Math::sqrt(p_in.x * p_in.x + p_in.z * p_in.z);
			return result = Vector3(-inverse * p_in.z, 0.f, inverse * p_in.x);
		}
		float inverse = 1.f / Math::sqrt(p_in.x * p_in.x + p_in.y * p_in.y);
		return result = Vector3(inverse * p_in.y, -inverse * p_in.x, 0.f);
	}
	return result;
}

void IKLimitCone3D::compute_triangles(Ref<IKLimitCone3D> p_next) {
	if (p_next.is_null()) {
		return;
	}
	first_triangle_next.write[1] = tangent_circle_center_next_1.normalized();
	first_triangle_next.write[0] = get_control_point().normalized();
	first_triangle_next.write[2] = p_next->get_control_point().normalized();

	second_triangle_next.write[1] = tangent_circle_center_next_2.normalized();
	second_triangle_next.write[0] = get_control_point().normalized();
	second_triangle_next.write[2] = p_next->get_control_point().normalized();
}

void IKLimitCone3D::update_tangent_handles(Ref<IKLimitCone3D> p_next) {
	if (p_next.is_null()) {
		return;
	}
	double radA = get_radius();
	double radB = p_next->get_radius();
	Vector3 A = get_control_point();
	Vector3 B = p_next->get_control_point();

	Vector3 arc_normal;
	Vector3 cross = A.cross(B);
	if (cross.is_zero_approx()) {
		arc_normal = get_orthogonal(A);
		if (arc_normal.is_zero_approx()) {
			arc_normal = Vector3(0, 1, 0);
		}
	} else {
		arc_normal = cross.normalized();
	}

	double tRadius = (Math::PI - (radA + radB)) / 2;
	double boundaryPlusTangentRadiusA = radA + tRadius;
	double boundaryPlusTangentRadiusB = radB + tRadius;

	Vector3 scaledAxisA = A * Math::cos(boundaryPlusTangentRadiusA);
	Quaternion temp_var = IKKusudama3D::get_quaternion_axis_angle(arc_normal, boundaryPlusTangentRadiusA);
	Vector3 planeDir1A = temp_var.xform(A);
	Quaternion tempVar2 = IKKusudama3D::get_quaternion_axis_angle(A, Math::PI / 2);
	Vector3 planeDir2A = tempVar2.xform(planeDir1A);

	Vector3 scaledAxisB = B * Math::cos(boundaryPlusTangentRadiusB);
	Quaternion tempVar3 = IKKusudama3D::get_quaternion_axis_angle(arc_normal, boundaryPlusTangentRadiusB);
	Vector3 planeDir1B = tempVar3.xform(B);
	Quaternion tempVar4 = IKKusudama3D::get_quaternion_axis_angle(B, Math::PI / 2);
	Vector3 planeDir2B = tempVar4.xform(planeDir1B);

	Ref<IKRay3D> r1B(memnew(IKRay3D(planeDir1B, scaledAxisB)));
	Ref<IKRay3D> r2B(memnew(IKRay3D(planeDir1B, planeDir2B)));
	r1B->elongate(99);
	r2B->elongate(99);

	Vector3 intersection1 = r1B->get_intersects_plane(scaledAxisA, planeDir1A, planeDir2A);
	Vector3 intersection2 = r2B->get_intersects_plane(scaledAxisA, planeDir1A, planeDir2A);

	Ref<IKRay3D> intersectionRay(memnew(IKRay3D(intersection1, intersection2)));
	intersectionRay->elongate(99);

	Vector3 sphereIntersect1, sphereIntersect2;
	Vector3 sphereCenter;
	intersectionRay->intersects_sphere(sphereCenter, 1.0f, &sphereIntersect1, &sphereIntersect2);

	set_tangent_circle_center_next_1(sphereIntersect1);
	set_tangent_circle_center_next_2(sphereIntersect2);
	set_tangent_circle_radius_next(tRadius);

	if (!tangent_circle_center_next_1.is_finite() || Math::is_zero_approx(tangent_circle_center_next_1.length_squared())) {
		tangent_circle_center_next_1 = get_orthogonal(control_point);
		if (Math::is_zero_approx(tangent_circle_center_next_1.length_squared())) {
			tangent_circle_center_next_1 = Vector3(0, 1, 0);
		}
		tangent_circle_center_next_1.normalize();
	}
	if (!tangent_circle_center_next_2.is_finite() || Math::is_zero_approx(tangent_circle_center_next_2.length_squared())) {
		Vector3 orthogonal_base = tangent_circle_center_next_1.is_finite() ? tangent_circle_center_next_1 : control_point;
		tangent_circle_center_next_2 = get_orthogonal(orthogonal_base);
		if (Math::is_zero_approx(tangent_circle_center_next_2.length_squared())) {
			tangent_circle_center_next_2 = Vector3(1, 0, 0);
		}
		tangent_circle_center_next_2.normalize();
	}
	if (p_next.is_valid()) {
		compute_triangles(p_next);
	}
}

Vector3 IKLimitCone3D::_closest_cone(Ref<IKLimitCone3D> next, Vector3 input) const {
	if (next.is_null()) {
		return control_point;
	}
	if (input.dot(control_point) > input.dot(next->control_point)) {
		return control_point;
	}
	return next->control_point;
}

bool IKLimitCone3D::_determine_if_in_bounds(Ref<IKLimitCone3D> next, Vector3 input) const {
	if (control_point.dot(input) >= radius_cosine) {
		return true;
	}
	if (next.is_valid() && next->control_point.dot(input) >= next->radius_cosine) {
		return true;
	}
	if (next.is_null()) {
		return false;
	}

	bool inTan1Rad = tangent_circle_center_next_1.dot(input) > tangent_circle_radius_next_cos;
	if (inTan1Rad) {
		return false;
	}
	bool inTan2Rad = tangent_circle_center_next_2.dot(input) > tangent_circle_radius_next_cos;
	if (inTan2Rad) {
		return false;
	}

	Vector3 c1xc2 = control_point.cross(next->control_point);
	double c1c2dir = input.dot(c1xc2);

	if (c1c2dir < 0.0) {
		Vector3 c1xt1 = control_point.cross(tangent_circle_center_next_1);
		Vector3 t1xc2 = tangent_circle_center_next_1.cross(next->control_point);
		return input.dot(c1xt1) > 0 && input.dot(t1xc2) > 0;
	} else {
		Vector3 t2xc1 = tangent_circle_center_next_2.cross(control_point);
		Vector3 c2xt2 = next->control_point.cross(tangent_circle_center_next_2);
		return input.dot(t2xc1) > 0 && input.dot(c2xt2) > 0;
	}
}

Vector3 IKLimitCone3D::get_on_great_tangent_triangle(Ref<IKLimitCone3D> next, Vector3 input) const {
	ERR_FAIL_COND_V(next.is_null(), input);

	Vector3 c1xc2 = control_point.cross(next->control_point);
	double c1c2dir = input.dot(c1xc2);

	if (c1c2dir < 0.0) {
		Vector3 c1xt1 = control_point.cross(tangent_circle_center_next_1);
		Vector3 t1xc2 = tangent_circle_center_next_1.cross(next->control_point);
		if (input.dot(c1xt1) > 0 && input.dot(t1xc2) > 0) {
			double to_next_cos = input.dot(tangent_circle_center_next_1);
			if (to_next_cos > tangent_circle_radius_next_cos) {
				Vector3 plane_normal = tangent_circle_center_next_1.cross(input);
				if (!plane_normal.is_finite() || Math::is_zero_approx(plane_normal.length_squared())) {
					plane_normal = Vector3(0, 1, 0);
				}
				plane_normal.normalize();
				Quaternion rotate_about_by(plane_normal, tangent_circle_radius_next);
				return rotate_about_by.xform(tangent_circle_center_next_1);
			}
			return input;
		}
		return Vector3(NAN, NAN, NAN);
	} else {
		Vector3 t2xc1 = tangent_circle_center_next_2.cross(control_point);
		Vector3 c2xt2 = next->control_point.cross(tangent_circle_center_next_2);
		if (input.dot(t2xc1) > 0 && input.dot(c2xt2) > 0) {
			if (input.dot(tangent_circle_center_next_2) > tangent_circle_radius_next_cos) {
				Vector3 plane_normal = tangent_circle_center_next_2.cross(input);
				if (!plane_normal.is_finite() || Math::is_zero_approx(plane_normal.length_squared())) {
					plane_normal = Vector3(0, 1, 0);
				}
				plane_normal.normalize();
				Quaternion rotate_about_by(plane_normal, tangent_circle_radius_next);
				return rotate_about_by.xform(tangent_circle_center_next_2);
			}
			return input;
		}
		return Vector3(NAN, NAN, NAN);
	}
}

Vector3 IKLimitCone3D::_get_on_path_sequence(Ref<IKLimitCone3D> next, Vector3 input) const {
	if (next.is_null()) {
		return Vector3(NAN, NAN, NAN);
	}

	Vector3 c1xc2 = control_point.cross(next->control_point);
	double c1c2dir = input.dot(c1xc2);

	if (c1c2dir < 0.0) {
		Vector3 c1xt1 = control_point.cross(tangent_circle_center_next_1);
		Vector3 t1xc2 = tangent_circle_center_next_1.cross(next->control_point);
		if (input.dot(c1xt1) > 0.0f && input.dot(t1xc2) > 0.0f) {
			Ref<IKRay3D> tan1ToInput(memnew(IKRay3D(tangent_circle_center_next_1, input)));
			Vector3 result = tan1ToInput->get_intersects_plane(Vector3(0.0f, 0.0f, 0.0f), get_control_point(), next->get_control_point());
			return result.normalized();
		}
		return Vector3(NAN, NAN, NAN);
	} else {
		Vector3 t2xc1 = tangent_circle_center_next_2.cross(control_point);
		Vector3 c2xt2 = next->control_point.cross(tangent_circle_center_next_2);
		if (input.dot(t2xc1) > 0 && input.dot(c2xt2) > 0) {
			Ref<IKRay3D> tan2ToInput(memnew(IKRay3D(tangent_circle_center_next_2, input)));
			Vector3 result = tan2ToInput->get_intersects_plane(Vector3(0.0f, 0.0f, 0.0f), get_control_point(), next->get_control_point());
			return result.normalized();
		}
		return Vector3(NAN, NAN, NAN);
	}
}

Vector3 IKLimitCone3D::_closest_point_on_closest_cone(Ref<IKLimitCone3D> next, Vector3 input, Vector<double> *in_bounds) const {
	ERR_FAIL_COND_V(next.is_null(), input);
	Vector3 closestToFirst = closest_to_cone(input, in_bounds);
	if (in_bounds != nullptr && (*in_bounds)[0] > 0.0) {
		return closestToFirst;
	}
	if (next.is_null()) {
		return closestToFirst;
	}
	Vector3 closestToSecond = next->closest_to_cone(input, in_bounds);
	if (in_bounds != nullptr && (*in_bounds)[0] > 0.0) {
		return closestToSecond;
	}
	double cosToFirst = input.dot(closestToFirst);
	double cosToSecond = input.dot(closestToSecond);
	if (cosToFirst > cosToSecond) {
		return closestToFirst;
	}
	return closestToSecond;
}

Vector3 IKLimitCone3D::closest_to_cone(Vector3 input, Vector<double> *in_bounds) const {
	Vector3 normalized_input = input.normalized();
	Vector3 normalized_control_point = get_control_point().normalized();
	if (normalized_input.dot(normalized_control_point) > get_radius_cosine()) {
		if (in_bounds != nullptr) {
			in_bounds->write[0] = 1.0;
		}
		return Vector3(NAN, NAN, NAN);
	}

	Vector3 axis = normalized_control_point.cross(normalized_input);
	if (!axis.is_finite() || Math::is_zero_approx(axis.length_squared())) {
		axis = get_orthogonal(normalized_control_point);
		if (Math::is_zero_approx(axis.length_squared())) {
			axis = Vector3(0, 1, 0);
		}
		axis.normalize();
	}

	Quaternion rot_to = IKKusudama3D::get_quaternion_axis_angle(axis, get_radius());
	Vector3 axis_control_point = normalized_control_point;
	if (Math::is_zero_approx(axis_control_point.length_squared())) {
		axis_control_point = Vector3(0, 1, 0);
	}
	Vector3 result = rot_to.xform(axis_control_point);
	if (in_bounds != nullptr) {
		in_bounds->write[0] = -1;
	}
	return result;
}

Vector3 IKLimitCone3D::get_closest_path_point(Ref<IKLimitCone3D> next, Vector3 input) const {
	Vector3 result;
	if (next.is_null()) {
		result = _closest_cone(Ref<IKLimitCone3D>(this), input);
	} else {
		result = _get_on_path_sequence(next, input);
		bool is_number = !(Math::is_nan(result.x) && Math::is_nan(result.y) && Math::is_nan(result.z));
		if (!is_number) {
			result = _closest_cone(next, input);
		}
	}
	return result;
}

// ===== IKKusudama3D Implementation =====

void IKKusudama3D::_update_constraint(Ref<IKNode3D> p_limiting_axes) {
	Vector<Vector3> directions;

	if (open_cones.size() == 1 && open_cones[0].is_valid()) {
		directions.push_back(open_cones[0]->get_control_point());
	} else {
		for (int i = 0; i < open_cones.size() - 1; i++) {
			if (open_cones[i].is_null() || open_cones[i + 1].is_null()) {
				continue;
			}
			Vector3 this_control_point = open_cones[i]->get_control_point();
			Vector3 next_control_point = open_cones[i + 1]->get_control_point();
			Quaternion this_to_next(this_control_point, next_control_point);
			Vector3 axis = this_to_next.get_axis();
			double angle = this_to_next.get_angle() / 2.0;
			Vector3 half_angle;
			if (Math::is_zero_approx(axis.length_squared())) {
				half_angle = this_control_point;
			} else {
				half_angle = this_control_point.rotated(axis, angle);
			}
			half_angle *= this_to_next.get_angle();
			half_angle.normalize();
			directions.push_back(half_angle);
		}
	}

	Vector3 new_y;
	for (Vector3 direction_vector : directions) {
		new_y += direction_vector;
	}

	if (!directions.is_empty()) {
		new_y /= directions.size();
		new_y.normalize();
	}

	Transform3D new_y_ray = Transform3D(Basis(), new_y);
	Vector3 old_y_norm = p_limiting_axes->get_global_transform().get_basis().get_column(Vector3::AXIS_Y).normalized();
	Vector3 new_y_global_norm = p_limiting_axes->get_global_transform().get_basis().xform(new_y_ray.origin).normalized();

	if (!(old_y_norm.is_zero_approx() || new_y_global_norm.is_zero_approx())) {
		Quaternion old_y_to_new_y = Quaternion(old_y_norm, new_y_global_norm);
		p_limiting_axes->rotate_local_with_global(old_y_to_new_y);
	}

	for (Ref<IKLimitCone3D> open_cone : open_cones) {
		if (open_cone.is_null()) {
			continue;
		}
		Vector3 control_point = open_cone->get_control_point();
		open_cone->set_control_point(control_point.normalized());
	}

	update_tangent_radii();
}

void IKKusudama3D::update_tangent_radii() {
	for (int i = 0; i < open_cones.size(); i++) {
		Ref<IKLimitCone3D> current = open_cones.write[i];
		Ref<IKLimitCone3D> next;
		if (i < open_cones.size() - 1) {
			next = open_cones.write[i + 1];
		}
		Ref<IKLimitCone3D> cone = open_cones[i];
		cone->update_tangent_handles(next);
	}
}

void IKKusudama3D::set_axial_limits(real_t min_angle, real_t in_range) {
	min_axial_angle = min_angle;
	range_angle = in_range;
	Vector3 y_axis = Vector3(0.0f, 1.0f, 0.0f);
	Vector3 z_axis = Vector3(0.0f, 0.0f, 1.0f);
	twist_min_rot = IKKusudama3D::get_quaternion_axis_angle(y_axis, min_axial_angle);
	twist_min_vec = twist_min_rot.xform(z_axis).normalized();
	twist_center_vec = twist_min_rot.xform(twist_min_vec).normalized();
	twist_center_rot = Quaternion(z_axis, twist_center_vec);
	twist_half_range_half_cos = Math::cos(in_range / real_t(4.0));
	twist_max_vec = IKKusudama3D::get_quaternion_axis_angle(y_axis, in_range).xform(twist_min_vec).normalized();
	twist_max_rot = Quaternion(z_axis, twist_max_vec);
}

void IKKusudama3D::set_snap_to_twist_limit(Ref<IKNode3D> p_bone_direction, Ref<IKNode3D> p_to_set, Ref<IKNode3D> p_constraint_axes, real_t p_dampening, real_t p_cos_half_dampen) {
	if (!is_axially_constrained()) {
		return;
	}
	Transform3D global_transform_constraint = p_constraint_axes->get_global_transform();
	Transform3D global_transform_to_set = p_to_set->get_global_transform();
	Basis parent_global_inverse = p_to_set->get_parent()->get_global_transform().basis.inverse();
	Basis global_twist_center = global_transform_constraint.basis * twist_center_rot;
	Basis align_rot = (global_twist_center.inverse() * global_transform_to_set.basis).orthonormalized();
	Quaternion twist_rotation, swing_rotation;
	get_swing_twist(align_rot.get_rotation_quaternion(), Vector3(0, 1, 0), swing_rotation, twist_rotation);
	twist_rotation = clamp_to_quadrance_angle(twist_rotation, twist_half_range_half_cos);
	Basis recomposition = (global_twist_center * (swing_rotation * twist_rotation)).orthonormalized();
	Basis rotation = parent_global_inverse * recomposition;
	p_to_set->set_transform(Transform3D(rotation, p_to_set->get_transform().origin));
}

void IKKusudama3D::get_swing_twist(Quaternion p_rotation, Vector3 p_axis, Quaternion &r_swing, Quaternion &r_twist) {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_MSG(!p_rotation.is_normalized(), "The quaternion must be normalized.");
#endif

	if (p_axis.length_squared() < CMP_EPSILON2) {
		r_swing = Quaternion();
		r_twist = Quaternion();
		return;
	}

	Quaternion rotation = p_rotation;
	Quaternion axis_quat = Quaternion(p_axis, 0);
	Quaternion conj_axis = axis_quat.inverse();
	Quaternion twist = rotation * conj_axis;
	r_twist = Quaternion(twist.x, twist.y, twist.z, 0).normalized();
	if (r_twist.w < 0) {
		r_twist = -r_twist;
	}
	r_swing = r_twist.inverse() * rotation;
}

void IKKusudama3D::add_open_cone(Ref<IKLimitCone3D> p_cone) {
	ERR_FAIL_COND(p_cone.is_null());
	ERR_FAIL_COND(p_cone->get_attached_to().is_null());
	open_cones.push_back(p_cone);
	update_tangent_radii();
}

void IKKusudama3D::remove_open_cone(Ref<IKLimitCone3D> limitCone) {
	ERR_FAIL_COND(limitCone.is_null());
	open_cones.erase(limitCone);
}

real_t IKKusudama3D::get_min_axial_angle() {
	return min_axial_angle;
}

real_t IKKusudama3D::get_range_angle() {
	return range_angle;
}

bool IKKusudama3D::is_axially_constrained() {
	return axially_constrained;
}

bool IKKusudama3D::is_orientationally_constrained() const {
	return orientationally_constrained;
}

void IKKusudama3D::disable_orientational_limits() {
	orientationally_constrained = false;
}

void IKKusudama3D::enable_orientational_limits() {
	orientationally_constrained = true;
}

void IKKusudama3D::toggle_orientational_limits() {
	orientationally_constrained = !orientationally_constrained;
}

void IKKusudama3D::disable_axial_limits() {
	axially_constrained = false;
}

void IKKusudama3D::enable_axial_limits() {
	axially_constrained = true;
}

void IKKusudama3D::toggle_axial_limits() {
	axially_constrained = !axially_constrained;
}

bool IKKusudama3D::is_enabled() const {
	return axially_constrained || orientationally_constrained;
}

void IKKusudama3D::disable() {
	axially_constrained = false;
	orientationally_constrained = false;
}

void IKKusudama3D::enable() {
	axially_constrained = true;
	orientationally_constrained = true;
}

TypedArray<IKLimitCone3D> IKKusudama3D::get_open_cones() const {
	TypedArray<IKLimitCone3D> cones;
	for (Ref<IKLimitCone3D> cone : open_cones) {
		cones.append(cone);
	}
	return cones;
}

Vector3 IKKusudama3D::local_point_on_path_sequence(Vector3 p_in_point, Ref<IKNode3D> p_limiting_axes) {
	double closest_point_dot = 0;
	Vector3 point = p_limiting_axes->get_transform().xform(p_in_point);
	point.normalize();
	Vector3 result = point;

	if (open_cones.size() == 1) {
		Ref<IKLimitCone3D> cone = open_cones[0];
		result = cone->get_control_point();
	} else {
		for (int i = 0; i < open_cones.size() - 1; i++) {
			Ref<IKLimitCone3D> next_cone = open_cones[i + 1];
			Ref<IKLimitCone3D> cone = open_cones[i];
			Vector3 closestPathPoint = cone->get_closest_path_point(next_cone, point);
			double closeDot = closestPathPoint.dot(point);
			if (closeDot > closest_point_dot) {
				result = closestPathPoint;
				closest_point_dot = closeDot;
			}
		}
	}

	return result;
}

Vector3 IKKusudama3D::get_local_point_in_limits(Vector3 in_point, Vector<double> *in_bounds) {
	Vector3 point = in_point.normalized();
	real_t closest_cos = -2.0;
	in_bounds->write[0] = -1;

	Vector3 closest_collision_point = in_point;

	for (int i = 0; i < open_cones.size(); i++) {
		Ref<IKLimitCone3D> cone = open_cones[i];
		Vector3 collision_point = cone->closest_to_cone(point, in_bounds);

		if (Math::is_nan(collision_point.x) || Math::is_nan(collision_point.y) || Math::is_nan(collision_point.z)) {
			in_bounds->write[0] = 1;
			return point;
		}

		real_t this_cos = collision_point.dot(point);

		if (closest_collision_point.is_zero_approx() || this_cos > closest_cos) {
			closest_collision_point = collision_point;
			closest_cos = this_cos;
		}
	}

	if ((*in_bounds)[0] == -1) {
		for (int i = 0; i < open_cones.size() - 1; i++) {
			Ref<IKLimitCone3D> currCone = open_cones[i];
			Ref<IKLimitCone3D> nextCone = open_cones[i + 1];
			Vector3 collision_point = currCone->get_on_great_tangent_triangle(nextCone, point);

			if (Math::is_nan(collision_point.x)) {
				continue;
			}

			real_t this_cos = collision_point.dot(point);

			if (Math::is_equal_approx(this_cos, real_t(1.0))) {
				in_bounds->write[0] = 1;
				return point;
			}

			if (this_cos > closest_cos) {
				closest_collision_point = collision_point;
				closest_cos = this_cos;
			}
		}
	}

	return closest_collision_point;
}

Vector3 IKKusudama3D::_solve(const Vector3 &p_direction) const {
	if (!is_enabled() || !is_orientationally_constrained()) {
		return p_direction;
	}

	Vector<double> bounds;
	bounds.resize(2);
	bounds.write[0] = -1.0;
	bounds.write[1] = 0.0;

	IKKusudama3D *mutable_this = const_cast<IKKusudama3D *>(this);
	Vector3 constrained = mutable_this->get_local_point_in_limits(p_direction, &bounds);
	return constrained.normalized();
}

void IKKusudama3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_open_cones"), &IKKusudama3D::get_open_cones);
	ClassDB::bind_method(D_METHOD("set_open_cones", "open_cones"), &IKKusudama3D::set_open_cones);
}

void IKKusudama3D::set_open_cones(TypedArray<IKLimitCone3D> p_cones) {
	open_cones.clear();
	open_cones.resize(p_cones.size());
	for (int32_t i = 0; i < p_cones.size(); i++) {
		open_cones.write[i] = p_cones[i];
	}
}

void IKKusudama3D::snap_to_orientation_limit(Ref<IKNode3D> bone_direction, Ref<IKNode3D> to_set, Ref<IKNode3D> limiting_axes, real_t p_dampening, real_t p_cos_half_angle_dampen) {
	if (bone_direction.is_null()) {
		return;
	}
	if (to_set.is_null()) {
		return;
	}
	if (limiting_axes.is_null()) {
		return;
	}
	Vector<double> in_bounds;
	in_bounds.resize(1);
	in_bounds.write[0] = 1.0;
	Vector3 limiting_origin = limiting_axes->get_global_transform().origin;
	Vector3 bone_dir_xform = bone_direction->get_global_transform().xform(Vector3(0.0, 1.0, 0.0));

	bone_ray->set_point_1(limiting_origin);
	bone_ray->set_point_2(bone_dir_xform);

	Vector3 bone_tip = limiting_axes->to_local(bone_ray->get_point_2());
	Vector3 in_limits = get_local_point_in_limits(bone_tip, &in_bounds);

	if (in_bounds[0] < 0) {
		constrained_ray->set_point_1(bone_ray->get_point_1());
		constrained_ray->set_point_2(limiting_axes->to_global(in_limits));

		Quaternion rectified_rot = Quaternion(bone_ray->get_heading(), constrained_ray->get_heading());
		to_set->rotate_local_with_global(rectified_rot);
	}
}

bool IKKusudama3D::is_nan_vector(const Vector3 &vec) {
	return Math::is_nan(vec.x) || Math::is_nan(vec.y) || Math::is_nan(vec.z);
}

void IKKusudama3D::set_resistance(float p_resistance) {
	resistance = p_resistance;
}

float IKKusudama3D::get_resistance() {
	return resistance;
}

Quaternion IKKusudama3D::clamp_to_quadrance_angle(Quaternion p_rotation, double p_cos_half_angle) {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_V_MSG(!p_rotation.is_normalized(), Quaternion(), "The quaternion must be normalized.");
#endif
	Quaternion rotation = p_rotation;
	double newCoeff = 1.0 - (p_cos_half_angle * Math::abs(p_cos_half_angle));
	double currentCoeff = rotation.x * rotation.x + rotation.y * rotation.y + rotation.z * rotation.z;
	if (newCoeff >= currentCoeff) {
		return rotation;
	}
	double over_limit = (currentCoeff - newCoeff) / (1.0 - newCoeff);
	Quaternion clamped_rotation = rotation;
	clamped_rotation.w = rotation.w < 0 ? -p_cos_half_angle : p_cos_half_angle;
	double compositeCoeff = sqrt(newCoeff / currentCoeff);
	clamped_rotation.x *= compositeCoeff;
	clamped_rotation.y *= compositeCoeff;
	clamped_rotation.z *= compositeCoeff;
	if (!rotation.is_finite() || !clamped_rotation.is_finite()) {
		return Quaternion();
	}
	return rotation.slerp(clamped_rotation, over_limit);
}

void IKKusudama3D::clear_open_cones() {
	open_cones.clear();
}

Quaternion IKKusudama3D::get_quaternion_axis_angle(const Vector3 &p_axis, real_t p_angle) {
	if (p_axis.length_squared() < CMP_EPSILON2) {
		return Quaternion();
	}

	if (Math::abs(p_angle) < CMP_EPSILON) {
		return Quaternion();
	}

	return Quaternion(p_axis, p_angle);
}</parameter>
<parameter name="task_progress">
- [x] Port kusudama to JointLimitation3D with multi-cone support
- [x] Create KusudamaMesh3D mesh primitive
- [x] Implement array inspector for cone definition
- [x] Remove tangent parameters from serialization
- [x] Analyze many_bone_ik module structure
- [x] Create core/math/ik_kusudama_3d.h with inlined IKOpenCone3D and IKRay3D
- [x] Create core/math/ik_kusudama_3d.cpp with implementations (incremental)
- [ ] Update includes in joint_limitation_cone_3d.cpp and kusudama_mesh_3d.cpp
- [ ] Register classes in core/register_core_types.cpp
- [ ] Move tests to tests/ directory
- [ ] Remove many_bone_ik module directory
- [ ] Verify build and dependencies
</parameter>
</write_to_file>
