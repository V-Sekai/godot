/**************************************************************************/
/*  kusudama_bone_constraint_3d.cpp                                       */
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

#include "kusudama_bone_constraint_3d.h"

#include "core/math/quaternion.h"
#include "ik_bone_segment_3d.h"
#include "ik_open_cone_3d.h"
#include "math/ik_node_3d_local.h"

void KusudamaBoneConstraint3D::_process_constraint(int p_index, Skeleton3D *p_skeleton, int p_apply_bone, int p_reference_bone, float p_amount) {
	if (!is_enabled() || !p_skeleton || p_apply_bone < 0 || p_apply_bone >= p_skeleton->get_bone_count()) {
		return;
	}

	// Get the current bone pose
	Transform3D bone_pose = p_skeleton->get_bone_pose(p_apply_bone);
	Quaternion current_rotation = bone_pose.get_basis().get_rotation_quaternion();

	// Get the reference bone transform if available
	Transform3D reference_transform;
	if (p_reference_bone >= 0 && p_reference_bone < p_skeleton->get_bone_count()) {
		reference_transform = p_skeleton->get_bone_global_pose(p_reference_bone);
	} else {
		// Use skeleton root as reference
		reference_transform = p_skeleton->get_global_transform();
	}

	// Convert to local space relative to reference
	Transform3D local_transform = reference_transform.affine_inverse() * p_skeleton->get_bone_global_pose(p_apply_bone);

	// Get the direction vector (assuming Y-axis is the bone direction)
	Vector3 direction = local_transform.get_basis().get_column(Vector3::AXIS_Y).normalized();

	// Apply kusudama constraint
	Vector3 constrained_direction = _solve(direction);

	// If the direction changed, update the bone rotation
	if (!constrained_direction.is_equal_approx(direction)) {
		Quaternion correction = Quaternion(direction, constrained_direction);
		Quaternion new_rotation = correction * current_rotation;

		// Apply damping
		if (p_amount < 1.0f) {
			new_rotation = current_rotation.slerp(new_rotation, p_amount);
		}

		p_skeleton->set_bone_pose_rotation(p_apply_bone, new_rotation);
	}
}

void KusudamaBoneConstraint3D::set_open_cones(const TypedArray<IKLimitCone3D> &p_cones) {
	open_cones.clear();
	open_cones.resize(p_cones.size());
	for (int32_t i = 0; i < p_cones.size(); i++) {
		open_cones.write[i] = p_cones[i];
	}
	update_tangent_radii();
}

TypedArray<IKLimitCone3D> KusudamaBoneConstraint3D::get_open_cones() const {
	TypedArray<IKLimitCone3D> cones;
	for (const Ref<IKLimitCone3D> &cone : open_cones) {
		cones.append(cone);
	}
	return cones;
}

void KusudamaBoneConstraint3D::add_open_cone(const Ref<IKLimitCone3D> &p_cone) {
	ERR_FAIL_COND(p_cone.is_null());
	ERR_FAIL_COND(p_cone->get_attached_to().is_null());
	open_cones.push_back(p_cone);
	update_tangent_radii();
}

void KusudamaBoneConstraint3D::remove_open_cone(const Ref<IKLimitCone3D> &limitCone) {
	ERR_FAIL_COND(limitCone.is_null());
	open_cones.erase(limitCone);
	update_tangent_radii();
}

void KusudamaBoneConstraint3D::clear_open_cones() {
	open_cones.clear();
}

void KusudamaBoneConstraint3D::set_axial_limits(real_t min_angle, real_t in_range) {
	min_axial_angle = min_angle;
	range_angle = in_range;
	Vector3 y_axis = Vector3(0.0f, 1.0f, 0.0f);
	Vector3 z_axis = Vector3(0.0f, 0.0f, 1.0f);
	twist_min_rot = get_quaternion_axis_angle(y_axis, min_axial_angle);
	twist_min_vec = twist_min_rot.xform(z_axis).normalized();
	twist_center_vec = twist_min_rot.xform(twist_min_vec).normalized();
	twist_center_rot = Quaternion(z_axis, twist_center_vec);
	twist_half_range_half_cos = Math::cos(in_range / real_t(4.0)); // For the quadrance angle. We need half the range angle since starting from the center, and half of that since quadrance takes cos(angle/2).
	twist_max_vec = get_quaternion_axis_angle(y_axis, in_range).xform(twist_min_vec).normalized();
	twist_max_rot = Quaternion(z_axis, twist_max_vec);
}

real_t KusudamaBoneConstraint3D::get_min_axial_angle() const {
	return min_axial_angle;
}

real_t KusudamaBoneConstraint3D::get_range_angle() const {
	return range_angle;
}

bool KusudamaBoneConstraint3D::is_axially_constrained() const {
	return axially_constrained;
}

void KusudamaBoneConstraint3D::disable_axial_limits() {
	axially_constrained = false;
}

void KusudamaBoneConstraint3D::enable_axial_limits() {
	axially_constrained = true;
}

void KusudamaBoneConstraint3D::toggle_axial_limits() {
	axially_constrained = !axially_constrained;
}

bool KusudamaBoneConstraint3D::is_orientationally_constrained() const {
	return orientationally_constrained;
}

void KusudamaBoneConstraint3D::disable_orientational_limits() {
	orientationally_constrained = false;
}

void KusudamaBoneConstraint3D::enable_orientational_limits() {
	orientationally_constrained = true;
}

void KusudamaBoneConstraint3D::toggle_orientational_limits() {
	orientationally_constrained = !orientationally_constrained;
}

bool KusudamaBoneConstraint3D::is_enabled() const {
	return axially_constrained || orientationally_constrained;
}

void KusudamaBoneConstraint3D::disable() {
	axially_constrained = false;
	orientationally_constrained = false;
}

void KusudamaBoneConstraint3D::enable() {
	axially_constrained = true;
	orientationally_constrained = true;
}

void KusudamaBoneConstraint3D::set_resistance(float p_resistance) {
	resistance = p_resistance;
}

float KusudamaBoneConstraint3D::get_resistance() const {
	return resistance;
}

Vector3 KusudamaBoneConstraint3D::_solve(const Vector3 &p_direction) const {
	// If constraints are disabled, return the original direction
	if (!is_enabled() || !is_orientationally_constrained()) {
		return p_direction;
	}

	// Use the existing sophisticated constraint solving algorithm
	Vector<double> bounds;
	bounds.resize(2);
	bounds.write[0] = -1.0; // Initialize as out of bounds
	bounds.write[1] = 0.0;

	// Cast away const for the existing method (this is safe as we're not modifying the object state)
	KusudamaBoneConstraint3D *mutable_this = const_cast<KusudamaBoneConstraint3D *>(this);
	Vector3 constrained = mutable_this->get_local_point_in_limits(p_direction, &bounds);

	// Ensure the result is normalized
	return constrained.normalized();
}

Vector3 KusudamaBoneConstraint3D::get_local_point_in_limits(Vector3 in_point, Vector<double> *in_bounds) const {
	// Normalize the input point
	Vector3 point = in_point.normalized();
	real_t closest_cos = -2.0;
	in_bounds->write[0] = -1;

	Vector3 closest_collision_point = in_point;

	// Loop through each limit cone
	for (int i = 0; i < open_cones.size(); i++) {
		const Ref<IKLimitCone3D> cone = open_cones[i];
		Vector3 collision_point = cone->closest_to_cone(point, in_bounds);

		// If the collision point is NaN, return the original point
		if (Math::is_nan(collision_point.x) || Math::is_nan(collision_point.y) || Math::is_nan(collision_point.z)) {
			in_bounds->write[0] = 1;
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
	if ((*in_bounds)[0] == -1) {
		for (int i = 0; i < open_cones.size() - 1; i++) {
			const Ref<IKLimitCone3D> currCone = open_cones[i];
			const Ref<IKLimitCone3D> nextCone = open_cones[i + 1];
			Vector3 collision_point = currCone->get_on_great_tangent_triangle(nextCone, point);

			// If the collision point is NaN, skip to the next iteration
			if (Math::is_nan(collision_point.x)) {
				continue;
			}

			real_t this_cos = collision_point.dot(point);

			// If the cosine is approximately 1, return the original point
			if (Math::is_equal_approx(this_cos, real_t(1.0))) {
				in_bounds->write[0] = 1;
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
	return closest_collision_point;
}

Vector3 KusudamaBoneConstraint3D::local_point_on_path_sequence(Vector3 p_in_point) const {
	double closest_point_dot = 0;
	Vector3 point = p_in_point.normalized();
	Vector3 result = point;

	if (open_cones.size() == 1) {
		const Ref<IKLimitCone3D> cone = open_cones[0];
		result = cone->get_control_point();
	} else {
		for (int i = 0; i < open_cones.size() - 1; i++) {
			const Ref<IKLimitCone3D> next_cone = open_cones[i + 1];
			const Ref<IKLimitCone3D> cone = open_cones[i];
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

void KusudamaBoneConstraint3D::snap_to_orientation_limit(Ref<IKNode3D> bone_direction, Ref<IKNode3D> to_set, Ref<IKNode3D> limiting_axes, real_t p_dampening, real_t p_cos_half_angle_dampen) {
	if (bone_direction.is_null() || to_set.is_null() || limiting_axes.is_null()) {
		return;
	}
	Vector<double> in_bounds;
	in_bounds.resize(1);
	in_bounds.write[0] = 1.0;

	// Create lightweight local snapshots to compute intersections without relying on IKNode3D runtime methods
	kusudama::IKNode3DLocal local_limiting_axes;
	kusudama::IKNode3DLocal local_bone_direction;
	kusudama::IKNode3DLocal local_to_set;

	// Copy essential transform state from Ref<IKNode3D> instances (if valid)
	if (!limiting_axes.is_null()) {
		local_limiting_axes.local_transform = limiting_axes->get_transform();
		local_limiting_axes.global_transform = limiting_axes->get_global_transform();
		local_limiting_axes.disable_scale = limiting_axes->is_scale_disabled();
	}
	if (!bone_direction.is_null()) {
		local_bone_direction.local_transform = bone_direction->get_transform();
		local_bone_direction.global_transform = bone_direction->get_global_transform();
	}
	if (!to_set.is_null()) {
		local_to_set.local_transform = to_set->get_transform();
		local_to_set.global_transform = to_set->get_global_transform();
	}

	// Use the local snapshots for geometric computations
	Vector3 limiting_origin = local_limiting_axes.get_global_transform().origin;
	Vector3 bone_dir_xform = local_bone_direction.get_global_transform().xform(Vector3(0.0, 1.0, 0.0));

	bone_ray->set_point_1(limiting_origin);
	bone_ray->set_point_2(bone_dir_xform);

	Vector3 bone_tip = local_limiting_axes.to_local(bone_ray->get_point_2());
	Vector3 in_limits = get_local_point_in_limits(bone_tip, &in_bounds);

	if (in_bounds[0] < 0) {
		constrained_ray->set_point_1(bone_ray->get_point_1());
		constrained_ray->set_point_2(local_limiting_axes.to_global(in_limits));

		// Compute rectified rotation and apply to real target (preserve existing mutation behavior)
		Quaternion rectified_rot = Quaternion(bone_ray->get_heading(), constrained_ray->get_heading());
		to_set->rotate_local_with_global(rectified_rot);
	}
}

void KusudamaBoneConstraint3D::set_snap_to_twist_limit(Ref<IKNode3D> p_bone_direction, Ref<IKNode3D> p_to_set, Ref<IKNode3D> p_constraint_axes, real_t p_dampening, real_t p_cos_half_dampen) {
	if (!is_axially_constrained()) {
		return;
	}
	if (p_to_set.is_null() || p_constraint_axes.is_null()) {
		return;
	}

	// Snapshot transforms into lightweight local objects to avoid relying on IKNode3D implementation.
	kusudama::IKNode3DLocal local_constraint_axes;
	kusudama::IKNode3DLocal local_to_set;
	kusudama::IKNode3DLocal local_parent;

	local_constraint_axes.local_transform = p_constraint_axes->get_transform();
	local_constraint_axes.global_transform = p_constraint_axes->get_global_transform();

	local_to_set.local_transform = p_to_set->get_transform();
	local_to_set.global_transform = p_to_set->get_global_transform();

	// Parent may be null
	Ref<IKNode3D> parent_ref = p_to_set->get_parent();
	if (!parent_ref.is_null()) {
		local_parent.local_transform = parent_ref->get_transform();
		local_parent.global_transform = parent_ref->get_global_transform();
	}

	// Perform same math using snapshots
	Transform3D global_transform_constraint = local_constraint_axes.get_global_transform();
	Transform3D global_transform_to_set = local_to_set.get_global_transform();
	Basis parent_global_inverse = local_parent.get_global_transform().basis.inverse();
	Basis global_twist_center = global_transform_constraint.basis * twist_center_rot;
	Basis align_rot = (global_twist_center.inverse() * global_transform_to_set.basis).orthonormalized();
	Quaternion twist_rotation, swing_rotation; // Hold the ik transform's decomposed swing and twist away from global_twist_centers's global basis.
	get_swing_twist(align_rot.get_rotation_quaternion(), Vector3(0, 1, 0), swing_rotation, twist_rotation);
	twist_rotation = IKBoneSegment3D::clamp_to_cos_half_angle(twist_rotation, twist_half_range_half_cos);
	Basis recomposition = (global_twist_center * (swing_rotation * twist_rotation)).orthonormalized();
	Basis rotation = parent_global_inverse * recomposition;

	// Apply the computed transform to the real target
	p_to_set->set_transform(Transform3D(rotation, p_to_set->get_transform().origin));
}

Quaternion KusudamaBoneConstraint3D::clamp_to_quadrance_angle(Quaternion p_rotation, double p_cos_half_angle) {
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

Quaternion KusudamaBoneConstraint3D::get_quaternion_axis_angle(const Vector3 &p_axis, real_t p_angle) {
	// Handle zero-length axis case
	if (p_axis.length_squared() < CMP_EPSILON2) {
		return Quaternion(); // Return identity quaternion
	}

	// Handle very small angle case
	if (Math::abs(p_angle) < CMP_EPSILON) {
		return Quaternion(); // Return identity quaternion
	}

	// Standard quaternion creation from axis-angle
	return Quaternion(p_axis, p_angle);
}

void KusudamaBoneConstraint3D::get_swing_twist(
		Quaternion p_rotation,
		Vector3 p_axis,
		Quaternion &r_swing,
		Quaternion &r_twist) {
#ifdef MATH_CHECKS
	ERR_FAIL_COND_MSG(!p_axis.is_normalized(), "The axis must be normalized.");
	ERR_FAIL_COND_MSG(!p_rotation.is_normalized(), "The quaternion must be normalized.");
#endif

	// Handle zero-length axis case
	if (p_axis.length_squared() < CMP_EPSILON2) {
		r_swing = Quaternion();
		r_twist = Quaternion();
		return;
	}

	// Standard swing-twist decomposition
	// Project rotation onto twist axis
	Vector3 axis = p_axis.normalized();
	Quaternion twist = Quaternion(axis, 0.0) * Quaternion(0.0, 0.0, 0.0, 1.0); // Pure twist around axis
	twist = (p_rotation * twist * p_rotation.inverse()).normalized();

	// Extract swing as the remaining rotation
	r_twist = twist;
	r_swing = p_rotation * twist.inverse();
}

void KusudamaBoneConstraint3D::update_tangent_radii() {
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

void KusudamaBoneConstraint3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_open_cones"), &KusudamaBoneConstraint3D::get_open_cones);
	ClassDB::bind_method(D_METHOD("set_open_cones", "open_cones"), &KusudamaBoneConstraint3D::set_open_cones);
	ClassDB::bind_method(D_METHOD("add_open_cone", "cone"), &KusudamaBoneConstraint3D::add_open_cone);
	ClassDB::bind_method(D_METHOD("remove_open_cone", "cone"), &KusudamaBoneConstraint3D::remove_open_cone);
	ClassDB::bind_method(D_METHOD("clear_open_cones"), &KusudamaBoneConstraint3D::clear_open_cones);

	ClassDB::bind_method(D_METHOD("set_axial_limits", "min_angle", "range"), &KusudamaBoneConstraint3D::set_axial_limits);
	ClassDB::bind_method(D_METHOD("get_min_axial_angle"), &KusudamaBoneConstraint3D::get_min_axial_angle);
	ClassDB::bind_method(D_METHOD("get_range_angle"), &KusudamaBoneConstraint3D::get_range_angle);

	ClassDB::bind_method(D_METHOD("is_axially_constrained"), &KusudamaBoneConstraint3D::is_axially_constrained);
	ClassDB::bind_method(D_METHOD("disable_axial_limits"), &KusudamaBoneConstraint3D::disable_axial_limits);
	ClassDB::bind_method(D_METHOD("enable_axial_limits"), &KusudamaBoneConstraint3D::enable_axial_limits);
	ClassDB::bind_method(D_METHOD("toggle_axial_limits"), &KusudamaBoneConstraint3D::toggle_axial_limits);

	ClassDB::bind_method(D_METHOD("is_orientationally_constrained"), &KusudamaBoneConstraint3D::is_orientationally_constrained);
	ClassDB::bind_method(D_METHOD("disable_orientational_limits"), &KusudamaBoneConstraint3D::disable_orientational_limits);
	ClassDB::bind_method(D_METHOD("enable_orientational_limits"), &KusudamaBoneConstraint3D::enable_orientational_limits);
	ClassDB::bind_method(D_METHOD("toggle_orientational_limits"), &KusudamaBoneConstraint3D::toggle_orientational_limits);

	ClassDB::bind_method(D_METHOD("is_enabled"), &KusudamaBoneConstraint3D::is_enabled);
	ClassDB::bind_method(D_METHOD("disable"), &KusudamaBoneConstraint3D::disable);
	ClassDB::bind_method(D_METHOD("enable"), &KusudamaBoneConstraint3D::enable);

	ClassDB::bind_method(D_METHOD("get_resistance"), &KusudamaBoneConstraint3D::get_resistance);
	ClassDB::bind_method(D_METHOD("set_resistance", "resistance"), &KusudamaBoneConstraint3D::set_resistance);
}

KusudamaBoneConstraint3D::KusudamaBoneConstraint3D() {
	bone_ray = Ref<IKRay3D>(memnew(IKRay3D()));
	constrained_ray = Ref<IKRay3D>(memnew(IKRay3D()));
}

KusudamaBoneConstraint3D::~KusudamaBoneConstraint3D() {
}
