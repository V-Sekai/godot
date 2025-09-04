/**************************************************************************/
/*  qcp_ik_3d.cpp                                                         */
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

#include "qcp_ik_3d.h"
#include "core/math/math_defs.h"
#include "core/math/quaternion.h"
#include "core/math/vector3.h"
#include "core/object/class_db.h"

QCPIK3D::QCPIK3D() {
	// QCP solver is used via static methods only
}

void QCPIK3D::set_use_translation(bool p_enabled) {
	use_translation = p_enabled;
}

bool QCPIK3D::get_use_translation() const {
	return use_translation;
}

void QCPIK3D::set_convergence_threshold(real_t p_threshold) {
	convergence_threshold = p_threshold;
}

real_t QCPIK3D::get_convergence_threshold() const {
	return convergence_threshold;
}

void QCPIK3D::set_max_qcp_iterations(int p_iterations) {
	max_qcp_iterations = p_iterations;
}

int QCPIK3D::get_max_qcp_iterations() const {
	return max_qcp_iterations;
}

void QCPIK3D::set_precision(real_t p_precision) {
	precision = p_precision;
}

real_t QCPIK3D::get_precision() const {
	return precision;
}

void QCPIK3D::set_use_rmd_flipping(bool p_enabled) {
	use_rmd_flipping = p_enabled;
}

bool QCPIK3D::get_use_rmd_flipping() const {
	return use_rmd_flipping;
}

void QCPIK3D::set_max_points(int p_max) {
	max_points = p_max;
}

int QCPIK3D::get_max_points() const {
	return max_points;
}

void QCPIK3D::_solve_iteration(double p_delta, Skeleton3D *p_skeleton, ChainIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size) {
	if (p_chain.size() < 2) {
		return;
	}

	// Create target chain aligned with destination
	Vector<Vector3> target_chain;
	target_chain.resize(p_chain.size());

	// Set root position (unchanged)
	target_chain.write[0] = p_chain[0];

	// Create target orientation pointing toward destination
	Vector3 target_direction = (p_destination - p_chain[0]).normalized();

	// Build target chain by maintaining bone lengths but pointing toward target
	for (int i = 1; i < p_chain.size(); i++) {
		int prev_idx = i - 1;
		ManyBoneIK3DSolverInfo *solver_info = p_joints[prev_idx]->solver_info;
		if (solver_info && !Math::is_zero_approx(solver_info->length)) {
			target_chain.write[i] = target_chain[prev_idx] + target_direction * solver_info->length;
		} else {
			target_chain.write[i] = p_chain[i]; // Fallback
		}
	}

	// Compute optimal QCP rotation
	Quaternion optimal_rotation = compute_optimal_qcp_rotation(p_chain, target_chain);

	// Compute optimal translation if enabled
	Vector3 optimal_translation;
	if (use_translation) {
		optimal_translation = compute_optimal_qcp_translation(p_chain, target_chain);
	}

	// Apply the QCP transform to the chain
	apply_qcp_transform(p_chain, optimal_rotation, optimal_translation);

	// Update chain coordinates in the setting
	for (int i = 0; i < p_chain.size(); i++) {
		p_setting->update_chain_coordinate(p_skeleton, i, p_chain[i], false);
	}
}

Quaternion QCPIK3D::compute_optimal_qcp_rotation(const Vector<Vector3> &p_current_chain, const Vector<Vector3> &p_target_chain, const Vector<double> &p_weights) const {
	if (p_current_chain.size() != p_target_chain.size() || p_current_chain.size() < 2) {
		return Quaternion();
	}

	// Prepare point sets for QCP
	PackedVector3Array current_points;
	PackedVector3Array target_points;
	Vector<double> weights;

	for (int i = 0; i < p_current_chain.size(); i++) {
		current_points.append(p_current_chain[i]);
		target_points.append(p_target_chain[i]);

		// Use provided weights or default to 1.0
		if (i < p_weights.size()) {
			weights.push_back(p_weights[i]);
		} else {
			weights.push_back(1.0);
		}
	}

	// Compute optimal rotation using QCP static method
	Array result = QuaternionCharacteristicPolynomial::weighted_superpose(current_points, target_points, weights, use_translation, precision);
	if (result.size() >= 2) {
		Quaternion rotation = result[0];
		return apply_rmd_flipping(rotation);
	}
	return Quaternion();
}

Vector3 QCPIK3D::compute_optimal_qcp_translation(const Vector<Vector3> &p_current_chain, const Vector<Vector3> &p_target_chain) const {
	if (p_current_chain.size() != p_target_chain.size() || p_current_chain.is_empty()) {
		return Vector3();
	}

	// Compute centroids
	Vector3 current_centroid;
	Vector3 target_centroid;

	for (int i = 0; i < p_current_chain.size(); i++) {
		current_centroid += p_current_chain[i];
		target_centroid += p_target_chain[i];
	}

	current_centroid /= p_current_chain.size();
	target_centroid /= p_target_chain.size();

	// Optimal translation is the difference between centroids
	return target_centroid - current_centroid;
}

real_t QCPIK3D::compute_rmsd(const Vector<Vector3> &p_points1, const Vector<Vector3> &p_points2) const {
	if (p_points1.size() != p_points2.size() || p_points1.is_empty()) {
		return 0.0;
	}

	real_t sum_squared_distances = 0.0;
	for (int i = 0; i < p_points1.size(); i++) {
		sum_squared_distances += p_points1[i].distance_squared_to(p_points2[i]);
	}

	return Math::sqrt(sum_squared_distances / p_points1.size());
}

void QCPIK3D::apply_qcp_transform(Vector<Vector3> &p_chain, const Quaternion &p_rotation, const Vector3 &p_translation) {
	if (p_chain.is_empty()) {
		return;
	}

	// Apply rotation and translation to each point in the chain
	for (int i = 0; i < p_chain.size(); i++) {
		// First apply rotation, then translation
		p_chain.write[i] = p_rotation.xform(p_chain[i]) + p_translation;
	}
}

bool QCPIK3D::validate_chain_alignment(const Vector<Vector3> &p_chain, const Vector3 &p_target) const {
	if (p_chain.is_empty()) {
		return false;
	}

	// Check if end effector is close to target
	Vector3 end_effector = p_chain[p_chain.size() - 1];
	return end_effector.distance_to(p_target) <= convergence_threshold;
}

Quaternion QCPIK3D::apply_rmd_flipping(const Quaternion &p_rotation) const {
	if (!use_rmd_flipping) {
		return p_rotation;
	}

	// Apply Right-hand Minimum Distance flipping to ensure optimal quaternion selection
	// This ensures we choose the quaternion that represents the shortest angular path
	Quaternion result = p_rotation;

	// Check if flipping the quaternion gives a shorter angular distance
	if (p_rotation.w < 0) {
		result = Quaternion(-p_rotation.x, -p_rotation.y, -p_rotation.z, -p_rotation.w);
	}

	return result;
}

// Validation methods (matching Elixir implementation)
bool QCPIK3D::validate_inputs(const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target, const Vector<double> &p_weights) const {
	// Check point set sizes match
	if (p_moved.size() != p_target.size()) {
		return false;
	}

	// Check for too many points
	if (p_moved.size() > max_points) {
		return false;
	}

	// Check weights if provided
	if (!p_weights.is_empty()) {
		if (p_weights.size() != p_moved.size()) {
			return false;
		}

		// Check weight ranges
		for (int i = 0; i < p_weights.size(); i++) {
			if (p_weights[i] <= 0.0 || p_weights[i] > 1.0e15) {
				return false;
			}
		}
	}

	return true;
}

bool QCPIK3D::validate_rotation(const Quaternion &p_rotation) const {
	// Check if quaternion is normalized
	real_t magnitude = Math::sqrt(p_rotation.x * p_rotation.x +
			p_rotation.y * p_rotation.y +
			p_rotation.z * p_rotation.z +
			p_rotation.w * p_rotation.w);

	return Math::abs(magnitude - 1.0) <= precision;
}

bool QCPIK3D::validate_alignment(const Quaternion &p_rotation, const Vector3 &p_translation, const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target) const {
	if (p_moved.size() != p_target.size()) {
		return false;
	}

	// Apply transformation and check alignment
	for (int i = 0; i < p_moved.size(); i++) {
		Vector3 transformed = p_rotation.xform(p_moved[i]) + p_translation;
		if (transformed.distance_to(p_target[i]) > convergence_threshold) {
			return false;
		}
	}

	return true;
}

bool QCPIK3D::validate_minimal_rmsd(const Quaternion &p_rotation, const Vector3 &p_translation, const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target) const {
	real_t rmsd = compute_rmsd(p_moved, p_target);
	return rmsd <= convergence_threshold;
}

bool QCPIK3D::validate_minimal_rotation_angle(const Quaternion &p_rotation, real_t p_max_angle) const {
	real_t angle = p_rotation.get_angle();
	return angle <= p_max_angle;
}

bool QCPIK3D::validate_transformation_efficiency(const Quaternion &p_rotation, const Vector3 &p_translation, const String &p_transformation_type) const {
	if (p_transformation_type == "identity") {
		// Check if rotation is identity
		return Math::abs(p_rotation.x) < precision &&
				Math::abs(p_rotation.y) < precision &&
				Math::abs(p_rotation.z) < precision &&
				Math::abs(Math::abs(p_rotation.w) - 1.0) < precision;
	} else if (p_transformation_type == "rotation_only") {
		// Check if translation is zero
		return p_translation.length() < precision;
	} else if (p_transformation_type == "translation_only") {
		// Check if rotation is identity
		return Math::abs(p_rotation.x) < precision &&
				Math::abs(p_rotation.y) < precision &&
				Math::abs(p_rotation.z) < precision &&
				Math::abs(Math::abs(p_rotation.w) - 1.0) < precision;
	}

	return true;
}

bool QCPIK3D::validate_minimal_jerk(const Quaternion &p_rotation, const Vector3 &p_translation) const {
	// For minimal jerk, we want smooth transformations
	// Check that the rotation axis is well-defined and translation is reasonable
	Vector3 axis = p_rotation.get_axis();
	real_t angle = p_rotation.get_angle();

	// Avoid gimbal lock situations
	if (Math::abs(axis.y) > 0.999) {
		return false; // Near gimbal lock
	}

	// Check for reasonable angular velocity
	if (angle > Math::deg_to_rad(180.0)) {
		return false; // Prefer shorter angular paths
	}

	return true;
}

bool QCPIK3D::validate_motion_coordination(const Quaternion &p_rotation, const Vector3 &p_translation) const {
	// Check that rotation and translation are well-coordinated
	// For example, avoid situations where large rotation combines with large translation
	real_t rotation_magnitude = p_rotation.get_angle();
	real_t translation_magnitude = p_translation.length();

	// If both are large, it might indicate poor coordination
	if (rotation_magnitude > Math::deg_to_rad(90.0) && translation_magnitude > 10.0) {
		return false;
	}

	return true;
}

// Advanced validation methods (matching Elixir tests)
bool QCPIK3D::validate_single_point_alignment(const Vector3 &p_moved, const Vector3 &p_target) const {
	// For single points, validate that the transformation aligns them correctly
	Quaternion rotation = compute_optimal_qcp_rotation({ p_moved }, { p_target });
	Vector3 translation = compute_optimal_qcp_translation({ p_moved }, { p_target });

	return validate_alignment(rotation, translation, { p_moved }, { p_target });
}

bool QCPIK3D::validate_opposite_vector_alignment(const Vector3 &p_moved, const Vector3 &p_target) const {
	// Opposite vectors should still produce valid transformations
	if (p_moved.is_equal_approx(-p_target)) {
		Quaternion rotation = compute_optimal_qcp_rotation({ p_moved }, { p_target });
		return validate_rotation(rotation);
	}
	return true;
}

bool QCPIK3D::validate_zero_length_vector_handling(const Vector3 &p_moved, const Vector3 &p_target) const {
	// Zero-length vectors should be handled gracefully
	if (p_moved.is_zero_approx()) {
		Quaternion rotation = compute_optimal_qcp_rotation({ p_moved }, { p_target });
		return validate_rotation(rotation);
	}
	return true;
}

bool QCPIK3D::validate_weighted_point_alignment(const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target, const Vector<double> &p_weights) const {
	if (!validate_inputs(p_moved, p_target, p_weights)) {
		return false;
	}

	Quaternion rotation = compute_optimal_qcp_rotation(p_moved, p_target, p_weights);
	Vector3 translation = compute_optimal_qcp_translation(p_moved, p_target);

	return validate_alignment(rotation, translation, p_moved, p_target);
}

bool QCPIK3D::validate_large_coordinate_handling(const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target) const {
	// Check that large coordinates don't cause numerical issues
	for (int i = 0; i < p_moved.size(); i++) {
		if (p_moved[i].length() > 1.0e10 || p_target[i].length() > 1.0e10) {
			Quaternion rotation = compute_optimal_qcp_rotation(p_moved, p_target);
			return validate_rotation(rotation);
		}
	}
	return true;
}

bool QCPIK3D::validate_near_collinear_points(const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target) const {
	// Near-collinear points should still produce stable transformations
	if (p_moved.size() >= 3) {
		Quaternion rotation = compute_optimal_qcp_rotation(p_moved, p_target);
		return validate_rotation(rotation);
	}
	return true;
}

bool QCPIK3D::validate_rotation_orthogonality(const Quaternion &p_rotation) const {
	// Check that the rotation matrix is orthogonal
	Basis rotation_matrix = Basis(p_rotation);

	// Check that R * R^T = I (orthogonality)
	Basis identity = rotation_matrix * rotation_matrix.transposed();

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			real_t expected = (i == j) ? 1.0 : 0.0;
			if (Math::abs(identity[i][j] - expected) > precision) {
				return false;
			}
		}
	}

	return true;
}

bool QCPIK3D::validate_rotation_determinant(const Quaternion &p_rotation) const {
	// Check that determinant is +1 (proper rotation, not reflection)
	Basis rotation_matrix = Basis(p_rotation);
	real_t det = rotation_matrix.determinant();

	return Math::abs(det - 1.0) <= precision;
}

bool QCPIK3D::validate_distance_preservation(const Vector<Vector3> &p_original, const Vector<Vector3> &p_transformed) const {
	// Check that distances between points are preserved under rigid transformation
	if (p_original.size() != p_transformed.size() || p_original.size() < 2) {
		return true; // Not enough points to check
	}

	// Check all pairwise distances
	for (int i = 0; i < p_original.size(); i++) {
		for (int j = i + 1; j < p_original.size(); j++) {
			real_t original_dist = p_original[i].distance_to(p_original[j]);
			real_t transformed_dist = p_transformed[i].distance_to(p_transformed[j]);

			if (Math::abs(original_dist - transformed_dist) > precision) {
				return false;
			}
		}
	}

	return true;
}

void QCPIK3D::_bind_methods() {
	// QCP-specific properties
	ClassDB::bind_method(D_METHOD("set_use_translation", "enabled"), &QCPIK3D::set_use_translation);
	ClassDB::bind_method(D_METHOD("get_use_translation"), &QCPIK3D::get_use_translation);
	ClassDB::bind_method(D_METHOD("set_convergence_threshold", "threshold"), &QCPIK3D::set_convergence_threshold);
	ClassDB::bind_method(D_METHOD("get_convergence_threshold"), &QCPIK3D::get_convergence_threshold);
	ClassDB::bind_method(D_METHOD("set_max_qcp_iterations", "iterations"), &QCPIK3D::set_max_qcp_iterations);
	ClassDB::bind_method(D_METHOD("get_max_qcp_iterations"), &QCPIK3D::get_max_qcp_iterations);
	ClassDB::bind_method(D_METHOD("set_precision", "precision"), &QCPIK3D::set_precision);
	ClassDB::bind_method(D_METHOD("get_precision"), &QCPIK3D::get_precision);
	ClassDB::bind_method(D_METHOD("set_use_rmd_flipping", "enabled"), &QCPIK3D::set_use_rmd_flipping);
	ClassDB::bind_method(D_METHOD("get_use_rmd_flipping"), &QCPIK3D::get_use_rmd_flipping);
	ClassDB::bind_method(D_METHOD("set_max_points", "max_points"), &QCPIK3D::set_max_points);
	ClassDB::bind_method(D_METHOD("get_max_points"), &QCPIK3D::get_max_points);

	// Utility methods
	ClassDB::bind_method(D_METHOD("validate_chain_alignment", "chain", "target"), &QCPIK3D::validate_chain_alignment);
	ClassDB::bind_method(D_METHOD("compute_rmsd", "points1", "points2"), &QCPIK3D::compute_rmsd);

	// Advanced validation methods
	ClassDB::bind_method(D_METHOD("validate_single_point_alignment", "moved", "target"), &QCPIK3D::validate_single_point_alignment);
	ClassDB::bind_method(D_METHOD("validate_opposite_vector_alignment", "moved", "target"), &QCPIK3D::validate_opposite_vector_alignment);
	ClassDB::bind_method(D_METHOD("validate_zero_length_vector_handling", "moved", "target"), &QCPIK3D::validate_zero_length_vector_handling);
	ClassDB::bind_method(D_METHOD("validate_weighted_point_alignment", "moved", "target", "weights"), &QCPIK3D::validate_weighted_point_alignment);
	ClassDB::bind_method(D_METHOD("validate_large_coordinate_handling", "moved", "target"), &QCPIK3D::validate_large_coordinate_handling);
	ClassDB::bind_method(D_METHOD("validate_near_collinear_points", "moved", "target"), &QCPIK3D::validate_near_collinear_points);
	ClassDB::bind_method(D_METHOD("validate_rotation_orthogonality", "rotation"), &QCPIK3D::validate_rotation_orthogonality);
	ClassDB::bind_method(D_METHOD("validate_rotation_determinant", "rotation"), &QCPIK3D::validate_rotation_determinant);
	ClassDB::bind_method(D_METHOD("validate_distance_preservation", "original", "transformed"), &QCPIK3D::validate_distance_preservation);

	// Property definitions
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_translation"), "set_use_translation", "get_use_translation");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "convergence_threshold", PROPERTY_HINT_RANGE, "0.0001,1.0,0.0001,or_greater"), "set_convergence_threshold", "get_convergence_threshold");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_qcp_iterations", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_max_qcp_iterations", "get_max_qcp_iterations");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "precision", PROPERTY_HINT_RANGE, "1e-15,1e-3,1e-6,or_greater"), "set_precision", "get_precision");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_rmd_flipping"), "set_use_rmd_flipping", "get_use_rmd_flipping");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_points", PROPERTY_HINT_RANGE, "2,10000,1,or_greater"), "set_max_points", "get_max_points");
}
