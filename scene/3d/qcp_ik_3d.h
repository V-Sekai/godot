/**************************************************************************/
/*  qcp_ik_3d.h                                                           */
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

#include "scene/3d/chain_ik_3d.h"
#include "scene/3d/qcp.h"

class QCPIK3D : public ChainIK3D {
	GDCLASS(QCPIK3D, ChainIK3D);

private:
	bool use_translation = true; // Enable full rigid body transform
	real_t convergence_threshold = 0.001; // RMSD threshold for convergence
	int max_qcp_iterations = 10; // Max iterations for QCP solving
	real_t precision = 1.0e-6; // Numerical precision for calculations
	bool use_rmd_flipping = true; // Enable Right-hand Minimum Distance flipping
	int max_points = 10000; // Maximum number of points to process

protected:
	virtual void _solve_iteration(double p_delta, Skeleton3D *p_skeleton, ChainIK3DSetting *p_setting, Vector<ManyBoneIK3DJointSetting *> &p_joints, Vector<Vector3> &p_chain, const Vector3 &p_destination, int p_joint_size, int p_chain_size) override;
	static void _bind_methods();

	// QCP-specific methods
	Quaternion compute_optimal_qcp_rotation(const Vector<Vector3> &p_current_chain, const Vector<Vector3> &p_target_chain, const Vector<double> &p_weights = Vector<double>()) const;
	Vector3 compute_optimal_qcp_translation(const Vector<Vector3> &p_current_chain, const Vector<Vector3> &p_target_chain) const;
	real_t compute_rmsd(const Vector<Vector3> &p_points1, const Vector<Vector3> &p_points2) const;
	void apply_qcp_transform(Vector<Vector3> &p_chain, const Quaternion &p_rotation, const Vector3 &p_translation);
	Quaternion apply_rmd_flipping(const Quaternion &p_rotation) const;

	// Validation methods (matching Elixir implementation)
	bool validate_inputs(const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target, const Vector<double> &p_weights) const;
	bool validate_rotation(const Quaternion &p_rotation) const;
	bool validate_alignment(const Quaternion &p_rotation, const Vector3 &p_translation, const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target) const;
	bool validate_minimal_rmsd(const Quaternion &p_rotation, const Vector3 &p_translation, const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target) const;
	bool validate_minimal_rotation_angle(const Quaternion &p_rotation, real_t p_max_angle) const;
	bool validate_transformation_efficiency(const Quaternion &p_rotation, const Vector3 &p_translation, const String &p_transformation_type) const;
	bool validate_minimal_jerk(const Quaternion &p_rotation, const Vector3 &p_translation) const;
	bool validate_motion_coordination(const Quaternion &p_rotation, const Vector3 &p_translation) const;

public:
	QCPIK3D();

	// QCP-specific properties
	void set_use_translation(bool p_enabled);
	bool get_use_translation() const;
	void set_convergence_threshold(real_t p_threshold);
	real_t get_convergence_threshold() const;
	void set_max_qcp_iterations(int p_iterations);
	int get_max_qcp_iterations() const;
	void set_precision(real_t p_precision);
	real_t get_precision() const;
	void set_use_rmd_flipping(bool p_enabled);
	bool get_use_rmd_flipping() const;
	void set_max_points(int p_max);
	int get_max_points() const;

	bool validate_chain_alignment(const Vector<Vector3> &p_chain, const Vector3 &p_target) const;

	// Advanced validation methods (matching Elixir tests)
	bool validate_single_point_alignment(const Vector3 &p_moved, const Vector3 &p_target) const;
	bool validate_opposite_vector_alignment(const Vector3 &p_moved, const Vector3 &p_target) const;
	bool validate_zero_length_vector_handling(const Vector3 &p_moved, const Vector3 &p_target) const;
	bool validate_weighted_point_alignment(const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target, const Vector<double> &p_weights) const;
	bool validate_large_coordinate_handling(const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target) const;
	bool validate_near_collinear_points(const Vector<Vector3> &p_moved, const Vector<Vector3> &p_target) const;
	bool validate_rotation_orthogonality(const Quaternion &p_rotation) const;
	bool validate_rotation_determinant(const Quaternion &p_rotation) const;
	bool validate_distance_preservation(const Vector<Vector3> &p_original, const Vector<Vector3> &p_transformed) const;
};
