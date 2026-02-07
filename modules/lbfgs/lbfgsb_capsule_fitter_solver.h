/**************************************************************************/
/*  lbfgsb_capsule_fitter_solver.h                                        */
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

#include "core/math/basis.h" // Added for Basis
#include "core/math/vector3.h"
#include "core/variant/dictionary.h"
#include "lbfgsbpp.h"
#include "scene/resources/mesh.h"

class LBFGSBCapsuleFitterSolver : public LBFGSBSolver {
	GDCLASS(LBFGSBCapsuleFitterSolver, LBFGSBSolver);

private:
	Ref<Mesh> source_mesh;
	int surface_index = 0;
	Vector3 axis_point_a = Vector3(0, -0.25, 0);
	Vector3 axis_point_b = Vector3(0, 0.25, 0);
	double radius = 0.1;
	Dictionary last_fit_result;

	PackedVector3Array current_cloud_points_for_objective;
	PackedVector3Array current_cloud_normals_for_objective; // Added for normals

	// For orientation optimization
	Vector3 _orientation_opt_initial_half_axis; // Stores half_axis = (B-A)/2 in initial capsule frame
	double _orientation_opt_fixed_radius = 0.5;

	// Configurable thresholds and penalty for orientation optimization
	double orientation_distance_threshold = 0.1;
	double orientation_angle_threshold_rad = Math::TAU / 8.0; // Stores radians (45 deg), using Math::TAU
	double orientation_penalty_factor = 1.0; // Default to 1 (no penalty) if not set otherwise

	double huber_delta = 0.1; // Delta parameter for Huber loss

	enum OptimizationMode {
		OPT_MODE_RADIUS,
		OPT_MODE_ORIENTATION
	};
	OptimizationMode _current_optimization_mode = OPT_MODE_RADIUS;

	// Helper function for closest point and its normal
	std::pair<Vector3, Vector3> _get_closest_point_and_normal_on_capsule_surface(
			const Vector3 &p_mesh_vertex,
			const Vector3 &p_cap_a,
			const Vector3 &p_cap_b,
			double p_cap_radius) const;

protected:
	static void _bind_methods();

public:
	virtual double call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) override;
	LBFGSBCapsuleFitterSolver();
	~LBFGSBCapsuleFitterSolver();

	void set_source_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_source_mesh() const;

	void set_surface_index(int p_index);
	int get_surface_index() const;

	// FIXME: Replace these methods with offsets from the bone
	void set_axis_point_a(const Vector3 &p_point);
	Vector3 get_axis_point_a() const;

	// FIXME: Replace these methods with offsets from the bone
	void set_axis_point_b(const Vector3 &p_point);
	Vector3 get_axis_point_b() const;

	void set_radius(double p_radius);
	double get_radius() const;

	void set_height(double p_height);
	double get_height() const;

	void set_orientation_distance_threshold(double p_threshold);
	double get_orientation_distance_threshold() const;

	void set_orientation_angle_threshold_rad(double p_threshold_rad);
	double get_orientation_angle_threshold_rad() const;

	void set_orientation_penalty_factor(double p_factor);
	double get_orientation_penalty_factor() const;

	void set_huber_delta(double p_delta);
	double get_huber_delta() const;

	Dictionary get_last_fit_result() const;

	void execute_fit();
	void execute_orientation_fit();

	Dictionary optimize_radius_for_fixed_axis();
	Dictionary optimize_orientation_for_fixed_size();

public: // Moved struct and static methods here for testability
	struct CapsuleSurfacePointDerivatives {
		Basis dC_dA; // Jacobian of closest point C w.r.t. endpoint A
		Basis dC_dB; // Jacobian of closest point C w.r.t. endpoint B
		Basis dN_dA; // Jacobian of capsule normal N w.r.t. endpoint A
		Basis dN_dB; // Jacobian of capsule normal N w.r.t. endpoint B
		bool is_valid = false; // Flag to indicate if derivatives are valid (e.g. point not on axis for normal derivative)
	};

	static std::pair<Vector3, Vector3> get_closest_point_and_normal_on_capsule_surface(
			const Vector3 &p_mesh_vertex,
			const Vector3 &p_cap_a,
			const Vector3 &p_cap_b,
			double p_cap_radius);

	static Basis d_vec_normalized_d_vec(const Vector3 &p_vec);
	static Basis d_basis_d_rot_comp(const Vector3 &p_rot_vec, int p_comp_idx);

	static CapsuleSurfacePointDerivatives get_capsule_surface_derivatives(
			const Vector3 &p_mesh_vertex,
			const Vector3 &p_cap_a,
			const Vector3 &p_cap_b,
			double p_cap_radius);

	static Basis outer_product(const Vector3 &p_v1, const Vector3 &p_v2);

private:
	static Basis _compute_rotation_matrix_from_rot_vec(const Vector3 &p_rot_vec);
	PackedVector3Array _generate_canonical_capsule_points(const Vector3 &p_cap_a, const Vector3 &p_cap_b, double p_cap_radius, int p_cylinder_rings, int p_radial_segments) const;
};
