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

#include "core/math/basis.h"
#include "core/math/vector3.h"
#include "core/variant/dictionary.h"
#include "lbfgsbpp.h"
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/mesh.h"

class LBFGSBCapsuleFitterSolverBase : public LBFGSBSolver {
	GDCLASS(LBFGSBCapsuleFitterSolverBase, LBFGSBSolver);

public:
	struct CapsuleInstance {
		Vector3 initial_axis_a;
		Vector3 initial_axis_b;
		double initial_radius = 0.1;

		Vector3 optimized_axis_a;
		Vector3 optimized_axis_b;
		double optimized_radius = 0.1;

		CapsuleInstance(const Vector3 &p_initial_axis_a, const Vector3 &p_initial_axis_b, double p_initial_radius) :
				initial_axis_a(p_initial_axis_a),
				initial_axis_b(p_initial_axis_b),
				initial_radius(p_initial_radius),
				optimized_axis_a(p_initial_axis_a),
				optimized_axis_b(p_initial_axis_b),
				optimized_radius(p_initial_radius) {}

		CapsuleInstance() :
				initial_axis_a(Vector3(0, 0, -0.5)),
				initial_axis_b(Vector3(0, 0, 0.5)),
				initial_radius(0.1),
				optimized_axis_a(Vector3(0, 0, -0.5)),
				optimized_axis_b(Vector3(0, 0, 0.5)),
				optimized_radius(0.1) {}
	};

	struct CapsuleSurfacePointDerivatives {
		Basis dC_dA;
		Basis dC_dB;
		Vector3 normal_on_surface;
		bool is_valid = false;
		double signed_distance = 0.0;
		double d_sd_d_radius = 0.0;
		Vector3 d_sd_d_axis_a;
		Vector3 d_sd_d_axis_b;
	};

protected: // Changed to protected to allow derived classes to access
	Vector<CapsuleInstance> capsules;
	Ref<Mesh> source_mesh;
	int surface_index = 0;
	Dictionary last_fit_result; // Each solver (radius/axis) will manage its own last_fit_result

	PackedVector3Array current_cloud_points_for_objective;
	PackedVector3Array current_cloud_normals_for_objective;

	double orientation_distance_threshold = 0.1;
	double orientation_angle_threshold_rad = Math::TAU / 8.0;
	double orientation_penalty_factor = 1.0;
	double huber_delta = 0.1;

	enum OptimizationMode {
		OPT_MODE_RADIUS,
		OPT_MODE_AXIS_A,
		OPT_MODE_AXIS_B
	};
	OptimizationMode _current_optimization_mode = OPT_MODE_RADIUS;
	int _current_capsule_idx_for_opt = -1;

	void _initialize_parameters_for_current_mode(
			PackedFloat64Array &r_x_initial,
			PackedFloat64Array &r_lower_bounds,
			PackedFloat64Array &r_upper_bounds) const;
	void _update_capsule_from_optimized_params(
			const PackedFloat64Array &p_optimized_params);

	std::pair<Vector3, Vector3> _get_closest_point_and_normal_on_capsule_surface(
			const Vector3 &p_mesh_vertex,
			const Vector3 &p_cap_a,
			const Vector3 &p_cap_b,
			double p_cap_radius) const;

	bool _validate_pre_optimization_conditions(Dictionary &r_result_dict);
	bool _prepare_objective_data(Dictionary &r_result_dict);
	bool _execute_lbfgsb_optimization(const PackedFloat64Array &p_initial_x, const PackedFloat64Array &p_lower_bounds, const PackedFloat64Array &p_upper_bounds, Dictionary &r_result_dict);

	static void _bind_methods();

public:
	virtual double call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) override;
	LBFGSBCapsuleFitterSolverBase();
	~LBFGSBCapsuleFitterSolverBase();

	void set_source_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_source_mesh() const;

	void set_surface_index(int p_index);
	int get_surface_index() const;

	void add_capsule_instance(const Vector3 &p_initial_axis_a_world, const Vector3 &p_initial_axis_b_world, double p_initial_radius);
	void clear_capsule_instances();
	int get_num_capsule_instances() const;
	Dictionary get_capsule_instance_data(int p_idx) const;

	void set_orientation_distance_threshold(double p_threshold);
	double get_orientation_distance_threshold() const;
	void set_orientation_angle_threshold_rad(double p_threshold_rad);
	double get_orientation_angle_threshold_rad() const;
	void set_orientation_penalty_factor(double p_factor);
	double get_orientation_penalty_factor() const;
	void set_huber_delta(double p_delta);
	double get_huber_delta() const;

	Dictionary get_last_fit_result() const; // Each derived solver will return its specific last result

	// Static helper methods, public for potential external use or easier access
	static Basis d_vec_normalized_d_vec(const Vector3 &p_vec);
	static Basis d_basis_d_rot_comp(const Vector3 &p_rot_vec, int p_comp_idx); // Kept if used, review usage
	static CapsuleSurfacePointDerivatives get_capsule_surface_derivatives(
			const Vector3 &p_mesh_vertex,
			const Vector3 &p_cap_a,
			const Vector3 &p_cap_b,
			double p_cap_radius);
	static Basis outer_product(const Vector3 &p_v1, const Vector3 &p_v2);
	static Basis _compute_rotation_matrix_from_rot_vec(const Vector3 &p_rot_vec); // Made static, review usage
	Array _generate_canonical_capsule_mesh_arrays(const Vector3 &p_cap_a, const Vector3 &p_cap_b, double p_cap_radius, int p_radial_segments, int p_rings, bool p_closed) const; // Keep as member
	Ref<ArrayMesh> _generate_result_mesh_with_capsules() const; // Keep as member
};

// Specialized solver for optimizing capsule radius
class LBFGSBCapsuleRadiusSolver : public LBFGSBCapsuleFitterSolverBase {
	GDCLASS(LBFGSBCapsuleRadiusSolver, LBFGSBCapsuleFitterSolverBase);

protected:
	static void _bind_methods();

public:
	LBFGSBCapsuleRadiusSolver();
	~LBFGSBCapsuleRadiusSolver();
	Dictionary optimize_radius(int p_capsule_idx);
};

// Specialized solver for optimizing capsule axes
class LBFGSBCapsuleAxisSolver : public LBFGSBCapsuleFitterSolverBase {
	GDCLASS(LBFGSBCapsuleAxisSolver, LBFGSBCapsuleFitterSolverBase);

protected:
	static void _bind_methods();

public:
	LBFGSBCapsuleAxisSolver();
	~LBFGSBCapsuleAxisSolver();
	Dictionary optimize_axes(int p_capsule_idx);
};
