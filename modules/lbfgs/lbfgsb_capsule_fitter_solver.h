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
#include "scene/resources/3d/importer_mesh.h"
#include "scene/resources/mesh.h"

class LBFGSBCapsuleFitterSolver : public LBFGSBSolver {
	GDCLASS(LBFGSBCapsuleFitterSolver, LBFGSBSolver);

public: // Made public for direct access in test, or provide getters
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
				optimized_radius(p_initial_radius) {}

		CapsuleInstance() :
				initial_radius(0.1),
				optimized_radius(0.1) {}
	};

private:
	Vector<CapsuleInstance> m_capsules; // Stores all capsules for multi-fit

	Ref<Mesh> source_mesh;
	int surface_index = 0;
	Dictionary last_fit_result;

	PackedVector3Array current_cloud_points_for_objective;
	PackedVector3Array current_cloud_normals_for_objective; // Added for normals

	// Configurable thresholds and penalty for orientation optimization
	double orientation_distance_threshold = 0.1;
	double orientation_angle_threshold_rad = Math::TAU / 8.0; // Stores radians (45 deg), using Math::TAU
	double orientation_penalty_factor = 1.0; // Default to 1 (no penalty) if not set otherwise

	double huber_delta = 0.1; // Delta parameter for Huber loss

	enum OptimizationMode {
		OPT_MODE_MULTI_ALL_PARAMS // Mode for optimizing all parameters of all capsules in m_capsules
	};
	OptimizationMode _current_optimization_mode = OPT_MODE_MULTI_ALL_PARAMS;

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

	// Methods for managing multiple capsules
	void add_capsule_instance(const Vector3 &p_initial_axis_a_world, const Vector3 &p_initial_axis_b_world, double p_initial_radius);
	void clear_capsule_instances();
	int get_num_capsule_instances() const;
	Dictionary get_capsule_instance_data(int p_idx) const; // For inspecting capsule data, useful for tests

	void set_orientation_distance_threshold(double p_threshold);
	double get_orientation_distance_threshold() const;

	void set_orientation_angle_threshold_rad(double p_threshold_rad);
	double get_orientation_angle_threshold_rad() const;

	void set_orientation_penalty_factor(double p_factor);
	double get_orientation_penalty_factor() const;

	void set_huber_delta(double p_delta);
	double get_huber_delta() const;

	Dictionary get_last_fit_result() const;

	Dictionary optimize_all_capsule_parameters(); // Declaration for the multi-capsule optimization method

public: // Moved struct and static methods here for testability
	struct CapsuleSurfacePointDerivatives {
		Basis dC_dA; // Jacobian of closest point C w.r.t. endpoint A
		Basis dC_dB; // Jacobian of closest point C w.r.t. endpoint B
		Vector3 normal_on_surface; // Added to store the normal itself, useful for d_dist_d_radius
		bool is_valid = false; // Flag to indicate if derivatives are valid (e.g. point not on axis for normal derivative)
	};

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
	Array _generate_canonical_capsule_mesh_arrays(const Vector3 &p_cap_a, const Vector3 &p_cap_b, double p_cap_radius, int p_radial_segments, int p_rings, bool p_closed) const;

	// Helper methods for pre-optimization validation and data preparation
	bool _validate_pre_optimization_conditions(Dictionary &r_result_dict);
	bool _prepare_objective_data(Dictionary &r_result_dict);

	// Helper methods for optimize_all_capsule_parameters
	void _initialize_optimization_parameters(PackedFloat64Array &r_local_x_initial, PackedFloat64Array &r_local_lower_bounds, PackedFloat64Array &r_local_upper_bounds) const;
	// _execute_lbfgsb_optimization runs the core L-BFGS-B solver.
	// It populates r_result_dict with "iterations", "final_fx", "optimized_params", and potentially "error" or "solver_error_message".
	// Returns true if solver ran and produced a structurally valid result (even if that result is an error message from the solver itself), false for pre-solver errors.
	bool _execute_lbfgsb_optimization(const PackedFloat64Array &p_initial_x, const PackedFloat64Array &p_lower_bounds, const PackedFloat64Array &p_upper_bounds, Dictionary &r_result_dict);
	void _process_optimization_result(const PackedFloat64Array &p_optimized_params, int p_num_total_params, Dictionary &r_actual_result_dict);
	Ref<ArrayMesh> _generate_result_mesh_with_capsules() const;
};
