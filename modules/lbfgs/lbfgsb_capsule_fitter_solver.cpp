/**************************************************************************/
/*  lbfgsb_capsule_fitter_solver.cpp                                      */
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

#include "lbfgsb_capsule_fitter_solver.h"
#include "core/io/json.h" // For stringifying results if debugging
#include "core/math/geometry_3d.h" // For Geometry3D::get_closest_points_between_segments if needed, though custom logic is used.

void LBFGSBCapsuleFitterSolver::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source_mesh", "p_mesh"), &LBFGSBCapsuleFitterSolver::set_source_mesh);
	ClassDB::bind_method(D_METHOD("get_source_mesh"), &LBFGSBCapsuleFitterSolver::get_source_mesh);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "source_mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_source_mesh", "get_source_mesh");

	ClassDB::bind_method(D_METHOD("set_surface_index", "p_index"), &LBFGSBCapsuleFitterSolver::set_surface_index);
	ClassDB::bind_method(D_METHOD("get_surface_index"), &LBFGSBCapsuleFitterSolver::get_surface_index);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "surface_index"), "set_surface_index", "get_surface_index");

	ClassDB::bind_method(D_METHOD("set_orientation_distance_threshold", "p_threshold"), &LBFGSBCapsuleFitterSolver::set_orientation_distance_threshold);
	ClassDB::bind_method(D_METHOD("get_orientation_distance_threshold"), &LBFGSBCapsuleFitterSolver::get_orientation_distance_threshold);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "orientation_distance_threshold"), "set_orientation_distance_threshold", "get_orientation_distance_threshold");

	ClassDB::bind_method(D_METHOD("set_orientation_angle_threshold_rad", "p_threshold_rad"), &LBFGSBCapsuleFitterSolver::set_orientation_angle_threshold_rad);
	ClassDB::bind_method(D_METHOD("get_orientation_angle_threshold_rad"), &LBFGSBCapsuleFitterSolver::get_orientation_angle_threshold_rad);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "orientation_angle_threshold_rad"), "set_orientation_angle_threshold_rad", "get_orientation_angle_threshold_rad");

	ClassDB::bind_method(D_METHOD("set_orientation_penalty_factor", "p_factor"), &LBFGSBCapsuleFitterSolver::set_orientation_penalty_factor);
	ClassDB::bind_method(D_METHOD("get_orientation_penalty_factor"), &LBFGSBCapsuleFitterSolver::get_orientation_penalty_factor);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "orientation_penalty_factor"), "set_orientation_penalty_factor", "get_orientation_penalty_factor");

	ClassDB::bind_method(D_METHOD("set_huber_delta", "p_delta"), &LBFGSBCapsuleFitterSolver::set_huber_delta);
	ClassDB::bind_method(D_METHOD("get_huber_delta"), &LBFGSBCapsuleFitterSolver::get_huber_delta);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "huber_delta"), "set_huber_delta", "get_huber_delta");

	ClassDB::bind_method(D_METHOD("get_last_fit_result"), &LBFGSBCapsuleFitterSolver::get_last_fit_result);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "last_fit_result", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "", "get_last_fit_result");

	ClassDB::bind_method(D_METHOD("optimize_all_capsule_parameters"), &LBFGSBCapsuleFitterSolver::optimize_all_capsule_parameters);

	ClassDB::bind_method(D_METHOD("add_capsule_instance", "initial_axis_a", "initial_axis_b", "initial_radius"), &LBFGSBCapsuleFitterSolver::add_capsule_instance);
	ClassDB::bind_method(D_METHOD("clear_capsule_instances"), &LBFGSBCapsuleFitterSolver::clear_capsule_instances);
	ClassDB::bind_method(D_METHOD("get_num_capsule_instances"), &LBFGSBCapsuleFitterSolver::get_num_capsule_instances);
	ClassDB::bind_method(D_METHOD("get_capsule_instance_data", "p_idx"), &LBFGSBCapsuleFitterSolver::get_capsule_instance_data);
}

LBFGSBCapsuleFitterSolver::LBFGSBCapsuleFitterSolver() {
	// Default constructor
}

LBFGSBCapsuleFitterSolver::~LBFGSBCapsuleFitterSolver() {
	// Destructor
}

void LBFGSBCapsuleFitterSolver::set_source_mesh(const Ref<Mesh> &p_mesh) {
	last_fit_result.clear();
	m_capsules.clear(); // Clear capsules when a new mesh is set.
	source_mesh = p_mesh; // Store the original user-provided mesh.
	current_cloud_points_for_objective.clear(); // Clear derived data.
	current_cloud_normals_for_objective.clear(); // Clear derived data.
}

Ref<Mesh> LBFGSBCapsuleFitterSolver::get_source_mesh() const {
	return source_mesh;
}

void LBFGSBCapsuleFitterSolver::set_surface_index(int p_index) {
	surface_index = p_index;
}

int LBFGSBCapsuleFitterSolver::get_surface_index() const {
	return surface_index;
}

void LBFGSBCapsuleFitterSolver::set_orientation_distance_threshold(double p_threshold) {
	orientation_distance_threshold = p_threshold;
}
double LBFGSBCapsuleFitterSolver::get_orientation_distance_threshold() const {
	return orientation_distance_threshold;
}

void LBFGSBCapsuleFitterSolver::set_orientation_angle_threshold_rad(double p_threshold_rad) {
	orientation_angle_threshold_rad = p_threshold_rad;
}
double LBFGSBCapsuleFitterSolver::get_orientation_angle_threshold_rad() const {
	return orientation_angle_threshold_rad;
}

void LBFGSBCapsuleFitterSolver::set_orientation_penalty_factor(double p_factor) {
	orientation_penalty_factor = p_factor;
}
double LBFGSBCapsuleFitterSolver::get_orientation_penalty_factor() const {
	return orientation_penalty_factor;
}

void LBFGSBCapsuleFitterSolver::set_huber_delta(double p_delta) {
	huber_delta = p_delta;
}
double LBFGSBCapsuleFitterSolver::get_huber_delta() const {
	return huber_delta;
}

Dictionary LBFGSBCapsuleFitterSolver::get_last_fit_result() const {
	return last_fit_result;
}

void LBFGSBCapsuleFitterSolver::add_capsule_instance(const Vector3 &p_initial_axis_a, const Vector3 &p_initial_axis_b, double p_initial_radius) {
	CapsuleInstance instance(p_initial_axis_a, p_initial_axis_b, p_initial_radius);
	m_capsules.push_back(instance);
}

void LBFGSBCapsuleFitterSolver::clear_capsule_instances() {
	m_capsules.clear();
}

int LBFGSBCapsuleFitterSolver::get_num_capsule_instances() const {
	return m_capsules.size();
}

Dictionary LBFGSBCapsuleFitterSolver::get_capsule_instance_data(int p_idx) const {
	Dictionary data;
	ERR_FAIL_INDEX_V_MSG(p_idx, m_capsules.size(), data, "Capsule index out of bounds.");
	const CapsuleInstance &instance = m_capsules[p_idx];
	data["initial_axis_a"] = instance.initial_axis_a;
	data["initial_axis_b"] = instance.initial_axis_b;
	data["initial_radius"] = instance.initial_radius;
	data["optimized_axis_a"] = instance.optimized_axis_a;
	data["optimized_axis_b"] = instance.optimized_axis_b;
	data["optimized_radius"] = instance.optimized_radius;
	return data;
}

// Helper method to initialize optimization parameters
void LBFGSBCapsuleFitterSolver::_initialize_optimization_parameters(PackedFloat64Array &r_local_x_initial, PackedFloat64Array &r_local_lower_bounds, PackedFloat64Array &r_local_upper_bounds) const {
	int num_total_params = m_capsules.size() * 7;
	r_local_x_initial.resize(num_total_params);
	r_local_lower_bounds.resize(num_total_params);
	r_local_upper_bounds.resize(num_total_params);

	for (int i = 0; i < m_capsules.size(); ++i) {
		const CapsuleInstance &capsule = m_capsules[i];
		int offset = i * 7;

		r_local_x_initial.write[offset + 0] = capsule.initial_radius;
		r_local_lower_bounds.write[offset + 0] = 0.001; // Min radius
		r_local_upper_bounds.write[offset + 0] = MAXFLOAT;

		r_local_x_initial.write[offset + 1] = capsule.initial_axis_a.x;
		r_local_x_initial.write[offset + 2] = capsule.initial_axis_a.y;
		r_local_x_initial.write[offset + 3] = capsule.initial_axis_a.z;
		for (int j = 0; j < 3; ++j) {
			r_local_lower_bounds.write[offset + 1 + j] = -MAXFLOAT;
			r_local_upper_bounds.write[offset + 1 + j] = MAXFLOAT;
		}

		r_local_x_initial.write[offset + 4] = capsule.initial_axis_b.x;
		r_local_x_initial.write[offset + 5] = capsule.initial_axis_b.y;
		r_local_x_initial.write[offset + 6] = capsule.initial_axis_b.z;
		for (int j = 0; j < 3; ++j) {
			r_local_lower_bounds.write[offset + 4 + j] = -MAXFLOAT;
			r_local_upper_bounds.write[offset + 4 + j] = MAXFLOAT;
		}
	}
}

// Helper method to process optimization results
void LBFGSBCapsuleFitterSolver::_process_optimization_result(const PackedFloat64Array &p_optimized_params, int p_num_total_params, Dictionary &r_actual_result_dict) {
	if (p_optimized_params.size() != p_num_total_params) {
		r_actual_result_dict["error"] = "Optimized params size mismatch. Expected " + itos(p_num_total_params) + ", got " + itos(p_optimized_params.size()) + ".";
		r_actual_result_dict["optimized_capsules_results"] = Array();
		return; // Return void
	}

	Array optimized_capsules_results_array;
	for (int i = 0; i < m_capsules.size(); ++i) {
		CapsuleInstance &capsule = m_capsules.write[i];
		int offset = i * 7;
		ERR_FAIL_COND_MSG(offset + 6 >= p_optimized_params.size(), "Offset calculation error during capsule result processing.");

		capsule.optimized_radius = p_optimized_params[offset + 0];
		capsule.optimized_axis_a = Vector3(p_optimized_params[offset + 1], p_optimized_params[offset + 2], p_optimized_params[offset + 3]);
		capsule.optimized_axis_b = Vector3(p_optimized_params[offset + 4], p_optimized_params[offset + 5], p_optimized_params[offset + 6]);

		Dictionary capsule_result_dict;
		capsule_result_dict["optimized_radius"] = capsule.optimized_radius;
		capsule_result_dict["optimized_axis_a"] = capsule.optimized_axis_a;
		capsule_result_dict["optimized_axis_b"] = capsule.optimized_axis_b;
		optimized_capsules_results_array.push_back(capsule_result_dict);
	}
	r_actual_result_dict["optimized_capsules_results"] = optimized_capsules_results_array;
	// No return here, it's void
}

// Helper method to generate the result mesh with capsules
Ref<ArrayMesh> LBFGSBCapsuleFitterSolver::_generate_result_mesh_with_capsules() const {
	Ref<ArrayMesh> combined_mesh;
	combined_mesh.instantiate();

	// 1. Add surfaces from the source mesh
	if (source_mesh.is_valid()) {
		for (int i = 0; i < source_mesh->get_surface_count(); ++i) {
			Array arrays = source_mesh->surface_get_arrays(i);
			if (arrays.is_empty() || arrays[Mesh::ARRAY_VERTEX].is_null() || (arrays[Mesh::ARRAY_VERTEX].get_type() == Variant::PACKED_VECTOR3_ARRAY && PackedVector3Array(arrays[Mesh::ARRAY_VERTEX]).is_empty())) {
				continue;
			}
			combined_mesh->add_surface_from_arrays(
					source_mesh->surface_get_primitive_type(i),
					arrays,
					source_mesh->surface_get_blend_shape_arrays(i),
					source_mesh->surface_get_lods(i),
					source_mesh->surface_get_format(i));
			int new_surf_idx = combined_mesh->get_surface_count() - 1;
			if (new_surf_idx >= 0) {
				Ref<Material> mat = source_mesh->surface_get_material(i);
				if (mat.is_valid()) {
					combined_mesh->surface_set_material(new_surf_idx, mat);
				}
			}
		}
	}

	// 2. Generate and add meshes for each optimized capsule
	for (int i = 0; i < m_capsules.size(); ++i) {
		const CapsuleInstance &capsule = m_capsules[i];
		Array capsule_geom_arrays = _generate_canonical_capsule_mesh_arrays(capsule.optimized_axis_a, capsule.optimized_axis_b, capsule.optimized_radius, 8, 16, true); // Example params

		if (!capsule_geom_arrays.is_empty() && capsule_geom_arrays.size() == 3) {
			Array capsule_mesh_arrays_for_surface; // Use a local variable for clarity
			capsule_mesh_arrays_for_surface.resize(Mesh::ARRAY_MAX);
			capsule_mesh_arrays_for_surface[Mesh::ARRAY_VERTEX] = capsule_geom_arrays[0]; // Vertices
			capsule_mesh_arrays_for_surface[Mesh::ARRAY_NORMAL] = capsule_geom_arrays[1]; // Normals
			capsule_mesh_arrays_for_surface[Mesh::ARRAY_INDEX] = capsule_geom_arrays[2]; // Indices

			if (!PackedVector3Array(capsule_mesh_arrays_for_surface[Mesh::ARRAY_VERTEX]).is_empty() &&
					!PackedInt32Array(capsule_mesh_arrays_for_surface[Mesh::ARRAY_INDEX]).is_empty()) {
				combined_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, capsule_mesh_arrays_for_surface);
			}
		}
	}
	return combined_mesh;
}

// Helper method for pre-optimization validation
bool LBFGSBCapsuleFitterSolver::_validate_pre_optimization_conditions(Dictionary &r_result_dict) {
	if (!source_mesh.is_valid()) {
		r_result_dict["error"] = "Internal source ArrayMesh is not set or invalid. Call set_source_mesh() first with a valid mesh.";
		r_result_dict["optimized_capsules_results"] = Array();
		return false;
	}

	if (surface_index < 0 || surface_index >= source_mesh->get_surface_count()) {
		r_result_dict["error"] = "Invalid surface_index " + itos(surface_index) +
							 ". Internal ArrayMesh has " + itos(source_mesh->get_surface_count()) + " surfaces.";
		r_result_dict["optimized_capsules_results"] = Array();
		return false;
	}

	if (m_capsules.is_empty()) {
		r_result_dict["message"] = "No capsule instances defined. Nothing to optimize.";
		r_result_dict["optimized_capsules_results"] = Array();
		r_result_dict["final_fx"] = 0.0;
		r_result_dict["iterations"] = 0;
		return false;
	}
	return true;
}

// Helper method to prepare data for the objective function
bool LBFGSBCapsuleFitterSolver::_prepare_objective_data(Dictionary &r_result_dict) {
	Array surface_arrays = source_mesh->surface_get_arrays(surface_index);
	if (surface_arrays.is_empty() || surface_arrays[Mesh::ARRAY_VERTEX].is_null() || (surface_arrays[Mesh::ARRAY_VERTEX].get_type() == Variant::PACKED_VECTOR3_ARRAY && PackedVector3Array(surface_arrays[Mesh::ARRAY_VERTEX]).is_empty())) {
		r_result_dict["error"] = "Internal source ArrayMesh surface arrays are empty or missing vertex data for surface index " + itos(surface_index) + ".";
		r_result_dict["optimized_capsules_results"] = Array();
		return false;
	}

	current_cloud_points_for_objective = surface_arrays[Mesh::ARRAY_VERTEX];
	if (source_mesh->surface_get_format(surface_index) & Mesh::ARRAY_FORMAT_NORMAL) {
		current_cloud_normals_for_objective = surface_arrays[Mesh::ARRAY_NORMAL];
	} else {
		current_cloud_normals_for_objective.clear(); // Ensure it's cleared if not present
	}

	if (current_cloud_points_for_objective.is_empty()) {
		r_result_dict["error"] = "Point cloud (from internal ArrayMesh surface " + itos(surface_index) + ") is empty.";
		r_result_dict["optimized_capsules_results"] = Array();
		return false;
	}
	return true;
}

// Helper method to execute the L-BFGS-B optimization
bool LBFGSBCapsuleFitterSolver::_execute_lbfgsb_optimization(const PackedFloat64Array &p_initial_x, const PackedFloat64Array &p_lower_bounds, const PackedFloat64Array &p_upper_bounds, Dictionary &r_result_dict) {
	_current_optimization_mode = OPT_MODE_MULTI_ALL_PARAMS;
	PackedFloat64Array dummy_gradient_array;
	double initial_fx = 0.0;
	if (p_initial_x.size() > 0) { // Avoid calling operator if no params (e.g. zero capsules, though handled earlier)
		initial_fx = call_operator(p_initial_x, dummy_gradient_array);
	}

	Array raw_solver_result_array = minimize(p_initial_x, initial_fx, p_lower_bounds, p_upper_bounds);

	if (raw_solver_result_array.size() != 3 ||
			raw_solver_result_array[0].get_type() != Variant::INT ||
			raw_solver_result_array[1].get_type() != Variant::PACKED_FLOAT64_ARRAY ||
			raw_solver_result_array[2].get_type() != Variant::FLOAT) {
		r_result_dict["error"] = "Base minimize method returned Array with unexpected structure or types. Expected [int, PackedFloat64Array, float].";
		if (raw_solver_result_array.size() == 1 && raw_solver_result_array[0].get_type() == Variant::STRING) { // Check if it's an error string from LBFGSBSolver
			r_result_dict["solver_error_message"] = raw_solver_result_array[0];
		}
		r_result_dict["optimized_capsules_results"] = Array();
		return false; // Indicates solver did not produce a valid structure
	}

	r_result_dict["iterations"] = raw_solver_result_array[0];
	r_result_dict["optimized_params"] = raw_solver_result_array[1]; // Store as PackedFloat64Array
	r_result_dict["final_fx"] = raw_solver_result_array[2];

	return true; // Solver ran and returned a structurally valid result
}

Dictionary LBFGSBCapsuleFitterSolver::optimize_all_capsule_parameters() {
	last_fit_result.clear();
	current_cloud_points_for_objective.clear();
	current_cloud_normals_for_objective.clear();

	if (!_validate_pre_optimization_conditions(last_fit_result)) {
		return last_fit_result;
	}

	if (!_prepare_objective_data(last_fit_result)) {
		return last_fit_result;
	}

	int num_total_params = m_capsules.size() * 7;
	PackedFloat64Array local_x_initial;
	PackedFloat64Array local_lower_bounds;
	PackedFloat64Array local_upper_bounds;

	_initialize_optimization_parameters(local_x_initial, local_lower_bounds, local_upper_bounds);

	Dictionary optimization_run_result_dict; // Temporary dict for _execute_lbfgsb_optimization
	if (!_execute_lbfgsb_optimization(local_x_initial, local_lower_bounds, local_upper_bounds, optimization_run_result_dict)) {
		last_fit_result.merge(optimization_run_result_dict); // Merge any error info
		return last_fit_result;
	}

	// Merge the successful execution results into last_fit_result
	last_fit_result.merge(optimization_run_result_dict);

	// Process the optimized parameters from the dictionary
	PackedFloat64Array optimized_params = last_fit_result["optimized_params"];
	_process_optimization_result(optimized_params, num_total_params, last_fit_result);

	// Check for errors from _process_optimization_result (e.g., param size mismatch)
	if (last_fit_result.has("error")) {
		return last_fit_result;
	}

	last_fit_result["result_mesh_with_capsules"] = _generate_result_mesh_with_capsules();

	return last_fit_result;
}

double LBFGSBCapsuleFitterSolver::call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) {
	if (r_grad.size() != p_x.size()) {
		r_grad.resize(p_x.size());
	}
	for (int i = 0; i < r_grad.size(); ++i) {
		r_grad.write[i] = 0.0;
	}

	if (_current_optimization_mode != OPT_MODE_MULTI_ALL_PARAMS) {
		ERR_FAIL_V_MSG(0.0, "call_operator called with unexpected optimization mode.");
	}

	if (m_capsules.is_empty()) {
		return 0.0;
	}

	int expected_params = m_capsules.size() * 7;
	ERR_FAIL_COND_V_MSG(p_x.size() != expected_params, 0.0, "Parameter vector size mismatch in call_operator.");

	double total_fx = 0.0;

	if (current_cloud_points_for_objective.is_empty()) {
		return 0.0;
	}

	const double MIN_RADIUS_GEOM = 1e-5; // Minimum radius for geometry functions to prevent division by zero or instability.
	const double MIN_RADIUS_PENALTY_THRESHOLD = 1e-4; // Penalize if radius is below this.
	const double RADIUS_PENALTY_COEFF = 1e12;

	const double AXIS_DIST_SQ_PENALTY_THRESHOLD = 1e-8; // Squared distance threshold (e.g., 1e-4 actual distance).
	const double AXIS_PENALTY_COEFF = 1e10;

	for (int cap_idx = 0; cap_idx < m_capsules.size(); ++cap_idx) {
		int offset = cap_idx * 7;

		double current_radius = p_x[offset + 0];
		Vector3 current_axis_a = Vector3(p_x[offset + 1], p_x[offset + 2], p_x[offset + 3]);
		Vector3 current_axis_b = Vector3(p_x[offset + 4], p_x[offset + 5], p_x[offset + 6]);

		// Penalty for very small or non-positive radius (quadratic)
		if (current_radius < MIN_RADIUS_PENALTY_THRESHOLD) {
			double radius_diff = MIN_RADIUS_PENALTY_THRESHOLD - current_radius;
			total_fx += RADIUS_PENALTY_COEFF * radius_diff * radius_diff;
			r_grad.write[offset + 0] += RADIUS_PENALTY_COEFF * 2.0 * radius_diff * (-1.0);
		}

		// Penalty for coincident/very close axis points (quadratic)
		Vector3 axis_vec = current_axis_b - current_axis_a;
		double axis_dist_sq = axis_vec.length_squared();
		if (axis_dist_sq < AXIS_DIST_SQ_PENALTY_THRESHOLD) {
			double dist_sq_diff = AXIS_DIST_SQ_PENALTY_THRESHOLD - axis_dist_sq;
			total_fx += AXIS_PENALTY_COEFF * dist_sq_diff * dist_sq_diff;
			double common_grad_term = 2.0 * AXIS_PENALTY_COEFF * dist_sq_diff * (-1.0); // d/dx (Threshold - x)^2 = 2*(Threshold-x)*(-1) * (dx/dparam)

			// d(axis_dist_sq)/d(axis_a.coord) = -2 * axis_vec.coord
			// d(axis_dist_sq)/d(axis_b.coord) =  2 * axis_vec.coord
			r_grad.write[offset + 1] += common_grad_term * (-2.0 * axis_vec.x);
			r_grad.write[offset + 2] += common_grad_term * (-2.0 * axis_vec.y);
			r_grad.write[offset + 3] += common_grad_term * (-2.0 * axis_vec.z);
			r_grad.write[offset + 4] += common_grad_term * (2.0 * axis_vec.x);
			r_grad.write[offset + 5] += common_grad_term * (2.0 * axis_vec.y);
			r_grad.write[offset + 6] += common_grad_term * (2.0 * axis_vec.z);
		}

		double capsule_fx_contribution = 0.0;
		double effective_radius_for_geom = MAX(current_radius, MIN_RADIUS_GEOM);

		for (int i = 0; i < current_cloud_points_for_objective.size(); ++i) {
			Vector3 mesh_vertex = current_cloud_points_for_objective[i];
			std::pair<Vector3, Vector3> closest_pair = _get_closest_point_and_normal_on_capsule_surface(
					mesh_vertex, current_axis_a, current_axis_b, effective_radius_for_geom);
			Vector3 closest_point_on_capsule = closest_pair.first;

			Vector3 diff_vec = mesh_vertex - closest_point_on_capsule;
			double distance = diff_vec.length();

			double huber_loss_val;
			double d_huber_loss_d_distance;
			if (Math::abs(distance) <= huber_delta) {
				huber_loss_val = 0.5 * distance * distance;
				d_huber_loss_d_distance = distance;
			} else {
				huber_loss_val = huber_delta * (Math::abs(distance) - 0.5 * huber_delta);
				d_huber_loss_d_distance = huber_delta * SIGN(distance);
			}
			capsule_fx_contribution += huber_loss_val;

			if (distance > 1e-9) {
				Vector3 d_dist_d_closest_point_normalized = -diff_vec.normalized();

				CapsuleSurfacePointDerivatives derivatives = get_capsule_surface_derivatives(
						mesh_vertex, current_axis_a, current_axis_b, effective_radius_for_geom);

				if (derivatives.is_valid) {
					double d_dist_d_radius = d_dist_d_closest_point_normalized.dot(derivatives.normal_on_surface);
					r_grad.write[offset + 0] += d_huber_loss_d_distance * d_dist_d_radius;

					Vector3 d_dist_d_axis_a_comps = derivatives.dC_dA.transposed().xform(d_dist_d_closest_point_normalized);
					r_grad.write[offset + 1] += d_huber_loss_d_distance * d_dist_d_axis_a_comps.x;
					r_grad.write[offset + 2] += d_huber_loss_d_distance * d_dist_d_axis_a_comps.y;
					r_grad.write[offset + 3] += d_huber_loss_d_distance * d_dist_d_axis_a_comps.z;

					Vector3 d_dist_d_axis_b_comps = derivatives.dC_dB.transposed().xform(d_dist_d_closest_point_normalized);
					r_grad.write[offset + 4] += d_huber_loss_d_distance * d_dist_d_axis_b_comps.x;
					r_grad.write[offset + 5] += d_huber_loss_d_distance * d_dist_d_axis_b_comps.y;
					r_grad.write[offset + 6] += d_huber_loss_d_distance * d_dist_d_axis_b_comps.z;
				}
			}
		}
		total_fx += capsule_fx_contribution;
	}
	return total_fx;
}

// Implementation for _get_closest_point_and_normal_on_capsule_surface
std::pair<Vector3, Vector3> LBFGSBCapsuleFitterSolver::_get_closest_point_and_normal_on_capsule_surface(
		const Vector3 &p_mesh_vertex,
		const Vector3 &p_cap_a,
		const Vector3 &p_cap_b,
		double p_cap_radius) const {
	Vector3 axis_vec = p_cap_b - p_cap_a;
	double axis_len_sq = axis_vec.length_squared();
	const double EPSILON_SQ_LEN = 1e-9; // For squared length checks

	if (axis_len_sq < EPSILON_SQ_LEN) { // Treat as sphere centered at p_cap_a
		Vector3 normal_to_center = p_mesh_vertex - p_cap_a;
		double dist_to_center = normal_to_center.length();
		Vector3 normal_on_surface = (dist_to_center < 1e-7) ? Vector3(0, 1, 0) : normal_to_center / dist_to_center; // Avoid division by zero
		Vector3 surface_point = p_cap_a + normal_on_surface * p_cap_radius;
		return { surface_point, normal_on_surface };
	}

	double t = (p_mesh_vertex - p_cap_a).dot(axis_vec) / axis_len_sq;
	Vector3 closest_point_on_axis;

	if (t <= 0.0) { // Clamped to A
		closest_point_on_axis = p_cap_a;
	} else if (t >= 1.0) { // Clamped to B
		closest_point_on_axis = p_cap_b;
	} else { // On the segment
		closest_point_on_axis = p_cap_a + t * axis_vec;
	}

	Vector3 vec_from_axis_to_vertex = p_mesh_vertex - closest_point_on_axis;
	double dist_to_axis = vec_from_axis_to_vertex.length();

	Vector3 normal_on_surface;
	if (dist_to_axis < 1e-7) { // Point is on the axis or very close
		// If on axis, normal is ill-defined for cylinder wall.
		// For caps (t=0 or t=1), normal is along axis.
		if (t < 0.01) { // Near A cap
			normal_on_surface = (p_cap_a - p_cap_b).normalized();
		} else if (t > 0.99) { // Near B cap
			normal_on_surface = (p_cap_b - p_cap_a).normalized();
		} else { // On the cylinder shaft axis, pick an arbitrary perpendicular
			normal_on_surface = axis_vec.get_any_perpendicular().normalized();
		}
	} else {
		normal_on_surface = vec_from_axis_to_vertex.normalized();
	}
	Vector3 closest_point_on_surface = closest_point_on_axis + normal_on_surface * p_cap_radius;
	return { closest_point_on_surface, normal_on_surface };
}

LBFGSBCapsuleFitterSolver::CapsuleSurfacePointDerivatives LBFGSBCapsuleFitterSolver::get_capsule_surface_derivatives(
		const Vector3 &p_mesh_vertex,
		const Vector3 &p_cap_a,
		const Vector3 &p_cap_b,
		double p_cap_radius) {
	CapsuleSurfacePointDerivatives derivatives;
	derivatives.is_valid = true; // Assume valid unless specific failure
	Vector3 V = p_cap_b - p_cap_a; // Axis vector V from A to B
	double len_sq_V = V.length_squared();
	const double EPSILON_SQ = 1e-9; // For squared length checks (e.g., 1e-4 or 1e-5 actual length)
	const double EPSILON_DIST = 1e-7; // For distance checks (e.g., point on axis, point at cap center)
	Basis id_matrix; // Identity matrix

	// Case 1: Degenerate capsule (essentially a sphere centered at p_cap_a)
	if (len_sq_V < EPSILON_SQ) {
		Vector3 U_pa = p_mesh_vertex - p_cap_a; // Vector from p_cap_a to p_mesh_vertex
		double dist_U_pa = U_pa.length();
		if (dist_U_pa < EPSILON_DIST) { // p_mesh_vertex is at the center of the sphere
			derivatives.is_valid = false; // Normal and derivatives are ill-defined
			return derivatives;
		}
		derivatives.normal_on_surface = U_pa / dist_U_pa; // U_pa.normalized()
		// C = A + R * normalize(P-A)
		// dC/dA = I + R * d(normalize(P-A))/dA = I - R * d_normalize(P-A)
		derivatives.dC_dA = id_matrix - d_vec_normalized_d_vec(U_pa) * p_cap_radius;
		derivatives.dC_dB.set_zero(); // p_cap_b has no distinct influence
		return derivatives;
	}

	// Parameter t for projection of (p_mesh_vertex - p_cap_a) onto V
	Vector3 U_va = p_mesh_vertex - p_cap_a; // Vector from p_cap_a to p_mesh_vertex
	double t = U_va.dot(V) / len_sq_V;

	Vector3 P_axis; // Closest point on the infinite line defined by A and B

	// Case 2: Closest point is on one of the spherical end caps
	if (t < 0.0) { // Closest to p_cap_a (spherical cap at A)
		P_axis = p_cap_a;
		Vector3 U_pa_cap = p_mesh_vertex - P_axis; // p_mesh_vertex - p_cap_a
		double dist_U_pa_cap = U_pa_cap.length();

		if (dist_U_pa_cap < EPSILON_DIST) { // p_mesh_vertex is at the center of cap A
			derivatives.is_valid = false;
			return derivatives;
		}
		derivatives.normal_on_surface = U_pa_cap / dist_U_pa_cap;
		// C = A + R * normalize(P-A)
		derivatives.dC_dA = id_matrix - d_vec_normalized_d_vec(U_pa_cap) * p_cap_radius;
		derivatives.dC_dB.set_zero();
	} else if (t > 1.0) { // Closest to p_cap_b (spherical cap at B)
		P_axis = p_cap_b;
		Vector3 U_pb_cap = p_mesh_vertex - P_axis; // p_mesh_vertex - p_cap_b
		double dist_U_pb_cap = U_pb_cap.length();

		if (dist_U_pb_cap < EPSILON_DIST) { // p_mesh_vertex is at the center of cap B
			derivatives.is_valid = false;
			return derivatives;
		}
		derivatives.normal_on_surface = U_pb_cap / dist_U_pb_cap;
		derivatives.dC_dA.set_zero();
		// C = B + R * normalize(P-B)
		derivatives.dC_dB = id_matrix - d_vec_normalized_d_vec(U_pb_cap) * p_cap_radius;
	} else {
		// Case 3: Closest point is on the cylindrical wall
		P_axis = p_cap_a + t * V;
		Vector3 N_raw = p_mesh_vertex - P_axis; // Vector from P_axis to p_mesh_vertex
		double dist_N_raw = N_raw.length();

		if (dist_N_raw < EPSILON_DIST) { // p_mesh_vertex is on the capsule axis segment
			derivatives.is_valid = false; // Normal and derivatives are ill-defined
			return derivatives;
		}
		derivatives.normal_on_surface = N_raw / dist_N_raw;

		// Factor matrix: (I - R * d_normalize(N_raw))
		Basis factor_matrix = id_matrix - d_vec_normalized_d_vec(N_raw) * p_cap_radius;

		// Gradient of t w.r.t. p_cap_a and p_cap_b
		// dt/dA = (-(U_va+V)*len_sq_V + 2*dot(U_va,V)*V) / (len_sq_V^2)
		// dt/dB = ( U_va*len_sq_V - 2*dot(U_va,V)*V) / (len_sq_V^2)
		Vector3 grad_t_dA_num = (-(U_va + V) * len_sq_V) + (2.0 * U_va.dot(V) * V);
		Vector3 grad_t_dA = grad_t_dA_num / (len_sq_V * len_sq_V); // Denominator is (len_sq_V)^2

		Vector3 grad_t_dB_num = (U_va * len_sq_V) - (2.0 * U_va.dot(V) * V);
		Vector3 grad_t_dB = grad_t_dB_num / (len_sq_V * len_sq_V);

		// dP_axis/dA = (1-t)I + outer_product(V, grad_t_dA)
		Basis dP_axis_dA_matrix = (id_matrix * (1.0 - t)) + outer_product(V, grad_t_dA);
		derivatives.dC_dA = factor_matrix * dP_axis_dA_matrix;

		// dP_axis/dB = t*I + outer_product(V, grad_t_dB)
		Basis dP_axis_dB_matrix = (id_matrix * t) + outer_product(V, grad_t_dB);
		derivatives.dC_dB = factor_matrix * dP_axis_dB_matrix;
	}
	return derivatives;
}

// Static helper: Jacobian of vector normalization: d( vec.normalized() ) / d(vec)
Basis LBFGSBCapsuleFitterSolver::d_vec_normalized_d_vec(const Vector3 &p_vec) {
	double len = p_vec.length();
	if (len < 1e-9) { // Avoid division by zero; derivative is undefined/infinite
		return Basis(); // Return zero matrix or handle as error
	}
	double len_inv = 1.0 / len;
	double len_cub_inv = len_inv * len_inv * len_inv;

	// (I / len) - (outer_product(vec, vec) / len^3)
	Basis term1 = Basis() * len_inv; // Identity matrix scaled by 1/len
	Basis term2 = outer_product(p_vec, p_vec) * (-len_cub_inv);
	return term1 + term2;
}

// Static helper: Outer product of two vectors v1 (col) and v2 (row) -> v1 * v2^T
Basis LBFGSBCapsuleFitterSolver::outer_product(const Vector3 &p_v1, const Vector3 &p_v2) {
	// Resulting basis has columns: (p_v1 * p_v2.x), (p_v1 * p_v2.y), (p_v1 * p_v2.z)
	return Basis(
			p_v1 * p_v2.x,
			p_v1 * p_v2.y,
			p_v1 * p_v2.z);
}

Basis LBFGSBCapsuleFitterSolver::_compute_rotation_matrix_from_rot_vec(const Vector3 &p_rot_vec) {
	real_t angle = p_rot_vec.length();
	if (angle < CMP_EPSILON) {
		return Basis(); // Identity
	}
	Vector3 axis = p_rot_vec / angle;
	return Basis(axis, angle);
}

Array LBFGSBCapsuleFitterSolver::_generate_canonical_capsule_mesh_arrays(const Vector3 &p_cap_a, const Vector3 &p_cap_b, double p_cap_radius, int p_radial_segments, int p_rings, bool p_closed) const {
	PackedVector3Array vertices;
	PackedVector3Array normals;
	PackedInt32Array indices;

	if (p_cap_radius <= 1e-6) {
		return Array(); // Return empty if radius is negligible
	}

	int radial_segments = MAX(3, p_radial_segments);
	int rings = MAX(1, p_rings); // Rings for the cylinder part. Hemispheres will adapt.
	int hemisphere_rings = MAX(1, radial_segments / 2); // Rings for each hemisphere cap.

	Vector3 axis_vec = p_cap_b - p_cap_a;
	real_t axis_len = axis_vec.length();

	Transform3D basis_transform; // To orient the capsule

	if (axis_len < 1e-6) { // Degenerate case: treat as a sphere centered at p_cap_a
		basis_transform.origin = p_cap_a;
		// Standard sphere generation (like a UV sphere)
		for (int i = 0; i <= hemisphere_rings * 2; ++i) { // Double rings for full sphere
			real_t v_angle = Math::PI * (real_t)i / (hemisphere_rings * 2);
			real_t y = p_cap_radius * Math::cos(v_angle);
			real_t current_ring_radius = p_cap_radius * Math::sin(v_angle);

			for (int j = 0; j <= radial_segments; ++j) {
				real_t u_angle = Math::TAU * (real_t)j / radial_segments;
				real_t x = current_ring_radius * Math::cos(u_angle);
				real_t z = current_ring_radius * Math::sin(u_angle);
				Vector3 point_local = Vector3(x, y, z);
				vertices.push_back(basis_transform.xform(point_local));
				normals.push_back(basis_transform.basis.xform(point_local.normalized()));
			}
		}

		for (int i = 0; i < hemisphere_rings * 2; ++i) {
			for (int j = 0; j < radial_segments; ++j) {
				int first = (i * (radial_segments + 1)) + j;
				int second = first + radial_segments + 1;

				indices.push_back(first);
				indices.push_back(second + 1);
				indices.push_back(second);

				indices.push_back(first);
				indices.push_back(first + 1);
				indices.push_back(second + 1);
			}
		}
	} else { // Proper capsule case
		Vector3 w_axis = axis_vec.normalized();
		Vector3 u_axis = w_axis.get_any_perpendicular().normalized();
		Vector3 v_axis = w_axis.cross(u_axis);
		basis_transform.basis.set_columns(u_axis, v_axis, w_axis); // u,v,w are x,y,z in local space

		// Hemisphere 1 (at p_cap_a, pointing "down" along -w_axis)
		int v_offset = 0;
		for (int i = 0; i <= hemisphere_rings; ++i) {
			real_t v_angle = (Math::PI / 2.0) * (real_t)i / hemisphere_rings; // 0 to PI/2
			real_t height_offset = -p_cap_radius * Math::sin(v_angle); // Along -w_axis
			real_t current_ring_radius = p_cap_radius * Math::cos(v_angle);

			for (int j = 0; j <= radial_segments; ++j) {
				real_t u_angle = Math::TAU * (real_t)j / radial_segments;
				real_t x_local = current_ring_radius * Math::cos(u_angle);
				real_t y_local = current_ring_radius * Math::sin(u_angle);
				Vector3 point_local = Vector3(x_local, y_local, height_offset); // Relative to cap_a center
				vertices.push_back(p_cap_a + basis_transform.basis.xform(point_local));
				Vector3 normal_local = Vector3(x_local, y_local, height_offset).normalized();
				normals.push_back(basis_transform.basis.xform(normal_local));
			}
		}
		for (int i = 0; i < hemisphere_rings; ++i) {
			for (int j = 0; j < radial_segments; ++j) {
				int first = v_offset + (i * (radial_segments + 1)) + j;
				int second = first + radial_segments + 1;
				indices.push_back(first);
				indices.push_back(second + 1);
				indices.push_back(second);

				indices.push_back(first);
				indices.push_back(first + 1);
				indices.push_back(second + 1);
			}
		}
		v_offset = vertices.size();

		// Cylinder Body
		if (axis_len > 0 && rings > 0) {
			for (int i = 0; i <= rings; ++i) { // From cap_a connection to cap_b connection
				real_t t = (real_t)i / rings;
				Vector3 current_axis_point = p_cap_a + t * axis_vec; // Point on the central axis for this ring
				for (int j = 0; j <= radial_segments; ++j) {
					real_t u_angle = Math::TAU * (real_t)j / radial_segments;
					real_t x_local = p_cap_radius * Math::cos(u_angle);
					real_t y_local = p_cap_radius * Math::sin(u_angle);
					Vector3 pt_on_circle = x_local * u_axis + y_local * v_axis;
					vertices.push_back(current_axis_point + pt_on_circle);
					Vector3 normal_local = (x_local * u_axis + y_local * v_axis).normalized();
					normals.push_back(normal_local); // Normals are radial from the axis
				}
			}

			for (int i = 0; i < rings; ++i) {
				for (int j = 0; j < radial_segments; ++j) {
					int first = v_offset + (i * (radial_segments + 1)) + j;
					int second = first + radial_segments + 1;
					indices.push_back(first);
					indices.push_back(second);
					indices.push_back(first + 1);

					indices.push_back(second);
					indices.push_back(second + 1);
					indices.push_back(first + 1);
				}
			}
			v_offset = vertices.size();
		}

		// Hemisphere 2 (at p_cap_b, pointing "up" along +w_axis)
		for (int i = 0; i <= hemisphere_rings; ++i) {
			real_t v_angle = (Math::PI / 2.0) * (1.0 - (real_t)i / hemisphere_rings); // PI/2 down to 0
			real_t height_offset = p_cap_radius * Math::sin(v_angle); // Along +w_axis
			real_t current_ring_radius = p_cap_radius * Math::cos(v_angle);

			for (int j = 0; j <= radial_segments; ++j) {
				real_t u_angle = Math::TAU * (real_t)j / radial_segments;
				real_t x_local = current_ring_radius * Math::cos(u_angle);
				real_t y_local = current_ring_radius * Math::sin(u_angle);
				Vector3 point_local = Vector3(x_local, y_local, height_offset); // Relative to cap_b center
				vertices.push_back(p_cap_b + basis_transform.basis.xform(point_local));
				Vector3 normal_local = Vector3(x_local, y_local, height_offset).normalized();
				normals.push_back(basis_transform.basis.xform(normal_local));
			}
		}
		for (int i = 0; i < hemisphere_rings; ++i) {
			for (int j = 0; j < radial_segments; ++j) {
				int first = v_offset + (i * (radial_segments + 1)) + j;
				int second = first + radial_segments + 1;
				indices.push_back(first);
				indices.push_back(second);
				indices.push_back(first + 1);

				indices.push_back(second);
				indices.push_back(second + 1);
				indices.push_back(first + 1);
			}
		}
	}

	Array mesh_arrays;
	mesh_arrays.resize(3);
	mesh_arrays[0] = vertices;
	mesh_arrays[1] = normals;
	mesh_arrays[2] = indices;
	return mesh_arrays;
}
