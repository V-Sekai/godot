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

// --- LBFGSBCapsuleFitterSolverBase Implementation ---

void LBFGSBCapsuleFitterSolverBase::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source_mesh", "p_mesh"), &LBFGSBCapsuleFitterSolverBase::set_source_mesh);
	ClassDB::bind_method(D_METHOD("get_source_mesh"), &LBFGSBCapsuleFitterSolverBase::get_source_mesh);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "source_mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_source_mesh", "get_source_mesh");

	ClassDB::bind_method(D_METHOD("set_surface_index", "p_index"), &LBFGSBCapsuleFitterSolverBase::set_surface_index);
	ClassDB::bind_method(D_METHOD("get_surface_index"), &LBFGSBCapsuleFitterSolverBase::get_surface_index);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "surface_index"), "set_surface_index", "get_surface_index");

	ClassDB::bind_method(D_METHOD("set_orientation_distance_threshold", "p_threshold"), &LBFGSBCapsuleFitterSolverBase::set_orientation_distance_threshold);
	ClassDB::bind_method(D_METHOD("get_orientation_distance_threshold"), &LBFGSBCapsuleFitterSolverBase::get_orientation_distance_threshold);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "orientation_distance_threshold"), "set_orientation_distance_threshold", "get_orientation_distance_threshold");

	ClassDB::bind_method(D_METHOD("set_orientation_angle_threshold_rad", "p_threshold_rad"), &LBFGSBCapsuleFitterSolverBase::set_orientation_angle_threshold_rad);
	ClassDB::bind_method(D_METHOD("get_orientation_angle_threshold_rad"), &LBFGSBCapsuleFitterSolverBase::get_orientation_angle_threshold_rad);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "orientation_angle_threshold_rad"), "set_orientation_angle_threshold_rad", "get_orientation_angle_threshold_rad");

	ClassDB::bind_method(D_METHOD("set_orientation_penalty_factor", "p_factor"), &LBFGSBCapsuleFitterSolverBase::set_orientation_penalty_factor);
	ClassDB::bind_method(D_METHOD("get_orientation_penalty_factor"), &LBFGSBCapsuleFitterSolverBase::get_orientation_penalty_factor);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "orientation_penalty_factor"), "set_orientation_penalty_factor", "get_orientation_penalty_factor");

	ClassDB::bind_method(D_METHOD("set_huber_delta", "p_delta"), &LBFGSBCapsuleFitterSolverBase::set_huber_delta);
	ClassDB::bind_method(D_METHOD("get_huber_delta"), &LBFGSBCapsuleFitterSolverBase::get_huber_delta);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "huber_delta"), "set_huber_delta", "get_huber_delta");

	ClassDB::bind_method(D_METHOD("get_last_fit_result"), &LBFGSBCapsuleFitterSolverBase::get_last_fit_result);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "last_fit_result", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY), "", "get_last_fit_result");

	// Note: optimize_capsule_radius and optimize_capsule_axes are bound in their respective derived classes.

	ClassDB::bind_method(D_METHOD("add_capsule_instance", "initial_axis_a", "initial_axis_b", "initial_radius"), &LBFGSBCapsuleFitterSolverBase::add_capsule_instance);
	ClassDB::bind_method(D_METHOD("clear_capsule_instances"), &LBFGSBCapsuleFitterSolverBase::clear_capsule_instances);
	ClassDB::bind_method(D_METHOD("get_num_capsule_instances"), &LBFGSBCapsuleFitterSolverBase::get_num_capsule_instances);
	ClassDB::bind_method(D_METHOD("get_capsule_instance_data", "p_idx"), &LBFGSBCapsuleFitterSolverBase::get_capsule_instance_data);
}

LBFGSBCapsuleFitterSolverBase::LBFGSBCapsuleFitterSolverBase() {
	// Default constructor
}

LBFGSBCapsuleFitterSolverBase::~LBFGSBCapsuleFitterSolverBase() {
	// Destructor
}

void LBFGSBCapsuleFitterSolverBase::set_source_mesh(const Ref<Mesh> &p_mesh) {
	last_fit_result.clear();
	capsules.clear(); // Clear capsules when a new mesh is set.
	source_mesh = p_mesh; // Store the original user-provided mesh.
	current_cloud_points_for_objective.clear(); // Clear derived data.
	current_cloud_normals_for_objective.clear(); // Clear derived data.
}

Ref<Mesh> LBFGSBCapsuleFitterSolverBase::get_source_mesh() const {
	return source_mesh;
}

void LBFGSBCapsuleFitterSolverBase::set_surface_index(int p_index) {
	surface_index = p_index;
}

int LBFGSBCapsuleFitterSolverBase::get_surface_index() const {
	return surface_index;
}

void LBFGSBCapsuleFitterSolverBase::set_orientation_distance_threshold(double p_threshold) {
	orientation_distance_threshold = p_threshold;
}
double LBFGSBCapsuleFitterSolverBase::get_orientation_distance_threshold() const {
	return orientation_distance_threshold;
}

void LBFGSBCapsuleFitterSolverBase::set_orientation_angle_threshold_rad(double p_threshold_rad) {
	orientation_angle_threshold_rad = p_threshold_rad;
}
double LBFGSBCapsuleFitterSolverBase::get_orientation_angle_threshold_rad() const {
	return orientation_angle_threshold_rad;
}

void LBFGSBCapsuleFitterSolverBase::set_orientation_penalty_factor(double p_factor) {
	orientation_penalty_factor = p_factor;
}
double LBFGSBCapsuleFitterSolverBase::get_orientation_penalty_factor() const {
	return orientation_penalty_factor;
}

void LBFGSBCapsuleFitterSolverBase::set_huber_delta(double p_delta) {
	huber_delta = p_delta;
}
double LBFGSBCapsuleFitterSolverBase::get_huber_delta() const {
	return huber_delta;
}

Dictionary LBFGSBCapsuleFitterSolverBase::get_last_fit_result() const {
	return last_fit_result;
}

void LBFGSBCapsuleFitterSolverBase::add_capsule_instance(const Vector3 &p_initial_axis_a, const Vector3 &p_initial_axis_b, double p_initial_radius) {
	CapsuleInstance instance(p_initial_axis_a, p_initial_axis_b, p_initial_radius);
	capsules.push_back(instance);
}

void LBFGSBCapsuleFitterSolverBase::clear_capsule_instances() {
	capsules.clear();
}

int LBFGSBCapsuleFitterSolverBase::get_num_capsule_instances() const {
	return capsules.size();
}

Dictionary LBFGSBCapsuleFitterSolverBase::get_capsule_instance_data(int p_idx) const {
	Dictionary data;
	ERR_FAIL_INDEX_V_MSG(p_idx, capsules.size(), data, "Capsule index out of bounds.");
	const CapsuleInstance &instance = capsules[p_idx];
	data["initial_axis_a"] = instance.initial_axis_a;
	data["initial_axis_b"] = instance.initial_axis_b;
	data["initial_radius"] = instance.initial_radius;
	data["optimized_axis_a"] = instance.optimized_axis_a;
	data["optimized_axis_b"] = instance.optimized_axis_b;
	data["optimized_radius"] = instance.optimized_radius;
	return data;
}

// New helper method: Initialize parameters for the current optimization mode and capsule
void LBFGSBCapsuleFitterSolverBase::_initialize_parameters_for_current_mode(
		PackedFloat64Array &r_x_initial,
		PackedFloat64Array &r_lower_bounds,
		PackedFloat64Array &r_upper_bounds) const {
	ERR_FAIL_INDEX_MSG(_current_capsule_idx_for_opt, capsules.size(), "Current capsule index out of bounds.");
	const CapsuleInstance &current_capsule = capsules[_current_capsule_idx_for_opt];
	AABB mesh_aabb;
	if (source_mesh.is_valid() && source_mesh->get_surface_count() > surface_index && surface_index >= 0) {
		mesh_aabb = source_mesh->get_aabb();
	} else {
		// Default AABB if mesh is not available or surface_index is invalid.
		// This situation should ideally be caught by _validate_pre_optimization_conditions.
		WARN_PRINT_ONCE("Mesh or surface not fully available for bounds calculation; using default AABB.");
		mesh_aabb = AABB(Vector3(-10, -10, -10), Vector3(20, 20, 20));
	}

	// Define a small epsilon for bounds to avoid issues with parameters being exactly on bounds.
	const double bound_epsilon = 1e-6;
	// Define a practical maximum for radius and coordinate values.
	const double practical_max_coord = 1000.0; // Adjust as needed.
	const double practical_min_coord = -1000.0; // Adjust as needed.

	switch (_current_optimization_mode) {
		case OPT_MODE_RADIUS:
			r_x_initial.resize(1);
			r_x_initial.write[0] = current_capsule.optimized_radius;
			r_lower_bounds.resize(1);
			r_lower_bounds.write[0] = 0.01; // Minimum radius
			r_upper_bounds.resize(1);
			r_upper_bounds.write[0] = MIN(MAX(mesh_aabb.get_longest_axis_size(), 2.0) * 2.0, practical_max_coord);
			break;
		case OPT_MODE_AXIS_A:
			r_x_initial.resize(3);
			r_x_initial.write[0] = current_capsule.optimized_axis_a.x;
			r_x_initial.write[1] = current_capsule.optimized_axis_a.y;
			r_x_initial.write[2] = current_capsule.optimized_axis_a.z;
			r_lower_bounds.resize(3);
			r_upper_bounds.resize(3);
			for (int i = 0; i < 3; ++i) {
				r_lower_bounds.write[i] = MIN(mesh_aabb.position[i] - mesh_aabb.size[i] * 0.5, practical_min_coord + bound_epsilon);
				r_upper_bounds.write[i] = MAX(mesh_aabb.position[i] + mesh_aabb.size[i] * 1.5, practical_max_coord - bound_epsilon);
			}
			break;
		case OPT_MODE_AXIS_B:
			r_x_initial.resize(3);
			r_x_initial.write[0] = current_capsule.optimized_axis_b.x;
			r_x_initial.write[1] = current_capsule.optimized_axis_b.y;
			r_x_initial.write[2] = current_capsule.optimized_axis_b.z;
			r_lower_bounds.resize(3);
			r_upper_bounds.resize(3);
			for (int i = 0; i < 3; ++i) {
				r_lower_bounds.write[i] = MIN(mesh_aabb.position[i] - mesh_aabb.size[i] * 0.5, practical_min_coord + bound_epsilon);
				r_upper_bounds.write[i] = MAX(mesh_aabb.position[i] + mesh_aabb.size[i] * 1.5, practical_max_coord - bound_epsilon);
			}
			break;
	}
}

// New helper method: Update the current capsule's optimized parameters from the solver's output
void LBFGSBCapsuleFitterSolverBase::_update_capsule_from_optimized_params(
		const PackedFloat64Array &p_optimized_params) {
	ERR_FAIL_INDEX_MSG(_current_capsule_idx_for_opt, capsules.size(), "Current capsule index out of bounds for update.");
	CapsuleInstance &current_capsule = capsules.write[_current_capsule_idx_for_opt];

	switch (_current_optimization_mode) {
		case OPT_MODE_RADIUS:
			ERR_FAIL_COND_MSG(p_optimized_params.size() != 1, "Optimized params size mismatch for radius update.");
			current_capsule.optimized_radius = MAX(0.01, p_optimized_params[0]); // Ensure radius stays positive
			break;
		case OPT_MODE_AXIS_A:
			ERR_FAIL_COND_MSG(p_optimized_params.size() != 3, "Optimized params size mismatch for axis A update.");
			current_capsule.optimized_axis_a = Vector3(p_optimized_params[0], p_optimized_params[1], p_optimized_params[2]);
			break;
		case OPT_MODE_AXIS_B:
			ERR_FAIL_COND_MSG(p_optimized_params.size() != 3, "Optimized params size mismatch for axis B update.");
			current_capsule.optimized_axis_b = Vector3(p_optimized_params[0], p_optimized_params[1], p_optimized_params[2]);
			break;
	}
}

// Helper method to generate the result mesh with capsules
Ref<ArrayMesh> LBFGSBCapsuleFitterSolverBase::_generate_result_mesh_with_capsules() const {
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
	for (int i = 0; i < capsules.size(); ++i) {
		const CapsuleInstance &capsule = capsules[i];
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
bool LBFGSBCapsuleFitterSolverBase::_validate_pre_optimization_conditions(Dictionary &r_result_dict) {
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

	if (capsules.is_empty()) {
		r_result_dict["message"] = "No capsule instances defined. Nothing to optimize.";
		r_result_dict["optimized_capsules_results"] = Array();
		r_result_dict["final_fx"] = 0.0;
		r_result_dict["iterations"] = 0;
		return false;
	}
	return true;
}

// Helper method to prepare data for the objective function
bool LBFGSBCapsuleFitterSolverBase::_prepare_objective_data(Dictionary &r_result_dict) {
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

bool LBFGSBCapsuleFitterSolverBase::_execute_lbfgsb_optimization(const PackedFloat64Array &p_initial_x, const PackedFloat64Array &p_lower_bounds, const PackedFloat64Array &p_upper_bounds, Dictionary &r_result_dict) {
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

double LBFGSBCapsuleFitterSolverBase::call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) {
	ERR_FAIL_COND_V_MSG(current_cloud_points_for_objective.is_empty(), 1e18, "Mesh points not prepared for objective function calculation."); // Return large error
	ERR_FAIL_INDEX_V_MSG(_current_capsule_idx_for_opt, capsules.size(), 1e18, "Current capsule index out of bounds for call_operator.");

	// Get the base parameters from the current state of the capsule being optimized.
	// These are the 'fixed' parameters for this specific sub-problem.
	const CapsuleInstance &base_capsule_for_opt = capsules[_current_capsule_idx_for_opt];
	Vector3 temp_axis_a = base_capsule_for_opt.optimized_axis_a;
	Vector3 temp_axis_b = base_capsule_for_opt.optimized_axis_b;
	double temp_radius = base_capsule_for_opt.optimized_radius;

	double *r_grad_ptr = r_grad.ptrw(); // Get writable pointer

	// Override the specific parameter(s) being optimized in this call from p_x.
	switch (_current_optimization_mode) {
		case OPT_MODE_RADIUS:
			ERR_FAIL_COND_V_MSG(p_x.size() != 1, 1e18, "Parameter vector p_x size mismatch for OPT_MODE_RADIUS.");
			temp_radius = p_x[0];
			r_grad.resize(1);
			r_grad_ptr = r_grad.ptrw(); // Re-acquire pointer after resize
			r_grad_ptr[0] = 0.0;
			break;
		case OPT_MODE_AXIS_A:
			ERR_FAIL_COND_V_MSG(p_x.size() != 3, 1e18, "Parameter vector p_x size mismatch for OPT_MODE_AXIS_A.");
			temp_axis_a = Vector3(p_x[0], p_x[1], p_x[2]);
			r_grad.resize(3);
			r_grad_ptr = r_grad.ptrw(); // Re-acquire pointer after resize
			for (int i = 0; i < 3; ++i) {
				r_grad_ptr[i] = 0.0;
			}
			break;
		case OPT_MODE_AXIS_B:
			ERR_FAIL_COND_V_MSG(p_x.size() != 3, 1e18, "Parameter vector p_x size mismatch for OPT_MODE_AXIS_B.");
			temp_axis_b = Vector3(p_x[0], p_x[1], p_x[2]);
			r_grad.resize(3);
			r_grad_ptr = r_grad.ptrw(); // Re-acquire pointer after resize
			for (int i = 0; i < 3; ++i) {
				r_grad_ptr[i] = 0.0;
			}
			break;
		default:
			ERR_FAIL_V_MSG(1e18, "Unknown optimization mode in call_operator.");
	}

	// Critical check: Ensure radius is positive to avoid NaNs and errors in geometric calculations.
	if (temp_radius <= 1e-3) { // Use a small epsilon for safety.
		// Penalize non-positive or very small radius heavily.
		// The gradient should strongly push the radius to become larger.
		if (_current_optimization_mode == OPT_MODE_RADIUS) {
			// If optimizing radius, make its gradient strongly negative (to increase radius).
			r_grad_ptr[0] = -1e6 * (1e-3 - temp_radius); // Gradient proportional to how much it's below threshold.
		}
		// For other modes, if radius is bad, the whole configuration is bad.
		return 1e12 + (1e-3 - temp_radius) * 1e9; // Return a very large error value.
	}
	// Prevent capsule inversion or zero height
	if ((temp_axis_a - temp_axis_b).length_squared() < 1e-6 && (temp_radius < 1e-2)) { // If it's basically a tiny sphere at a point
		// This configuration is degenerate. Penalize.
		double penalty = 1e10;
		if (_current_optimization_mode == OPT_MODE_AXIS_A) {
			// Push A away from B. If A = p_x, then d_penalty / d_ax = some_vector_pointing_away_from_B
			Vector3 dir = (temp_axis_a - temp_axis_b).normalized();
			if (dir.is_zero_approx()) {
				dir = Vector3(1, 0, 0); // arbitrary direction if coincident
			}
			r_grad_ptr[0] += dir.x * 1e4;
			r_grad_ptr[1] += dir.y * 1e4;
			r_grad_ptr[2] += dir.z * 1e4;
		} else if (_current_optimization_mode == OPT_MODE_AXIS_B) {
			Vector3 dir = (temp_axis_b - temp_axis_a).normalized();
			if (dir.is_zero_approx()) {
				dir = Vector3(1, 0, 0);
			}
			r_grad_ptr[0] += dir.x * 1e4;
			r_grad_ptr[1] += dir.y * 1e4;
			r_grad_ptr[2] += dir.z * 1e4;
		} else if (_current_optimization_mode == OPT_MODE_RADIUS) {
			r_grad_ptr[0] -= 1e4; // Penalize small radius in this state
		}
		return penalty;
	}

	double total_objective_value = 0.0;
	// Accumulators for gradients, specific to the parameter(s) being optimized in this call.
	double grad_radius_accumulator = 0.0;
	Vector3 grad_axis_a_accumulator;
	Vector3 grad_axis_b_accumulator;

	// Iterate over all points in the source mesh surface.
	for (int i = 0; i < current_cloud_points_for_objective.size(); ++i) {
		Vector3 mesh_vertex = current_cloud_points_for_objective[i];
		// Vector3 mesh_normal = current_cloud_normals_for_objective[i]; // Available if needed for orientation penalty.

		// Calculate signed distance and its derivatives w.r.t. all 7 params of the *current* capsule.
		CapsuleSurfacePointDerivatives derivatives = get_capsule_surface_derivatives(
				mesh_vertex, temp_axis_a, temp_axis_b, temp_radius);

		double signed_dist = derivatives.signed_distance;
		double loss_value_for_point;
		double d_loss_d_signed_dist; // Derivative of the loss w.r.t. signed_distance.

		// Huber loss for robustness against outliers.
		if (Math::abs(signed_dist) <= huber_delta) {
			loss_value_for_point = 0.5 * signed_dist * signed_dist;
			d_loss_d_signed_dist = signed_dist;
		} else {
			loss_value_for_point = huber_delta * (Math::abs(signed_dist) - 0.5 * huber_delta);
			d_loss_d_signed_dist = huber_delta * (signed_dist > 0 ? 1.0 : -1.0); // Sign of signed_dist.
		}
		total_objective_value += loss_value_for_point;

		// Accumulate gradients using the chain rule:
		// d_loss_d_param = (d_loss_d_signed_dist) * (d_signed_dist_d_param)
		grad_radius_accumulator += d_loss_d_signed_dist * derivatives.d_sd_d_radius;
		grad_axis_a_accumulator += d_loss_d_signed_dist * derivatives.d_sd_d_axis_a;
		grad_axis_b_accumulator += d_loss_d_signed_dist * derivatives.d_sd_d_axis_b;

		// --- Optional: Orientation Penalty (if configured and relevant) ---
		// This was part of the original thought but needs careful integration with sequential optimization.
		// If get_capsule_surface_derivatives also computes orientation_penalty and its derivatives:
		// total_objective_value += derivatives.orientation_penalty * orientation_penalty_factor;
		// grad_radius_accumulator += derivatives.d_orientation_penalty_d_radius * orientation_penalty_factor;
		// grad_axis_a_accumulator += derivatives.d_orientation_penalty_d_axis_a * orientation_penalty_factor;
		// grad_axis_b_accumulator += derivatives.d_orientation_penalty_d_axis_b * orientation_penalty_factor;
	}

	// Assign the accumulated gradients to r_grad based on the current optimization mode.
	switch (_current_optimization_mode) {
		case OPT_MODE_RADIUS:
			r_grad_ptr[0] = grad_radius_accumulator;
			break;
		case OPT_MODE_AXIS_A:
			r_grad_ptr[0] = grad_axis_a_accumulator.x;
			r_grad_ptr[1] = grad_axis_a_accumulator.y;
			r_grad_ptr[2] = grad_axis_a_accumulator.z;
			break;
		case OPT_MODE_AXIS_B:
			r_grad_ptr[0] = grad_axis_b_accumulator.x;
			r_grad_ptr[1] = grad_axis_b_accumulator.y;
			r_grad_ptr[2] = grad_axis_b_accumulator.z;
			break;
	}
	return total_objective_value;
}

// Implementation for _get_closest_point_and_normal_on_capsule_surface
std::pair<Vector3, Vector3> LBFGSBCapsuleFitterSolverBase::_get_closest_point_and_normal_on_capsule_surface(
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

LBFGSBCapsuleFitterSolverBase::CapsuleSurfacePointDerivatives LBFGSBCapsuleFitterSolverBase::get_capsule_surface_derivatives(
		const Vector3 &p_mesh_vertex,
		const Vector3 &p_cap_a,
		const Vector3 &p_cap_b,
		double p_cap_radius) {
	CapsuleSurfacePointDerivatives derivatives;
	derivatives.is_valid = true;
	derivatives.d_sd_d_radius = -1.0; // sd = dist - R, so d(sd)/dR = -1

	// Initialize dC_dA and dC_dB to zero as they are not currently used by the objective function.
	// If an orientation penalty or other terms were to use them, their calculation would be needed here.
	derivatives.dC_dA.set_zero();
	derivatives.dC_dB.set_zero();

	Vector3 V = p_cap_b - p_cap_a; // Axis vector from A to B
	double len_sq_V = V.length_squared();
	const double EPSILON_SQ = 1e-9; // For squared length checks (e.g., 1e-4 or 1e-5 actual length)
	const double EPSILON_DIST = 1e-7; // For distance checks (e.g., point on axis, point at cap center)

	// Case 1: Degenerate capsule (essentially a sphere centered at p_cap_a)
	if (len_sq_V < EPSILON_SQ) {
		Vector3 U_pa = p_mesh_vertex - p_cap_a; // Vector from p_cap_a to p_mesh_vertex
		double dist_U_pa = U_pa.length();

		if (dist_U_pa < EPSILON_DIST) { // p_mesh_vertex is at the center of the sphere
			derivatives.is_valid = false;
			derivatives.signed_distance = -p_cap_radius; // Point is at the center, deep inside
			derivatives.normal_on_surface = Vector3(0, 1, 0); // Arbitrary valid normal
			derivatives.d_sd_d_axis_a.zero();
			derivatives.d_sd_d_axis_b.zero();
			return derivatives;
		}
		derivatives.normal_on_surface = U_pa / dist_U_pa; // Normal points from center to mesh_vertex
		derivatives.signed_distance = dist_U_pa - p_cap_radius;
		derivatives.d_sd_d_axis_a = -derivatives.normal_on_surface; // Moving A in direction of normal decreases distance
		derivatives.d_sd_d_axis_b.zero(); // B is effectively coincident with A
		return derivatives;
	}

	// Parameter t_param for projection of (p_mesh_vertex - p_cap_a) onto V
	Vector3 U_va = p_mesh_vertex - p_cap_a; // Vector from p_cap_a to p_mesh_vertex
	double t_param = U_va.dot(V) / len_sq_V;

	Vector3 P_axis; // Closest point on the infinite line defined by A and B to p_mesh_vertex

	// Case 2: Closest point is on one of the spherical end caps
	if (t_param <= 0.0) { // Closest to p_cap_a (spherical cap at A)
		P_axis = p_cap_a;
		Vector3 U_cap_center_P = p_mesh_vertex - P_axis; // Vector from cap center (A) to p_mesh_vertex
		double dist_U_cap_center_P = U_cap_center_P.length();

		if (dist_U_cap_center_P < EPSILON_DIST) { // p_mesh_vertex is at the center of cap A
			derivatives.is_valid = false;
			derivatives.signed_distance = -p_cap_radius;
			// Normal along axis, pointing away from B if possible, else arbitrary
			derivatives.normal_on_surface = (p_cap_a - p_cap_b).normalized();
			if (derivatives.normal_on_surface.is_zero_approx()) {
				derivatives.normal_on_surface = Vector3(0, 1, 0); // Fallback
			}
			derivatives.d_sd_d_axis_a.zero();
			derivatives.d_sd_d_axis_b.zero();
			return derivatives;
		}
		derivatives.normal_on_surface = U_cap_center_P / dist_U_cap_center_P;
		derivatives.signed_distance = dist_U_cap_center_P - p_cap_radius;
		derivatives.d_sd_d_axis_a = -derivatives.normal_on_surface;
		derivatives.d_sd_d_axis_b.zero();

	} else if (t_param >= 1.0) { // Closest to p_cap_b (spherical cap at B)
		P_axis = p_cap_b;
		Vector3 U_cap_center_P = p_mesh_vertex - P_axis; // Vector from cap center (B) to p_mesh_vertex
		double dist_U_cap_center_P = U_cap_center_P.length();

		if (dist_U_cap_center_P < EPSILON_DIST) { // p_mesh_vertex is at the center of cap B
			derivatives.is_valid = false;
			derivatives.signed_distance = -p_cap_radius;
			// Normal along axis, pointing away from A if possible, else arbitrary
			derivatives.normal_on_surface = (p_cap_b - p_cap_a).normalized();
			if (derivatives.normal_on_surface.is_zero_approx()) {
				derivatives.normal_on_surface = Vector3(0, 1, 0); // Fallback
			}
			derivatives.d_sd_d_axis_a.zero();
			derivatives.d_sd_d_axis_b.zero();
			return derivatives;
		}
		derivatives.normal_on_surface = U_cap_center_P / dist_U_cap_center_P;
		derivatives.signed_distance = dist_U_cap_center_P - p_cap_radius;
		derivatives.d_sd_d_axis_a.zero();
		derivatives.d_sd_d_axis_b = -derivatives.normal_on_surface;
	} else {
		// Case 3: Closest point is on the cylindrical wall
		P_axis = p_cap_a + t_param * V;
		Vector3 N_cyl_wall = p_mesh_vertex - P_axis; // Vector from P_axis to p_mesh_vertex (normal to axis)
		double dist_N_cyl_wall = N_cyl_wall.length();

		if (dist_N_cyl_wall < EPSILON_DIST) { // p_mesh_vertex is on the capsule axis segment
			derivatives.is_valid = false;
			derivatives.signed_distance = -p_cap_radius; // Point is on the axis, deep inside
			// Normal is ill-defined, pick one perpendicular to V
			derivatives.normal_on_surface = V.get_any_perpendicular().normalized();
			if (derivatives.normal_on_surface.is_zero_approx()) { // Should not happen if V is not zero
				derivatives.normal_on_surface = Vector3(0, 1, 0); // Absolute fallback
			}
			derivatives.d_sd_d_axis_a.zero();
			derivatives.d_sd_d_axis_b.zero();
			return derivatives;
		}
		derivatives.normal_on_surface = N_cyl_wall / dist_N_cyl_wall;
		derivatives.signed_distance = dist_N_cyl_wall - p_cap_radius;

		// Derivatives for cylinder wall:
		// P_axis = A + t_param * (B-A)
		// dt/dA_vec = (-(U_va+V)*len_sq_V + 2*dot(U_va,V)*V) / (len_sq_V^2)
		// dt/dB_vec = (  U_va   *len_sq_V - 2*dot(U_va,V)*V) / (len_sq_V^2)
		// where U_va = p_mesh_vertex - p_cap_a
		// V = p_cap_b - p_cap_a

		// Avoid division by zero if len_sq_V is extremely small (though handled by degenerate case)
		double inv_len_sq_V_sq = (len_sq_V > EPSILON_SQ) ? (1.0 / (len_sq_V * len_sq_V)) : 0.0;

		Vector3 grad_t_dA_vec = (-(U_va + V) * len_sq_V + 2.0 * U_va.dot(V) * V) * inv_len_sq_V_sq;
		Vector3 grad_t_dB_vec = (U_va * len_sq_V - 2.0 * U_va.dot(V) * V) * inv_len_sq_V_sq;

		// d(sd)/dA = -(1-t_param)*normal - dot(normal,V)*grad_t_dA_vec
		// d(sd)/dB = -t_param*normal    - dot(normal,V)*grad_t_dB_vec
		derivatives.d_sd_d_axis_a = -(1.0 - t_param) * derivatives.normal_on_surface - (derivatives.normal_on_surface.dot(V)) * grad_t_dA_vec;
		derivatives.d_sd_d_axis_b = -t_param * derivatives.normal_on_surface - (derivatives.normal_on_surface.dot(V)) * grad_t_dB_vec;
	}
	return derivatives;
}

// Static helper: Jacobian of vector normalization: d( vec.normalized() ) / d(vec)
Basis LBFGSBCapsuleFitterSolverBase::d_vec_normalized_d_vec(const Vector3 &p_vec) {
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
Basis LBFGSBCapsuleFitterSolverBase::outer_product(const Vector3 &p_v1, const Vector3 &p_v2) {
	// Resulting basis has columns: (p_v1 * p_v2.x), (p_v1 * p_v2.y), (p_v1 * p_v2.z)
	return Basis(
			p_v1 * p_v2.x,
			p_v1 * p_v2.y,
			p_v1 * p_v2.z);
}

Basis LBFGSBCapsuleFitterSolverBase::_compute_rotation_matrix_from_rot_vec(const Vector3 &p_rot_vec) {
	real_t angle = p_rot_vec.length();
	if (angle < CMP_EPSILON) {
		return Basis(); // Identity
	}
	Vector3 axis = p_rot_vec / angle;
	return Basis(axis, angle);
}

Array LBFGSBCapsuleFitterSolverBase::_generate_canonical_capsule_mesh_arrays(const Vector3 &p_cap_a, const Vector3 &p_cap_b, double p_cap_radius, int p_radial_segments, int p_rings, bool p_closed) const {
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

// --- LBFGSBCapsuleRadiusSolver Implementation ---

void LBFGSBCapsuleRadiusSolver::_bind_methods() {
	ClassDB::bind_method(D_METHOD("optimize_radius", "capsule_index"), &LBFGSBCapsuleRadiusSolver::optimize_radius);
}

LBFGSBCapsuleRadiusSolver::LBFGSBCapsuleRadiusSolver() {}
LBFGSBCapsuleRadiusSolver::~LBFGSBCapsuleRadiusSolver() {}

Dictionary LBFGSBCapsuleRadiusSolver::optimize_radius(int p_capsule_idx) {
	Dictionary result_dict;
	_current_capsule_idx_for_opt = p_capsule_idx;
	_current_optimization_mode = OPT_MODE_RADIUS;

	if (!_validate_pre_optimization_conditions(result_dict)) {
		last_fit_result = result_dict;
		return result_dict;
	}
	ERR_FAIL_INDEX_V_MSG(p_capsule_idx, capsules.size(), result_dict, "Capsule index out of bounds.");

	if (!_prepare_objective_data(result_dict)) {
		last_fit_result = result_dict;
		return result_dict;
	}

	PackedFloat64Array x_initial_for_mode;
	PackedFloat64Array lower_bounds_for_mode;
	PackedFloat64Array upper_bounds_for_mode;
	_initialize_parameters_for_current_mode(x_initial_for_mode, lower_bounds_for_mode, upper_bounds_for_mode);

	Dictionary optimization_run_result_dict;
	bool success_optimization_run = _execute_lbfgsb_optimization(x_initial_for_mode, lower_bounds_for_mode, upper_bounds_for_mode, optimization_run_result_dict);

	if (success_optimization_run && !optimization_run_result_dict.has("error") && !optimization_run_result_dict.has("solver_error_message")) {
		PackedFloat64Array optimized_params_from_solver = optimization_run_result_dict["optimized_params"];
		_update_capsule_from_optimized_params(optimized_params_from_solver);

		result_dict["message"] = "Radius optimization completed for capsule " + itos(p_capsule_idx) + ".";
		result_dict["iterations"] = optimization_run_result_dict["iterations"];
		result_dict["final_fx"] = optimization_run_result_dict["final_fx"];
		result_dict["optimized_radius"] = capsules[p_capsule_idx].optimized_radius;
		result_dict["optimized_axis_a"] = capsules[p_capsule_idx].optimized_axis_a;
		result_dict["optimized_axis_b"] = capsules[p_capsule_idx].optimized_axis_b;
	} else {
		String error_msg = "Radius optimization failed for capsule " + itos(p_capsule_idx) + ".";
		if (optimization_run_result_dict.has("error")) {
			error_msg += " Solver error: " + String(optimization_run_result_dict["error"]);
		} else if (optimization_run_result_dict.has("solver_error_message")) {
			error_msg += " Solver message: " + String(optimization_run_result_dict["solver_error_message"]);
		}
		result_dict["error"] = error_msg;
		if (optimization_run_result_dict.has("iterations")) {
			result_dict["iterations"] = optimization_run_result_dict["iterations"];
		}
		if (optimization_run_result_dict.has("final_fx")) {
			result_dict["final_fx"] = optimization_run_result_dict["final_fx"];
		}
	}

	last_fit_result = result_dict;
	return result_dict;
}

// --- LBFGSBCapsuleAxisSolver Implementation ---

void LBFGSBCapsuleAxisSolver::_bind_methods() {
	ClassDB::bind_method(D_METHOD("optimize_axes", "capsule_index"), &LBFGSBCapsuleAxisSolver::optimize_axes);
}

LBFGSBCapsuleAxisSolver::LBFGSBCapsuleAxisSolver() {}
LBFGSBCapsuleAxisSolver::~LBFGSBCapsuleAxisSolver() {}

Dictionary LBFGSBCapsuleAxisSolver::optimize_axes(int p_capsule_idx) {
	Dictionary result_dict;
	_current_capsule_idx_for_opt = p_capsule_idx;

	if (!_validate_pre_optimization_conditions(result_dict)) {
		last_fit_result = result_dict;
		return result_dict;
	}
	ERR_FAIL_INDEX_V_MSG(p_capsule_idx, capsules.size(), result_dict, "Capsule index out of bounds.");

	if (!_prepare_objective_data(result_dict)) {
		last_fit_result = result_dict;
		return result_dict;
	}

	int total_iterations = 0;
	double final_fx_axis_b = 0.0;

	// --- Optimize Axis A ---
	_current_optimization_mode = OPT_MODE_AXIS_A;
	PackedFloat64Array x_initial_axis_a;
	PackedFloat64Array lower_bounds_axis_a;
	PackedFloat64Array upper_bounds_axis_a;
	_initialize_parameters_for_current_mode(x_initial_axis_a, lower_bounds_axis_a, upper_bounds_axis_a);

	Dictionary axis_a_opt_result_dict;
	bool success_axis_a = _execute_lbfgsb_optimization(x_initial_axis_a, lower_bounds_axis_a, upper_bounds_axis_a, axis_a_opt_result_dict);

	if (success_axis_a && !axis_a_opt_result_dict.has("error") && !axis_a_opt_result_dict.has("solver_error_message")) {
		PackedFloat64Array optimized_params_axis_a = axis_a_opt_result_dict["optimized_params"];
		_update_capsule_from_optimized_params(optimized_params_axis_a);
		total_iterations += int(axis_a_opt_result_dict.get("iterations", 0));
	} else {
		String error_msg = "Axis A optimization failed for capsule " + itos(p_capsule_idx) + ".";
		if (axis_a_opt_result_dict.has("error")) {
			error_msg += " Solver error: " + String(axis_a_opt_result_dict["error"]);
		} else if (axis_a_opt_result_dict.has("solver_error_message")) {
			error_msg += " Solver message: " + String(axis_a_opt_result_dict["solver_error_message"]);
		}
		result_dict["error"] = error_msg;
		last_fit_result = result_dict;
		return result_dict; // Early exit on failure
	}

	// --- Optimize Axis B ---
	_current_optimization_mode = OPT_MODE_AXIS_B;
	PackedFloat64Array x_initial_axis_b;
	PackedFloat64Array lower_bounds_axis_b;
	PackedFloat64Array upper_bounds_axis_b;
	_initialize_parameters_for_current_mode(x_initial_axis_b, lower_bounds_axis_b, upper_bounds_axis_b);

	Dictionary axis_b_opt_result_dict;
	bool success_axis_b = _execute_lbfgsb_optimization(x_initial_axis_b, lower_bounds_axis_b, upper_bounds_axis_b, axis_b_opt_result_dict);

	if (success_axis_b && !axis_b_opt_result_dict.has("error") && !axis_b_opt_result_dict.has("solver_error_message")) {
		PackedFloat64Array optimized_params_axis_b = axis_b_opt_result_dict["optimized_params"];
		_update_capsule_from_optimized_params(optimized_params_axis_b);
		total_iterations += int(axis_b_opt_result_dict.get("iterations", 0));
		final_fx_axis_b = double(axis_b_opt_result_dict.get("final_fx", 0.0));

		result_dict["message"] = "Axes optimization completed for capsule " + itos(p_capsule_idx) + ".";
		result_dict["iterations"] = total_iterations;
		result_dict["final_fx"] = final_fx_axis_b;
		result_dict["optimized_radius"] = capsules[p_capsule_idx].optimized_radius;
		result_dict["optimized_axis_a"] = capsules[p_capsule_idx].optimized_axis_a;
		result_dict["optimized_axis_b"] = capsules[p_capsule_idx].optimized_axis_b;
	} else {
		String error_msg = "Axis B optimization failed for capsule " + itos(p_capsule_idx) + ".";
		if (axis_b_opt_result_dict.has("error")) {
			error_msg += " Solver error: " + String(axis_b_opt_result_dict["error"]);
		} else if (axis_b_opt_result_dict.has("solver_error_message")) {
			error_msg += " Solver message: " + String(axis_b_opt_result_dict["solver_error_message"]);
		}
		result_dict["error"] = error_msg;
		if (axis_b_opt_result_dict.has("iterations")) {
			result_dict["iterations"] = total_iterations + int(axis_b_opt_result_dict.get("iterations", 0));
		}
		if (axis_b_opt_result_dict.has("final_fx")) {
			result_dict["final_fx"] = axis_b_opt_result_dict["final_fx"];
		}
	}

	last_fit_result = result_dict;
	return result_dict;
}
