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

#include "core/math/basis.h"
#include "core/math/geometry_3d.h" // For Geometry3D::get_closest_point_to_segment
#include "core/math/quaternion.h"
#include "core/object/class_db.h"
#include "core/object/object.h"

#include "core/variant/variant.h"
#include "thirdparty/LBFGSpp/include/LBFGSB.h"
#include "thirdparty/eigen/Eigen/Core"

LBFGSBCapsuleFitterSolver::LBFGSBCapsuleFitterSolver() {
}

LBFGSBCapsuleFitterSolver::~LBFGSBCapsuleFitterSolver() {
}

void LBFGSBCapsuleFitterSolver::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_source_mesh", "p_mesh"), &LBFGSBCapsuleFitterSolver::set_source_mesh);
	ClassDB::bind_method(D_METHOD("get_source_mesh"), &LBFGSBCapsuleFitterSolver::get_source_mesh);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "source_mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_source_mesh", "get_source_mesh");

	ClassDB::bind_method(D_METHOD("set_surface_index", "p_index"), &LBFGSBCapsuleFitterSolver::set_surface_index);
	ClassDB::bind_method(D_METHOD("get_surface_index"), &LBFGSBCapsuleFitterSolver::get_surface_index);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "surface_index"), "set_surface_index", "get_surface_index");

	ClassDB::bind_method(D_METHOD("set_axis_point_a", "p_point"), &LBFGSBCapsuleFitterSolver::set_axis_point_a);
	ClassDB::bind_method(D_METHOD("get_axis_point_a"), &LBFGSBCapsuleFitterSolver::get_axis_point_a);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis_point_a"), "set_axis_point_a", "get_axis_point_a");

	ClassDB::bind_method(D_METHOD("set_axis_point_b", "p_point"), &LBFGSBCapsuleFitterSolver::set_axis_point_b);
	ClassDB::bind_method(D_METHOD("get_axis_point_b"), &LBFGSBCapsuleFitterSolver::get_axis_point_b);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "axis_point_b"), "set_axis_point_b", "get_axis_point_b");

	ClassDB::bind_method(D_METHOD("set_radius", "p_radius"), &LBFGSBCapsuleFitterSolver::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &LBFGSBCapsuleFitterSolver::get_radius);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");

	ClassDB::bind_method(D_METHOD("set_height", "p_height"), &LBFGSBCapsuleFitterSolver::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &LBFGSBCapsuleFitterSolver::get_height);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");

	ClassDB::bind_method(D_METHOD("get_last_fit_result"), &LBFGSBCapsuleFitterSolver::get_last_fit_result);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "last_fit_result", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY | PROPERTY_USAGE_EDITOR), "", "get_last_fit_result");

	ClassDB::bind_method(D_METHOD("execute_fit"), &LBFGSBCapsuleFitterSolver::execute_fit);
	ClassDB::bind_method(D_METHOD("execute_orientation_fit"), &LBFGSBCapsuleFitterSolver::execute_orientation_fit);

	ClassDB::bind_method(D_METHOD("optimize_radius_for_fixed_axis"), &LBFGSBCapsuleFitterSolver::optimize_radius_for_fixed_axis);
	ClassDB::bind_method(D_METHOD("optimize_orientation_for_fixed_size"), &LBFGSBCapsuleFitterSolver::optimize_orientation_for_fixed_size);

	ClassDB::bind_method(D_METHOD("set_huber_delta", "p_delta"), &LBFGSBCapsuleFitterSolver::set_huber_delta);
	ClassDB::bind_method(D_METHOD("get_huber_delta"), &LBFGSBCapsuleFitterSolver::get_huber_delta);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "huber_delta", PROPERTY_HINT_RANGE, "0.001,1.0,0.001"), "set_huber_delta", "get_huber_delta");
}

void LBFGSBCapsuleFitterSolver::set_source_mesh(const Ref<Mesh> &p_mesh) {
	source_mesh = p_mesh;
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

void LBFGSBCapsuleFitterSolver::set_axis_point_a(const Vector3 &p_point) {
	axis_point_a = p_point;
}

Vector3 LBFGSBCapsuleFitterSolver::get_axis_point_a() const {
	return axis_point_a;
}

void LBFGSBCapsuleFitterSolver::set_axis_point_b(const Vector3 &p_point) {
	axis_point_b = p_point;
}

Vector3 LBFGSBCapsuleFitterSolver::get_axis_point_b() const {
	return axis_point_b;
}

void LBFGSBCapsuleFitterSolver::set_radius(double p_radius) {
	radius = p_radius;
}

double LBFGSBCapsuleFitterSolver::get_radius() const {
	return radius;
}

void LBFGSBCapsuleFitterSolver::set_height(double p_height) {
	double actual_height = MAX(0.0, p_height);

	Vector3 center = (axis_point_a + axis_point_b) * 0.5;
	Vector3 dir = axis_point_b - axis_point_a;

	if (dir.length_squared() < CMP_EPSILON * CMP_EPSILON) {
		if (actual_height > CMP_EPSILON) {
			dir = Vector3(0, 1, 0);
		} else {
			axis_point_a = center;
			axis_point_b = center;
			notify_property_list_changed();
			return;
		}
	}
	dir.normalize();

	axis_point_a = center - dir * (actual_height * 0.5);
	axis_point_b = center + dir * (actual_height * 0.5);
}

double LBFGSBCapsuleFitterSolver::get_height() const {
	return (axis_point_b - axis_point_a).length();
}

Dictionary LBFGSBCapsuleFitterSolver::get_last_fit_result() const {
	return last_fit_result;
}

void LBFGSBCapsuleFitterSolver::execute_fit() {
	if (source_mesh.is_valid()) {
		last_fit_result = optimize_radius_for_fixed_axis();
		if (last_fit_result.has("optimized_radius")) {
			radius = last_fit_result["optimized_radius"];
		}
	} else {
		last_fit_result = Dictionary();
		last_fit_result["error"] = "Source mesh is not set.";
	}
}

void LBFGSBCapsuleFitterSolver::execute_orientation_fit() {
	if (source_mesh.is_valid()) {
		last_fit_result = optimize_orientation_for_fixed_size();
	} else {
		last_fit_result = Dictionary();
		last_fit_result["error"] = "Source mesh is not set.";
	}
}

double LBFGSBCapsuleFitterSolver::call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad) {
	ERR_FAIL_COND_V_MSG(p_x.is_empty(), std::numeric_limits<double>::quiet_NaN(), "Input parameters (p_x) are empty.");
	ERR_FAIL_COND_V_MSG(r_grad.size() != p_x.size(), std::numeric_limits<double>::quiet_NaN(), "Gradient array (r_grad) size does not match input parameters (p_x) size.");

	double current_radius_param = p_x[0];
	double fx = 0.0;

	for (int i = 0; i < r_grad.size(); ++i) {
		r_grad.write[i] = 0.0;
	}

	if (_current_optimization_mode == OPT_MODE_RADIUS) {
		ERR_FAIL_COND_V_MSG(p_x.size() != 1, std::numeric_limits<double>::quiet_NaN(), "Radius optimization expects 1 parameter.");
		double current_radius_param_for_radius_mode = p_x[0]; // Explicitly use p_x[0] here for clarity

		if (current_radius_param_for_radius_mode < 1e-4) {
			fx = std::numeric_limits<double>::max() / 2.0; // Large penalty for near-zero or negative radius
			// Ensure gradient points towards increasing radius if it's too small.
			r_grad.write[0] = -std::numeric_limits<double>::max() / (2.0 * (current_radius_param_for_radius_mode + 1e-5));
			return fx;
		}

		for (int i = 0; i < current_cloud_points_for_objective.size(); ++i) {
			Vector3 p = current_cloud_points_for_objective[i];
			Vector3 closest_point_on_line_segment = Geometry3D::get_closest_point_to_segment(p, axis_point_a, axis_point_b);

			double dist_to_axis = p.distance_to(closest_point_on_line_segment);
			double a = dist_to_axis - current_radius_param_for_radius_mode; // 'a' is the error term (dist_to_axis - radius)

			double huber_loss_term;
			double dL_da; // Derivative of Huber loss L_delta(a) w.r.t. a

			if (Math::abs(a) <= huber_delta) {
				huber_loss_term = 0.5 * a * a;
				dL_da = a;
			} else {
				huber_loss_term = huber_delta * (Math::abs(a) - 0.5 * huber_delta);
				dL_da = huber_delta * SIGN(a);
			}
			fx += huber_loss_term;
			r_grad.write[0] += dL_da * (-1.0); // Derivative of 'a' w.r.t. current_radius_param is -1
		}
	} else if (_current_optimization_mode == OPT_MODE_ORIENTATION) {
		ERR_FAIL_COND_V_MSG(p_x.size() != 6, std::numeric_limits<double>::quiet_NaN(), "Orientation optimization expects 6 parameters (3 exp_map, 3 translation).");

		Vector3 rot_vec(p_x[0], p_x[1], p_x[2]);
		Vector3 current_translation(p_x[3], p_x[4], p_x[5]);

		Basis current_basis;
		double angle = rot_vec.length();
		if (angle < CMP_EPSILON) {
			current_basis = Basis(); // Identity
		} else {
			current_basis = Basis(rot_vec / angle, angle);
		}

		Vector3 current_world_half_axis = current_basis.xform(_orientation_opt_initial_half_axis);
		Vector3 current_axis_a = current_translation - current_world_half_axis;
		Vector3 current_axis_b = current_translation + current_world_half_axis;

		const double distance_threshold_sq = orientation_distance_threshold * orientation_distance_threshold;
		const double cos_angle_threshold = Math::cos(orientation_angle_threshold_rad);
		const double penalty = orientation_penalty_factor;

		fx = 0.0;
		bool normals_are_available = !current_cloud_normals_for_objective.is_empty();

		// Initialize gradients to zero
		for (int j = 0; j < 6; ++j) {
			r_grad.write[j] = 0.0;
		}

		for (int i = 0; i < current_cloud_points_for_objective.size(); ++i) {
			Vector3 P_i = current_cloud_points_for_objective[i];
			Vector3 N_mesh_i;
			if (normals_are_available) {
				N_mesh_i = current_cloud_normals_for_objective[i];
			}

			std::pair<Vector3, Vector3> cap_data = LBFGSBCapsuleFitterSolver::get_closest_point_and_normal_on_capsule_surface(
					P_i, current_axis_a, current_axis_b, _orientation_opt_fixed_radius);
			Vector3 C_i = cap_data.first;
			Vector3 N_cap_i = cap_data.second;

			double dist = P_i.distance_to(C_i);

			double huber_loss_term_val;
			double dL_ddist; // Derivative of Huber loss L_delta(dist) w.r.t. dist

			if (dist <= huber_delta) {
				huber_loss_term_val = 0.5 * dist * dist;
				dL_ddist = dist;
			} else {
				huber_loss_term_val = huber_delta * (dist - 0.5 * huber_delta);
				dL_ddist = huber_delta;
			}

			Vector3 d_dist_dC;
			if (dist < CMP_EPSILON) {
				d_dist_dC = Vector3(); // Zero gradient if P_i is on C_i
			} else {
				d_dist_dC = (C_i - P_i) / dist;
			}
			Vector3 d_huber_loss_dC = dL_ddist * d_dist_dC;

			bool match_is_good = true;
			if (dist * dist > distance_threshold_sq) {
				match_is_good = false;
			}

			if (normals_are_available && N_mesh_i.length_squared() > CMP_EPSILON && N_cap_i.length_squared() > CMP_EPSILON) {
				double cos_angle = N_mesh_i.normalized().dot(N_cap_i.normalized());
				if (cos_angle < cos_angle_threshold) {
					match_is_good = false;
				}
			} else if (normals_are_available) { // If normals should be there but one is zero length, consider it a bad match
				match_is_good = false;
			}

			double current_point_fx_contribution = huber_loss_term_val;
			Vector3 final_d_loss_dC = d_huber_loss_dC;

			if (!match_is_good) {
				current_point_fx_contribution *= penalty;
				final_d_loss_dC *= penalty;
			}
			fx += current_point_fx_contribution;

			LBFGSBCapsuleFitterSolver::CapsuleSurfacePointDerivatives C_derivs = LBFGSBCapsuleFitterSolver::get_capsule_surface_derivatives(P_i, current_axis_a, current_axis_b, _orientation_opt_fixed_radius);

			if (C_derivs.is_valid) {
				for (int k_param = 0; k_param < 6; ++k_param) {
					Vector3 dA_dparam_k;
					Vector3 dB_dparam_k;

					if (k_param < 3) { // Derivative w.r.t. rot_vec component k_param
						Basis dR_drot_k = LBFGSBCapsuleFitterSolver::d_basis_d_rot_comp(rot_vec, k_param);
						dA_dparam_k = dR_drot_k.xform(_orientation_opt_initial_half_axis) * -1.0; // dA = -dR*h
						dB_dparam_k = dR_drot_k.xform(_orientation_opt_initial_half_axis); // dB =  dR*h
					} else { // Derivative w.r.t. translation component (k_param - 3)
						Vector3 dTrans_dparam_k;
						dTrans_dparam_k[k_param - 3] = 1.0;
						dA_dparam_k = dTrans_dparam_k;
						dB_dparam_k = dTrans_dparam_k;
					}

					Vector3 dC_dparam_k = C_derivs.dC_dA.xform(dA_dparam_k) + C_derivs.dC_dB.xform(dB_dparam_k);
					r_grad.write[k_param] += final_d_loss_dC.dot(dC_dparam_k);
				}
			}
		}
	}

	return fx;
}

Dictionary LBFGSBCapsuleFitterSolver::optimize_radius_for_fixed_axis() {
	Dictionary result;
	_current_optimization_mode = OPT_MODE_RADIUS;

	if (source_mesh.is_null()) {
		result["error"] = "Input mesh is null.";
		return result;
	}
	if (surface_index < 0 || surface_index >= source_mesh->get_surface_count()) {
		result["error"] = "Invalid surface index.";
		return result;
	}

	Array surface_arrays = source_mesh->surface_get_arrays(surface_index);
	if (surface_arrays.is_empty() || !(source_mesh->surface_get_format(surface_index) & Mesh::ARRAY_FORMAT_VERTEX)) {
		result["error"] = "Surface has no vertex data or invalid format.";
		return result;
	}
	current_cloud_points_for_objective = surface_arrays[Mesh::ARRAY_VERTEX];

	if (current_cloud_points_for_objective.is_empty()) {
		result["error"] = "Point cloud (from mesh surface) is empty.";
		return result;
	}

	PackedFloat64Array x_initial_td;
	x_initial_td.push_back(radius);

	PackedFloat64Array lower_bounds_td;
	lower_bounds_td.push_back(1e-5);

	PackedFloat64Array upper_bounds_td;
	upper_bounds_td.push_back(std::numeric_limits<double>::infinity());

	PackedFloat64Array grad_initial_td;
	grad_initial_td.resize(x_initial_td.size());
	double fx_initial = call_operator(x_initial_td, grad_initial_td);

	Array solver_result_array = minimize(x_initial_td, fx_initial, lower_bounds_td, upper_bounds_td);

	if (solver_result_array.size() >= 3) {
		PackedFloat64Array optimized_params_td = solver_result_array[1];
		if (optimized_params_td.size() == 1) {
			result["optimized_radius"] = optimized_params_td[0];
			result["final_fx"] = solver_result_array[2];
			result["iterations"] = solver_result_array[0];
			result["height"] = get_height();
		} else {
			result["error"] = "Solver returned incorrect number of parameters.";
		}
	} else {
		result["error"] = "Solver did not return a valid result.";
	}
	return result;
}

Dictionary LBFGSBCapsuleFitterSolver::optimize_orientation_for_fixed_size() {
	_current_optimization_mode = OPT_MODE_ORIENTATION;
	Dictionary result;

	// Ensure the source mesh and its vertex data are available.
	if (source_mesh.is_null() || source_mesh->get_surface_count() == 0 || surface_index >= source_mesh->get_surface_count()) {
		result["error"] = "Source mesh not set or surface index invalid.";
		return result;
	}

	Array surface_arrays = source_mesh->surface_get_arrays(surface_index);
	if (surface_arrays.is_empty() || !(source_mesh->surface_get_format(surface_index) & Mesh::ARRAY_FORMAT_VERTEX)) {
		result["error"] = "Surface has no vertex data or invalid format.";
		return result;
	}
	current_cloud_points_for_objective = surface_arrays[Mesh::ARRAY_VERTEX];
	if (source_mesh->surface_get_format(surface_index) & Mesh::ARRAY_FORMAT_NORMAL) {
		current_cloud_normals_for_objective = surface_arrays[Mesh::ARRAY_NORMAL];
	} else {
		current_cloud_normals_for_objective.clear(); // Ensure it's empty if no normals
	}

	if (current_cloud_points_for_objective.is_empty()) {
		result["error"] = "Point cloud (from mesh surface) is empty.";
		return result;
	}

	_current_optimization_mode = OPT_MODE_ORIENTATION;
	_orientation_opt_fixed_radius = radius; // Use the current radius of the solver
	double current_height = get_height();
	if (current_height < CMP_EPSILON) {
		result["error"] = "Capsule height is near zero, cannot define orientation meaningfully.";
		return result;
	}
	_orientation_opt_initial_half_axis = Vector3(0, current_height / 2.0, 0); // Canonical half-axis along Y

	PackedFloat64Array x_initial_td;
	x_initial_td.resize(6);
	// Initial rotation (exp_map_x, exp_map_y, exp_map_z) = (0,0,0) -> identity rotation
	x_initial_td.write[0] = 0.0;
	x_initial_td.write[1] = 0.0;
	x_initial_td.write[2] = 0.0;
	// Initial translation (tx, ty, tz) = current capsule center
	Vector3 initial_center = (axis_point_a + axis_point_b) / 2.0;
	x_initial_td.write[3] = initial_center.x;
	x_initial_td.write[4] = initial_center.y;
	x_initial_td.write[5] = initial_center.z;

	PackedFloat64Array lower_bounds_td;
	lower_bounds_td.resize(6);
	PackedFloat64Array upper_bounds_td;
	upper_bounds_td.resize(6);

	// Bounds for rotation (e.g., +/- 2*PI for each component of exp_map)
	for (int i = 0; i < 3; ++i) {
		lower_bounds_td.write[i] = -2.0 * Math::TAU; // Allow multiple full rotations if needed
		upper_bounds_td.write[i] = 2.0 * Math::TAU;
	}
	// Bounds for translation (can be quite large, or based on scene dimensions if known)
	for (int i = 3; i < 6; ++i) {
		lower_bounds_td.write[i] = -1000.0; // Example large bounds
		upper_bounds_td.write[i] = 1000.0;
	}

	PackedFloat64Array grad_initial_td;
	grad_initial_td.resize(x_initial_td.size());
	// Temporarily set current_cloud_points_for_objective and normals if not already set by a previous call
	// This is crucial because call_operator relies on these members.
	// The main setup for these is now at the beginning of this function.
	double fx_initial = call_operator(x_initial_td, grad_initial_td); // Calculate initial fx and grad

	Array solver_result_array = minimize(x_initial_td, fx_initial, lower_bounds_td, upper_bounds_td);

	if (solver_result_array.size() >= 3) {
		PackedFloat64Array optimized_params_td = solver_result_array[1];
		if (optimized_params_td.size() == 6) {
			Vector3 optimized_rot_vec(optimized_params_td[0], optimized_params_td[1], optimized_params_td[2]);
			Vector3 optimized_translation(optimized_params_td[3], optimized_params_td[4], optimized_params_td[5]);

			Basis optimized_basis;
			double angle = optimized_rot_vec.length();
			if (angle < CMP_EPSILON) {
				optimized_basis = Basis(); // Identity
			} else {
				optimized_basis = Basis(optimized_rot_vec / angle, angle);
			}

			Vector3 final_world_half_axis = optimized_basis.xform(_orientation_opt_initial_half_axis);
			axis_point_a = optimized_translation - final_world_half_axis;
			axis_point_b = optimized_translation + final_world_half_axis;

			notify_property_list_changed();

			result["optimized_axis_a"] = axis_point_a;
			result["optimized_axis_b"] = axis_point_b;
			result["final_fx"] = solver_result_array[2];
			result["iterations"] = solver_result_array[0];
			result["status_message"] = solver_result_array.size() > 3 ? String(solver_result_array[3]) : String("Optimization completed.");
		} else {
			result["error"] = "Solver returned incorrect number of parameters for orientation.";
		}
	} else {
		result["error"] = "Solver did not return a valid result for orientation.";
	}
	return result;
}

Vector<Vector3> LBFGSBCapsuleFitterSolver::_generate_canonical_capsule_points(const Vector3 &p_cap_a, const Vector3 &p_cap_b, double p_cap_radius, int p_cylinder_rings, int p_radial_segments) const {
	Vector<Vector3> points;

	Vector3 axis_vec = p_cap_b - p_cap_a;
	double axis_length = axis_vec.length();
	Vector3 axis_dir = (axis_length > CMP_EPSILON) ? axis_vec / axis_length : Vector3(0, 1, 0);

	Basis cylinder_basis = Basis::looking_at(axis_dir);

	for (int i = 0; i <= p_cylinder_rings; ++i) {
		double t_cyl = (p_cylinder_rings > 0) ? (double)i / p_cylinder_rings : 0.5;
		Vector3 current_axis_point = p_cap_a + axis_vec * t_cyl;
		for (int j = 0; j < p_radial_segments; ++j) {
			double angle = (p_radial_segments > 0) ? (double)j / p_radial_segments * Math::TAU : 0.0;
			Vector3 radial_offset = Vector3(Math::cos(angle) * p_cap_radius, Math::sin(angle) * p_cap_radius, 0);
			points.push_back(current_axis_point + cylinder_basis.xform(radial_offset));
		}
	}

	int num_cap_points_per_hemisphere = MAX(1, p_radial_segments * MAX(1, p_cylinder_rings / 2 + 1));
	double golden_angle = Math::PI * (3.0 - Math::sqrt(5.0));

	for (int cap_idx = 0; cap_idx < 2; ++cap_idx) {
		Vector3 cap_center = (cap_idx == 0) ? p_cap_a : p_cap_b;
		Vector3 cap_normal_direction = (cap_idx == 0) ? -axis_dir : axis_dir;
		Basis cap_orientation_basis = Basis::looking_at(cap_normal_direction);

		points.push_back(cap_center + cap_normal_direction * p_cap_radius);

		for (int i = 0; i < num_cap_points_per_hemisphere; ++i) {
			double z_local_on_unit_sphere = (double)i / (num_cap_points_per_hemisphere);
			double radius_at_this_z_local = Math::sqrt(1.0 - z_local_on_unit_sphere * z_local_on_unit_sphere);
			double theta_fib = golden_angle * i;

			Vector3 p_local_hemisphere;
			p_local_hemisphere.x = Math::cos(theta_fib) * radius_at_this_z_local;
			p_local_hemisphere.y = Math::sin(theta_fib) * radius_at_this_z_local;
			p_local_hemisphere.z = z_local_on_unit_sphere;

			points.push_back(cap_center + cap_orientation_basis.xform(p_local_hemisphere * p_cap_radius));
		}
	}

	return points;
}

void LBFGSBCapsuleFitterSolver::set_huber_delta(double p_delta) {
	huber_delta = MAX(1e-5, p_delta); // Ensure delta is positive and not too small
}

double LBFGSBCapsuleFitterSolver::get_huber_delta() const {
	return huber_delta;
}

// Definition for get_closest_point_and_normal_on_capsule_surface
std::pair<Vector3, Vector3> LBFGSBCapsuleFitterSolver::get_closest_point_and_normal_on_capsule_surface(
		const Vector3 &p_mesh_vertex,
		const Vector3 &p_cap_a,
		const Vector3 &p_cap_b,
		double p_cap_radius) {
	// 1. Find the closest point on the capsule's central axis segment.
	Vector3 axis_vec = p_cap_b - p_cap_a;
	double axis_length_sq = axis_vec.length_squared();
	Vector3 closest_point_on_axis; // This is Q in some notations

	bool on_sphere_cap_a = false;
	bool on_sphere_cap_b = false;

	if (axis_length_sq < CMP_EPSILON * CMP_EPSILON) { // Capsule is a sphere centered at p_cap_a
		closest_point_on_axis = p_cap_a;
		on_sphere_cap_a = true; // Treat as cap A for normal calculation logic
	} else {
		Vector3 ap = p_mesh_vertex - p_cap_a;
		double t = axis_vec.dot(ap) / axis_length_sq;
		if (t <= 0.0) {
			closest_point_on_axis = p_cap_a;
			on_sphere_cap_a = true;
		} else if (t >= 1.0) {
			closest_point_on_axis = p_cap_b;
			on_sphere_cap_b = true;
		} else {
			closest_point_on_axis = p_cap_a + t * axis_vec;
			// On cylinder body axis segment part
		}
	}

	// 2. Determine the normal vector (N) and the closest point on surface (C).
	Vector3 normal_on_surface;
	Vector3 vec_from_axis_to_vertex = p_mesh_vertex - closest_point_on_axis;

	if (vec_from_axis_to_vertex.length_squared() < CMP_EPSILON * CMP_EPSILON) {
		// Mesh vertex P_i is ON the capsule axis segment (at closest_point_on_axis).
		if (on_sphere_cap_a) { // Includes sphere case where closest_point_on_axis is p_cap_a
			if (axis_length_sq < CMP_EPSILON * CMP_EPSILON) { // True sphere case, P_i is at center
				normal_on_surface = Vector3(1, 0, 0); // Arbitrary default normal for point at sphere center
			} else {
				normal_on_surface = (p_cap_a - p_cap_b).normalized(); // Normal points outwards from cap A along axis
			}
		} else if (on_sphere_cap_b) {
			normal_on_surface = (p_cap_b - p_cap_a).normalized(); // Normal points outwards from cap B along axis
		} else { // On the cylinder body's axis (not at the caps)
			// Normal is arbitrary, perpendicular to the axis.
			// This case can be problematic for unique derivatives if not handled carefully by the caller.
			// Ensure axis_vec is not zero before calling get_any_perpendicular.
			// If axis_vec is zero, it's a sphere, handled above.
			if (axis_vec.length_squared() > CMP_EPSILON * CMP_EPSILON) {
				normal_on_surface = axis_vec.get_any_perpendicular().normalized();
			} else { // Should not happen if logic is correct, but as a fallback for P_i on axis of a zero-length axis capsule (sphere center)
				normal_on_surface = Vector3(1, 0, 0);
			}
		}
	} else {
		// Mesh vertex P_i is NOT on the capsule axis. Normal is from axis point to P_i.
		normal_on_surface = vec_from_axis_to_vertex.normalized();
	}

	Vector3 closest_point_on_surface = closest_point_on_axis + normal_on_surface * p_cap_radius;
	return { closest_point_on_surface, normal_on_surface };
}

Basis LBFGSBCapsuleFitterSolver::d_vec_normalized_d_vec(const Vector3 &p_vec) {
	double len_sq = p_vec.length_squared();
	if (len_sq < CMP_EPSILON * CMP_EPSILON) {
		return Basis(); // Zero matrix for zero vector
	}
	double len = Math::sqrt(len_sq);
	double inv_len = 1.0 / len;
	double inv_len_cubed = inv_len * inv_len * inv_len;

	// J = (I - v v^T / |v|^2) / |v|
	Basis I; // Identity
	Basis vvT = outer_product(p_vec, p_vec);

	Basis J = (I - vvT * (1.0 / len_sq)) * inv_len;
	return J;
}

Basis LBFGSBCapsuleFitterSolver::d_basis_d_rot_comp(const Vector3 &p_rot_vec, int p_comp_idx) {
	double theta = p_rot_vec.length();
	Basis R = _compute_rotation_matrix_from_rot_vec(p_rot_vec); // R = exp([p_rot_vec]_x)

	Vector3 e_k;
	e_k[p_comp_idx] = 1.0;

	if (theta < 1e-6) { // Threshold for very small angle. For exactly zero, R=I.
		// For phi -> 0, dR/dphi_k -> [e_k]_x
		Basis e_k_x;
		e_k_x.set_column(0, e_k.cross(Vector3(1, 0, 0)));
		e_k_x.set_column(1, e_k.cross(Vector3(0, 1, 0)));
		e_k_x.set_column(2, e_k.cross(Vector3(0, 0, 1)));
		// Correction: Basis().set_skew_symmetric(v) or v.skew()
		// Basis stores columns. Skew symmetric matrix [v]_x has columns v x e1, v x e2, v x e3
		// For [e_k]_x, columns are e_k x e1, e_k x e2, e_k x e3
		Vector3 c0 = e_k.cross(Vector3(1, 0, 0)); // e_k x i
		Vector3 c1 = e_k.cross(Vector3(0, 1, 0)); // e_k x j
		Vector3 c2 = e_k.cross(Vector3(0, 0, 1)); // e_k x k
		return Basis(c0, c1, c2);
	}

	double theta_sq = theta * theta;
	double sin_theta = Math::sin(theta);
	double cos_theta = Math::cos(theta);

	double A_coeff, B_coeff;
	// Using a slightly larger threshold for Taylor expansion for stability, as used in previous numerical version.
	if (theta < 1e-2) {
		A_coeff = 0.5 - theta_sq / 24.0; // (1 - cos_theta) / theta_sq
		B_coeff = 1.0 - theta_sq / 6.0; // sin_theta / theta
	} else {
		A_coeff = (1.0 - cos_theta) / theta_sq;
		B_coeff = sin_theta / theta;
	}

	// Formula: dR/dphi_k = A_coeff * (phi_k [phi]_x + [(I-R)e_k]_x R) + B_coeff * [e_k]_x R
	// where phi is p_rot_vec.
	double phi_k_val = p_rot_vec[p_comp_idx];

	Basis phi_x; // Skew symmetric matrix for p_rot_vec
	phi_x.set_column(0, p_rot_vec.cross(Vector3(1, 0, 0)));
	phi_x.set_column(1, p_rot_vec.cross(Vector3(0, 1, 0)));
	phi_x.set_column(2, p_rot_vec.cross(Vector3(0, 0, 1)));

	Basis e_k_x; // Skew symmetric matrix for e_k
	e_k_x.set_column(0, e_k.cross(Vector3(1, 0, 0)));
	e_k_x.set_column(1, e_k.cross(Vector3(0, 1, 0)));
	e_k_x.set_column(2, e_k.cross(Vector3(0, 0, 1)));

	Basis I; // Identity
	Vector3 I_minus_R_ek = e_k - R.xform(e_k); // (I-R)e_k

	Basis I_minus_R_ek_x; // Skew symmetric matrix for (I-R)e_k
	I_minus_R_ek_x.set_column(0, I_minus_R_ek.cross(Vector3(1, 0, 0)));
	I_minus_R_ek_x.set_column(1, I_minus_R_ek.cross(Vector3(0, 1, 0)));
	I_minus_R_ek_x.set_column(2, I_minus_R_ek.cross(Vector3(0, 0, 1)));

	Basis term1 = (phi_x * phi_k_val + I_minus_R_ek_x * R) * A_coeff;
	Basis term2 = (e_k_x * R) * B_coeff;

	return term1 + term2;
}

Basis LBFGSBCapsuleFitterSolver::_compute_rotation_matrix_from_rot_vec(const Vector3 &p_rot_vec) {
	Basis R;
	double angle = p_rot_vec.length();
	if (angle < CMP_EPSILON) {
		R = Basis(); // Identity
	} else {
		Vector3 axis = p_rot_vec / angle;
		R.set_axis_angle(axis, angle);
	}
	return R;
}

LBFGSBCapsuleFitterSolver::CapsuleSurfacePointDerivatives LBFGSBCapsuleFitterSolver::get_capsule_surface_derivatives(
		const Vector3 &p_mesh_vertex,
		const Vector3 &p_cap_a,
		const Vector3 &p_cap_b,
		double p_cap_radius) {
	CapsuleSurfacePointDerivatives derivs;
	derivs.is_valid = false; // Default to invalid

	std::pair<Vector3, Vector3> cn_data = get_closest_point_and_normal_on_capsule_surface(p_mesh_vertex, p_cap_a, p_cap_b, p_cap_radius);
	Vector3 C = cn_data.first;
	Vector3 N = cn_data.second;

	Vector3 P = p_mesh_vertex;
	Vector3 A = p_cap_a;
	Vector3 B = p_cap_b;
	double r = p_cap_radius;

	Vector3 axis = B - A;
	double h_sq = axis.length_squared();

	// Check for degenerate cases where derivatives are not well-defined
	if (h_sq < CMP_EPSILON * CMP_EPSILON) { // Sphere case
		if ((P - A).length_squared() < CMP_EPSILON * CMP_EPSILON) {
			return derivs; // P at sphere center
		}
	} else { // Cylinder or caps
		// Determine t_unclamped for projection on infinite line
		Vector3 AP = P - A;
		double t_unclamped = axis.is_normalized() ? AP.dot(axis) : (h_sq > CMP_EPSILON ? AP.dot(axis) / h_sq : 0.0);
		Vector3 Q_on_line = A + t_unclamped * axis;
		if ((P - Q_on_line).length_squared() < CMP_EPSILON * CMP_EPSILON) {
			return derivs; // P on infinite axis line
		}
	}

	// Determine region (Cap A, Cap B, Cylinder)
	// Q_cap is the closest point on the *segment* AB to P
	// This logic is implicitly handled by get_closest_point_and_normal_on_capsule_surface determining Q (closest_point_on_axis)
	// We need to know where Q lies relative to A and B to pick the formula.

	Vector3 Q; // Closest point on axis segment AB to P
	double t_param; // Parameter along segment, t_param in [0,1] if on segment
	// Re-evaluate Q and t_param based on Geometry3D logic for clarity here
	if (h_sq < CMP_EPSILON * CMP_EPSILON) { // Sphere
		Q = A;
		t_param = 0; // Treat as Cap A
	} else {
		t_param = (P - A).dot(axis) / h_sq;
		if (t_param <= 0.0) {
			Q = A; // Projects to Cap A
		} else if (t_param >= 1.0) {
			Q = B; // Projects to Cap B
		} else {
			Q = A + t_param * axis; // Projects to Cylinder body
		}
	}

	Vector3 P_minus_Q = P - Q;
	// If P_minus_Q is zero, it means P is on the axis segment.
	// This was checked earlier by (P - Q_on_line).length_squared() for the infinite line.
	// If Q is an endpoint (A or B) and P=Q, then P_minus_Q is zero.
	// get_closest_point_and_normal_on_capsule_surface gives a default normal in this case.
	// The derivatives of N might be problematic.
	// However, the earlier check for P on infinite axis should catch this.

	Basis I; // Identity

	if (h_sq < CMP_EPSILON * CMP_EPSILON || t_param <= 0.0) { // Sphere case (A=B) or Cap A (Q=A)
		// Derivatives for Cap A (Q = A)
		// N = normalize(P-A), C = A + r*N
		if ((P - A).length_squared() < CMP_EPSILON * CMP_EPSILON) {
			return derivs;
		} // P at cap center A
		Basis dNdA = d_vec_normalized_d_vec(P - A) * -1.0; // d(normalize(P-X))/dX = -d_normalize_dvec(P-X)
		derivs.dN_dA = dNdA;
		derivs.dC_dA = I + dNdA * r;
		derivs.dN_dB = Basis(); // Zero
		derivs.dC_dB = Basis(); // Zero
		derivs.is_valid = true;

	} else if (t_param >= 1.0) { // Cap B (Q=B)
		// Derivatives for Cap B (Q = B)
		// N = normalize(P-B), C = B + r*N
		if ((P - B).length_squared() < CMP_EPSILON * CMP_EPSILON) {
			return derivs;
		} // P at cap center B
		Basis dNdB = d_vec_normalized_d_vec(P - B) * -1.0;
		derivs.dN_dB = dNdB;
		derivs.dC_dB = I + dNdB * r;
		derivs.dN_dA = Basis(); // Zero
		derivs.dC_dA = Basis(); // Zero
		derivs.is_valid = true;

	} else { // Cylinder body (0 < t_param < 1)
		// Q = A*(1-t) + B*t, where t = dot(P-A, B-A) / dot(B-A, B-A)
		// N = normalize(P-Q)
		// C = Q + r*N

		// Gradient of t with respect to A and B (these are row vectors / 1x3 Jacobians)
		// grad_A t = ( (B-A) x ( (P-A) x (B-A) ) ) / ||B-A||^4
		// grad_B t = ( (B-A) x ( (P-B) x (B-A) ) ) / ||B-A||^4
		// Let V = B-A (axis). Let AP = P-A. Let BP = P-B.
		// grad_A t = (V x (AP x V)) / h_sq^2
		// grad_B t = (V x (BP x V)) / h_sq^2 (Note: P-B for grad_B t, not AP)
		Vector3 grad_t_wrt_A_vec = axis.cross((P - A).cross(axis)) / (h_sq * h_sq);
		Vector3 grad_t_wrt_B_vec = axis.cross((P - B).cross(axis)) / (h_sq * h_sq);

		// dQ/dA = (1-t)I - outer_product(axis, grad_t_wrt_A_vec)
		// dQ/dB = tI   + outer_product(axis, grad_t_wrt_B_vec)
		Basis dQdA = I * (1.0 - t_param) - outer_product(axis, grad_t_wrt_A_vec);
		Basis dQdB = I * t_param + outer_product(axis, grad_t_wrt_B_vec);

		if (P_minus_Q.length_squared() < CMP_EPSILON * CMP_EPSILON) {
			return derivs;
		} // P on axis segment (already checked but good to be safe)

		Basis d_norm_P_minus_Q = d_vec_normalized_d_vec(P_minus_Q);

		// dN/dA = d(normalize(P-Q))/d(P-Q) * d(P-Q)/dA = d_norm_P_minus_Q * (-dQdA)
		derivs.dN_dA = d_norm_P_minus_Q * (dQdA * -1.0);
		derivs.dN_dB = d_norm_P_minus_Q * (dQdB * -1.0);

		// dC/dA = dQ/dA + r * dN/dA
		derivs.dC_dA = dQdA + derivs.dN_dA * r;
		derivs.dC_dB = dQdB + derivs.dN_dB * r;
		derivs.is_valid = true;
	}
	return derivs;
}

// Helper function for outer product v1 * v2^T, result is a Basis
Basis LBFGSBCapsuleFitterSolver::outer_product(const Vector3 &p_v1, const Vector3 &p_v2) {
	return Basis(p_v1 * p_v2.x, p_v1 * p_v2.y, p_v1 * p_v2.z);
}
