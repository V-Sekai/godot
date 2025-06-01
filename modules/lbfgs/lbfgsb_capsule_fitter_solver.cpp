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
#include "core/math/geometry_3d.h" // For Geometry3D::get_closest_points_between_segments if needed, though custom logic is used.
#include "core/io/json.h"         // For stringifying results if debugging

void LBFGSBCapsuleFitterSolver::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_source_mesh", "p_mesh"), &LBFGSBCapsuleFitterSolver::set_source_mesh);
    ClassDB::bind_method(D_METHOD("get_source_mesh"), &LBFGSBCapsuleFitterSolver::get_source_mesh);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "source_mesh", PROPERTY_HINT_RESOURCE_TYPE, "ImporterMesh"), "set_source_mesh", "get_source_mesh");

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

void LBFGSBCapsuleFitterSolver::set_source_mesh(const Ref<ImporterMesh> &p_mesh) {
    source_mesh = p_mesh;
}

Ref<ImporterMesh> LBFGSBCapsuleFitterSolver::get_source_mesh() const {
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

Dictionary LBFGSBCapsuleFitterSolver::optimize_all_capsule_parameters() {
    last_fit_result.clear();

    if (!source_mesh.is_valid()) {
        last_fit_result["error"] = "Source mesh is not set.";
        return last_fit_result;
    }

    ERR_FAIL_COND_V_MSG(surface_index < 0 || surface_index >= source_mesh->get_surface_count(),
            last_fit_result, "Invalid surface_index " + itos(surface_index) +
                                 ". Mesh has " + itos(source_mesh->get_surface_count()) + " surfaces.");

    if (m_capsules.is_empty()) {
        last_fit_result["message"] = "No capsule instances defined. Nothing to optimize.";
        last_fit_result["final_fx"] = 0.0;
        last_fit_result["iterations"] = 0;
        last_fit_result["optimized_capsules_results"] = Array();
        return last_fit_result;
    }

    Array surface_arrays = source_mesh->get_surface_arrays(surface_index);
    if (surface_arrays.is_empty()) {
        last_fit_result["error"] = "Source mesh surface arrays are empty for surface index " + itos(surface_index) + ".";
        return last_fit_result;
    }

    current_cloud_points_for_objective = surface_arrays[Mesh::ARRAY_VERTEX];
    if (source_mesh->get_surface_format(surface_index) & Mesh::ARRAY_FORMAT_NORMAL) {
        current_cloud_normals_for_objective = surface_arrays[Mesh::ARRAY_NORMAL];
    } else {
        current_cloud_normals_for_objective.clear();
    }

    if (current_cloud_points_for_objective.is_empty()) {
        last_fit_result["error"] = "Point cloud (from mesh surface " + itos(surface_index) + ") is empty.";
        return last_fit_result;
    }

    _current_optimization_mode = OPT_MODE_MULTI_ALL_PARAMS;
    int num_total_params = m_capsules.size() * 7;

    PackedFloat64Array local_x_initial;
    PackedFloat64Array local_lower_bounds;
    PackedFloat64Array local_upper_bounds;

    local_x_initial.resize(num_total_params);
    local_lower_bounds.resize(num_total_params);
    local_upper_bounds.resize(num_total_params);

    for (int i = 0; i < m_capsules.size(); ++i) {
        const CapsuleInstance &capsule = m_capsules[i];
        int offset = i * 7;

        local_x_initial.write[offset + 0] = capsule.initial_radius;
        local_lower_bounds.write[offset + 0] = 0.001; // Min radius
        local_upper_bounds.write[offset + 0] = MAXFLOAT;

        local_x_initial.write[offset + 1] = capsule.initial_axis_a.x;
        local_x_initial.write[offset + 2] = capsule.initial_axis_a.y;
        local_x_initial.write[offset + 3] = capsule.initial_axis_a.z;
        for (int j = 0; j < 3; ++j) {
            local_lower_bounds.write[offset + 1 + j] = -MAXFLOAT;
            local_upper_bounds.write[offset + 1 + j] = MAXFLOAT;
        }

        local_x_initial.write[offset + 4] = capsule.initial_axis_b.x;
        local_x_initial.write[offset + 5] = capsule.initial_axis_b.y;
        local_x_initial.write[offset + 6] = capsule.initial_axis_b.z;
        for (int j = 0; j < 3; ++j) {
            local_lower_bounds.write[offset + 4 + j] = -MAXFLOAT;
            local_upper_bounds.write[offset + 4 + j] = MAXFLOAT;
        }
    }

    PackedFloat64Array dummy_gradient_array;
    if (num_total_params > 0) {
        dummy_gradient_array.resize(num_total_params);
    }
    double initial_fx = call_operator(local_x_initial, dummy_gradient_array);

    Array result_array = minimize(local_x_initial, initial_fx, local_lower_bounds, local_upper_bounds);

    if (result_array.size() != 3 ||
            result_array[0].get_type() != Variant::INT ||
            result_array[1].get_type() != Variant::PACKED_FLOAT64_ARRAY ||
            result_array[2].get_type() != Variant::FLOAT) {
        last_fit_result["error"] = "Base minimize method returned Array with unexpected structure or types. Expected [int, PackedFloat64Array, float].";
        if (result_array.size() == 1 && result_array[0].get_type() == Variant::STRING) { // Check if it's an error string from LBFGSBSolver
            last_fit_result["solver_error_message"] = result_array[0];
        }
        return last_fit_result;
    }

    Dictionary actual_result_dict;
    actual_result_dict["iterations"] = result_array[0];
    PackedFloat64Array optimized_params = result_array[1];
    actual_result_dict["final_fx"] = result_array[2];
    actual_result_dict["optimized_params"] = optimized_params;

    if (optimized_params.size() != num_total_params) {
        actual_result_dict["error"] = "Optimized params size mismatch. Expected " + itos(num_total_params) + ", got " + itos(optimized_params.size()) + ".";
        last_fit_result = actual_result_dict;
        return last_fit_result;
    }

    Array optimized_capsules_results_array;
    for (int i = 0; i < m_capsules.size(); ++i) {
        CapsuleInstance &capsule = m_capsules.write[i];
        int offset = i * 7;
        ERR_FAIL_COND_V_MSG(offset + 6 >= optimized_params.size(), actual_result_dict, "Offset calculation error during capsule result processing.");

        capsule.optimized_radius = optimized_params[offset + 0];
        capsule.optimized_axis_a = Vector3(optimized_params[offset + 1], optimized_params[offset + 2], optimized_params[offset + 3]);
        capsule.optimized_axis_b = Vector3(optimized_params[offset + 4], optimized_params[offset + 5], optimized_params[offset + 6]);

        Dictionary capsule_result_dict;
        capsule_result_dict["optimized_radius"] = capsule.optimized_radius;
        capsule_result_dict["optimized_axis_a"] = capsule.optimized_axis_a;
        capsule_result_dict["optimized_axis_b"] = capsule.optimized_axis_b;
        optimized_capsules_results_array.push_back(capsule_result_dict);
    }
    actual_result_dict["optimized_capsules_results"] = optimized_capsules_results_array;

    last_fit_result = actual_result_dict;
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
        return {surface_point, normal_on_surface};
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
    return {closest_point_on_surface, normal_on_surface};
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

PackedVector3Array LBFGSBCapsuleFitterSolver::_generate_canonical_capsule_points(const Vector3 &p_cap_a, const Vector3 &p_cap_b, double p_cap_radius, int p_cylinder_rings, int p_radial_segments) const {
    PackedVector3Array points;
    return points;
}
