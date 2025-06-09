/**************************************************************************/
/*  lbfgsbpp.cpp                                                          */
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

#include "lbfgsbpp.h"

#include "thirdparty/LBFGSpp/include/LBFGSB.h"
#include "thirdparty/LBFGSpp/include/LBFGSpp/Param.h"
#include "thirdparty/eigen/Eigen/Core"

#include <functional>
#include <limits>

// Anonymous namespace for file-static helper functions
namespace {
static Eigen::VectorXd godot_to_eigen(const PackedFloat64Array &p_packed_array) {
	if (p_packed_array.is_empty()) {
		return Eigen::VectorXd(0);
	}
	Eigen::VectorXd vector(p_packed_array.size());
	for (int i = 0; i < p_packed_array.size(); ++i) {
		vector[i] = p_packed_array[i];
	}
	return vector;
}

static PackedFloat64Array eigen_to_godot(const Eigen::VectorXd &p_vector) {
	if (p_vector.size() == 0) {
		return PackedFloat64Array();
	}
	int size = p_vector.size();
	PackedFloat64Array array;
	array.resize(size);
	for (int i = 0; i < size; ++i) {
		array.write[i] = p_vector[i]; // Corrected to use .write[]
	}
	return array;
}
} // end anonymous namespace

struct LBFGSBSolver::PImpl {
	LBFGSpp::LBFGSBParam<double> param;
	LBFGSpp::LBFGSBSolver<double> solver;
	std::function<double(const Eigen::VectorXd &, Eigen::VectorXd &)> native_eval_func;

	PImpl() :
			solver(param) {}
};

static double static_native_operator_wrapper(LBFGSBSolver *solver_instance, const Eigen::VectorXd &r_x, Eigen::VectorXd &r_gradient) {
	ERR_FAIL_NULL_V_MSG(solver_instance, std::numeric_limits<double>::quiet_NaN(), "LBFGSBSolver instance is null in static_native_operator_wrapper.");

	PackedFloat64Array x_godot = eigen_to_godot(r_x); // Now calls file-static version
	PackedFloat64Array gradient_godot;
	gradient_godot.resize(r_gradient.size());

	double fx = solver_instance->call_operator(x_godot, gradient_godot);

	if (std::isnan(fx)) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	Eigen::VectorXd updated_grad_eigen = godot_to_eigen(gradient_godot); // Now calls file-static version
	ERR_FAIL_COND_V_MSG(updated_grad_eigen.size() != r_gradient.size(), std::numeric_limits<double>::quiet_NaN(),
			"Gradient size mismatch after _call_operator.");
	r_gradient = updated_grad_eigen;
	return fx;
}

Array LBFGSBSolver::minimize(const PackedFloat64Array &p_x,
		const double &p_fx_initial_guess, const PackedFloat64Array &p_lower_bound, const PackedFloat64Array &p_upper_bound) {
	Array ret;
	ret.push_back(0);
	ret.push_back(PackedFloat64Array());
	ret.push_back(0.0);

	ERR_FAIL_COND_V_MSG(p_x.is_empty(), ret, "Initial parameters (p_x) cannot be empty.");
	ERR_FAIL_COND_V_MSG(p_lower_bound.is_empty(), ret, "Lower bounds cannot be empty.");
	ERR_FAIL_COND_V_MSG(p_upper_bound.is_empty(), ret, "Upper bounds cannot be empty.");
	ERR_FAIL_COND_V_MSG(p_x.size() != p_lower_bound.size() || p_lower_bound.size() != p_upper_bound.size(), ret,
			"Parameter and bound arrays must have the same size.");

	ERR_FAIL_NULL_V_MSG(pimpl, ret, "LBFGSBSolver PImpl not initialized.");

	pimpl->param.epsilon = epsilon;
	pimpl->param.max_iterations = max_iterations;

	Eigen::VectorXd x = godot_to_eigen(p_x); // Now calls file-static version
	Eigen::VectorXd lower_bound = godot_to_eigen(p_lower_bound); // Now calls file-static version
	Eigen::VectorXd upper_bound = godot_to_eigen(p_upper_bound); // Now calls file-static version

	double fx = p_fx_initial_guess;

	if (!pimpl->native_eval_func) { // Should ideally be set in constructor only
		pimpl->native_eval_func = [this](const Eigen::VectorXd &x_eigen, Eigen::VectorXd &grad_eigen) -> double {
			return static_native_operator_wrapper(this, x_eigen, grad_eigen);
		};
	}

	int number_of_iterations = pimpl->solver.minimize(pimpl->native_eval_func, x, fx, lower_bound, upper_bound);

	ret.clear();
	ret.push_back(number_of_iterations);
	ret.push_back(eigen_to_godot(x)); // Now calls file-static version
	ret.push_back(fx);
	return ret;
}

LBFGSBSolver::LBFGSBSolver() {
	pimpl = memnew(LBFGSBSolver::PImpl);
	// Use a lambda to capture 'this' and pass it to the static wrapper function.
	pimpl->native_eval_func = [this](const Eigen::VectorXd &x_eigen, Eigen::VectorXd &grad_eigen) -> double {
		return static_native_operator_wrapper(this, x_eigen, grad_eigen);
	};
	epsilon = 1e-6;
	max_iterations = 100;
}

LBFGSBSolver::~LBFGSBSolver() {
	if (pimpl) {
		memdelete(pimpl);
		pimpl = nullptr;
	}
}

void LBFGSBSolver::_bind_methods() {
	GDVIRTUAL_BIND(_call_operator, "x", "gradient");
	ClassDB::bind_method(D_METHOD("minimize", "x", "fx", "lower_bound", "upper_bound"), &LBFGSBSolver::minimize);

	ClassDB::bind_method(D_METHOD("set_epsilon", "p_epsilon"), &LBFGSBSolver::set_epsilon);
	ClassDB::bind_method(D_METHOD("get_epsilon"), &LBFGSBSolver::get_epsilon);
	ClassDB::add_property("LBFGSBSolver", PropertyInfo(Variant::FLOAT, "epsilon"), "set_epsilon", "get_epsilon");

	ClassDB::bind_method(D_METHOD("set_max_iterations", "p_iterations"), &LBFGSBSolver::set_max_iterations);
	ClassDB::bind_method(D_METHOD("get_max_iterations"), &LBFGSBSolver::get_max_iterations);
	ClassDB::add_property("LBFGSBSolver", PropertyInfo(Variant::INT, "max_iterations"), "set_max_iterations", "get_max_iterations");
}

double LBFGSBSolver::call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_gradient) {
	double ret = 0;
	if (!get_script_instance() || !get_script_instance()->has_method("_call_operator")) {
		WARN_PRINT("LBFGSBSolver: _call_operator method not implemented in script. Returning NaN. This may lead to unexpected behavior.");
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (GDVIRTUAL_CALL(_call_operator, p_x, r_gradient, ret)) {
		return ret;
	};
	WARN_PRINT("LBFGSBSolver: Failed to call _call_operator in script. Returning NaN.");
	return std::numeric_limits<double>::quiet_NaN();
}

void LBFGSBSolver::set_epsilon(double p_epsilon) {
	epsilon = p_epsilon;
}

double LBFGSBSolver::get_epsilon() const {
	return epsilon;
}

void LBFGSBSolver::set_max_iterations(int p_iterations) {
	max_iterations = p_iterations;
}

int LBFGSBSolver::get_max_iterations() const {
	return max_iterations;
}
