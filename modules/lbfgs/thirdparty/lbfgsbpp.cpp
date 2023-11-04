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
#include "LBFGSpp/Param.h"

Array LBFGSBSolver::minimize(const TypedArray<double> &p_x,
		const double &p_fx, const TypedArray<double> &p_lower_bound, const TypedArray<double> &p_upper_bound) {
	Array ret;
	ret.push_back(0);
	ret.push_back(TypedArray<double>());
	ret.push_back(0);
	ERR_FAIL_COND_V(!p_x.size(), ret);
	ERR_FAIL_COND_V(!p_lower_bound.size(), ret);
	ERR_FAIL_COND_V(!p_upper_bound.size(), ret);
	ERR_FAIL_COND_V(p_x.size() != p_lower_bound.size() || p_lower_bound.size() != p_upper_bound.size(), ret);
	LBFGSpp::LBFGSBParam<double> param;
	param.epsilon = 1e-6;
	param.max_iterations = 100;
	LBFGSpp::LBFGSBSolver<double> solver(param);
	Eigen::VectorXd lower_bound = godot_to_eigen(p_lower_bound);

	Eigen::VectorXd x = godot_to_eigen(p_x);
	Eigen::VectorXd upper_bound = godot_to_eigen(p_upper_bound);

	double fx = p_fx;
	int number_of_iterations = solver.minimize(operator_pointer, x, fx, lower_bound, upper_bound);
	ret.clear();
	ret.push_back(number_of_iterations);
	ret.push_back(eigen_to_godot(x));
	ret.push_back(fx);
	return ret;
}

LBFGSBSolver::LBFGSBSolver() {
	operator_pointer = std::bind(&LBFGSBSolver::native_operator, this, std::placeholders::_1, std::placeholders::_2);
}

void LBFGSBSolver::_bind_methods() {
	GDVIRTUAL_BIND(_call_operator, "x", "gradient");
	ClassDB::bind_method(D_METHOD("minimize", "x", "fx", "lower_bound", "upper_bound"), &LBFGSBSolver::minimize);
}

double LBFGSBSolver::native_operator(const Eigen::VectorXd &r_x, Eigen::VectorXd &r_gradient) {
	TypedArray<double> x = LBFGSBSolver::eigen_to_godot(r_x);
	TypedArray<double> gradient = LBFGSBSolver::eigen_to_godot(r_gradient);
	Eigen::VectorXd vec(r_gradient.size());
	for (int i = 0; i < r_gradient.size(); ++i) {
		vec[i] = r_gradient[i];
	}
	double fx = call_operator(x, gradient);
	r_gradient = godot_to_eigen(gradient);
	return fx;
}

double LBFGSBSolver::call_operator(const TypedArray<double> &p_x, TypedArray<double> &r_gradient) {
	double ret = 0;
	if (GDVIRTUAL_CALL(_call_operator, p_x, r_gradient, ret)) {
		return ret;
	};
	return 0;
}

Eigen::VectorXd LBFGSBSolver::godot_to_eigen(const TypedArray<double> &p_packed_array) {
	ERR_FAIL_COND_V(!p_packed_array.size(), Eigen::VectorXd());
	Eigen::VectorXd vector(p_packed_array.size());
	for (int i = 0; i < p_packed_array.size(); ++i) {
		vector[i] = p_packed_array[i];
	}
	return vector;
}

TypedArray<double> LBFGSBSolver::eigen_to_godot(const Eigen::VectorXd &p_vector) {
	ERR_FAIL_COND_V(!p_vector.size(), TypedArray<double>());
	int size = p_vector.size();
	TypedArray<double> array;
	array.resize(size);
	for (int i = 0; i < size; ++i) {
		array[i] = p_vector[i];
	}
	return array;
};
