/**************************************************************************/
/*  lbfgsbpp.h                                                            */
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

#ifndef LBFGSBPP_H
#define LBFGSBPP_H

#include "core/error/error_macros.h"
#include "core/object/gdvirtual.gen.inc"
#include "core/object/ref_counted.h"
#include "core/object/script_language.h"

#include "../thirdparty/LBFGSpp/include/LBFGSB.h"
#include "scene/3d/node_3d.h"
#include "thirdparty/eigen/Eigen/Core"

#include <functional>
#include <iostream>

class LBFGSBSolver : public RefCounted {
	GDCLASS(LBFGSBSolver, RefCounted);

protected:
	static void _bind_methods();
	GDVIRTUAL2R(double, _call_operator, const TypedArray<double> &, TypedArray<double>);

	std::function<double(const Eigen::VectorXd &p_x, Eigen::VectorXd &r_grad)> operator_pointer;

	static Eigen::VectorXd godot_to_eigen(const TypedArray<double> &p_array);
	static TypedArray<double> eigen_to_godot(const Eigen::VectorXd &p_vector);
	double operator_call(const TypedArray<double> &p_x, TypedArray<double> &r_grad);
	double native_operator(const Eigen::VectorXd &r_x, Eigen::VectorXd &r_grad);
public:
	LBFGSBSolver();
	double call_operator(const TypedArray<double> &p_x, TypedArray<double> &r_grad);
	Array minimize(const TypedArray<double> &p_x,
			const double &p_fx, const TypedArray<double> &p_lower_bound, const TypedArray<double> &p_upper_bound);
};

#endif // LBFGSBPP_H
