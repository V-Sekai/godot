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

#pragma once

#include "core/error/error_macros.h"
#include "core/object/gdvirtual.gen.inc"
#include "core/object/ref_counted.h"
#include "core/object/script_language.h"

#include "scene/3d/node_3d.h"

class LBFGSBSolver : public Node3D {
	GDCLASS(LBFGSBSolver, Node3D);

private:
	double epsilon;
	int max_iterations;

	struct PImpl;
	PImpl *pimpl = nullptr;

protected:
	static void _bind_methods();
	GDVIRTUAL2R(double, _call_operator, const PackedFloat64Array &, PackedFloat64Array);

public:
	LBFGSBSolver();
	~LBFGSBSolver();

	void set_epsilon(double p_epsilon);
	double get_epsilon() const;

	void set_max_iterations(int p_iterations);
	int get_max_iterations() const;

	virtual double call_operator(const PackedFloat64Array &p_x, PackedFloat64Array &r_grad);
	Array minimize(const PackedFloat64Array &p_x,
			const double &p_fx, const PackedFloat64Array &p_lower_bound, const PackedFloat64Array &p_upper_bound);
};
