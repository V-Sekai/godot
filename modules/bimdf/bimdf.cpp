/**************************************************************************/
/*  bimdf.cpp                                                             */
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

#include "bimdf.h"

#include <libTimekeeper/StopWatchPrinting.hh>
#include <libsatsuma/Extra/Highlevel.hh>
#include <libsatsuma/Problems/BiMDF.hh>
#include <map>

void BIMDF::solve() {
	using BiMDF = Satsuma::BiMDF;
	BiMDF bimdf;
	std::map<std::string, BiMDF::Edge> edges;

	auto x = bimdf.add_node(); // boundary
	auto a = bimdf.add_node();
	auto b = bimdf.add_node();
	auto c = bimdf.add_node();

	using Abs = Satsuma::CostFunction::AbsDeviation;
	using Quad = Satsuma::CostFunction::QuadDeviation;
	using Zero = Satsuma::CostFunction::Zero;
	edges["x_x"] = bimdf.add_edge({
			.u = x,
			.v = x,
			.u_head = false,
			.v_head = false,
			.cost_function = Zero{},
	});
	edges["x_a"] = bimdf.add_edge({ .u = x, .v = a, .u_head = true, .v_head = true, .cost_function = Quad{ .target = 4, .weight = 1 }, .lower = 1 });
	edges["a_b"] = bimdf.add_edge({ .u = a, .v = b, .u_head = false, .v_head = false, .cost_function = Abs{ .target = .7, .weight = 1 }, .lower = 0 });
	edges["a_c"] = bimdf.add_edge({ .u = a, .v = c, .u_head = false, .v_head = false, .cost_function = Abs{ .target = .4, .weight = 1 }, .lower = 0 });
	edges["b_c"] = bimdf.add_edge({ .u = b, .v = c, .u_head = true, .v_head = true, .cost_function = Abs{ .target = .2, .weight = 1 }, .lower = 0 });

	auto config = Satsuma::BiMDFSolverConfig{
		.double_cover = Satsuma::BiMDFDoubleCoverConfig(),
		.matching_solver = Satsuma::MatchingSolver::Lemon,
	};
	auto result = Satsuma::solve_bimdf(bimdf, config);
	std::ostringstream buffer;

	buffer << "Total cost: " << result.cost << "\n";
	for (const auto &[name, edge] : edges) {
		buffer << "Flow on " << name << ": " << (*result.solution)[edge] << "\n";
	}
	buffer << result.stopwatch << std::endl;
	std::string output = buffer.str();
	print_line(output.c_str());
}
