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
#include "core/error/error_macros.h"

#include <cstdint>
#include <libTimekeeper/StopWatchPrinting.hh>
#include <libsatsuma/Extra/Highlevel.hh>
#include <libsatsuma/Problems/BiMDF.hh>

void MinimumDeviationFlow::_bind_methods() {
	ClassDB::bind_static_method("MinimumDeviationFlow", D_METHOD("add_node", "state"), &MinimumDeviationFlow::add_node);
	ClassDB::bind_static_method("MinimumDeviationFlow", D_METHOD("add_edge_abs", "state", "u", "v", "target", "weight", "lower", "upper", "u_head", "v_head"), &MinimumDeviationFlow::add_edge_abs);
	ClassDB::bind_static_method("MinimumDeviationFlow", D_METHOD("add_edge_quad", "state", "u", "v", "target", "weight", "lower", "upper", "u_head", "v_head"), &MinimumDeviationFlow::add_edge_quad);
	ClassDB::bind_static_method("MinimumDeviationFlow", D_METHOD("add_edge_zero", "state", "u", "v", "lower", "upper", "u_head", "v_head"), &MinimumDeviationFlow::add_edge_zero);
	ClassDB::bind_static_method("MinimumDeviationFlow", D_METHOD("solve", "state"), &MinimumDeviationFlow::solve);
}

int MinimumDeviationFlow::add_node(Ref<MinimumDeviationFlowState> p_state) {
	ERR_FAIL_COND_V_MSG(p_state.is_null(), -1, "State is null. Cannot add node.");
	p_state->nodes.push_back(p_state->bimdf.add_node());
	return p_state->nodes.size() - 1;
}

void MinimumDeviationFlow::add_edge_abs(Ref<MinimumDeviationFlowState> p_state, int p_u, int p_v, double p_target, double p_weight, int p_lower, int p_upper, bool p_u_head, bool p_v_head) {
	ERR_FAIL_COND_MSG(p_state.is_null(), "State is null. Cannot add edge with absolute deviation.");
	ERR_FAIL_COND_MSG(p_lower >= p_upper, "Invalid bounds: Lower bound cannot be greater than or equal to upper bound.");
	ERR_FAIL_COND_MSG(p_lower < 0, "Invalid bounds: Lower bound cannot be negative.");
	using Abs = Satsuma::CostFunction::AbsDeviation;
	Satsuma::BiMDF::Edge edge = p_state->bimdf.add_edge({ .u = p_state->nodes[p_u],
			.v = p_state->nodes[p_v],
			.u_head = p_u_head,
			.v_head = p_v_head,
			.cost_function = Abs{ .target = p_target, .weight = p_weight },
			.lower = static_cast<int>(p_lower),
			.upper = static_cast<int>(p_upper) });
	p_state->edges[std::to_string(p_u) + "_" + std::to_string(p_v)] = edge;
}

void MinimumDeviationFlow::add_edge_quad(Ref<MinimumDeviationFlowState> p_state, int p_u, int p_v, double p_target, double p_weight, int p_lower, int p_upper, bool p_u_head, bool p_v_head) {
	ERR_FAIL_COND_MSG(p_state.is_null(), "State is null. Cannot add edge with quadratic deviation.");
	ERR_FAIL_COND_MSG(p_lower >= p_upper, "Invalid bounds: Lower bound cannot be greater than or equal to upper bound.");
	ERR_FAIL_COND_MSG(p_lower < 0, "Invalid bounds: Lower bound cannot be negative.");
	using Quad = Satsuma::CostFunction::QuadDeviation;
	Satsuma::BiMDF::Edge edge = p_state->bimdf.add_edge({ .u = p_state->nodes[p_u],
			.v = p_state->nodes[p_v],
			.u_head = p_u_head,
			.v_head = p_v_head,
			.cost_function = Quad{ .target = p_target, .weight = p_weight },
			.lower = static_cast<int>(p_lower),
			.upper = static_cast<int>(p_upper) });
	p_state->edges[std::to_string(p_u) + "_" + std::to_string(p_v)] = edge;
}

void MinimumDeviationFlow::add_edge_zero(Ref<MinimumDeviationFlowState> p_state, int p_u, int p_v, int p_lower, int p_upper, bool p_u_head, bool p_v_head) {
	ERR_FAIL_COND_MSG(p_state.is_null(), "State is null. Cannot add edge with zero deviation.");
	ERR_FAIL_COND_MSG(p_lower >= p_upper, "Invalid bounds: Lower bound cannot be greater than or equal to upper bound.");
	ERR_FAIL_COND_MSG(p_lower < 0, "Invalid bounds: Lower bound cannot be negative.");
	using Zero = Satsuma::CostFunction::Zero;
	Satsuma::BiMDF::Edge edge = p_state->bimdf.add_edge({ .u = p_state->nodes[p_u],
			.v = p_state->nodes[p_v],
			.u_head = p_u_head,
			.v_head = p_v_head,
			.cost_function = Zero{},
			.lower = static_cast<int>(p_lower),
			.upper = static_cast<int>(p_upper) });
	p_state->edges[std::to_string(p_u) + "_" + std::to_string(p_v)] = edge;
}

void MinimumDeviationFlow::solve(Ref<MinimumDeviationFlowState> p_state) {
	ERR_FAIL_COND_MSG(p_state.is_null(), "State is null. Cannot solve BIMDF.");
	Satsuma::BiMDFSolverConfig config = Satsuma::BiMDFSolverConfig{
		.double_cover = Satsuma::BiMDFDoubleCoverConfig(),
		.matching_solver = Satsuma::MatchingSolver::Lemon,
	};
	try {
		Satsuma::BiMDFFullResult result = Satsuma::solve_bimdf(p_state->bimdf, config);
		print_solution(result, p_state->edges);
	} catch (const std::exception &e) {
		print_line(e.what());
	}
}

void MinimumDeviationFlow::print_solution(const Satsuma::BiMDFFullResult &p_result, const std::map<std::string, Satsuma::BiMDF::Edge> &p_edges) {
	std::string output = format_solution(p_result, p_edges);
	print_line(output.c_str());
}

std::string MinimumDeviationFlow::format_solution(const Satsuma::BiMDFFullResult &p_result, const std::map<std::string, Satsuma::BiMDF::Edge> &p_edges) {
	std::ostringstream buffer;
	buffer << "Total cost: " << p_result.cost << "\n";
	for (const std::pair<const std::string, Satsuma::BiMDF::Edge> &entry : p_edges) {
		const std::string &name = entry.first;
		const Satsuma::BiMDF::Edge &edge = entry.second;
		buffer << "Flow on " << name << ": " << (*p_result.solution)[edge] << "\n";
	}
	buffer << p_result.stopwatch << std::endl;
	return buffer.str();
}

MinimumDeviationFlow::MinimumDeviationFlow() {
}
