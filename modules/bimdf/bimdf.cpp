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

void MinimumDeviationFlow::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_node", "name"), &MinimumDeviationFlow::add_node);
	ClassDB::bind_method(D_METHOD("add_edge_abs", "u", "v", "target", "weight", "lower", "upper", "u_head", "v_head"), &MinimumDeviationFlow::add_edge_abs);
	ClassDB::bind_method(D_METHOD("add_edge_quad",  "u", "v", "target", "weight", "lower", "upper", "u_head", "v_head"), &MinimumDeviationFlow::add_edge_quad);
	ClassDB::bind_method(D_METHOD("add_edge_zero", "u", "v", "lower", "upper", "u_head", "v_head"), &MinimumDeviationFlow::add_edge_zero);
	ClassDB::bind_method(D_METHOD("solve"), &MinimumDeviationFlow::solve);
	ClassDB::bind_method(D_METHOD("clear"), &MinimumDeviationFlow::clear);
}

void MinimumDeviationFlow::solve() {
	Satsuma::BiMDFSolverConfig config = Satsuma::BiMDFSolverConfig{
		.double_cover = Satsuma::BiMDFDoubleCoverConfig(),
		.matching_solver = Satsuma::MatchingSolver::Lemon,
	};
	try {
		Satsuma::BiMDFFullResult result = Satsuma::solve_bimdf(bimdf, config);
		print_solution(result, edges);
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
	buffer << "====================\n";
	buffer << "Solution Summary\n";
	buffer << "====================\n";
	buffer << "Total cost: " << p_result.cost << "\n\n";
	buffer << "Edge Flows:\n";
	buffer << "--------------------\n";
	for (const auto &entry : p_edges) {
		const std::string &name = entry.first;
		String godot_name = String(name.c_str());
		buffer << "Edge " << name << ":\n";
		buffer << "  Flow: " << (*p_result.solution)[entry.second] << "\n";
		buffer << "  Lower bound: " << edge_lowers[godot_name] << "\n";
		buffer << "  Upper bound: " << edge_uppers[godot_name] << "\n";
		buffer << "  Weight: " << edge_weights[godot_name] << "\n";
		buffer << "  Parent node: " << node_names[edge_nodes[godot_name].first].utf8().get_data() << "\n";
		buffer << "  Next node: " << node_names[edge_nodes[godot_name].second].utf8().get_data() << "\n";
		buffer << "--------------------\n";
	}
	buffer << "\nStopwatch:\n";
	buffer << "--------------------\n";
	buffer << "graph TD\n";
	for (const auto &entry : edge_nodes) {
		const String &edge_key = entry.key;
		int u = entry.value.first;
		int v = entry.value.second;
		double flow = (*p_result.solution)[edges[edge_key.utf8().get_data()]];
		buffer << "    " << node_names[u].utf8().get_data() << " -->|" << edge_key.utf8().get_data() << " (" << flow << ")| " << node_names[v].utf8().get_data() << "\n";
	}
	buffer << p_result.stopwatch << "\n";
	buffer << "====================\n";
	return buffer.str();
}

int MinimumDeviationFlow::add_node(const String &name) {
	nodes.push_back(bimdf.add_node());
	int node_index = nodes.size() - 1;
	node_names[node_index] = name;
	return node_index;
}
void MinimumDeviationFlow::add_edge_abs(int p_u, int p_v, double p_target, double p_weight, int p_lower, int p_upper, bool p_u_head, bool p_v_head) {
	using Abs = Satsuma::CostFunction::AbsDeviation;
	Satsuma::BiMDF::Edge edge = bimdf.add_edge({ .u = nodes[p_u],
			.v = nodes[p_v],
			.u_head = p_u_head,
			.v_head = p_v_head,
			.cost_function = Abs{ .target = p_target, .weight = p_weight },
			.lower = static_cast<int>(p_lower),
			.upper = static_cast<int>(p_upper) });
	String edge_key = node_names[p_u] + "_" + node_names[p_v];
	edges[edge_key.utf8().get_data()] = edge;
	edge_nodes[edge_key] = { p_u, p_v };
	edge_targets[edge_key] = p_target;
	edge_weights[edge_key] = p_weight;
	edge_lowers[edge_key] = p_lower;
	edge_uppers[edge_key] = p_upper;
	edge_u_heads[edge_key] = p_u_head;
	edge_v_heads[edge_key] = p_v_head;
}

void MinimumDeviationFlow::add_edge_quad(int p_u, int p_v, double p_target, double p_weight, int p_lower, int p_upper, bool p_u_head, bool p_v_head) {
	using Quad = Satsuma::CostFunction::QuadDeviation;
	Satsuma::BiMDF::Edge edge = bimdf.add_edge({ .u = nodes[p_u],
			.v = nodes[p_v],
			.u_head = p_u_head,
			.v_head = p_v_head,
			.cost_function = Quad{ .target = p_target, .weight = p_weight },
			.lower = static_cast<int>(p_lower),
			.upper = static_cast<int>(p_upper) });
	String edge_key = node_names[p_u] + "_" + node_names[p_v];
	edges[edge_key.utf8().get_data()] = edge;
	edge_nodes[edge_key] = { p_u, p_v };
	edge_targets[edge_key] = p_target;
	edge_weights[edge_key] = p_weight;
	edge_lowers[edge_key] = p_lower;
	edge_uppers[edge_key] = p_upper;
	edge_u_heads[edge_key] = p_u_head;
	edge_v_heads[edge_key] = p_v_head;
}

void MinimumDeviationFlow::add_edge_zero(int p_u, int p_v, int p_lower, int p_upper, bool p_u_head, bool p_v_head) {
	using Zero = Satsuma::CostFunction::Zero;
	Satsuma::BiMDF::Edge edge = bimdf.add_edge({ .u = nodes[p_u],
			.v = nodes[p_v],
			.u_head = p_u_head,
			.v_head = p_v_head,
			.cost_function = Zero{},
			.lower = static_cast<int>(p_lower),
			.upper = static_cast<int>(p_upper) });
	String edge_key = node_names[p_u] + "_" + node_names[p_v];
	edges[edge_key.utf8().get_data()] = edge;
	edge_nodes[edge_key] = { p_u, p_v };
	edge_lowers[edge_key] = p_lower;
	edge_uppers[edge_key] = p_upper;
	edge_u_heads[edge_key] = p_u_head;
	edge_v_heads[edge_key] = p_v_head;
}

const std::map<std::string, Satsuma::BiMDF::Edge> &MinimumDeviationFlow::get_edges() const {
	return edges;
}
const std::vector<Satsuma::BiMDF::Node> &MinimumDeviationFlow::get_nodes() const {
	return nodes;
}
const  HashMap<int, String> &MinimumDeviationFlow::get_node_names() const {
	return node_names;
}
const  HashMap<String, Pair<int, int>> &MinimumDeviationFlow::get_edge_nodes() const {
	return edge_nodes;
}
const  HashMap<String, double> &MinimumDeviationFlow::get_edge_targets() const {
	return edge_targets;
}
const  HashMap<String, double> &MinimumDeviationFlow::get_edge_weights() const {
	return edge_weights;
}
const  HashMap<String, int> &MinimumDeviationFlow::get_edge_lowers() const {
	return edge_lowers;
}
const  HashMap<String, int> &MinimumDeviationFlow::get_edge_uppers() const {
	return edge_uppers;
}
const  HashMap<String, bool> &MinimumDeviationFlow::get_edge_u_heads() const {
	return edge_u_heads;
}
const  HashMap<String, bool> &MinimumDeviationFlow::get_edge_v_heads() const {
	return edge_v_heads;
}

void MinimumDeviationFlow::clear() {
	node_names.clear();
	edge_nodes.clear();
	edge_targets.clear();
	edge_weights.clear();
	edge_lowers.clear();
	edge_uppers.clear();
	edge_u_heads.clear();
	edge_v_heads.clear();
	edges.clear();
	nodes.clear();
}
