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

#include "thirdparty/libsatsuma/src/libsatsuma/Extra/Highlevel.hh"
#include <libTimekeeper/StopWatchPrinting.hh>
#include <libsatsuma/Problems/BiMDF.hh>
#include <map>

std::string format_solution(const Satsuma::BiMDFFullResult &result, const std::map<std::string, Satsuma::BiMDF::Edge> &edges) {
	std::ostringstream buffer;
	buffer << "Total cost: " << result.cost << "\n";
	for (const auto &[name, edge] : edges) {
		buffer << "Flow on " << name << ": " << (*result.solution)[edge] << "\n";
	}
	buffer << result.stopwatch << std::endl;
	return buffer.str();
}

void print_solution(const Satsuma::BiMDFFullResult &result, const std::map<std::string, Satsuma::BiMDF::Edge> &edges) {
	std::string output = format_solution(result, edges);
	print_line(output.c_str());
}

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
	Satsuma::BiMDFFullResult result = Satsuma::solve_bimdf(bimdf, config);
	print_solution(result, edges);
}

// https://developers.google.com/optimization/flow/mincostflow

// Minimum Cost Flows

// Closely related to the max flow problem is the minimum cost (min cost) flow problem, in which each arc in the graph has a unit cost for transporting material across it. The problem is to find a flow with the least total cost.

// The min cost flow problem also has special nodes, called supply nodes or demand nodes, which are similar to the source and sink in the max flow problem. Material is transported from supply nodes to demand nodes.

//     At a supply node, a positive amount — the supply — is added to the flow. A supply could represent production at that node, for example.
//     At a demand node, a negative amount — the demand — is taken away from the flow. A demand could represent consumption at that node, for example.

// For convenience, we'll assume that all nodes, other than supply or demand nodes, have zero supply (and demand).

// For the min cost flow problem, we have the following flow conservation rule, which takes the supplies and demands into account:
// Note: At each node, the total flow leading out of the node minus the total flow leading in to the node equals the supply (or demand) at that node.

// The graph below shows a min cost flow problem. The arcs are labeled with pairs of numbers: the first number is the capacity and the second number is the cost. The numbers in parentheses next to the nodes represent supplies or demands. Node 0 is a supply node with supply 20, while nodes 3 and 4 are demand nodes, with demands -5 and -15, respectively.

// network cost flow graph
// Import the libraries

// import numpy as np

// from ortools.graph.python import min_cost_flow

// Declare the solver

// # Instantiate a SimpleMinCostFlow solver.
// smcf = min_cost_flow.SimpleMinCostFlow()

// Define the data

// # Define four parallel arrays: sources, destinations, capacities,
// # and unit costs between each pair. For instance, the arc from node 0
// # to node 1 has a capacity of 15.
// start_nodes = np.array([0, 0, 1, 1, 1, 2, 2, 3, 4])
// end_nodes = np.array([1, 2, 2, 3, 4, 3, 4, 4, 2])
// capacities = np.array([15, 8, 20, 4, 10, 15, 4, 20, 5])
// unit_costs = np.array([4, 4, 2, 2, 6, 1, 3, 2, 3])

// # Define an array of supplies at each node.
// supplies = [20, 0, 0, -5, -15]

// Add the arcs

// For each start node and end node, we create an arc from start node to end node with the given capacity and unit cost, using the method AddArcWithCapacityAndUnitCost.

// The solver's SetNodeSupply method creates a vector of supplies for the nodes.

// # Add arcs, capacities and costs in bulk using numpy.
// all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
//     start_nodes, end_nodes, capacities, unit_costs
// )

// # Add supply for each nodes.
// smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)
