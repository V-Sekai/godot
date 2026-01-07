/**************************************************************************/
/*  planner_result.cpp                                                    */
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

#include "planner_result.h"

#include "core/object/class_db.h"
#include "graph_operations.h"
#include "solution_graph.h"

PlannerResult::PlannerResult() :
		success(false) {
}

void PlannerResult::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_final_state"), &PlannerResult::get_final_state);
	ClassDB::bind_method(D_METHOD("set_final_state", "state"), &PlannerResult::set_final_state);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "final_state"), "set_final_state", "get_final_state");

	ClassDB::bind_method(D_METHOD("get_solution_graph"), &PlannerResult::get_solution_graph);
	ClassDB::bind_method(D_METHOD("set_solution_graph", "graph"), &PlannerResult::set_solution_graph);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "solution_graph"), "set_solution_graph", "get_solution_graph");

	ClassDB::bind_method(D_METHOD("get_success"), &PlannerResult::get_success);
	ClassDB::bind_method(D_METHOD("set_success", "success"), &PlannerResult::set_success);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "success"), "set_success", "get_success");

	ClassDB::bind_method(D_METHOD("extract_plan", "verbose"), &PlannerResult::extract_plan, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("find_failed_nodes"), &PlannerResult::find_failed_nodes);
	ClassDB::bind_method(D_METHOD("get_node", "node_id"), &PlannerResult::get_node);
	ClassDB::bind_method(D_METHOD("get_all_nodes"), &PlannerResult::get_all_nodes);
	ClassDB::bind_method(D_METHOD("has_node", "node_id"), &PlannerResult::has_node);
}

Array PlannerResult::extract_plan(int p_verbose) const {
	if (p_verbose >= 3) {
		// Flush output immediately
		fflush(stdout);
		fflush(stderr);
		print_line("[EXTRACT_PLAN] Starting extract_plan()");
		fflush(stdout);
	}

	// Safety check: ensure solution_graph is valid
	if (p_verbose >= 3) {
		print_line("[EXTRACT_PLAN] Checking if solution_graph is empty...");
		fflush(stdout);
	}
	if (solution_graph.is_empty()) {
		if (p_verbose >= 1) {
			print_line("[EXTRACT_PLAN] ERROR: solution_graph is empty");
		}
		return Array(); // Return empty plan if graph is empty
	}

	if (p_verbose >= 3) {
		print_line(vformat("[EXTRACT_PLAN] solution_graph has %d keys", solution_graph.keys().size()));
	}

	// Reconstruct PlannerSolutionGraph from the stored graph Dictionary
	PlannerSolutionGraph graph;
	// Copy the stored graph Dictionary into the PlannerSolutionGraph
	// get_graph() returns a reference, so we can assign directly
	// The stored solution_graph Dictionary already contains all nodes including root
	if (p_verbose >= 3) {
		print_line("[EXTRACT_PLAN] Duplicating solution_graph...");
	}
	Dictionary graph_dict = solution_graph.duplicate();
	if (p_verbose >= 3) {
		print_line(vformat("[EXTRACT_PLAN] Duplicated graph has %d keys", graph_dict.keys().size()));
	}

	// Safety check: ensure root node (0) exists
	if (!graph_dict.has(0)) {
		if (p_verbose >= 1) {
			print_line("[EXTRACT_PLAN] ERROR: Root node (0) missing from graph");
		}
		return Array(); // Return empty plan if root node is missing
	}

	if (p_verbose >= 3) {
		print_line("[EXTRACT_PLAN] Setting graph...");
	}
	graph.load_from_dictionary(graph_dict);

	// Safety check: ensure graph_dict is not empty before processing
	if (graph_dict.is_empty()) {
		if (p_verbose >= 1) {
			print_line("[EXTRACT_PLAN] WARNING: graph_dict is empty, returning empty plan");
		}
		return Array();
	}

	// Set next_node_id to prevent issues (find max node ID)
	int max_id = -1;
	Array graph_keys = graph_dict.keys();
	// Safety check: ensure graph_keys is not empty
	if (graph_keys.is_empty() || graph_keys.size() < 1) {
		if (p_verbose >= 1) {
			print_line("[EXTRACT_PLAN] WARNING: graph_keys is empty, returning empty plan");
		}
		return Array();
	}
	for (int i = 0; i < graph_keys.size(); i++) {
		Variant key = graph_keys[i];
		if (key.get_type() == Variant::INT) {
			int node_id = key;
			if (node_id > max_id) {
				max_id = node_id;
			}
		}
	}
	graph.set_next_node_id((max_id >= 0) ? (max_id + 1) : 1);
	if (p_verbose >= 3) {
		print_line(vformat("[EXTRACT_PLAN] Max node ID: %d, next_node_id: %d", max_id, graph.get_next_node_id()));
	}

	// Extract the plan using the existing graph operations
	if (p_verbose >= 3) {
		print_line("[EXTRACT_PLAN] Calling extract_solution_plan()...");
	}
	Array result = PlannerGraphOperations::extract_solution_plan(graph, p_verbose);
	if (p_verbose >= 3) {
		print_line(vformat("[EXTRACT_PLAN] extract_solution_plan() returned %d actions", result.size()));
	}
	return result;
}

Array PlannerResult::find_failed_nodes() const {
	Array failed_nodes;

	// Safety check: ensure solution_graph is not empty
	if (solution_graph.is_empty()) {
		return failed_nodes;
	}

	Array graph_keys = solution_graph.keys();

	// Safety check: ensure graph_keys is not empty
	if (graph_keys.is_empty() || graph_keys.size() < 1) {
		return failed_nodes;
	}

	for (int i = 0; i < graph_keys.size(); i++) {
		Variant key = graph_keys[i];
		if (key.get_type() != Variant::INT) {
			continue; // Skip invalid node IDs
		}
		int node_id = key;
		if (!solution_graph.has(node_id)) {
			continue; // Skip missing nodes
		}
		Dictionary node = solution_graph[node_id];
		// Safety check: ensure node is not empty
		if (node.is_empty()) {
			continue; // Skip empty nodes
		}
		if (node.has("status")) {
			int status = node["status"];
			if (status == static_cast<int>(PlannerNodeStatus::STATUS_FAILED)) {
				Dictionary node_info;
				node_info["node_id"] = node_id;
				node_info["type"] = node.get("type", Variant());
				node_info["info"] = node.get("info", Variant());
				failed_nodes.push_back(node_info);
			}
		}
	}

	return failed_nodes;
}

Dictionary PlannerResult::get_node(int p_node_id) const {
	if (!solution_graph.has(p_node_id)) {
		return Dictionary();
	}
	return solution_graph[p_node_id];
}

Array PlannerResult::get_all_nodes() const {
	Array nodes;

	// Safety check: ensure solution_graph is not empty
	if (solution_graph.is_empty()) {
		return nodes;
	}

	Array graph_keys = solution_graph.keys();

	// Safety check: ensure graph_keys is not empty
	if (graph_keys.is_empty() || graph_keys.size() < 1) {
		return nodes;
	}

	for (int i = 0; i < graph_keys.size(); i++) {
		Variant key = graph_keys[i];
		if (key.get_type() != Variant::INT) {
			continue; // Skip invalid node IDs
		}
		int node_id = key;
		if (!solution_graph.has(node_id)) {
			continue; // Skip missing nodes
		}
		Dictionary node = solution_graph[node_id];
		// Safety check: ensure node is not empty
		if (node.is_empty()) {
			continue; // Skip empty nodes
		}
		Dictionary node_info;
		node_info["node_id"] = node_id;
		node_info["type"] = node.get("type", Variant());
		node_info["status"] = node.get("status", Variant());
		node_info["info"] = node.get("info", Variant());
		node_info["tag"] = node.get("tag", Variant("new"));
		nodes.push_back(node_info);
	}

	return nodes;
}

bool PlannerResult::has_node(int p_node_id) const {
	return solution_graph.has(p_node_id);
}
