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

	ClassDB::bind_method(D_METHOD("extract_plan"), &PlannerResult::extract_plan);
	ClassDB::bind_method(D_METHOD("find_failed_nodes"), &PlannerResult::find_failed_nodes);
	ClassDB::bind_method(D_METHOD("get_node", "node_id"), &PlannerResult::get_node);
	ClassDB::bind_method(D_METHOD("get_all_nodes"), &PlannerResult::get_all_nodes);
	ClassDB::bind_method(D_METHOD("has_node", "node_id"), &PlannerResult::has_node);
}

Array PlannerResult::extract_plan() const {
	// Reconstruct PlannerSolutionGraph from the stored graph Dictionary
	PlannerSolutionGraph graph;
	// Copy the stored graph Dictionary into the PlannerSolutionGraph
	// get_graph() returns a reference, so we can assign directly
	// The stored solution_graph Dictionary already contains all nodes including root
	graph.get_graph() = solution_graph.duplicate();

	// Extract the plan using the existing graph operations
	return PlannerGraphOperations::extract_solution_plan(graph);
}

Array PlannerResult::find_failed_nodes() const {
	Array failed_nodes;
	Array graph_keys = solution_graph.keys();
	
	for (int i = 0; i < graph_keys.size(); i++) {
		int node_id = graph_keys[i];
		Dictionary node = solution_graph[node_id];
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
	Array graph_keys = solution_graph.keys();
	
	for (int i = 0; i < graph_keys.size(); i++) {
		int node_id = graph_keys[i];
		Dictionary node = solution_graph[node_id];
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
