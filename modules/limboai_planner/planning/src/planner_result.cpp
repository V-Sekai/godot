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
#include "core/templates/hash_set.h"
#include "core/variant/variant.h"
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

	// Plan explanation and debugging methods
	ClassDB::bind_method(D_METHOD("explain_plan"), &PlannerResult::explain_plan);
	ClassDB::bind_method(D_METHOD("get_alternative_methods", "node_id"), &PlannerResult::get_alternative_methods);
	ClassDB::bind_method(D_METHOD("get_decision_path", "node_id"), &PlannerResult::get_decision_path);
	ClassDB::bind_method(D_METHOD("to_graph_json"), &PlannerResult::to_graph_json);
	ClassDB::bind_method(D_METHOD("get_node_explanation", "node_id"), &PlannerResult::get_node_explanation);
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

Dictionary PlannerResult::explain_plan() const {
	Dictionary explanation;
	explanation["success"] = success;
	explanation["total_nodes"] = solution_graph.size();

	// Count nodes by type and status
	Dictionary node_counts;
	int action_count = 0;
	int task_count = 0;
	int failed_count = 0;
	int closed_count = 0;

	Array all_nodes = get_all_nodes();
	for (int i = 0; i < all_nodes.size(); i++) {
		Dictionary node = all_nodes[i];
		int node_type = node.get("type", -1);
		int node_status = node.get("status", -1);

		if (node_type == static_cast<int>(PlannerNodeType::TYPE_COMMAND)) {
			action_count++;
		} else if (node_type == static_cast<int>(PlannerNodeType::TYPE_TASK)) {
			task_count++;
		}

		if (node_status == static_cast<int>(PlannerNodeStatus::STATUS_FAILED)) {
			failed_count++;
		} else if (node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
			closed_count++;
		}
	}

	node_counts["actions"] = action_count;
	node_counts["tasks"] = task_count;
	node_counts["failed"] = failed_count;
	node_counts["closed"] = closed_count;
	explanation["node_counts"] = node_counts;

	// Extract plan summary
	Array plan = extract_plan();
	explanation["plan_length"] = plan.size();
	explanation["plan_actions"] = plan;

	// Get decision points (nodes with alternatives)
	Array decision_points;
	for (int i = 0; i < all_nodes.size(); i++) {
		Dictionary node = all_nodes[i];
		int node_id = node.get("node_id", -1);
		if (node_id >= 0 && has_node(node_id)) {
			Dictionary full_node = get_node(node_id);
			if (full_node.has("decision_info")) {
				Dictionary decision_info = full_node["decision_info"];
				Variant alternatives_var = decision_info.get("alternatives", Array());
				if (alternatives_var.get_type() == Variant::ARRAY) {
					Array alternatives_array = alternatives_var;
					if (alternatives_array.size() > 0) {
						Dictionary decision_point;
						decision_point["node_id"] = node_id;
						decision_point["type"] = node.get("type", -1);
						decision_point["info"] = node.get("info", Variant());
						decision_point["selected_method"] = decision_info.get("selected_method_id", "");
						decision_point["selected_score"] = decision_info.get("selected_score", 0.0);
						decision_point["alternatives_count"] = decision_info.get("total_candidates", 0);
						decision_points.push_back(decision_point);
					}
				}
			}
		}
	}
	explanation["decision_points"] = decision_points;

	return explanation;
}

Array PlannerResult::get_alternative_methods(int p_node_id) const {
	Array alternatives;
	if (!has_node(p_node_id)) {
		return alternatives;
	}

	Dictionary node = get_node(p_node_id);
	if (node.has("decision_info")) {
		Dictionary decision_info = node["decision_info"];
		if (decision_info.has("alternatives")) {
			alternatives = decision_info["alternatives"];
		}
	}

	return alternatives;
}

Dictionary PlannerResult::get_decision_path(int p_node_id) const {
	Dictionary path;
	Array path_nodes;

	if (!has_node(p_node_id)) {
		return path;
	}

	// Traverse from node to root following parent relationships
	int current_id = p_node_id;
	HashSet<int> visited;

	while (current_id >= 0 && has_node(current_id) && !visited.has(current_id)) {
		visited.insert(current_id);
		Dictionary node = get_node(current_id);

		Dictionary path_node;
		path_node["node_id"] = current_id;
		path_node["type"] = node.get("type", -1);
		path_node["status"] = node.get("status", -1);
		path_node["info"] = node.get("info", Variant());

		// Add decision info if available
		if (node.has("decision_info")) {
			path_node["decision_info"] = node["decision_info"];
		}

		path_nodes.push_back(path_node);

		// Find parent by searching for nodes that have current_id as successor
		int parent_id = -1;
		Array graph_keys = solution_graph.keys();
		for (int i = 0; i < graph_keys.size(); i++) {
			Variant key = graph_keys[i];
			if (key.get_type() != Variant::INT) {
				continue;
			}
			int candidate_id = key;
			Dictionary candidate = solution_graph[candidate_id];
			if (candidate.has("successors")) {
				TypedArray<int> successors = candidate["successors"];
				for (int j = 0; j < successors.size(); j++) {
					Variant succ_variant = successors[j];
					int succ_id = succ_variant;
					if (succ_id == current_id) {
						parent_id = candidate_id;
						break;
					}
				}
				if (parent_id >= 0) {
					break;
				}
			}
		}

		if (parent_id < 0 || parent_id == current_id) {
			break; // Reached root or cycle
		}
		current_id = parent_id;
	}

	path["path"] = path_nodes;
	path["length"] = path_nodes.size();
	path["target_node_id"] = p_node_id;

	return path;
}

Dictionary PlannerResult::to_graph_json() const {
	Dictionary graph_json;
	graph_json["type"] = "SolutionGraph";
	graph_json["success"] = success;

	Array nodes;
	Array edges;

	Array graph_keys = solution_graph.keys();

	// Add nodes
	for (int i = 0; i < graph_keys.size(); i++) {
		Variant key = graph_keys[i];
		if (key.get_type() != Variant::INT) {
			continue;
		}
		int node_id = key;
		Dictionary node = solution_graph[node_id];

		Dictionary node_json;
		node_json["id"] = node_id;
		node_json["type"] = node.get("type", -1);
		node_json["status"] = node.get("status", -1);
		node_json["info"] = node.get("info", Variant());
		node_json["tag"] = node.get("tag", "new");

		if (node.has("decision_info")) {
			node_json["decision_info"] = node["decision_info"];
		}

		if (node.has("successors")) {
			TypedArray<int> successors = node["successors"];
			for (int j = 0; j < successors.size(); j++) {
				Dictionary edge;
				edge["from"] = node_id;
				edge["to"] = successors[j];
				edges.push_back(edge);
			}
		}

		nodes.push_back(node_json);
	}

	graph_json["nodes"] = nodes;
	graph_json["edges"] = edges;

	return graph_json;
}

String PlannerResult::get_node_explanation(int p_node_id) const {
	if (!has_node(p_node_id)) {
		return vformat("Node %d does not exist in the solution graph.", p_node_id);
	}

	Dictionary node = get_node(p_node_id);
	int node_type = node.get("type", -1);
	int node_status = node.get("status", -1);
	Variant info = node.get("info", Variant());

	String explanation = vformat("Node %d: ", p_node_id);

	// Add type description
	String type_name = "Unknown";
	switch (static_cast<PlannerNodeType>(node_type)) {
		case PlannerNodeType::TYPE_ROOT:
			type_name = "Root";
			break;
		case PlannerNodeType::TYPE_COMMAND:
			type_name = "Command";
			break;
		case PlannerNodeType::TYPE_TASK:
			type_name = "Task";
			break;
		case PlannerNodeType::TYPE_UNIGOAL:
			type_name = "Unigoal";
			break;
		case PlannerNodeType::TYPE_MULTIGOAL:
			type_name = "Multigoal";
			break;
		default:
			break;
	}
	explanation += type_name + " - " + String(info) + "\n";

	// Add status
	String status_name = "Unknown";
	switch (static_cast<PlannerNodeStatus>(node_status)) {
		case PlannerNodeStatus::STATUS_CLOSED:
			status_name = "Completed";
			break;
		case PlannerNodeStatus::STATUS_FAILED:
			status_name = "Failed";
			break;
		case PlannerNodeStatus::STATUS_OPEN:
			status_name = "Open";
			break;
		case PlannerNodeStatus::STATUS_NOT_APPLICABLE:
			status_name = "Not Applicable";
			break;
		default:
			break;
	}
	explanation += "Status: " + status_name + "\n";

	// Add decision info if available
	if (node.has("decision_info")) {
		Dictionary decision_info = node["decision_info"];
		if (decision_info.has("selected_method_id")) {
			explanation += vformat("Selected Method: %s\n", String(decision_info["selected_method_id"]));
		}
		if (decision_info.has("selected_score")) {
			explanation += vformat("Selection Score: %.2f\n", double(decision_info["selected_score"]));
		}
		if (decision_info.has("selected_reason")) {
			explanation += vformat("Reason: %s\n", String(decision_info["selected_reason"]));
		}
		if (decision_info.has("alternatives")) {
			Array alternatives = decision_info["alternatives"];
			explanation += vformat("Alternative Methods Considered: %d\n", alternatives.size());
			for (int i = 0; i < alternatives.size(); i++) {
				Dictionary alt = alternatives[i];
				explanation += vformat("  - %s (score: %.2f)\n",
						String(alt.get("method_id", "")),
						double(alt.get("score", 0.0)));
			}
		}
	}

	// Add successors info
	if (node.has("successors")) {
		TypedArray<int> successors = node["successors"];
		explanation += vformat("Child Nodes: %d\n", successors.size());
	}

	return explanation;
}
