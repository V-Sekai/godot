/**************************************************************************/
/*  graph_operations.cpp                                                  */
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

#include "graph_operations.h"
#include "domain.h"
#include "multigoal.h"

PlannerNodeType PlannerGraphOperations::get_node_type(Variant p_node_info, Dictionary p_action_dict, Dictionary p_task_dict, Dictionary p_unigoal_dict) {
	// Check if it's a String - look up in dictionaries
	if (p_node_info.get_type() == Variant::STRING) {
		String node_str = p_node_info;
		// Check action dictionary
		if (p_action_dict.has(node_str)) {
			return PlannerNodeType::TYPE_ACTION;
		}
		// Check task method dictionary
		if (p_task_dict.has(node_str)) {
			return PlannerNodeType::TYPE_TASK;
		}
		// Check unigoal method dictionary
		if (p_unigoal_dict.has(node_str)) {
			return PlannerNodeType::TYPE_UNIGOAL;
		}
		// Not found in any dictionary, return ROOT
		return PlannerNodeType::TYPE_ROOT;
	}

	// Check if it's a Dictionary-wrapped item (with constraints)
	if (p_node_info.get_type() == Variant::DICTIONARY) {
		Dictionary dict = p_node_info;
		if (dict.has("item")) {
			// Unwrap and recursively check the item
			Variant unwrapped_item = dict["item"];
			return get_node_type(unwrapped_item, p_action_dict, p_task_dict, p_unigoal_dict);
		}
		// If it's a dictionary without "item", it's not a valid node (multigoals are now Arrays)
		return PlannerNodeType::TYPE_ROOT;
	}

	// Check if it's an Array (task/goal/action/multigoal)
	if (p_node_info.get_type() == Variant::ARRAY) {
		Array arr = p_node_info;
		if (arr.is_empty()) {
			return PlannerNodeType::TYPE_ROOT;
		}

		Variant first = arr[0];

		// Check if it's a multigoal (Array of unigoal arrays)
		// A multigoal is an Array where the first element is also an Array
		if (first.get_type() == Variant::ARRAY) {
			return PlannerNodeType::TYPE_MULTIGOAL;
		}

		// Otherwise, it's a single unigoal/action/task - check first element as string
		String first_str = first;

		// Check action dictionary
		if (p_action_dict.has(first_str)) {
			return PlannerNodeType::TYPE_ACTION;
		}

		// Check task method dictionary
		if (p_task_dict.has(first_str)) {
			return PlannerNodeType::TYPE_TASK;
		}

		// Check unigoal method dictionary
		if (p_unigoal_dict.has(first_str)) {
			return PlannerNodeType::TYPE_UNIGOAL;
		}
	}

	return PlannerNodeType::TYPE_ROOT;
}

int PlannerGraphOperations::add_nodes_and_edges(PlannerSolutionGraph &p_graph, int p_parent_node_id, Array p_children_node_info_list, Dictionary p_action_dict, Dictionary p_task_dict, Dictionary p_unigoal_dict, TypedArray<Callable> p_multigoal_methods) {
	int current_id = p_graph.next_node_id - 1;

	for (int i = 0; i < p_children_node_info_list.size(); i++) {
		Variant child_info = p_children_node_info_list[i];
		PlannerNodeType node_type = get_node_type(child_info, p_action_dict, p_task_dict, p_unigoal_dict);

		TypedArray<Callable> available_methods;
		Callable action;

		// Extract actual item if wrapped in dictionary
		Variant actual_item = child_info;
		if (child_info.get_type() == Variant::DICTIONARY) {
			Dictionary dict = child_info;
			if (dict.has("item")) {
				actual_item = dict["item"];
			}
		}

		// Set up node attributes based on type
		if (node_type == PlannerNodeType::TYPE_TASK) {
			Array arr = actual_item;
			if (!arr.is_empty()) {
				String task_name = arr[0];
				if (p_task_dict.has(task_name)) {
					Variant methods_var = p_task_dict[task_name];
					available_methods = TypedArray<Callable>(methods_var);
				}
			}
		} else if (node_type == PlannerNodeType::TYPE_UNIGOAL) {
			Array arr = actual_item;
			if (!arr.is_empty()) {
				String goal_name = arr[0];
				if (p_unigoal_dict.has(goal_name)) {
					Variant methods_var = p_unigoal_dict[goal_name];
					available_methods = TypedArray<Callable>(methods_var);
				}
			}
		} else if (node_type == PlannerNodeType::TYPE_ACTION) {
			Array arr = actual_item;
			if (!arr.is_empty()) {
				String action_name = arr[0];
				if (p_action_dict.has(action_name)) {
					action = p_action_dict[action_name];
				}
			}
		} else if (node_type == PlannerNodeType::TYPE_MULTIGOAL) {
			// MultiGoal methods are in a list
			available_methods = p_multigoal_methods;
		}

		int child_id = p_graph.create_node(node_type, child_info, available_methods, action);
		p_graph.add_successor(p_parent_node_id, child_id);
		current_id = child_id;
	}

	// Add verification nodes for Unigoals and MultiGoals
	Dictionary parent_node = p_graph.get_node(p_parent_node_id);
	int parent_type = parent_node["type"];

	if (parent_type == static_cast<int>(PlannerNodeType::TYPE_UNIGOAL)) {
		int verify_id = p_graph.create_node(PlannerNodeType::TYPE_VERIFY_GOAL, Variant("VerifyUnigoal"), TypedArray<Callable>(), Callable());
		p_graph.add_successor(p_parent_node_id, verify_id);
		current_id = verify_id;
	} else if (parent_type == static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
		int verify_id = p_graph.create_node(PlannerNodeType::TYPE_VERIFY_MULTIGOAL, Variant("VerifyMultiGoal"), TypedArray<Callable>(), Callable());
		p_graph.add_successor(p_parent_node_id, verify_id);
		current_id = verify_id;
	}

	return current_id;
}

Variant PlannerGraphOperations::find_open_node(PlannerSolutionGraph &p_graph, int p_parent_node_id) {
	Dictionary parent_node = p_graph.get_node(p_parent_node_id);
	TypedArray<int> successors = parent_node["successors"];

	for (int i = 0; i < successors.size(); i++) {
		int node_id = successors[i];
		Dictionary node = p_graph.get_node(node_id);
		int status = node["status"];

		if (status == static_cast<int>(PlannerNodeStatus::STATUS_OPEN)) {
			return node_id;
		}
	}

	return Variant(); // No open node found
}

int PlannerGraphOperations::find_predecessor(PlannerSolutionGraph &p_graph, int p_node_id) {
	Array keys = p_graph.graph.keys();

	for (int i = 0; i < keys.size(); i++) {
		int parent_id = keys[i];
		Dictionary parent_node = p_graph.get_node(parent_id);
		TypedArray<int> successors = parent_node["successors"];

		if (successors.has(p_node_id)) {
			return parent_id;
		}
	}

	return -1; // No predecessor found
}

void PlannerGraphOperations::remove_descendants(PlannerSolutionGraph &p_graph, int p_node_id, bool p_also_remove_from_parent) {
	TypedArray<int> to_remove;
	TypedArray<int> visited;

	// Start from the node's successors
	Dictionary node = p_graph.get_node(p_node_id);
	TypedArray<int> successors = node["successors"];

	do_get_descendants(p_graph, successors, visited, to_remove);

	// Remove nodes from graph
	Dictionary &graph_dict = p_graph.get_graph();
	for (int i = 0; i < to_remove.size(); i++) {
		int node_id_to_remove = to_remove[i];
		if (node_id_to_remove != p_node_id) { // Don't remove the node itself
			graph_dict.erase(node_id_to_remove);
		}
	}

	// Clear successors of the node
	successors.clear();
	node["successors"] = successors;
	p_graph.update_node(p_node_id, node);

	// Optionally remove the node itself from its parent's successors list
	if (p_also_remove_from_parent) {
		int parent_id = find_predecessor(p_graph, p_node_id);
		if (parent_id >= 0) {
			Dictionary parent_node = p_graph.get_node(parent_id);
			TypedArray<int> parent_successors = parent_node["successors"];
			// Remove the node from parent's successors
			parent_successors.erase(p_node_id);
			parent_node["successors"] = parent_successors;
			p_graph.update_node(parent_id, parent_node);
		}
	}
}

void PlannerGraphOperations::do_get_descendants(PlannerSolutionGraph &p_graph, TypedArray<int> p_current_nodes, TypedArray<int> &p_visited, TypedArray<int> &p_result) {
	for (int i = 0; i < p_current_nodes.size(); i++) {
		int node_id = p_current_nodes[i];

		if (p_visited.has(node_id)) {
			continue;
		}

		p_visited.push_back(node_id);
		p_result.push_back(node_id);

		Dictionary node = p_graph.get_node(node_id);
		TypedArray<int> successors = node["successors"];

		if (successors.size() > 0) {
			do_get_descendants(p_graph, successors, p_visited, p_result);
		}
	}
}

Array PlannerGraphOperations::extract_solution_plan(PlannerSolutionGraph &p_graph) {
	Array plan;
	Array to_visit;
	to_visit.push_back(0); // Start from root
	TypedArray<int> visited; // Track visited nodes to prevent revisiting

	while (!to_visit.is_empty()) {
		int node_id = to_visit.pop_back();
		
		// Skip if already visited
		if (visited.has(node_id)) {
			continue;
		}
		visited.push_back(node_id);
		
		Dictionary node = p_graph.get_node(node_id);

		int node_type = node["type"];
		int node_status = node["status"];

		// Only extract actions that are closed (successful)
		if (node_type == static_cast<int>(PlannerNodeType::TYPE_ACTION) &&
				node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
			Variant info = node["info"];
			// Unwrap if dictionary-wrapped (has constraints)
			if (info.get_type() == Variant::DICTIONARY) {
				Dictionary dict = info;
				if (dict.has("item")) {
					info = dict["item"];
				}
			}
			plan.push_back(info);
		}

		// Only visit successors of closed nodes (skip failed branches)
		// This ensures we only follow the final successful path
		// Failed nodes are removed from their parent's successors during backtracking,
		// so only nodes in the final successful path will be in the successors list
		// Also verify that each successor is actually still in its parent's successors list
		// (this prevents including nodes from backtracked paths that weren't fully cleaned up)
		if (node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) ||
				node_id == 0) { // Root is NA status, but we need to visit it
			TypedArray<int> successors = node["successors"];
			// Add successors in reverse order to maintain DFS order (last added = first visited)
			// This ensures we process tasks in the order they appear in the todo list
			for (int i = successors.size() - 1; i >= 0; i--) {
				int succ_id = successors[i];
				// Only visit if not already visited
				if (!visited.has(succ_id)) {
					// Verify this successor is actually in its parent's successors list
					// (double-check to ensure we're only following the final successful path)
					int parent_of_succ = find_predecessor(p_graph, succ_id);
					if (parent_of_succ == node_id) {
						// This successor is actually a child of the current node
						Dictionary succ_node = p_graph.get_node(succ_id);
						int succ_status = succ_node["status"];
						// Only follow closed nodes (or root which is NA)
						// Failed nodes should have been removed from successors, so they won't be here
						if (succ_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) ||
								succ_id == 0) {
							to_visit.push_back(succ_id);
						}
					}
					// If parent doesn't match, skip this successor (it was removed during backtracking)
				}
			}
		}
	}

	return plan;
}

void PlannerGraphOperations::do_dfs_preorder(PlannerSolutionGraph &p_graph, int p_node_id, TypedArray<int> &p_visited, TypedArray<int> &p_result) {
	// Check if already visited (prevent cycles)
	if (p_visited.has(p_node_id)) {
		return;
	}
	
	// Mark as visited and add to result
	p_visited.push_back(p_node_id);
	p_result.push_back(p_node_id);
	
	// Get node and its successors
	Dictionary node = p_graph.get_node(p_node_id);
	TypedArray<int> successors = node["successors"];
	
	// Recursively visit all successors
	for (int i = 0; i < successors.size(); i++) {
		int succ_id = successors[i];
		// Only visit if node exists in graph (might have been removed)
		if (p_graph.get_graph().has(succ_id)) {
			do_dfs_preorder(p_graph, succ_id, p_visited, p_result);
		}
	}
}

bool PlannerGraphOperations::is_retriable_closed_node(PlannerSolutionGraph &p_graph, int p_node_id) {
	// Check if node exists in graph
	if (!p_graph.get_graph().has(p_node_id)) {
		return false;
	}
	
	Dictionary node = p_graph.get_node(p_node_id);
	int status = node["status"];
	int node_type = node["type"];
	
	// Must be CLOSED and of type TASK/UNIGOAL/MULTIGOAL
	if (status != static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) ||
			(node_type != static_cast<int>(PlannerNodeType::TYPE_TASK) &&
			 node_type != static_cast<int>(PlannerNodeType::TYPE_UNIGOAL) &&
			 node_type != static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL))) {
		return false;
	}
	
	// Must have available methods
	if (!node.has("available_methods")) {
		return false;
	}
	
	Variant methods_var = node["available_methods"];
	if (methods_var.get_type() != Variant::ARRAY) {
		return false;
	}
	
	Array methods_array = methods_var;
	TypedArray<Callable> available_methods = TypedArray<Callable>(methods_array);
	if (available_methods.size() == 0) {
		return false;
	}
	
	// Must have successors (IPyHOP only retries if it has descendants)
	TypedArray<int> successors = node["successors"];
	return successors.size() > 0;
}

int PlannerGraphOperations::find_first_closed_node_dfs(PlannerSolutionGraph &p_graph, int p_start_node_id, int p_failed_node_id) {
	// If start node is root, check all root children first (siblings of failed node)
	if (p_start_node_id == 0) {
		Dictionary root_node = p_graph.get_node(0);
		TypedArray<int> root_successors = root_node["successors"];
		
		for (int i = 0; i < root_successors.size(); i++) {
			int child_id = root_successors[i];
			if (is_retriable_closed_node(p_graph, child_id)) {
				return child_id;
			}
		}
	}
	
	// Traverse up parent chain from failed node to start node
	// Check each ancestor for retriable CLOSED nodes
	int current_node = p_failed_node_id;
	
	while (current_node >= 0) {
		int parent = find_predecessor(p_graph, current_node);
		
		// Stop if we've reached invalid node or gone beyond start node
		if (parent < 0) {
			break;
		}
		
		// Stop if we've reached root (parent == 0) and start node is not root
		// Or if we've reached start node
		if (parent == 0 && p_start_node_id > 0) {
			break;
		}
		if (parent == p_start_node_id) {
			break;
		}
		
		// Check if this ancestor node is retriable
		if (is_retriable_closed_node(p_graph, parent)) {
			return parent;
		}
		
		current_node = parent;
	}
	
	return -1; // No retriable CLOSED node found
}
