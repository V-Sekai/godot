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

// Determine node type from node_info
// Supports all planner element types: actions, tasks, unigoals (goals), and multigoals
// Methods can return Arrays containing any of these types
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

	// Check if it's a Dictionary-wrapped item (with constraints/metadata)
	if (p_node_info.get_type() == Variant::DICTIONARY) {
		Dictionary dict = p_node_info;
		if (dict.has("item")) {
			// Unwrap and recursively check the item
			Variant unwrapped_item = dict["item"];
			return get_node_type(unwrapped_item, p_action_dict, p_task_dict, p_unigoal_dict);
		}
		// If it's a dictionary without "item", it's not a valid node (multigoals are Arrays)
		return PlannerNodeType::TYPE_ROOT;
	}

	// Check if it's an Array (can be task/goal/action/multigoal)
	// Methods return Arrays containing any planner elements
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

// Add nodes and edges to solution graph
// p_children_node_info_list can contain any planner elements: goals (unigoals), PlannerMultigoal, tasks, and actions
// Methods return Arrays of these elements, which are processed here
int PlannerGraphOperations::add_nodes_and_edges(PlannerSolutionGraph &p_graph, int p_parent_node_id, Array p_children_node_info_list, Dictionary p_action_dict, Dictionary p_task_dict, Dictionary p_unigoal_dict, TypedArray<Callable> p_multigoal_methods) {
	int current_id = p_graph.next_node_id - 1;

	for (int i = 0; i < p_children_node_info_list.size(); i++) {
		Variant child_info = p_children_node_info_list[i];
		// Determine type of planner element (action, task, unigoal, multigoal)
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

		// Check for duplicate tasks with same info to prevent infinite recursion
		// For tasks like move_blocks that recursively create themselves
		int existing_node_id = -1;
		if (node_type == PlannerNodeType::TYPE_TASK) {
			Array arr = actual_item;
			if (!arr.is_empty()) {
				String task_name = arr[0];
				// For recursive tasks like move_blocks, check if a node with same info already exists
				// Only check if task_name is "move_blocks" to avoid false positives
				if (task_name == "move_blocks") {
					const Dictionary &graph = p_graph.get_graph();
					Array graph_keys = graph.keys();
					for (int j = 0; j < graph_keys.size(); j++) {
						int node_id = graph_keys[j];
						Dictionary node = p_graph.get_node(node_id);
						if (node.is_empty() || !node.has("type") || !node.has("info")) {
							continue;
						}
						int node_type_check = node["type"];
						if (node_type_check == static_cast<int>(PlannerNodeType::TYPE_TASK)) {
							Array node_info = node["info"];
							if (!node_info.is_empty() && node_info[0] == task_name) {
								// Compare task info - for move_blocks, compare the goal state
								if (arr.size() >= 2 && node_info.size() >= 2) {
									Variant arr_goal = arr[1];
									Variant node_goal = node_info[1];
									if (arr_goal == node_goal) {
										// Same task with same goal - reuse existing node
										existing_node_id = node_id;
										break;
									}
								}
							}
						}
					}
				}
			}
		}

		int child_id;
		if (existing_node_id >= 0) {
			// Reuse existing node instead of creating new one
			child_id = existing_node_id;
		} else {
			child_id = p_graph.create_node(node_type, child_info, available_methods, action);
		}
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
	Dictionary graph_dict = p_graph.get_graph();
	Array keys = graph_dict.keys();

	for (int i = 0; i < keys.size(); i++) {
		int parent_id = keys[i];
		Dictionary parent_node = p_graph.get_node(parent_id);
		
		// Validate parent node exists and has required fields
		if (parent_node.is_empty() || !parent_node.has("successors")) {
			continue; // Skip invalid parent nodes
		}
		
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
	// Convert from recursive to iterative to prevent stack overflow with deep graphs
	// Use a stack (TypedArray) instead of recursion
	TypedArray<int> to_process = p_current_nodes;
	
	while (!to_process.is_empty()) {
		int node_id = to_process.pop_back();
		
		// Skip if already visited
		if (p_visited.has(node_id)) {
			continue;
		}
		
		// Mark as visited and add to result
		p_visited.push_back(node_id);
		p_result.push_back(node_id);
		
		// Get node and its successors
		Dictionary node = p_graph.get_node(node_id);
		
		// Validate node exists and has successors field
		if (node.is_empty() || !node.has("successors")) {
			continue; // Skip invalid nodes
		}
		
		TypedArray<int> successors = node["successors"];
		
		// Add successors to stack instead of recursing
		// Process in reverse order to maintain DFS order (last added = first processed)
		for (int i = successors.size() - 1; i >= 0; i--) {
			int succ_id = successors[i];
			if (!p_visited.has(succ_id)) {
				to_process.push_back(succ_id);
			}
		}
	}
}

Array PlannerGraphOperations::extract_solution_plan(PlannerSolutionGraph &p_graph) {
	print_line("[EXTRACT_SOLUTION_PLAN] Starting extract_solution_plan()");
	Array plan;
	Array to_visit;
	to_visit.push_back(0); // Start from root
	TypedArray<int> visited; // Track visited nodes to prevent revisiting

	// Optimize: Precompute parent map once instead of calling find_predecessor() repeatedly
	// This follows the Nostradamus Distributor principle: reduce expensive indirect operations
	// in tight loops (http://www.emulators.com/docs/nx25_nostradamus.htm)
	print_line("[EXTRACT_SOLUTION_PLAN] Building parent map...");
	Dictionary parent_map; // child_id -> parent_id
	Dictionary graph_dict = p_graph.get_graph();
	Array graph_keys = graph_dict.keys();
	print_line(vformat("[EXTRACT_SOLUTION_PLAN] Graph has %d nodes", graph_keys.size()));
	
	int parent_map_count = 0;
	for (int i = 0; i < graph_keys.size(); i++) {
		Variant key = graph_keys[i];
		if (key.get_type() != Variant::INT) {
			continue;
		}
		int parent_id = key;
		Dictionary parent_node = p_graph.get_node(parent_id);
		if (parent_node.is_empty() || !parent_node.has("successors")) {
			continue;
		}
		TypedArray<int> successors = parent_node["successors"];
		for (int j = 0; j < successors.size(); j++) {
			int child_id = successors[j];
			parent_map[child_id] = parent_id; // O(1) lookup instead of O(n) search
			parent_map_count++;
		}
	}
	print_line(vformat("[EXTRACT_SOLUTION_PLAN] Parent map built with %d entries", parent_map_count));

	while (!to_visit.is_empty()) {
		int node_id = to_visit.pop_back();

		// Skip if already visited
		if (visited.has(node_id)) {
			continue;
		}
		visited.push_back(node_id);

		Dictionary node = p_graph.get_node(node_id);
		
		// Validate node exists and has required fields
		if (node.is_empty() || !node.has("type") || !node.has("status")) {
			// Skip invalid nodes (may have been removed during backtracking)
			continue;
		}

		int node_type = node["type"];
		int node_status = node["status"];

		// Only extract actions that are closed (successful)
		if (node_type == static_cast<int>(PlannerNodeType::TYPE_ACTION) &&
				node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
			// Validate info field exists
			if (!node.has("info")) {
				continue; // Skip nodes without info field
			}
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
			// Validate successors field exists
			if (!node.has("successors")) {
				continue; // Skip nodes without successors field
			}
			TypedArray<int> successors = node["successors"];
			// Add successors in reverse order to maintain DFS order (last added = first visited)
			// This ensures we process tasks in the order they appear in the todo list
			for (int i = successors.size() - 1; i >= 0; i--) {
				int succ_id = successors[i];
				// Only visit if not already visited
				if (!visited.has(succ_id)) {
					// Verify this successor is actually in its parent's successors list
					// Use O(1) lookup from precomputed parent_map instead of O(n) find_predecessor()
					int parent_of_succ = parent_map.get(succ_id, -1);
					if (parent_of_succ == node_id) {
						// This successor is actually a child of the current node
						Dictionary succ_node = p_graph.get_node(succ_id);
						// Validate successor node exists and has required fields
						if (succ_node.is_empty() || !succ_node.has("status")) {
							// Skip invalid successor nodes (may have been removed)
							continue;
						}
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

Array PlannerGraphOperations::extract_new_actions(PlannerSolutionGraph &p_graph) {
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
		
		// Validate node exists and has required fields
		if (node.is_empty() || !node.has("type") || !node.has("status")) {
			// Skip invalid nodes (may have been removed during backtracking)
			continue;
		}

		int node_type = node["type"];
		int node_status = node["status"];
		String node_tag = p_graph.get_node_tag(node_id);

		// Only extract actions that are closed (successful) and tagged as "new"
		if (node_type == static_cast<int>(PlannerNodeType::TYPE_ACTION) &&
				node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) &&
				node_tag == "new") {
			// Validate info field exists
			if (!node.has("info")) {
				continue; // Skip nodes without info field
			}
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

		// Visit successors (only closed nodes or root)
		if (node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) ||
				node_id == 0) { // Root is NA status, but we need to visit it
			// Validate successors field exists
			if (!node.has("successors")) {
				continue; // Skip nodes without successors field
			}
			TypedArray<int> successors = node["successors"];
			// Add successors in reverse order to maintain DFS order
			for (int i = successors.size() - 1; i >= 0; i--) {
				int succ_id = successors[i];
				if (!visited.has(succ_id) && p_graph.get_graph().has(succ_id)) {
					to_visit.push_back(succ_id);
				}
			}
		}
	}

	return plan;
}
