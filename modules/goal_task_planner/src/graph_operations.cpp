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
PlannerNodeType PlannerGraphOperations::get_node_type(Variant p_node_info, Dictionary p_action_dict, Dictionary p_task_dict, Dictionary p_unigoal_dict, int p_verbose) {
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
			return get_node_type(unwrapped_item, p_action_dict, p_task_dict, p_unigoal_dict, p_verbose);
		}
		// If it's a dictionary without "item", it's not a valid node (multigoals are Arrays)
		return PlannerNodeType::TYPE_ROOT;
	}

	// Check if it's an Array (can be task/goal/action/multigoal)
	// Methods return Arrays containing any planner elements
	if (p_node_info.get_type() == Variant::ARRAY) {
		Array arr = p_node_info;
		if (arr.is_empty() || arr.size() < 1) {
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

		// CRITICAL: Check action dictionary FIRST (actions take priority over tasks)
		// This ensures that when methods return actions like ["action_pickup", "a"],
		// they are correctly classified as ACTION nodes, not TASK nodes
		if (p_action_dict.has(first_str)) {
			// Debug: Log classification for action names
			if (p_verbose >= 3 && first_str.begins_with("action_")) {
				print_line(vformat("[GET_NODE_TYPE] Returning TYPE_ACTION (1) for '%s'", first_str));
			}
			return PlannerNodeType::TYPE_ACTION;
		}

		// Check task method dictionary
		if (p_task_dict.has(first_str)) {
			// Debug: Log if action name is incorrectly in task_dict
			if (p_verbose >= 2 && first_str.begins_with("action_")) {
				print_line(vformat("[GET_NODE_TYPE] WARNING: '%s' is in task_dict but should be in action_dict! Returning TYPE_TASK", first_str));
			}
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
int PlannerGraphOperations::add_nodes_and_edges(PlannerSolutionGraph &p_graph, int p_parent_node_id, Array p_children_node_info_list, Dictionary p_action_dict, Dictionary p_task_dict, Dictionary p_unigoal_dict, TypedArray<Callable> p_multigoal_methods, int p_verbose) {
	int current_id = p_graph.next_node_id - 1;

	for (int i = 0; i < p_children_node_info_list.size(); i++) {
		Variant child_info = p_children_node_info_list[i];
		// Determine type of planner element (action, task, unigoal, multigoal)
		PlannerNodeType node_type = get_node_type(child_info, p_action_dict, p_task_dict, p_unigoal_dict, p_verbose);

		// Debug: Log node type determination for action arrays
		if (p_verbose >= 3 && child_info.get_type() == Variant::ARRAY) {
			Array arr = child_info;
			if (!arr.is_empty() && arr[0].get_type() == Variant::STRING) {
				String first_str = arr[0];
				if (first_str.begins_with("action_")) {
					print_line(vformat("[ADD_NODES] After get_node_type: child_info=%s, node_type=%d (ACTION=1, TASK=2), expected=ACTION(1)",
							String(Variant(child_info)), static_cast<int>(node_type)));
					if (static_cast<int>(node_type) != 1) {
						print_line(vformat("[ADD_NODES] ERROR: node_type is %d but should be 1 (TYPE_ACTION)!", static_cast<int>(node_type)));
					}
				}
			}
		}

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
		// Handle strings directly (most frequent use case) and arrays
		if (node_type == PlannerNodeType::TYPE_TASK) {
			String task_name;
			if (actual_item.get_type() == Variant::STRING) {
				// Most frequent: string task name
				task_name = actual_item;
			} else if (actual_item.get_type() == Variant::ARRAY) {
				Array arr = actual_item;
				if (!arr.is_empty()) {
					task_name = arr[0];
				} else {
					if (p_verbose >= 2) {
						print_line("[ADD_NODES] WARNING: Task array is empty");
					}
					continue;
				}
			} else {
				if (p_verbose >= 2) {
					print_line(vformat("[ADD_NODES] WARNING: Task actual_item has invalid type %d", actual_item.get_type()));
				}
				continue;
			}

			if (p_task_dict.has(task_name)) {
				Variant methods_var = p_task_dict[task_name];
				available_methods = TypedArray<Callable>(methods_var);
				// Debug: Verify we got the right methods
				if (p_verbose >= 3 && available_methods.size() > 0) {
					Callable first_method = available_methods[0];
					String method_name = first_method.get_method();
					print_line(vformat("[ADD_NODES] Task '%s' has %d methods, first method: '%s'", task_name, available_methods.size(), method_name));
				}
			} else {
				if (p_verbose >= 2) {
					print_line(vformat("[ADD_NODES] WARNING: Task '%s' not found in task_dict (has %d keys), skipping node creation", task_name, p_task_dict.keys().size()));
				}
				continue; // Skip creating node if task not found in domain
			}
		} else if (node_type == PlannerNodeType::TYPE_UNIGOAL) {
			String goal_name;
			if (actual_item.get_type() == Variant::STRING) {
				goal_name = actual_item;
			} else if (actual_item.get_type() == Variant::ARRAY) {
				Array arr = actual_item;
				if (!arr.is_empty()) {
					goal_name = arr[0];
				} else {
					continue;
				}
			} else {
				continue;
			}
			if (p_unigoal_dict.has(goal_name)) {
				Variant methods_var = p_unigoal_dict[goal_name];
				available_methods = TypedArray<Callable>(methods_var);
			}
		} else if (node_type == PlannerNodeType::TYPE_ACTION) {
			String action_name;
			if (actual_item.get_type() == Variant::STRING) {
				action_name = actual_item;
			} else if (actual_item.get_type() == Variant::ARRAY) {
				Array arr = actual_item;
				if (!arr.is_empty()) {
					action_name = arr[0];
				} else {
					continue;
				}
			} else {
				continue;
			}
			if (p_action_dict.has(action_name)) {
				action = p_action_dict[action_name];
			}
		} else if (node_type == PlannerNodeType::TYPE_MULTIGOAL) {
			// MultiGoal methods are in a list
			available_methods = p_multigoal_methods;
		}

		// Check for duplicate tasks with same info to prevent infinite recursion
		// For tasks like move_blocks that recursively create themselves
		int existing_node_id = -1;
		if (node_type == PlannerNodeType::TYPE_TASK) {
			// Get task name and args for comparison
			String task_name;
			Array task_args;
			if (actual_item.get_type() == Variant::STRING) {
				task_name = actual_item;
			} else if (actual_item.get_type() == Variant::ARRAY) {
				Array arr = actual_item;
				if (!arr.is_empty()) {
					task_name = arr[0];
					if (arr.size() > 1) {
						task_args = arr.slice(1);
					}
				}
			}
			if (!task_name.is_empty()) {
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
							Variant node_info_variant = node["info"];
							// Handle both strings and arrays
							String node_task_name;
							Array node_info;
							if (node_info_variant.get_type() == Variant::STRING) {
								node_task_name = node_info_variant;
								node_info.push_back(node_task_name);
							} else if (node_info_variant.get_type() == Variant::ARRAY) {
								node_info = node_info_variant;
								if (!node_info.is_empty()) {
									node_task_name = node_info[0];
								} else {
									continue;
								}
							} else {
								continue;
							}
							if (node_task_name == task_name) {
								// Compare all task arguments (including nested fluents)
								// For tasks with arguments, compare all arguments element by element
								// This properly handles nested arrays/fluents like [5, 5]
								if (task_args.size() > 0 && node_info.size() >= 2) {
									// Extract node args (skip task name at index 0)
									Array node_args = node_info.slice(1);
									// Compare all arguments, including nested structures
									if (task_args.size() == node_args.size()) {
										bool args_match = true;
										for (int k = 0; k < task_args.size(); k++) {
											Variant task_arg = task_args[k];
											Variant node_arg = node_args[k];
											// Use Variant comparison which handles nested arrays/dictionaries
											if (task_arg != node_arg) {
												args_match = false;
												break;
											}
										}
										if (args_match) {
											// Same task with same arguments (including nested fluents) - reuse existing node
											existing_node_id = node_id;
											break;
										}
									}
								} else if (task_args.size() == 0 && node_info.size() == 1) {
									// Both are simple tasks with no args - reuse
									existing_node_id = node_id;
									break;
								}
							}
						}
					}
				}
			}
		}

		// Convert to array format for node storage (nodes store info as arrays)
		// Strings are most frequent, but nodes need arrays for consistency
		// CRITICAL: For array tasks with fluents (e.g., ["move_task", "r1", [5, 5]]),
		// we preserve the entire array structure including nested arrays/fluents
		Variant normalized_info;
		if (node_type == PlannerNodeType::TYPE_TASK || node_type == PlannerNodeType::TYPE_ACTION || node_type == PlannerNodeType::TYPE_UNIGOAL) {
			if (actual_item.get_type() == Variant::STRING) {
				// Convert string to array for node storage
				Array arr;
				arr.push_back(actual_item);
				normalized_info = arr;
			} else if (actual_item.get_type() == Variant::ARRAY) {
				// Preserve full array structure including nested arrays/fluents
				// This ensures tasks like ["move_task", "r1", [5, 5]] maintain their structure
				normalized_info = actual_item;
			} else {
				normalized_info = actual_item;
			}
		} else {
			normalized_info = actual_item;
		}

		int child_id;
		if (existing_node_id >= 0) {
			// Reuse existing node instead of creating new one
			child_id = existing_node_id;
		} else {
			child_id = p_graph.create_node(node_type, normalized_info, available_methods, action);
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

	// First check direct successors
	for (int i = 0; i < successors.size(); i++) {
		int node_id = successors[i];
		Dictionary node = p_graph.get_node(node_id);
		int status = node["status"];

		if (status == static_cast<int>(PlannerNodeStatus::STATUS_OPEN)) {
			return node_id;
		}
	}

	// If no OPEN node in direct successors, recursively search through CLOSED nodes
	// This is needed because a CLOSED task may have OPEN action children
	for (int i = 0; i < successors.size(); i++) {
		int node_id = successors[i];
		Dictionary node = p_graph.get_node(node_id);
		int status = node["status"];

		// Recursively search through CLOSED nodes (they may have OPEN descendants)
		if (status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
			Variant found = find_open_node(p_graph, node_id);
			if (found.get_type() != Variant::NIL) {
				return found;
			}
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

Array PlannerGraphOperations::extract_solution_plan(PlannerSolutionGraph &p_graph, int p_verbose) {
	if (p_verbose >= 3) {
		print_line("[EXTRACT_SOLUTION_PLAN] Starting extract_solution_plan()");
	}
	Array plan;
	
	// Safety check: ensure graph is not empty
	Dictionary graph_dict = p_graph.get_graph();
	if (graph_dict.is_empty()) {
		if (p_verbose >= 1) {
			print_line("[EXTRACT_SOLUTION_PLAN] WARNING: Graph is empty, returning empty plan");
		}
		return plan;
	}
	
	// Safety check: ensure root node exists
	Dictionary root_node_check = p_graph.get_node(0);
	if (root_node_check.is_empty()) {
		if (p_verbose >= 1) {
			print_line("[EXTRACT_SOLUTION_PLAN] WARNING: Root node (0) is empty, returning empty plan");
		}
		return plan;
	}
	
	Array to_visit;
	to_visit.push_back(0); // Start from root
	TypedArray<int> visited; // Track visited nodes to prevent revisiting

	// Optimize: Precompute parent map once instead of calling find_predecessor() repeatedly
	// This follows the Nostradamus Distributor principle: reduce expensive indirect operations
	// in tight loops (http://www.emulators.com/docs/nx25_nostradamus.htm)
	if (p_verbose >= 3) {
		print_line("[EXTRACT_SOLUTION_PLAN] Building parent map...");
	}
	Dictionary parent_map; // child_id -> parent_id
	Array graph_keys = graph_dict.keys();
	if (p_verbose >= 3) {
		print_line(vformat("[EXTRACT_SOLUTION_PLAN] Graph has %d nodes", graph_keys.size()));
	}

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
			// Skip self-references (nodes that have themselves in their successors list)
			// This is a graph construction bug, but we can work around it here
			if (child_id == parent_id) {
				if (p_verbose >= 2) {
					print_line(vformat("[EXTRACT_SOLUTION_PLAN] WARNING: Node %d has itself in successors list, skipping", parent_id));
				}
				continue;
			}
			// In a proper tree, each node should have exactly one parent
			// If a node appears in multiple successors lists, that's a graph construction bug
			if (!parent_map.has(child_id)) {
				parent_map[child_id] = parent_id; // O(1) lookup instead of O(n) search
				parent_map_count++;
			} else {
				// Node already has a parent - this indicates a graph construction bug
				// The solution graph should be a tree where each node has exactly one parent
				int existing_parent = parent_map[child_id];
				if (p_verbose >= 2) {
					print_line(vformat("[EXTRACT_SOLUTION_PLAN] WARNING: Node %d appears in multiple successors lists (parents: %d and %d). This is a graph construction bug. Using first parent (%d).",
							child_id, existing_parent, parent_id, existing_parent));
				}
				// Keep the first parent we encountered (don't change it)
				// This is more conservative than assuming smaller IDs are closer to root
			}
		}
	}
	if (p_verbose >= 3) {
		print_line(vformat("[EXTRACT_SOLUTION_PLAN] Parent map built with %d entries", parent_map_count));

		// Debug: Print parent map for nodes 0-6 and check each node's successors
		print_line("[EXTRACT_SOLUTION_PLAN] Parent map contents:");
		for (int i = 0; i <= 6; i++) {
			if (parent_map.has(i)) {
				print_line(vformat("[EXTRACT_SOLUTION_PLAN]   node %d -> parent %d", i, parent_map[i]));
			}
			// Also check each node's successors to see why parent map is wrong
			Dictionary node = p_graph.get_node(i);
			if (!node.is_empty() && node.has("successors")) {
				TypedArray<int> succs = node["successors"];
				print_line(vformat("[EXTRACT_SOLUTION_PLAN]   node %d has successors: %s", i, String(Variant(succs))));
			}
		}
	}

	// Debug: Check if root node exists and is valid
	Dictionary root_node = p_graph.get_node(0);
	if (root_node.is_empty()) {
		if (p_verbose >= 1) {
			print_line("[EXTRACT_SOLUTION_PLAN] ERROR: Root node (0) is empty!");
		}
		return plan;
	}
	if (p_verbose >= 3) {
		print_line(vformat("[EXTRACT_SOLUTION_PLAN] Root node (0) exists, has type=%s, status=%s, successors=%s",
				root_node.has("type") ? itos(root_node["type"]) : "NO_TYPE",
				root_node.has("status") ? itos(root_node["status"]) : "NO_STATUS",
				root_node.has("successors") ? String(Variant(root_node["successors"])) : "NO_SUCCESSORS"));
		print_line(vformat("[EXTRACT_SOLUTION_PLAN] Starting traversal, to_visit.size()=%d", to_visit.size()));
	}

	while (!to_visit.is_empty()) {
		int node_id = to_visit.pop_back();
		if (p_verbose >= 3) {
			print_line(vformat("[EXTRACT_SOLUTION_PLAN] Popped node %d from to_visit (remaining=%d)", node_id, to_visit.size()));
		}

		// Skip if already visited
		if (visited.has(node_id)) {
			if (p_verbose >= 3) {
				print_line(vformat("[EXTRACT_SOLUTION_PLAN] Node %d already visited, skipping", node_id));
			}
			continue;
		}
		visited.push_back(node_id);
		if (p_verbose >= 3) {
			print_line(vformat("[EXTRACT_SOLUTION_PLAN] Processing node %d", node_id));
		}

		Dictionary node = p_graph.get_node(node_id);

		// Validate node exists and has required fields
		if (node.is_empty() || !node.has("type") || !node.has("status")) {
			// Skip invalid nodes (may have been removed during backtracking)
			if (p_verbose >= 2) {
				print_line(vformat("[EXTRACT_SOLUTION_PLAN] Node %d is invalid (empty=%s, has_type=%s, has_status=%s), skipping",
						node_id, node.is_empty() ? "YES" : "NO",
						node.has("type") ? "YES" : "NO",
						node.has("status") ? "YES" : "NO"));
			}
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
			// Debug: Log action extraction
			if (p_verbose >= 3) {
				if (info.get_type() == Variant::ARRAY) {
					Array info_arr = info;
					if (!info_arr.is_empty() && info_arr.size() > 0 && info_arr[0].get_type() == Variant::STRING) {
						String action_name = info_arr[0];
						if (action_name.begins_with("action_")) {
							print_line(vformat("[EXTRACT_SOLUTION_PLAN] Extracting action node %d: %s", node_id, String(Variant(info))));
						}
					}
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
		// Debug: Log why we're visiting or skipping successors
		bool should_visit_successors = (node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) ||
				node_id == 0); // Root is NA status, but we need to visit it
		if (p_verbose >= 3) {
			print_line(vformat("[EXTRACT_SOLUTION_PLAN] Node %d: type=%d, status=%d, should_visit_successors=%s",
					node_id, node_type, node_status, should_visit_successors ? "YES" : "NO"));
		}
		if (should_visit_successors) {
			// Validate successors field exists
			if (!node.has("successors")) {
				if (p_verbose >= 2) {
					print_line(vformat("[EXTRACT_SOLUTION_PLAN] Node %d has no successors field, skipping", node_id));
				}
				continue; // Skip nodes without successors field
			}
			TypedArray<int> successors = node["successors"];
			if (p_verbose >= 3) {
				print_line(vformat("[EXTRACT_SOLUTION_PLAN] Node %d has %d successors: %s", node_id, successors.size(), String(Variant(successors))));
			}
			// Add successors in reverse order to maintain DFS order (last added = first visited)
			// This ensures we process tasks in the order they appear in the todo list
			for (int i = successors.size() - 1; i >= 0; i--) {
				int succ_id = successors[i];
				if (p_verbose >= 3) {
					print_line(vformat("[EXTRACT_SOLUTION_PLAN] Checking successor %d of node %d", succ_id, node_id));
				}
				// Only visit if not already visited
				if (!visited.has(succ_id)) {
					// Verify this successor is actually in its parent's successors list
					// Use O(1) lookup from precomputed parent_map instead of O(n) find_predecessor()
					int parent_of_succ = parent_map.get(succ_id, -1);
					if (p_verbose >= 3) {
						print_line(vformat("[EXTRACT_SOLUTION_PLAN] Successor %d: parent_from_map=%d, expected_parent=%d", succ_id, parent_of_succ, node_id));
					}
					if (parent_of_succ == node_id) {
						// This successor is actually a child of the current node
						if (p_verbose >= 3) {
							print_line(vformat("[EXTRACT_SOLUTION_PLAN] Adding successor %d to to_visit", succ_id));
						}
						Dictionary succ_node = p_graph.get_node(succ_id);
						// Validate successor node exists and has required fields
						if (succ_node.is_empty() || !succ_node.has("status")) {
							// Skip invalid successor nodes (may have been removed)
							if (p_verbose >= 2) {
								print_line(vformat("[EXTRACT_SOLUTION_PLAN] Successor %d is invalid (empty=%s, has_status=%s), skipping",
										succ_id, succ_node.is_empty() ? "YES" : "NO",
										succ_node.has("status") ? "YES" : "NO"));
							}
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

Dictionary PlannerGraphOperations::execute_solution_graph(PlannerSolutionGraph &p_graph, Dictionary p_initial_state, Ref<PlannerDomain> p_domain) {
	if (!p_domain.is_valid()) {
		ERR_PRINT("PlannerGraphOperations::execute_solution_graph: domain is invalid");
		return p_initial_state.duplicate(true);
	}

	Dictionary state = p_initial_state.duplicate(true);
	Dictionary action_dict = p_domain->action_dictionary;
	Array to_visit;
	to_visit.push_back(0); // Start from root
	TypedArray<int> visited; // Track visited nodes to prevent revisiting

	// Build parent map (same as extract_solution_plan)
	Dictionary parent_map; // child_id -> parent_id
	Dictionary graph_dict = p_graph.get_graph();
	Array graph_keys = graph_dict.keys();

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
			// Skip self-references
			if (child_id == parent_id) {
				continue;
			}
			// In a proper tree, each node should have exactly one parent
			if (!parent_map.has(child_id)) {
				parent_map[child_id] = parent_id;
			}
			// If node already has a parent, keep the first one (graph construction bug)
		}
	}

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
			continue;
		}

		int node_type = node["type"];
		int node_status = node["status"];

		// Execute actions that are closed (successful)
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

			// Extract action info
			if (info.get_type() != Variant::ARRAY) {
				continue;
			}
			Array action_arr = info;
			if (action_arr.is_empty() || action_arr.size() < 1) {
				continue;
			}

			// Safety check: ensure action_arr is not empty
			if (action_arr.is_empty() || action_arr.size() < 1) {
				continue;
			}
			String action_name = action_arr[0];
			if (!action_dict.has(action_name)) {
				// Action not found in dictionary, skip
				continue;
			}

			// Get action callable and execute
			Callable action_callable = action_dict[action_name];
			Array args;
			args.push_back(state);
			args.append_array(action_arr.slice(1, action_arr.size()));

			Variant result = action_callable.callv(args);
			if (result.get_type() == Variant::DICTIONARY) {
				// Action succeeded, update state
				state = result;
			} else {
				// Action failed, stop execution
				break;
			}
		}

		// Visit successors of closed nodes (or root)
		if (node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) ||
				node_id == 0) {
			if (!node.has("successors")) {
				continue;
			}
			TypedArray<int> successors = node["successors"];
			// Add successors in reverse order to maintain DFS order
			for (int i = successors.size() - 1; i >= 0; i--) {
				int succ_id = successors[i];
				if (!visited.has(succ_id)) {
					// Verify this successor is actually a child of the current node
					int parent_of_succ = parent_map.get(succ_id, -1);
					if (parent_of_succ == node_id) {
						Dictionary succ_node = p_graph.get_node(succ_id);
						if (succ_node.is_empty() || !succ_node.has("status")) {
							continue;
						}
						int succ_status = succ_node["status"];
						// Only follow closed nodes (or root)
						if (succ_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) ||
								succ_id == 0) {
							to_visit.push_back(succ_id);
						}
					}
				}
			}
		}
	}

	return state;
}
