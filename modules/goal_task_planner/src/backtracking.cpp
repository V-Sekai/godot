/**************************************************************************/
/*  backtracking.cpp                                                      */
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

#include "backtracking.h"
#include "graph_operations.h"

PlannerBacktracking::BacktrackResult PlannerBacktracking::backtrack(PlannerSolutionGraph p_graph, int p_parent_node_id, int p_current_node_id, Dictionary p_state, TypedArray<Variant> p_blacklisted_commands, int p_verbose) {
	if (p_verbose >= 2) {
		print_line(vformat("Backtracking: parent_node_id=%d, current_node_id=%d", p_parent_node_id, p_current_node_id));
	}

	// Mark current node as failed first
	p_graph.set_node_status(p_current_node_id, PlannerNodeStatus::STATUS_FAILED);

	// If backtracking from root, check for OPEN nodes (excluding the failed node)
	// This ensures we try other OPEN tasks before retrying CLOSED ones or giving up
	if (p_parent_node_id == 0) {
		Dictionary root_node = p_graph.get_node(0);
		TypedArray<int> root_successors = root_node["successors"];
		int open_node_id = -1;
		// Check root children for OPEN nodes, excluding the failed node
		for (int i = 0; i < root_successors.size(); i++) {
			int child_id = root_successors[i];
			if (child_id == p_current_node_id) {
				continue; // Skip the failed node
			}
			if (!p_graph.get_graph().has(child_id)) {
				continue;
			}
			Dictionary child_node = p_graph.get_node(child_id);
			int status = child_node["status"];
			if (status == static_cast<int>(PlannerNodeStatus::STATUS_OPEN)) {
				open_node_id = child_id;
				break;
			}
		}
		if (p_verbose >= 3) {
			if (open_node_id >= 0) {
				print_line(vformat("Backtracking from root: Found OPEN node %d at root (excluding failed node %d)", open_node_id, p_current_node_id));
			} else {
				print_line(vformat("Backtracking from root: No OPEN nodes found at root (excluding failed node %d)", p_current_node_id));
			}
		}
		if (open_node_id >= 0) {
			// Found an OPEN node at root, remove failed node and return OPEN node for retry
			if (p_verbose >= 2) {
				print_line(vformat("Backtracking: Returning OPEN node %d at root (failed node %d)", open_node_id, p_current_node_id));
			}
			PlannerGraphOperations::remove_descendants(p_graph, p_current_node_id, true);
			BacktrackResult result;
			result.parent_node_id = 0;
			result.current_node_id = open_node_id;
			result.graph = p_graph;
			result.state = p_state;
			result.blacklisted_commands = p_blacklisted_commands;
			return result;
		}
	}

	// Reset current node's selected_method and state (IPyHOP-style)
	Dictionary current_node = p_graph.get_node(p_current_node_id);

	// Guard: Node must have type field
	if (!current_node.has("type")) {
		// Invalid node structure, return failure
		BacktrackResult result;
		result.parent_node_id = -1;
		result.current_node_id = -1;
		result.graph = p_graph;
		result.state = p_state;
		result.blacklisted_commands = p_blacklisted_commands;
		return result;
	}
	int current_node_type = current_node["type"];
	if (current_node_type == static_cast<int>(PlannerNodeType::TYPE_TASK) ||
			current_node_type == static_cast<int>(PlannerNodeType::TYPE_UNIGOAL) ||
			current_node_type == static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
		current_node["selected_method"] = Variant();
		current_node["state"] = Dictionary(); // Clear state - will be restored from saved snapshot if needed
		// Note: available_methods will be re-fetched from domain when node is retried
		p_graph.update_node(p_current_node_id, current_node);
	}

	// Remove descendants of the failed node and remove it from parent's successors list
	// This prevents failed nodes from being considered as part of the solution path
	PlannerGraphOperations::remove_descendants(p_graph, p_current_node_id, true);

	// IPyHOP-style backtracking: Do reverse DFS from parent to find first CLOSED node
	// This matches IPyHOP's _backtrack behavior (ipyhop/planner.py lines 401-410)
	// The DFS includes ALL nodes in the subtree rooted at parent, including siblings
	if (p_verbose >= 3) {
		print_line(vformat("Backtracking: Finding first CLOSED node (start=%d, failed=%d)", p_parent_node_id, p_current_node_id));
	}

	// Do DFS preorder traversal from start node
	TypedArray<int> visited;
	TypedArray<int> dfs_list;
	// Helper function for DFS preorder (inlined from graph_operations)
	struct DFSHelper {
		PlannerSolutionGraph &graph;
		TypedArray<int> &visited;
		TypedArray<int> &dfs_list;

		void do_dfs_preorder(int node_id) {
			if (visited.has(node_id)) {
				return;
			}
			visited.push_back(node_id);
			dfs_list.push_back(node_id);
			Dictionary node = graph.get_node(node_id);
			TypedArray<int> successors = node["successors"];
			for (int i = 0; i < successors.size(); i++) {
				int succ_id = successors[i];
				if (graph.get_graph().has(succ_id)) {
					do_dfs_preorder(succ_id);
				}
			}
		}
	};
	DFSHelper dfs_helper = { p_graph, visited, dfs_list };
	dfs_helper.do_dfs_preorder(p_parent_node_id);

	if (p_verbose >= 3) {
		print_line(vformat("Backtracking: DFS collected %d nodes", dfs_list.size()));
	}

	// Traverse in reverse order to find first CLOSED node with descendants
	int closed_node_id = -1;
	for (int i = dfs_list.size() - 1; i >= 0; i--) {
		int node_id = dfs_list[i];
		if (node_id == p_current_node_id) {
			continue; // Skip the failed node
		}

		// Check if node is retriable (CLOSED with descendants and available methods)
		if (!p_graph.get_graph().has(node_id)) {
			continue;
		}
		Dictionary node = p_graph.get_node(node_id);
		int status = node["status"];
		int node_type = node["type"];

		// Must be CLOSED and of type TASK/UNIGOAL/MULTIGOAL
		if (status != static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) ||
				(node_type != static_cast<int>(PlannerNodeType::TYPE_TASK) &&
						node_type != static_cast<int>(PlannerNodeType::TYPE_UNIGOAL) &&
						node_type != static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL))) {
			continue;
		}

		// Must have available methods
		if (!node.has("available_methods")) {
			continue;
		}
		Variant methods_var = node["available_methods"];
		if (methods_var.get_type() != Variant::ARRAY) {
			continue;
		}
		Array methods_array = methods_var;
		TypedArray<Callable> available_methods = TypedArray<Callable>(methods_array);
		if (available_methods.size() == 0) {
			continue;
		}

		// Must have successors (IPyHOP only retries if it has descendants)
		TypedArray<int> successors = node["successors"];
		if (successors.size() > 0) {
			closed_node_id = node_id;
			if (p_verbose >= 3) {
				print_line(vformat("Backtracking: Found retriable CLOSED node %d", closed_node_id));
			}
			break;
		}
	}

	if (p_verbose >= 3 && closed_node_id < 0) {
		print_line("Backtracking: No retriable CLOSED node found");
	}

	if (p_verbose >= 3) {
		if (closed_node_id >= 0) {
			print_line(vformat("Backtracking: Found CLOSED node %d to retry", closed_node_id));
		} else {
			print_line("Backtracking: No CLOSED node found, falling back to parent chain traversal");
		}
	}

	if (closed_node_id >= 0) {
		// Found a CLOSED node with available methods, retry it
		Dictionary closed_node = p_graph.get_node(closed_node_id);

		// CRITICAL: Before reopening, blacklist the method array that this node used
		// This ensures that when the node is reopened, it will skip the method that led to failure
		// and try the next method instead (TLA+ model insight)
		TypedArray<Variant> updated_blacklist = p_blacklisted_commands;
		if (closed_node.has("created_subtasks")) {
			Array created_subtasks = closed_node["created_subtasks"];
			if (created_subtasks.size() > 0) {
				// Store a deep copy to avoid reference issues
				Array subtasks_copy = created_subtasks.duplicate(true);
				// Check if not already blacklisted to avoid duplicates (using nested comparison)
				bool already_blacklisted = false;
				for (int i = 0; i < updated_blacklist.size(); i++) {
					Variant blacklisted = updated_blacklist[i];
					if (blacklisted.get_type() == Variant::ARRAY) {
						Array blacklisted_arr = blacklisted;
						if (blacklisted_arr.size() == subtasks_copy.size()) {
							bool match = true;
							for (int j = 0; j < subtasks_copy.size(); j++) {
								Variant subtask_elem = subtasks_copy[j];
								Variant blacklisted_elem = blacklisted_arr[j];
								// Nested array comparison
								if (subtask_elem.get_type() == Variant::ARRAY && blacklisted_elem.get_type() == Variant::ARRAY) {
									Array subtask_elem_arr = subtask_elem;
									Array blacklisted_elem_arr = blacklisted_elem;
									if (subtask_elem_arr.size() != blacklisted_elem_arr.size()) {
										match = false;
										break;
									}
									for (int k = 0; k < subtask_elem_arr.size(); k++) {
										if (subtask_elem_arr[k] != blacklisted_elem_arr[k]) {
											match = false;
											break;
										}
									}
									if (!match) {
										break;
									}
								} else if (subtask_elem != blacklisted_elem) {
									match = false;
									break;
								}
							}
							if (match) {
								already_blacklisted = true;
								break;
							}
						}
					}
				}
				if (!already_blacklisted) {
					updated_blacklist.push_back(subtasks_copy);
					if (p_verbose >= 2) {
						print_line(vformat("Backtracking: Blacklisted reopened node %d's created_subtasks (size %d)",
								closed_node_id, subtasks_copy.size()));
					}
				}
			}
		}

		// Set to OPEN
		p_graph.set_node_status(closed_node_id, PlannerNodeStatus::STATUS_OPEN);

		// Remove old descendants before retrying (like IPyHOP line 406-408)
		PlannerGraphOperations::remove_descendants(p_graph, closed_node_id);

		// Reset selected_method (IPyHOP-style)
		// Clear state snapshot so we use the current state (with successful actions) instead of restoring old state
		// This is critical: when reopening to try a different method, we want to preserve progress from previous attempts
		closed_node["selected_method"] = Variant();
		closed_node["state"] = Dictionary(); // Clear state snapshot - use current state instead
		closed_node["stn_snapshot"] = Variant(); // Clear STN snapshot too
		// Clear created_subtasks since we're trying a different method
		closed_node["created_subtasks"] = Variant();
		p_graph.update_node(closed_node_id, closed_node);

		// Find the predecessor of the closed node to return as parent
		int closed_node_parent = PlannerGraphOperations::find_predecessor(p_graph, closed_node_id);

		BacktrackResult result;
		result.parent_node_id = closed_node_parent >= 0 ? closed_node_parent : p_parent_node_id;
		result.current_node_id = closed_node_id;
		result.graph = p_graph;
		// Preserve the current state which includes successful actions from this method
		// This is the state at the point of failure, which includes all successful actions
		result.state = p_state;
		result.blacklisted_commands = updated_blacklist;
		return result;
	}

	// No CLOSED node found in DFS, fall back to parent chain traversal
	// Find the nearest ancestor that can be retried
	int new_parent_node_id = p_parent_node_id;

	// Traverse up the tree to find a node that can be retried
	while (new_parent_node_id >= 0) {
		Dictionary node = p_graph.get_node(new_parent_node_id);
		// Validate required dictionary keys exist
		if (!node.has("type")) {
			// Invalid node structure, return failure
			BacktrackResult result;
			result.parent_node_id = -1;
			result.current_node_id = -1;
			result.graph = p_graph;
			result.state = p_state;
			result.blacklisted_commands = p_blacklisted_commands;
			return result;
		}
		int node_type = node["type"];

		// Validate available_methods exists before accessing
		// Some node types may not have available_methods (e.g., ACTION nodes)
		TypedArray<Callable> available_methods;
		if (node.has("available_methods")) {
			Variant methods_var = node["available_methods"];
			// Try to convert to TypedArray<Callable> - handle both Array and TypedArray
			if (methods_var.get_type() == Variant::ARRAY) {
				Array methods_array = methods_var;
				// Convert to TypedArray and check size after conversion
				available_methods = TypedArray<Callable>(methods_array);
			}
		}

		// Check if this node has alternative methods
		bool can_retry = false;

		if (node_type == static_cast<int>(PlannerNodeType::TYPE_TASK) ||
				node_type == static_cast<int>(PlannerNodeType::TYPE_UNIGOAL) ||
				node_type == static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
			// Check if there are available methods
			if (available_methods.size() > 0) {
				can_retry = true;
			}
		}

		if (can_retry) {
			// Found a node with available methods, retry it
			// CRITICAL: Get node BEFORE modifying status to preserve created_subtasks for blacklisting
			Dictionary node_to_retry = p_graph.get_node(new_parent_node_id);

			// CRITICAL: Before reopening, blacklist the method array that this node used
			// This ensures that when the node is reopened, it will skip the method that led to failure
			// and try the next method instead (same logic as CLOSED node case)
			TypedArray<Variant> updated_blacklist = p_blacklisted_commands;
			if (node_to_retry.has("created_subtasks")) {
				Array created_subtasks = node_to_retry["created_subtasks"];
				if (created_subtasks.size() > 0) {
					// Store a deep copy to avoid reference issues
					Array subtasks_copy = created_subtasks.duplicate(true);
					// Check if not already blacklisted to avoid duplicates (using nested comparison)
					bool already_blacklisted = false;
					for (int i = 0; i < updated_blacklist.size(); i++) {
						Variant blacklisted = updated_blacklist[i];
						if (blacklisted.get_type() == Variant::ARRAY) {
							Array blacklisted_arr = blacklisted;
							if (blacklisted_arr.size() == subtasks_copy.size()) {
								bool match = true;
								for (int j = 0; j < subtasks_copy.size(); j++) {
									Variant subtask_elem = subtasks_copy[j];
									Variant blacklisted_elem = blacklisted_arr[j];
									// Nested array comparison
									if (subtask_elem.get_type() == Variant::ARRAY && blacklisted_elem.get_type() == Variant::ARRAY) {
										Array subtask_elem_arr = subtask_elem;
										Array blacklisted_elem_arr = blacklisted_elem;
										if (subtask_elem_arr.size() != blacklisted_elem_arr.size()) {
											match = false;
											break;
										}
										for (int k = 0; k < subtask_elem_arr.size(); k++) {
											if (subtask_elem_arr[k] != blacklisted_elem_arr[k]) {
												match = false;
												break;
											}
										}
										if (!match) {
											break;
										}
									} else if (subtask_elem != blacklisted_elem) {
										match = false;
										break;
									}
								}
								if (match) {
									already_blacklisted = true;
									break;
								}
							}
						}
					}
					if (!already_blacklisted) {
						updated_blacklist.push_back(subtasks_copy);
						if (p_verbose >= 2) {
							print_line(vformat("Backtracking: Blacklisted reopened node %d's created_subtasks (size %d) in parent chain",
									new_parent_node_id, subtasks_copy.size()));
						}
					}
				}
			}

			// Set to OPEN (after blacklisting, before clearing created_subtasks)
			p_graph.set_node_status(new_parent_node_id, PlannerNodeStatus::STATUS_OPEN);

			// Remove old descendants before retrying (like IPyHOP line 406-408)
			// This ensures old nodes from previous attempts are removed from the graph
			PlannerGraphOperations::remove_descendants(p_graph, new_parent_node_id);

			// Reset selected_method and clear state snapshot (IPyHOP-style)
			// Clear state snapshot so we use the current state (with successful actions) instead of restoring old state
			// This allows the node to try all methods again from the beginning, but with current state
			// IPyHOP behavior: available_methods = iter(c_node['methods']) - reset iterator
			// In C++, we retrieve methods fresh from domain each time, so this is already handled
			// But we need to clear any cached method selection
			// Use the node we already fetched (node_to_retry) to avoid losing data
			node_to_retry["selected_method"] = Variant();
			node_to_retry["state"] = Dictionary(); // Clear state snapshot - use current state instead
			node_to_retry["stn_snapshot"] = Variant(); // Clear STN snapshot too
			node_to_retry["created_subtasks"] = Variant(); // Clear created_subtasks
			p_graph.update_node(new_parent_node_id, node_to_retry);

			BacktrackResult result;
			// Return the retriable node as parent_node_id so planning can continue from it
			result.parent_node_id = new_parent_node_id;
			result.current_node_id = new_parent_node_id;
			result.graph = p_graph;
			result.state = p_state;
			result.blacklisted_commands = updated_blacklist; // Use updated blacklist instead of original
			return result;
		} else {
			// No more methods, this node also fails, continue backtracking
			p_graph.set_node_status(new_parent_node_id, PlannerNodeStatus::STATUS_FAILED);
			new_parent_node_id = PlannerGraphOperations::find_predecessor(p_graph, new_parent_node_id);
		}
	}

	// Reached root, return failure
	if (p_verbose >= 2) {
		print_line("Backtracking: Reached root, no retriable nodes found, returning failure");
	}
	BacktrackResult result;
	result.parent_node_id = -1;
	result.current_node_id = -1;
	result.graph = p_graph;
	result.state = p_state;
	result.blacklisted_commands = p_blacklisted_commands;
	return result;
}
