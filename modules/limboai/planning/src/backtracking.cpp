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

	// When a command fails, remove the entire method expansion (all children of the parent task),
	// so the task can be retried with the next method. Otherwise we only remove the failed node.
	if (current_node_type == static_cast<int>(PlannerNodeType::TYPE_COMMAND)) {
		// Derive method array from parent's successors before removing them (no created_subtasks).
		Array subtasks_to_blacklist = PlannerGraphOperations::get_successors_info_array(p_graph, p_parent_node_id);
		PlannerGraphOperations::remove_descendants(p_graph, p_parent_node_id, false);
		// Early return: retry the parent task (do not run DFS/fallback so we never return root).
		if (p_verbose >= 2) {
			print_line(vformat("Backtracking: COMMAND failure, early return with parent_node_id=%d", p_parent_node_id));
		}
		Dictionary parent_task_node = p_graph.get_node(p_parent_node_id);
		TypedArray<Variant> updated_blacklist = p_blacklisted_commands;
		if (subtasks_to_blacklist.size() > 0) {
			Array subtasks_copy = subtasks_to_blacklist.duplicate(true);
			bool already_blacklisted = false;
			for (int i = 0; i < updated_blacklist.size(); i++) {
				Variant blacklisted = updated_blacklist[i];
				if (blacklisted.get_type() == Variant::ARRAY) {
					Array blacklisted_arr = blacklisted;
					if (blacklisted_arr.size() == subtasks_copy.size()) {
						bool match = true;
						for (int j = 0; j < subtasks_copy.size(); j++) {
							Variant task_elem = subtasks_copy[j];
							Variant bl_elem = blacklisted_arr[j];
							if (task_elem.get_type() == Variant::ARRAY && bl_elem.get_type() == Variant::ARRAY) {
								Array ta = task_elem;
								Array ba = bl_elem;
								if (ta.size() != ba.size()) {
									match = false;
									break;
								}
								for (int k = 0; k < ta.size(); k++) {
									if (ta[k] != ba[k]) {
										match = false;
										break;
									}
								}
							} else if (task_elem != bl_elem) {
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
					print_line(vformat("Backtracking: Blacklisted parent task %d method expansion (command failure)", p_parent_node_id));
				}
			}
		}
		p_graph.set_node_status(p_parent_node_id, PlannerNodeStatus::STATUS_OPEN);
		parent_task_node["selected_method"] = Variant();
		parent_task_node["state"] = Dictionary();
		parent_task_node["stn_snapshot"] = Variant();
		p_graph.update_node(p_parent_node_id, parent_task_node);
		BacktrackResult result;
		result.parent_node_id = p_parent_node_id;
		result.current_node_id = p_parent_node_id;
		result.graph = p_graph;
		result.state = p_state;
		result.blacklisted_commands = updated_blacklist;
		return result;
	} else {
		// Remove descendants of the failed node and remove it from parent's successors list
		PlannerGraphOperations::remove_descendants(p_graph, p_current_node_id, true);
	}

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
			print_line("Backtracking: No CLOSED node found, returning failure");
		}
	}

	if (closed_node_id >= 0) {
		// Found a CLOSED node with available methods, retry it
		Dictionary closed_node = p_graph.get_node(closed_node_id);

		// CRITICAL: Before reopening, blacklist the method array that this node used (derive from successors).
		// This ensures that when the node is reopened, it will skip the method that led to failure
		// and try the next method instead (TLA+ model insight). No created_subtasks - use graph.
		Array subtasks_to_blacklist = PlannerGraphOperations::get_successors_info_array(p_graph, closed_node_id);
		TypedArray<Variant> updated_blacklist = p_blacklisted_commands;
		if (subtasks_to_blacklist.size() > 0) {
				Array subtasks_copy = subtasks_to_blacklist.duplicate(true);
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
						print_line(vformat("Backtracking: Blacklisted reopened node %d method expansion (size %d)",
								closed_node_id, subtasks_copy.size()));
					}
				}
		}

		// Set to OPEN
		p_graph.set_node_status(closed_node_id, PlannerNodeStatus::STATUS_OPEN);

		// Remove old descendants before retrying (like IPyHOP line 406-408)
		PlannerGraphOperations::remove_descendants(p_graph, closed_node_id);

		// Reset selected_method (IPyHOP-style)
		// Clear state snapshot so we use the current state (with successful actions) instead of restoring old state
		// Re-fetch node after remove_descendants so we don't overwrite OPEN status with stale closed_node
		closed_node = p_graph.get_node(closed_node_id);
		closed_node["selected_method"] = Variant();
		closed_node["state"] = Dictionary();
		closed_node["stn_snapshot"] = Variant();
		closed_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_OPEN);
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

	// No CLOSED node found in DFS: clean tree like IPyHOP (remove all descendants of root), then return failure
	PlannerGraphOperations::remove_descendants(p_graph, 0, false);
	if (p_verbose >= 2) {
		print_line("Backtracking: No CLOSED node found, returning failure");
	}
	BacktrackResult result;
	result.parent_node_id = -1;
	result.current_node_id = -1;
	result.graph = p_graph;
	result.state = p_state;
	result.blacklisted_commands = p_blacklisted_commands;
	return result;
}
