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

namespace {
// Returns true if subtasks_copy was already in p_blacklist (by deep comparison). Otherwise adds it and returns false.
bool blacklist_contains_array(const TypedArray<Variant> &p_blacklist, const Array &p_subtasks) {
	for (int i = 0; i < p_blacklist.size(); i++) {
		Variant bl = p_blacklist[i];
		if (bl.get_type() != Variant::ARRAY) {
			continue;
		}
		Array bl_arr = bl;
		if (bl_arr.size() != p_subtasks.size()) {
			continue;
		}
		bool match = true;
		for (int j = 0; j < p_subtasks.size(); j++) {
			Variant a = p_subtasks[j];
			Variant b = bl_arr[j];
			if (a.get_type() == Variant::ARRAY && b.get_type() == Variant::ARRAY) {
				Array aa = a;
				Array ab = b;
				if (aa.size() != ab.size()) {
					match = false;
					break;
				}
				for (int k = 0; k < aa.size(); k++) {
					if (aa[k] != ab[k]) {
						match = false;
						break;
					}
				}
			} else if (a != b) {
				match = false;
				break;
			}
		}
		if (match) {
			return true;
		}
	}
	return false;
}

void add_to_blacklist_if_new(TypedArray<Variant> &p_blacklist, const Array &p_subtasks_copy) {
	if (blacklist_contains_array(p_blacklist, p_subtasks_copy)) {
		return;
	}
	p_blacklist.push_back(p_subtasks_copy.duplicate(true));
}
} // namespace

static PlannerBacktracking::BacktrackResult make_failure_result(PlannerSolutionGraph &p_graph, const Dictionary &p_state, const TypedArray<Variant> &p_blacklist) {
	PlannerBacktracking::BacktrackResult result;
	result.parent_node_id = -1;
	result.current_node_id = -1;
	result.graph = p_graph;
	result.state = p_state;
	result.blacklisted_commands = p_blacklist;
	return result;
}

PlannerBacktracking::BacktrackResult PlannerBacktracking::backtrack(PlannerSolutionGraph p_graph, int p_parent_node_id, int p_current_node_id, Dictionary p_state, TypedArray<Variant> p_blacklisted_commands, int p_verbose) {
	if (p_verbose >= 2) {
		print_line(vformat("Backtracking: parent_node_id=%d, current_node_id=%d", p_parent_node_id, p_current_node_id));
	}

	p_graph.set_node_status(p_current_node_id, PlannerNodeStatus::STATUS_FAILED);

	// Guard: from root, try OPEN sibling first
	if (p_parent_node_id == 0) {
		int open_node_id = -1;
		Dictionary root_node = p_graph.get_node(0);
		TypedArray<int> root_successors = root_node["successors"];
		for (int i = 0; i < root_successors.size(); i++) {
			int child_id = root_successors[i];
			if (child_id == p_current_node_id || !p_graph.get_graph().has(child_id)) {
				continue;
			}
			Dictionary child_node = p_graph.get_node(child_id);
			if (child_node["status"] == Variant(PlannerNodeStatus::STATUS_OPEN)) {
				open_node_id = child_id;
				break;
			}
		}
		if (p_verbose >= 3) {
			print_line(open_node_id >= 0
					? vformat("Backtracking from root: Found OPEN node %d at root (excluding failed node %d)", open_node_id, p_current_node_id)
					: vformat("Backtracking from root: No OPEN nodes found at root (excluding failed node %d)", p_current_node_id));
		}
		if (open_node_id >= 0) {
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

	Dictionary current_node = p_graph.get_node(p_current_node_id);

	// Guard: invalid node
	if (!current_node.has("type")) {
		return make_failure_result(p_graph, p_state, p_blacklisted_commands);
	}

	int current_node_type = current_node["type"];

	// Guard: COMMAND failure → retry parent task only
	if (current_node_type == static_cast<int>(PlannerNodeType::TYPE_COMMAND)) {
		Array subtasks_to_blacklist = PlannerGraphOperations::get_successors_info_array(p_graph, p_parent_node_id);
		PlannerGraphOperations::remove_descendants(p_graph, p_parent_node_id, false);
		if (p_verbose >= 2) {
			print_line(vformat("Backtracking: COMMAND failure, early return with parent_node_id=%d", p_parent_node_id));
		}
		TypedArray<Variant> updated_blacklist = p_blacklisted_commands;
		if (subtasks_to_blacklist.size() > 0) {
			Array copy = subtasks_to_blacklist.duplicate(true);
			add_to_blacklist_if_new(updated_blacklist, copy);
			if (p_verbose >= 2 && updated_blacklist.size() > p_blacklisted_commands.size()) {
				print_line(vformat("Backtracking: Blacklisted parent task %d method expansion (command failure)", p_parent_node_id));
			}
		}
		p_graph.set_node_status(p_parent_node_id, PlannerNodeStatus::STATUS_OPEN);
		Dictionary parent_task_node = p_graph.get_node(p_parent_node_id);
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
	}

	// Reset TASK/UNIGOAL/MULTIGOAL node (IPyHOP-style)
	if (current_node_type == static_cast<int>(PlannerNodeType::TYPE_TASK) ||
			current_node_type == static_cast<int>(PlannerNodeType::TYPE_UNIGOAL) ||
			current_node_type == static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
		current_node["selected_method"] = Variant();
		current_node["state"] = Dictionary();
		p_graph.update_node(p_current_node_id, current_node);
	}

	// Remove failed node and its descendants from parent
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

	int closed_node_id = -1;
	for (int i = dfs_list.size() - 1; i >= 0; i--) {
		int node_id = dfs_list[i];
		if (node_id == p_current_node_id || !p_graph.get_graph().has(node_id)) {
			continue;
		}
		Dictionary node = p_graph.get_node(node_id);
		int status = node["status"];
		int node_type = node["type"];
		if (status != static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
			continue;
		}
		if (node_type != static_cast<int>(PlannerNodeType::TYPE_TASK) &&
				node_type != static_cast<int>(PlannerNodeType::TYPE_UNIGOAL) &&
				node_type != static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
			continue;
		}
		if (!node.has("available_methods")) {
			continue;
		}
		Variant methods_var = node["available_methods"];
		if (methods_var.get_type() != Variant::ARRAY) {
			continue;
		}
		TypedArray<Callable> available_methods = TypedArray<Callable>(methods_var);
		if (available_methods.size() == 0) {
			continue;
		}
		TypedArray<int> successors = node["successors"];
		if (successors.size() == 0) {
			continue;
		}
		closed_node_id = node_id;
		if (p_verbose >= 3) {
			print_line(vformat("Backtracking: Found retriable CLOSED node %d", closed_node_id));
		}
		break;
	}

	// Guard: no CLOSED node to retry → return failure
	if (closed_node_id < 0) {
		if (p_verbose >= 3) {
			print_line("Backtracking: No retriable CLOSED node found");
		}
		if (p_verbose >= 2) {
			print_line("Backtracking: No CLOSED node found, returning failure");
		}
		PlannerGraphOperations::remove_descendants(p_graph, 0, false);
		return make_failure_result(p_graph, p_state, p_blacklisted_commands);
	}

	if (p_verbose >= 3) {
		print_line(vformat("Backtracking: Found CLOSED node %d to retry", closed_node_id));
	}

	// Reopen CLOSED node: blacklist its method expansion, set OPEN, clear descendants and reset fields.
	// If we're reopening a sibling of the failed branch (e.g. failed in tm_2, reopening tm_1), clear
	// the blacklist so the other branch (tm_2) can retry all methods with the new state (IPyHOP sample_1).
	int closed_node_parent = PlannerGraphOperations::find_predecessor(p_graph, closed_node_id);
	bool reopening_sibling = (closed_node_parent >= 0 && closed_node_parent == PlannerGraphOperations::find_predecessor(p_graph, p_parent_node_id));
	TypedArray<Variant> updated_blacklist = p_blacklisted_commands;
	if (reopening_sibling) {
		updated_blacklist.clear();
		if (p_verbose >= 2) {
			print_line(vformat("Backtracking: Cleared blacklist (reopening sibling node %d of failed branch)", closed_node_id));
		}
	}
	Array subtasks_to_blacklist = PlannerGraphOperations::get_successors_info_array(p_graph, closed_node_id);
	if (subtasks_to_blacklist.size() > 0) {
		Array copy = subtasks_to_blacklist.duplicate(true);
		int n_before = updated_blacklist.size();
		add_to_blacklist_if_new(updated_blacklist, copy);
		if (p_verbose >= 2 && updated_blacklist.size() > n_before) {
			print_line(vformat("Backtracking: Blacklisted reopened node %d method expansion (size %d)", closed_node_id, copy.size()));
		}
	}

	p_graph.set_node_status(closed_node_id, PlannerNodeStatus::STATUS_OPEN);
	PlannerGraphOperations::remove_descendants(p_graph, closed_node_id);

	Dictionary closed_node = p_graph.get_node(closed_node_id);
	closed_node["selected_method"] = Variant();
	closed_node["state"] = Dictionary();
	closed_node["stn_snapshot"] = Variant();
	closed_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_OPEN);
	p_graph.update_node(closed_node_id, closed_node);

	BacktrackResult result;
	result.parent_node_id = closed_node_parent >= 0 ? closed_node_parent : p_parent_node_id;
	result.current_node_id = closed_node_id;
	result.graph = p_graph;
	result.state = p_state;
	result.blacklisted_commands = updated_blacklist;
	return result;
}
