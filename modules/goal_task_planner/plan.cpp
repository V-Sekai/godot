/**************************************************************************/
/*  plan.cpp                                                              */
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

#include "plan.h"

#include "core/io/json.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/variant/callable.h"
#include "core/variant/typed_array.h"

#include "backtracking.h"
#include "domain.h"
#include "graph_operations.h"
#include "multigoal.h"
#include "stn_constraints.h"

int PlannerPlan::get_verbose() const {
	return verbose;
}

void PlannerPlan::set_verbose(int p_verbose) {
	verbose = p_verbose;
}

TypedArray<PlannerDomain> PlannerPlan::get_domains() const {
	return domains;
}

Ref<PlannerDomain> PlannerPlan::get_current_domain() const {
	return current_domain;
}

void PlannerPlan::set_domains(TypedArray<PlannerDomain> p_domain) {
	domains = p_domain;
}

Variant PlannerPlan::find_plan(Dictionary p_state, Array p_todo_list) {
	// Note: Array is a value type in Godot and cannot be null, so no null check needed

	if (verbose >= 1) {
		print_line("verbose=" + itos(verbose) + ":");
		print_line("    state = " + _item_to_string(p_state));
		print_line("    todo_list = " + _item_to_string(p_todo_list));
		if (verbose >= 2 && current_domain.is_valid()) {
			Dictionary action_dict = current_domain->action_dictionary;
			Array action_keys = action_dict.keys();
			print_line("    Available actions: " + _item_to_string(action_keys));
		}
	}

	// Initialize solution graph
	solution_graph = PlannerSolutionGraph();
	blacklisted_commands.clear();

	// Initialize STN solver (optional, but keep for consistency)
	stn.clear();
	stn.add_time_point("origin");

	// Initialize time range if not already set
	if (time_range.get_start_time() == 0) {
		time_range.set_start_time(PlannerTimeRange::now_microseconds());
	}

	// Anchor origin to current absolute time
	PlannerSTNConstraints::anchor_to_origin(stn, "origin", time_range.get_start_time());

	// Validate that current_domain is set before accessing its members
	if (!current_domain.is_valid()) {
		if (verbose >= 1) {
			print_line("result = false (no domain set)");
		}
		ERR_PRINT("PlannerPlan::find_plan: current_domain is not set. Call set_current_domain() before planning.");
		return false;
	}

	// Add initial tasks to the solution graph
	int parent_node_id = 0; // Root node
	PlannerGraphOperations::add_nodes_and_edges(
			solution_graph,
			parent_node_id,
			p_todo_list,
			current_domain->action_dictionary,
			current_domain->task_method_dictionary,
			current_domain->unigoal_method_dictionary,
			current_domain->multigoal_method_list);

	// Start planning loop
	Dictionary final_state = _planning_loop_recursive(parent_node_id, p_state, 0);

	// Check if planning succeeded (if we got back to root with a valid state)
	// Planning succeeds if all nodes are closed and we're back at root
	Dictionary root_node = solution_graph.get_node(0);

	// Check if all nodes reachable from root are closed (planning succeeded)
	// Only consider nodes that are reachable from root via closed nodes
	// This way, failed nodes that were removed from their parent's successors don't cause planning to fail
	bool planning_succeeded = true;
	Dictionary graph = solution_graph.get_graph();
	Array graph_keys = graph.keys();
	Array failed_nodes;
	Array open_nodes;
	TypedArray<int> reachable_nodes;
	TypedArray<int> to_visit;
	to_visit.push_back(0); // Start from root
	TypedArray<int> visited;

	// Find all nodes reachable from root via closed nodes
	while (!to_visit.is_empty()) {
		int node_id = to_visit.pop_back();
		if (visited.has(node_id)) {
			continue;
		}
		visited.push_back(node_id);
		reachable_nodes.push_back(node_id);

		Dictionary node = graph[node_id];
		int status = node["status"];
		// Only traverse through closed nodes (or root which is NA)
		if (status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED) ||
				node_id == 0) {
			TypedArray<int> successors = node["successors"];
			for (int j = 0; j < successors.size(); j++) {
				int succ_id = successors[j];
				if (!visited.has(succ_id)) {
					to_visit.push_back(succ_id);
				}
			}
		}
	}

	// Check reachable nodes for failures
	bool has_reachable_closed_nodes = false;
	for (int i = 0; i < reachable_nodes.size(); i++) {
		int node_id = reachable_nodes[i];
		if (node_id == 0) {
			continue; // Skip root
		}
		Dictionary node = graph[node_id];
		int status = node["status"];
		// Planning fails if any reachable node is open or failed
		if (status == static_cast<int>(PlannerNodeStatus::STATUS_OPEN)) {
			planning_succeeded = false;
			open_nodes.push_back(node_id);
		} else if (status == static_cast<int>(PlannerNodeStatus::STATUS_FAILED)) {
			planning_succeeded = false;
			failed_nodes.push_back(node_id);
		} else if (status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
			has_reachable_closed_nodes = true;
		}
	}

	// If no reachable closed nodes (besides root), planning failed
	if (!has_reachable_closed_nodes) {
		planning_succeeded = false;
	}

	if (planning_succeeded && !final_state.is_empty()) {
		// Mark root node as CLOSED when planning succeeds so extract_solution_plan can traverse from it
		Dictionary root_node = solution_graph.get_node(0);
		root_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
		solution_graph.update_node(0, root_node);

		// Extract the plan from the graph
		Array plan = PlannerGraphOperations::extract_solution_plan(solution_graph);

		if (verbose >= 1) {
			print_line("result = " + _item_to_string(plan));
		}

		return plan;
	} else {
		if (verbose >= 1) {
			print_line("result = false (planning failed)");
			if (verbose >= 2 || !failed_nodes.is_empty() || !open_nodes.is_empty()) {
				// Print solution graph for debugging
				print_line("Solution graph structure:");
				for (int i = 0; i < graph_keys.size(); i++) {
					int node_id = graph_keys[i];
					Dictionary node = graph[node_id];
					int node_type = node["type"];
					int node_status = node["status"];
					Variant node_info = node["info"];
					TypedArray<int> successors = node["successors"];

					String type_str;
					switch (static_cast<PlannerNodeType>(node_type)) {
						case PlannerNodeType::TYPE_ROOT:
							type_str = "ROOT";
							break;
						case PlannerNodeType::TYPE_ACTION:
							type_str = "ACTION";
							break;
						case PlannerNodeType::TYPE_TASK:
							type_str = "TASK";
							break;
						case PlannerNodeType::TYPE_GOAL:
							type_str = "GOAL";
							break;
						case PlannerNodeType::TYPE_MULTIGOAL:
							type_str = "MULTIGOAL";
							break;
						case PlannerNodeType::TYPE_VERIFY_GOAL:
							type_str = "VERIFY_GOAL";
							break;
						case PlannerNodeType::TYPE_VERIFY_MULTIGOAL:
							type_str = "VERIFY_MULTIGOAL";
							break;
						default:
							type_str = "UNKNOWN";
							break;
					}

					String status_str;
					switch (static_cast<PlannerNodeStatus>(node_status)) {
						case PlannerNodeStatus::STATUS_OPEN:
							status_str = "OPEN";
							break;
						case PlannerNodeStatus::STATUS_CLOSED:
							status_str = "CLOSED";
							break;
						case PlannerNodeStatus::STATUS_FAILED:
							status_str = "FAILED";
							break;
						case PlannerNodeStatus::STATUS_NOT_APPLICABLE:
							status_str = "NA";
							break;
						default:
							status_str = "UNKNOWN";
							break;
					}

					String info_str = _item_to_string(node_info);
					String successors_str = "[";
					for (int j = 0; j < successors.size(); j++) {
						if (j > 0) {
							successors_str += ", ";
						}
						successors_str += itos(successors[j]);
					}
					successors_str += "]";

					print_line(vformat("  Node %d: type=%s, status=%s, info=%s, successors=%s",
							node_id, type_str, status_str, info_str, successors_str));
				}
				if (!failed_nodes.is_empty()) {
					print_line("Failed nodes: " + _item_to_string(failed_nodes));
				}
				if (!open_nodes.is_empty()) {
					print_line("Open nodes: " + _item_to_string(open_nodes));
				}
			}
		}
		return false;
	}
}

String PlannerPlan::_item_to_string(Variant p_item) {
	return String(p_item);
}

Dictionary PlannerPlan::run_lazy_lookahead(Dictionary p_state, Array p_todo_list, int p_max_tries) {
	// Note: Array is a value type in Godot and cannot be null, so no null check needed

	// Input validation: validate max_tries is positive
	if (p_max_tries <= 0) {
		if (verbose >= 1) {
			ERR_PRINT(vformat("PlannerPlan::run_lazy_lookahead: max_tries must be positive, got %d", p_max_tries));
		}
		return Dictionary(); // Return empty dictionary on error
	}

	if (verbose >= 1) {
		print_line(vformat("run_lazy_lookahead: verbose = %s, max_tries = %s", verbose, p_max_tries));
		print_line(vformat("Initial state: %s", p_state.keys()));
		print_line(vformat("To do: %s", p_todo_list));
	}

	Dictionary ordinals;
	ordinals[1] = "st";
	ordinals[2] = "nd";
	ordinals[3] = "rd";

	for (int tries = 1; tries <= p_max_tries; tries++) {
		if (verbose >= 1) {
			print_line(vformat("run_lazy_lookahead: %sth call to find_plan: %s", tries, ordinals.get(tries, "")));
		}

		Variant plan = find_plan(p_state, p_todo_list);
		if (plan == Variant(false)) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("run_lazy_lookahead: find_plan has failed after %s calls.", tries));
			}
			return p_state;
		}

		if (plan.is_array() && Array(plan).is_empty()) {
			if (verbose >= 1) {
				print_line(vformat("run_lazy_lookahead: Empty plan => success\nafter %s calls to find_plan.", tries));
			}
			if (verbose >= 2) {
				print_line(vformat("run_lazy_lookahead: final state %s", p_state));
			}
			return p_state;
		}

		if (plan.is_array()) {
			Array action_list = plan;
			for (int i = 0; i < action_list.size(); i++) {
				Array action = action_list[i];
				// Validate action array is not empty before accessing first element
				if (action.is_empty()) {
					if (verbose >= 1) {
						ERR_PRINT("run_lazy_lookahead: Found empty action in plan, skipping");
					}
					continue;
				}
				Callable action_name = current_domain->action_dictionary[action[0]];
				if (verbose >= 1) {
					String action_arguments;
					Array actions = action.slice(1, action.size());
					for (Variant element : actions) {
						action_arguments += String(" ") + String(element);
					}
					print_line(vformat("run_lazy_lookahead: Task: %s, %s", action_name.get_method(), action_arguments));
				}

				Variant result = _apply_task_and_continue(p_state, action_name, action.slice(1, action.size()));
				if (result.get_type() == Variant::DICTIONARY) {
					Dictionary new_state = result;
					if (verbose >= 2) {
						print_line(new_state);
					}
					p_state = new_state;
				} else {
					if (verbose >= 1) {
						ERR_PRINT(vformat("run_lazy_lookahead: WARNING: action %s failed; will call find_plan.", action_name));
					}
					break;
				}
			}
		}

		if (verbose >= 1 && !p_state.is_empty()) {
			print_line("RunLazyLookahead> Plan ended; will call find_plan again.");
		}
	}

	if (verbose >= 1) {
		ERR_PRINT("run_lazy_lookahead: Too many tries, giving up.");
	}
	if (verbose >= 2) {
		print_line(vformat("run_lazy_lookahead: final state %s", p_state));
	}

	return p_state;
}

Variant PlannerPlan::_apply_task_and_continue(Dictionary p_state, Callable p_command, Array p_arguments) {
	if (verbose >= 3) {
		print_line(vformat("_apply_task_and_continue %s, args = %s", p_command.get_method(), _item_to_string(p_arguments)));
	}
	Array argument;
	argument.push_back(p_state);
	argument.append_array(p_arguments);
	Variant next_state = p_command.callv(argument);
	if (!next_state) {
		if (verbose >= 3) {
			print_line(vformat("Not applicable command %s", argument));
		}
		return false;
	}

	if (verbose >= 3) {
		print_line("Applied");
		print_line(next_state);
	}
	return next_state;
}

void PlannerPlan::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_verify_goals"), &PlannerPlan::get_verify_goals);
	ClassDB::bind_method(D_METHOD("set_verify_goals", "value"), &PlannerPlan::set_verify_goals);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "verify_goals"), "set_verify_goals", "get_verify_goals");

	ClassDB::bind_method(D_METHOD("get_verbose"), &PlannerPlan::get_verbose);
	ClassDB::bind_method(D_METHOD("set_verbose", "level"), &PlannerPlan::set_verbose);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "verbose"), "set_verbose", "get_verbose");

	ClassDB::bind_method(D_METHOD("get_max_depth"), &PlannerPlan::get_max_depth);
	ClassDB::bind_method(D_METHOD("set_max_depth", "max_depth"), &PlannerPlan::set_max_depth);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_depth"), "set_max_depth", "get_max_depth");

	ClassDB::bind_method(D_METHOD("get_domains"), &PlannerPlan::get_domains);
	ClassDB::bind_method(D_METHOD("set_domains", "domain"), &PlannerPlan::set_domains);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "domains", PROPERTY_HINT_RESOURCE_TYPE, "Domain"), "set_domains", "get_domains");

	ClassDB::bind_method(D_METHOD("get_current_domain"), &PlannerPlan::get_current_domain);
	ClassDB::bind_method(D_METHOD("set_current_domain", "current_domain"), &PlannerPlan::set_current_domain);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "current_domain", PROPERTY_HINT_RESOURCE_TYPE, "Domain"), "set_current_domain", "get_current_domain");

	ClassDB::bind_method(D_METHOD("find_plan", "state", "todo_list"), &PlannerPlan::find_plan);
	ClassDB::bind_method(D_METHOD("run_lazy_lookahead", "state", "todo_list", "max_tries"), &PlannerPlan::run_lazy_lookahead, DEFVAL(10));
	ClassDB::bind_method(D_METHOD("run_lazy_refineahead", "state", "todo_list"), &PlannerPlan::run_lazy_refineahead);
	ClassDB::bind_method(D_METHOD("generate_plan_id"), &PlannerPlan::generate_plan_id);
	ClassDB::bind_method(D_METHOD("submit_operation", "operation"), &PlannerPlan::submit_operation);

	ADD_SIGNAL(MethodInfo("plan_id_generated", PropertyInfo(Variant::STRING, "plan_id")));
}

// Temporal method implementations
String PlannerPlan::generate_plan_id() {
	String plan_id = get_name();
	if (plan_id.is_empty()) {
		plan_id = get_path();
	}
	print_line("Generated plan ID: " + plan_id);
	emit_signal("plan_id_generated", plan_id);
	return plan_id;
}

Dictionary PlannerPlan::submit_operation(Dictionary p_operation) {
	String transaction_id = "";

	// Get absolute time in microseconds
	int64_t current_time = PlannerTimeRange::now_microseconds();

	Dictionary consensus_result;
	consensus_result["operation_id"] = transaction_id;
	consensus_result["agreed_at"] = current_time; // Absolute microseconds
	Array participants;
	participants.push_back("node_1");
	consensus_result["participants"] = participants;

	print_line("Operation submitted [" + transaction_id + "]: " + String(Variant(p_operation)));
	emit_signal("operation_submitted", consensus_result);
	return consensus_result;
}

bool PlannerPlan::get_verify_goals() const {
	return verify_goals;
}

void PlannerPlan::set_verify_goals(bool p_value) {
	verify_goals = p_value;
}

int PlannerPlan::get_max_depth() const {
	return max_depth;
}

void PlannerPlan::set_max_depth(int p_max_depth) {
	max_depth = p_max_depth;
}

// Graph-based lazy refinement (Elixir-style)
Dictionary PlannerPlan::run_lazy_refineahead(Dictionary p_state, Array p_todo_list) {
	// Note: Array is a value type in Godot and cannot be null, so no null check needed

	if (verbose >= 1) {
		print_line("run_lazy_refineahead: Starting graph-based planning");
		print_line("Initial state keys: " + String(Variant(p_state.keys())));
		print_line("Todo list: " + _item_to_string(p_todo_list));
	}

	// Initialize solution graph
	solution_graph = PlannerSolutionGraph();
	blacklisted_commands.clear();

	// Initialize STN solver
	stn.clear();
	stn.add_time_point("origin"); // Origin time point (plan start)

	// Initialize time range if not already set
	if (time_range.get_start_time() == 0) {
		time_range.set_start_time(PlannerTimeRange::now_microseconds());
	}

	// Anchor origin to current absolute time
	PlannerSTNConstraints::anchor_to_origin(stn, "origin", time_range.get_start_time());

	// Validate that current_domain is set before accessing its members
	if (!current_domain.is_valid()) {
		if (verbose >= 1) {
			print_line("run_lazy_refineahead: Error - no domain set");
		}
		ERR_PRINT("PlannerPlan::run_lazy_refineahead: current_domain is not set. Call set_current_domain() before planning.");
		return Dictionary(); // Return empty dictionary on error
	}

	// Add initial tasks to the solution graph
	int parent_node_id = 0; // Root node
	PlannerGraphOperations::add_nodes_and_edges(
			solution_graph,
			parent_node_id,
			p_todo_list,
			current_domain->action_dictionary,
			current_domain->task_method_dictionary,
			current_domain->unigoal_method_dictionary,
			current_domain->multigoal_method_list);

	// Start planning loop
	Dictionary final_state = _planning_loop_recursive(parent_node_id, p_state, 0);

	// Update time range with end time
	time_range.set_end_time(PlannerTimeRange::now_microseconds());
	time_range.calculate_duration();

	if (verbose >= 1) {
		print_line("run_lazy_refineahead: Completed graph-based planning");
		print_line("Duration: " + itos(time_range.get_duration()) + " microseconds");
	}

	return final_state;
}

Dictionary PlannerPlan::_planning_loop_recursive(int p_parent_node_id, Dictionary p_state, int p_iter) {
	// Check depth limit to prevent infinite recursion
	if (p_iter >= max_depth) {
		if (verbose >= 1) {
			ERR_PRINT(vformat("Planning depth limit (%d) exceeded, aborting", max_depth));
		}
		return p_state;
	}

	// Validate that current_domain is set before accessing its members
	// This defensive check protects against invalid state even if called from unexpected contexts
	if (!current_domain.is_valid()) {
		if (verbose >= 1) {
			ERR_PRINT("PlannerPlan::_planning_loop_recursive: current_domain is not set. Aborting planning loop.");
		}
		return p_state;
	}

	if (verbose >= 2) {
		print_line(vformat("_planning_loop_recursive: parent_node_id=%d, iter=%d", p_parent_node_id, p_iter));
	}

	// Find the first Open node
	Variant open_node_result = PlannerGraphOperations::find_open_node(solution_graph, p_parent_node_id);

	if (open_node_result.get_type() == Variant::NIL) {
		// No open node found, check if parent is root
		Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
		int parent_type = parent_node["type"];

		if (parent_type == static_cast<int>(PlannerNodeType::TYPE_ROOT)) {
			// Planning complete
			if (verbose >= 1) {
				print_line("Planning complete, returning final state");
			}
			return p_state;
		} else {
			// Move to predecessor
			int new_parent = PlannerGraphOperations::find_predecessor(solution_graph, p_parent_node_id);
			if (new_parent >= 0) {
				return _planning_loop_recursive(new_parent, p_state, p_iter + 1);
			}
			return p_state;
		}
	}

	int curr_node_id = open_node_result;
	Dictionary curr_node = solution_graph.get_node(curr_node_id);

	if (verbose >= 2) {
		print_line(vformat("Iteration %d: Refining node %d", p_iter, curr_node_id));
	}

	// Save current state if first visit (state is empty)
	Dictionary node_state = solution_graph.get_state_snapshot(curr_node_id);
	if (node_state.is_empty()) {
		solution_graph.save_state_snapshot(curr_node_id, p_state.duplicate());
		// Also save STN snapshot on first visit
		PlannerSTNSolver::Snapshot snapshot = stn.create_snapshot();
		curr_node["stn_snapshot"] = snapshot.to_dictionary();
		solution_graph.update_node(curr_node_id, curr_node);
	} else {
		// Restore state if backtracking
		p_state = node_state.duplicate();
		// Also restore STN snapshot
		_restore_stn_from_node(curr_node_id);
	}

	// Validate required dictionary keys exist
	if (!curr_node.has("type")) {
		if (verbose >= 1) {
			ERR_PRINT(vformat("PlannerPlan::_planning_loop_recursive: Node %d missing 'type' field", curr_node_id));
		}
		return p_state;
	}
	if (!curr_node.has("info")) {
		if (verbose >= 1) {
			ERR_PRINT(vformat("PlannerPlan::_planning_loop_recursive: Node %d missing 'info' field", curr_node_id));
		}
		return p_state;
	}

	int node_type = curr_node["type"];

	// Handle different node types
	switch (static_cast<PlannerNodeType>(node_type)) {
		case PlannerNodeType::TYPE_TASK: {
			// Try to refine task with available methods (like Elixir's Enum.find_value)
			Variant task_info = curr_node["info"];

			// Extract metadata and validate entity requirements (use original task_info for metadata extraction to preserve constraints)
			PlannerMetadata metadata = _extract_metadata(task_info);
			if (!_validate_entity_requirements(p_state, metadata)) {
				if (verbose >= 2) {
					print_line("Task entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}

			// Validate available_methods exists before accessing
			if (!curr_node.has("available_methods")) {
				if (verbose >= 1) {
					ERR_PRINT(vformat("PlannerPlan::_planning_loop_recursive: Task node %d missing 'available_methods' field", curr_node_id));
				}
				// Mark as failed and backtrack
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(curr_node_id, curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			TypedArray<Callable> available_methods = curr_node["available_methods"];

			// Unwrap task_info if it's in dictionary format
			Variant actual_task_info = task_info;
			if (task_info.get_type() == Variant::DICTIONARY) {
				Dictionary dict = task_info;
				if (dict.has("item")) {
					actual_task_info = dict["item"];
				}
			}

			// Check if this task is blacklisted (IPyHOP-style)
			if (_is_command_blacklisted(actual_task_info)) {
				if (verbose >= 2) {
					print_line("Task is blacklisted, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}

			// Try all available methods (like Elixir's Enum.find_value)
			// Don't modify available_methods - keep full list for backtracking
			Callable selected_method;
			Array subtasks;
			bool found_working_method = false;

			for (int i = 0; i < available_methods.size(); i++) {
				Callable method = available_methods[i];
				Array task_arr = actual_task_info;
				Array args;
				args.push_back(p_state);
				args.append_array(task_arr.slice(1));

				Variant result = method.callv(args);
				if (result.get_type() == Variant::ARRAY) {
					Array candidate_subtasks = result;
					// Check if this exact subtasks array is blacklisted (not its contents)
					// IPyHOP is reentrant - methods can return different results, so we only
					// blacklist the exact array that failed, not arrays that contain blacklisted actions
					if (_is_command_blacklisted(candidate_subtasks)) {
						if (verbose >= 2) {
							print_line("Method returned blacklisted subtasks array, skipping this method");
						}
						continue; // Skip this method, try next
					}
					subtasks = candidate_subtasks;
					selected_method = method;
					found_working_method = true;
					break; // Found working method, stop trying
				}
				// Method failed, continue to next (like Enum.find_value)
			}

			if (found_working_method) {
				// Successfully refined - like Elixir's {method, subtasks}
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				curr_node["selected_method"] = selected_method;
				// Store the subtasks that were created by this method for potential blacklisting
				curr_node["created_subtasks"] = subtasks;
				// Don't modify available_methods - keep full list for potential backtracking
				solution_graph.update_node(curr_node_id, curr_node);

				// Add subtasks to graph
				PlannerGraphOperations::add_nodes_and_edges(
						solution_graph,
						curr_node_id,
						subtasks,
						current_domain->action_dictionary,
						current_domain->task_method_dictionary,
						current_domain->unigoal_method_dictionary,
						current_domain->multigoal_method_list);

				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}

			// Failed to refine, backtrack
			if (verbose >= 2) {
				print_line("Task refinement failed, backtracking");
			}
			// Blacklist the task info since all methods failed (IPyHOP-style)
			_blacklist_command(actual_task_info);
			if (verbose >= 2) {
				print_line("Blacklisted task info since all methods failed");
			}
			// If this node was created by a parent's method subtasks, blacklist those subtasks
			if (p_parent_node_id >= 0) {
				Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
				if (parent_node.has("created_subtasks")) {
					Array parent_subtasks = parent_node["created_subtasks"];
					_blacklist_command(parent_subtasks);
					if (verbose >= 2) {
						print_line("Blacklisted parent subtasks that led to failure");
					}
				}
			}
			PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				// Restore STN snapshot from the node we're backtracking to
				_restore_stn_from_node(backtrack_result.parent_node_id);
				return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
			}
			return p_state;
		}

		case PlannerNodeType::TYPE_ACTION: {
			Variant action_info = curr_node["info"];

			// Check if blacklisted
			if (_is_command_blacklisted(action_info)) {
				if (verbose >= 2) {
					print_line("Action is blacklisted, backtracking");
				}
				// If this action was created by a parent's method subtasks, blacklist those subtasks
				if (p_parent_node_id >= 0) {
					Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
					if (parent_node.has("created_subtasks")) {
						Array parent_subtasks = parent_node["created_subtasks"];
						_blacklist_command(parent_subtasks);
						if (verbose >= 2) {
							print_line("Blacklisted parent subtasks that contained blacklisted action");
						}
					}
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					// Restore STN snapshot from the node we're backtracking to
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}

			// Create STN snapshot before action execution and store with node
			stn_snapshot = stn.create_snapshot();
			curr_node["stn_snapshot"] = stn_snapshot.to_dictionary();
			solution_graph.update_node(curr_node_id, curr_node);

			// Check for temporal constraints and entity requirements in action
			PlannerMetadata metadata = _extract_metadata(action_info);
			Dictionary temporal_metadata;
			if (_has_temporal_constraints(action_info)) {
				temporal_metadata = _get_temporal_constraints(action_info);
			}

			// Validate entity requirements before executing action
			if (!_validate_entity_requirements(p_state, metadata)) {
				if (verbose >= 2) {
					print_line("Action entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}

			// Execute action with temporal tracking
			// Validate action field exists before accessing
			if (!curr_node.has("action")) {
				if (verbose >= 1) {
					ERR_PRINT(vformat("PlannerPlan::_planning_loop_recursive: Action node %d missing 'action' field", curr_node_id));
				}
				// Mark as failed and backtrack
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(curr_node_id, curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			Callable action = curr_node["action"];
			// Unwrap action_info if it's in dictionary format
			Variant actual_action_info = action_info;
			if (action_info.get_type() == Variant::DICTIONARY) {
				Dictionary dict = action_info;
				if (dict.has("item")) {
					actual_action_info = dict["item"];
				}
			}
			Array action_arr = actual_action_info;

			// Validate that action was found
			if (!action.is_valid() || action.is_null()) {
				if (verbose >= 1) {
					String action_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
					print_line(vformat("Action '%s' not found in domain, marking as failed", action_name));
				}
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(curr_node_id, curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}

			Array args;
			args.push_back(p_state);
			args.append_array(action_arr.slice(1));

			// Use temporal metadata start_time if provided, otherwise use current time
			int64_t action_start_time;
			if (temporal_metadata.has("start_time")) {
				action_start_time = temporal_metadata["start_time"];
			} else {
				action_start_time = PlannerTimeRange::now_microseconds();
			}

			if (verbose >= 2) {
				String action_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
				print_line(vformat("Executing action '%s' with args: %s", action_name, _item_to_string(args.slice(1))));
			}

			Variant result = action.callv(args);

			// Use temporal metadata end_time if provided, otherwise use current time
			int64_t action_end_time;
			if (temporal_metadata.has("end_time")) {
				action_end_time = temporal_metadata["end_time"];
			} else {
				action_end_time = PlannerTimeRange::now_microseconds();
			}

			// Use temporal metadata duration if provided, otherwise calculate from start/end times
			int64_t action_duration;
			if (temporal_metadata.has("duration")) {
				action_duration = temporal_metadata["duration"];
			} else {
				action_duration = action_end_time - action_start_time;
			}

			if (result.get_type() == Variant::DICTIONARY) {
				Dictionary new_state = result;
				if (verbose >= 2) {
					String action_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
					print_line(vformat("Action '%s' succeeded, new state keys: %s", action_name, String(Variant(new_state.keys()))));
				}

				// Add action to STN only if it has temporal metadata
				// Actions without temporal metadata can occur at any time and don't need STN constraints
				bool has_temporal = temporal_metadata.has("start_time") || temporal_metadata.has("end_time") || temporal_metadata.has("duration");

				if (has_temporal) {
					// Validate action_arr is not empty before accessing first element
					if (action_arr.is_empty()) {
						if (verbose >= 2) {
							ERR_PRINT("Action array is empty, cannot create STN interval");
						}
						// Action failed, backtrack
						curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
						solution_graph.update_node(curr_node_id, curr_node);
						PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
								solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
						solution_graph = backtrack_result.graph;
						if (backtrack_result.parent_node_id >= 0) {
							_restore_stn_from_node(backtrack_result.parent_node_id);
							return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
						}
						return p_state;
					}
					String action_id = action_arr[0];
					int64_t metadata_start = temporal_metadata.get("start_time", 0);
					int64_t metadata_end = temporal_metadata.get("end_time", 0);
					int64_t metadata_duration = temporal_metadata.get("duration", action_duration);

					bool stn_success = PlannerSTNConstraints::add_interval(
							stn, action_id, metadata_start, metadata_end, metadata_duration);

					if (!stn_success) {
						if (verbose >= 2) {
							print_line("Failed to add interval to STN, backtracking");
						}
						_blacklist_command(action_info);
						stn.restore_snapshot(stn_snapshot);
						PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
								solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
						solution_graph = backtrack_result.graph;
						if (backtrack_result.parent_node_id >= 0) {
							_restore_stn_from_node(backtrack_result.parent_node_id);
							return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
						}
						return p_state;
					}

					// Check STN consistency only if we added temporal constraints
					stn.check_consistency();
					if (!stn.is_consistent()) {
						// STN inconsistent, backtrack
						if (verbose >= 2) {
							print_line("STN inconsistent after action, backtracking");
						}
						_blacklist_command(action_info);
						PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
								solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
						solution_graph = backtrack_result.graph;
						if (backtrack_result.parent_node_id >= 0) {
							// Restore STN snapshot from the node we're backtracking to
							_restore_stn_from_node(backtrack_result.parent_node_id);
							return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
						}
						return p_state;
					}
				} else {
					// Action has no temporal constraints - can occur at any time
					// Skip STN addition entirely
					if (verbose >= 3) {
						String action_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
						print_line(vformat("Action '%s' has no temporal constraints, skipping STN addition", action_name));
					}
				}

				// Action successful and STN consistent
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				curr_node["start_time"] = action_start_time;
				curr_node["end_time"] = action_end_time;
				curr_node["duration"] = action_duration;
				solution_graph.update_node(curr_node_id, curr_node);

				// Update plan time range
				time_range.set_end_time(action_end_time);
				time_range.calculate_duration();

				return _planning_loop_recursive(p_parent_node_id, new_state, p_iter + 1);
			} else {
				// Action failed, backtrack and restore STN
				String action_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
				if (verbose >= 1) {
					print_line(vformat("Action '%s' failed (returned %s, expected Dictionary), backtracking",
							action_name, Variant::get_type_name(result.get_type())));
					if (verbose >= 2) {
						print_line(vformat("  Action args: %s", _item_to_string(args.slice(1))));
						print_line(vformat("  Current state: %s", _item_to_string(p_state)));
					}
				}
				_blacklist_command(action_info);
				// If this action was created by a parent's method subtasks, blacklist those subtasks
				if (p_parent_node_id >= 0) {
					Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
					if (parent_node.has("created_subtasks")) {
						Array parent_subtasks = parent_node["created_subtasks"];
						_blacklist_command(parent_subtasks);
						if (verbose >= 2) {
							print_line("Blacklisted parent subtasks that contained failing action");
						}
					}
				}
				stn.restore_snapshot(stn_snapshot);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					// Restore STN snapshot from the node we're backtracking to
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
		}

		case PlannerNodeType::TYPE_GOAL: {
			Variant goal_info = curr_node["info"];

			// Unwrap goal_info if it's in dictionary format
			Variant actual_goal_info = goal_info;
			if (goal_info.get_type() == Variant::DICTIONARY) {
				Dictionary dict = goal_info;
				if (dict.has("item")) {
					actual_goal_info = dict["item"];
				}
			}

			// Check if this goal is blacklisted (IPyHOP-style)
			if (_is_command_blacklisted(actual_goal_info)) {
				if (verbose >= 2) {
					print_line("Goal is blacklisted, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}

			Array goal_arr = actual_goal_info;
			if (goal_arr.size() < 3) {
				// Invalid goal format
				return p_state;
			}

			String state_var_name = goal_arr[0];
			String argument = goal_arr[1];
			Variant desired_value = goal_arr[2];

			// Extract metadata and validate entity requirements (use original goal_info for metadata extraction)
			PlannerMetadata metadata = _extract_metadata(goal_info);
			if (!_validate_entity_requirements(p_state, metadata)) {
				if (verbose >= 2) {
					print_line("Goal entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}

			// Check if goal already achieved
			Dictionary state_var = p_state[state_var_name];
			if (state_var[argument] == desired_value) {
				// Goal already achieved
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				solution_graph.update_node(curr_node_id, curr_node);
				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}

			// Try to refine goal (like Elixir's Enum.find_value)
			// Validate available_methods exists before accessing
			if (!curr_node.has("available_methods")) {
				if (verbose >= 1) {
					ERR_PRINT(vformat("PlannerPlan::_planning_loop_recursive: Goal node %d missing 'available_methods' field", curr_node_id));
				}
				// Mark as failed and backtrack
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(curr_node_id, curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			TypedArray<Callable> available_methods = curr_node["available_methods"];

			// Try all available methods - don't modify available_methods
			Callable selected_method;
			Array subgoals;
			bool found_working_method = false;

			for (int i = 0; i < available_methods.size(); i++) {
				Callable method = available_methods[i];
				Variant result = method.call(p_state, argument, desired_value);
				if (result.get_type() == Variant::ARRAY) {
					Array candidate_subgoals = result;
					// Check if this exact subgoals array is blacklisted (not its contents)
					// IPyHOP is reentrant - methods can return different results, so we only
					// blacklist the exact array that failed, not arrays that contain blacklisted actions
					if (_is_command_blacklisted(candidate_subgoals)) {
						if (verbose >= 2) {
							print_line("Method returned blacklisted subgoals array, skipping this method");
						}
						continue; // Skip this method, try next
					}
					subgoals = candidate_subgoals;
					selected_method = method;
					found_working_method = true;
					break;
				}
				// Method failed, continue to next
			}

			if (found_working_method) {
				// Successfully refined
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				curr_node["selected_method"] = selected_method;
				// Store the subgoals that were created by this method for potential blacklisting
				curr_node["created_subtasks"] = subgoals;
				// Don't modify available_methods
				solution_graph.update_node(curr_node_id, curr_node);

				// Add subgoals to graph
				PlannerGraphOperations::add_nodes_and_edges(
						solution_graph,
						curr_node_id,
						subgoals,
						current_domain->action_dictionary,
						current_domain->task_method_dictionary,
						current_domain->unigoal_method_dictionary,
						current_domain->multigoal_method_list);

				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}

			// Failed to refine, backtrack
			if (verbose >= 2) {
				print_line("Goal refinement failed, backtracking");
			}
			// Blacklist the goal info since all methods failed (IPyHOP-style)
			_blacklist_command(actual_goal_info);
			if (verbose >= 2) {
				print_line("Blacklisted goal info since all methods failed");
			}
			// If this node was created by a parent's method subgoals, blacklist those subgoals
			if (p_parent_node_id >= 0) {
				Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
				if (parent_node.has("created_subtasks")) {
					Array parent_subgoals = parent_node["created_subtasks"];
					_blacklist_command(parent_subgoals);
					if (verbose >= 2) {
						print_line("Blacklisted parent subgoals that led to failure");
					}
				}
			}
			PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				// Restore STN snapshot from the node we're backtracking to
				_restore_stn_from_node(backtrack_result.parent_node_id);
				return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
			}
			return p_state;
		}

		case PlannerNodeType::TYPE_MULTIGOAL: {
			Variant multigoal_variant = curr_node["info"];

			// Unwrap if dictionary-wrapped
			if (multigoal_variant.get_type() == Variant::DICTIONARY) {
				Dictionary dict = multigoal_variant;
				if (dict.has("item")) {
					multigoal_variant = dict["item"];
				}
			}

			if (!PlannerMultigoal::is_multigoal_dict(multigoal_variant)) {
				return p_state;
			}
			Dictionary multigoal = multigoal_variant;

			// Check if this multigoal is blacklisted (IPyHOP-style)
			// Note: multigoal_variant is a Dictionary, but blacklister works with Arrays
			// If multigoal_variant is actually an Array, check it; otherwise skip blacklist check
			if (multigoal_variant.get_type() == Variant::ARRAY) {
				if (_is_command_blacklisted(multigoal_variant)) {
					if (verbose >= 2) {
						print_line("MultiGoal is blacklisted, backtracking");
					}
					PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
							solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
					solution_graph = backtrack_result.graph;
					if (backtrack_result.parent_node_id >= 0) {
						_restore_stn_from_node(backtrack_result.parent_node_id);
						return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
					}
					return p_state;
				}
			}

			// Extract metadata from multigoal and validate entity requirements
			// Multigoal metadata might be stored in the multigoal dictionary itself
			PlannerMetadata metadata = _extract_metadata(multigoal_variant);
			if (!_validate_entity_requirements(p_state, metadata)) {
				if (verbose >= 2) {
					print_line("MultiGoal entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}

			// Check if multigoal already achieved
			Dictionary goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, multigoal);
			if (goals_not_achieved.is_empty()) {
				// All goals are already achieved
				if (verbose >= 1) {
					print_line("MultiGoal already achieved, marking as closed");
				}
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				solution_graph.update_node(curr_node_id, curr_node);
				// Add empty subgoals for verification node (like Elixir)
				Array empty_subgoals;
				PlannerGraphOperations::add_nodes_and_edges(
						solution_graph,
						curr_node_id,
						empty_subgoals,
						current_domain->action_dictionary,
						current_domain->task_method_dictionary,
						current_domain->unigoal_method_dictionary,
						current_domain->multigoal_method_list);
				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}

			// Try to refine multigoal (like IPyHOP - rely on backtracking and blacklisting, not recursive split detection)
			// Validate available_methods exists before accessing
			if (!curr_node.has("available_methods")) {
				if (verbose >= 1) {
					ERR_PRINT(vformat("PlannerPlan::_planning_loop_recursive: MultiGoal node %d missing 'available_methods' field", curr_node_id));
				}
				// Mark as failed and backtrack
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(curr_node_id, curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			TypedArray<Callable> available_methods = curr_node["available_methods"];

			// Try all available methods - don't modify available_methods
			Callable selected_method;
			Array subgoals;
			bool found_working_method = false;

			for (int i = 0; i < available_methods.size(); i++) {
				Callable method = available_methods[i];
				Variant result = method.call(p_state, multigoal);
				if (result.get_type() == Variant::ARRAY) {
					Array candidate_subgoals = result;
					// Check if this exact subgoals array is blacklisted (not its contents)
					// IPyHOP is reentrant - methods can return different results, so we only
					// blacklist the exact array that failed, not arrays that contain blacklisted actions
					if (_is_command_blacklisted(candidate_subgoals)) {
						if (verbose >= 2) {
							print_line("Method returned blacklisted subgoals array, skipping this method");
						}
						continue; // Skip this method, try next
					}
					subgoals = candidate_subgoals;
					selected_method = method;
					found_working_method = true;
					break;
				}
				// Method failed, continue to next
			}

			if (found_working_method) {
				// Successfully refined
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				curr_node["selected_method"] = selected_method;
				// Don't modify available_methods
				
				// Store the subgoals that were created by this method for potential blacklisting
				curr_node["created_subtasks"] = subgoals;
				solution_graph.update_node(curr_node_id, curr_node);

				// Add subgoals to graph
				PlannerGraphOperations::add_nodes_and_edges(
						solution_graph,
						curr_node_id,
						subgoals,
						current_domain->action_dictionary,
						current_domain->task_method_dictionary,
						current_domain->unigoal_method_dictionary,
						current_domain->multigoal_method_list);

				return _planning_loop_recursive(curr_node_id, p_state, p_iter + 1);
			}

			// Failed to refine, backtrack
			if (verbose >= 2) {
				print_line("MultiGoal refinement failed, backtracking");
			}
			// Blacklist the multigoal info since all methods failed (IPyHOP-style)
			// Only blacklist if it's an Array (blacklister works with Arrays)
			if (multigoal_variant.get_type() == Variant::ARRAY) {
				_blacklist_command(multigoal_variant);
				if (verbose >= 2) {
					print_line("Blacklisted multigoal info since all methods failed");
				}
			}
			// If this node was created by a parent's method subgoals, blacklist those subgoals
			if (p_parent_node_id >= 0) {
				Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
				if (parent_node.has("created_subtasks")) {
					Array parent_subgoals = parent_node["created_subtasks"];
					_blacklist_command(parent_subgoals);
					if (verbose >= 2) {
						print_line("Blacklisted parent subgoals that led to failure");
					}
				}
			}
			PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				// Restore STN snapshot from the node we're backtracking to
				_restore_stn_from_node(backtrack_result.parent_node_id);
				return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
			}
			return p_state;
		}

		case PlannerNodeType::TYPE_VERIFY_GOAL: {
			// Verify the parent goal
			Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
			Array goal_arr = parent_node["info"];
			if (goal_arr.size() >= 3) {
				String state_var_name = goal_arr[0];
				String argument = goal_arr[1];
				Variant desired_value = goal_arr[2];

				Dictionary state_var = p_state[state_var_name];
				if (state_var[argument] == desired_value) {
					// Verification successful
					curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
					solution_graph.update_node(curr_node_id, curr_node);
					return _planning_loop_recursive(p_parent_node_id, p_state, p_iter + 1);
				}
			}

			// Verification failed, backtrack
			if (verbose >= 2) {
				print_line("Goal verification failed, backtracking");
			}
			PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
					solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				// Restore STN snapshot from the node we're backtracking to
				_restore_stn_from_node(backtrack_result.parent_node_id);
				return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
			}
			return p_state;
		}

		case PlannerNodeType::TYPE_VERIFY_MULTIGOAL: {
			// Verify the parent multigoal
			Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
			Variant multigoal_variant = parent_node["info"];

			// Unwrap if dictionary-wrapped
			if (multigoal_variant.get_type() == Variant::DICTIONARY) {
				Dictionary dict = multigoal_variant;
				if (dict.has("item")) {
					multigoal_variant = dict["item"];
				}
			}

			if (!PlannerMultigoal::is_multigoal_dict(multigoal_variant)) {
				// Invalid parent, backtrack
				if (verbose >= 2) {
					print_line("MultiGoal verification failed: invalid parent multigoal, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					// Restore STN snapshot from the node we're backtracking to
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
			Dictionary multigoal = multigoal_variant;

			Dictionary goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, multigoal);
			if (goals_not_achieved.is_empty()) {
				// Verification successful - all goals are achieved
				if (verbose >= 1) {
					print_line("MultiGoal verified successfully");
				}
				curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				solution_graph.update_node(curr_node_id, curr_node);
				return _planning_loop_recursive(p_parent_node_id, p_state, p_iter + 1);
			} else {
				// Verification failed - some goals not achieved
				if (verbose >= 2) {
					print_line("MultiGoal verification failed: some goals not achieved, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				if (backtrack_result.parent_node_id >= 0) {
					// Restore STN snapshot from the node we're backtracking to
					_restore_stn_from_node(backtrack_result.parent_node_id);
					return _planning_loop_recursive(backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1);
				}
				return p_state;
			}
		}

		default:
			return p_state;
	}
}

void PlannerPlan::_restore_stn_from_node(int p_node_id) {
	if (p_node_id >= 0) {
		Dictionary node = solution_graph.get_node(p_node_id);
		if (node.has("stn_snapshot")) {
			Dictionary snapshot_dict = node["stn_snapshot"];
			PlannerSTNSolver::Snapshot snapshot = PlannerSTNSolver::Snapshot::from_dictionary(snapshot_dict);
			stn.restore_snapshot(snapshot);
			if (verbose >= 3) {
				print_line("Restored STN snapshot from node " + itos(p_node_id));
			}
		}
	}
}

bool PlannerPlan::_is_command_blacklisted(Variant p_command) const {
	// Unwrap if dictionary-wrapped
	Variant actual_command = p_command;
	if (p_command.get_type() == Variant::DICTIONARY) {
		Dictionary dict = p_command;
		if (dict.has("item")) {
			actual_command = dict["item"];
		}
	}

	// Compare Arrays properly - need to check if it's an Array and compare elements
	if (actual_command.get_type() != Variant::ARRAY) {
		return false;
	}

	Array action_arr = actual_command;

	// Check each blacklisted command
	for (int i = 0; i < blacklisted_commands.size(); i++) {
		Variant blacklisted = blacklisted_commands[i];
		if (blacklisted.get_type() != Variant::ARRAY) {
			continue;
		}

		Array blacklisted_arr = blacklisted;

		// Compare Arrays element by element
		if (blacklisted_arr.size() != action_arr.size()) {
			continue;
		}

		bool match = true;
		for (int j = 0; j < action_arr.size(); j++) {
			if (action_arr[j] != blacklisted_arr[j]) {
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

bool PlannerPlan::_contains_blacklisted_action(Array p_subtasks) const {
	// Check if the subtasks array itself is blacklisted
	if (_is_command_blacklisted(p_subtasks)) {
		return true;
	}
	
	// Check if any action/task/goal in the subtasks array is blacklisted
	for (int i = 0; i < p_subtasks.size(); i++) {
		Variant subtask = p_subtasks[i];
		// Unwrap if dictionary-wrapped
		Variant actual_subtask = subtask;
		if (subtask.get_type() == Variant::DICTIONARY) {
			Dictionary dict = subtask;
			if (dict.has("item")) {
				actual_subtask = dict["item"];
			}
		}
		if (_is_command_blacklisted(actual_subtask)) {
			return true;
		}
	}
	return false;
}

void PlannerPlan::_blacklist_command(Variant p_command) {
	if (!_is_command_blacklisted(p_command)) {
		blacklisted_commands.push_back(p_command);
	}
}

PlannerMetadata PlannerPlan::_extract_temporal_constraints(const Variant &p_item) const {
	// Extract only temporal constraints (for backward compatibility)
	PlannerMetadata metadata = _extract_metadata(p_item);
	// Clear entity requirements to return only temporal constraints
	metadata.requires_entities.clear();
	return metadata;
}

PlannerMetadata PlannerPlan::_extract_metadata(const Variant &p_item) const {
	PlannerMetadata metadata;

	// Check if item has temporal_constraints field
	if (p_item.get_type() == Variant::DICTIONARY) {
		Dictionary item_dict = p_item;
		const Variant *temporal_constraints_var = item_dict.getptr("temporal_constraints");
		if (temporal_constraints_var && temporal_constraints_var->get_type() == Variant::DICTIONARY) {
			Dictionary constraints_dict = *temporal_constraints_var;
			metadata = PlannerMetadata::from_dictionary(constraints_dict);
		}
		// Also check for entity requirements in constraints field (for combined format)
		const Variant *constraints_var = item_dict.getptr("constraints");
		if (constraints_var && constraints_var->get_type() == Variant::DICTIONARY) {
			Dictionary constraints_dict = *constraints_var;
			if (constraints_dict.has("requires_entities")) {
				Variant entities_var = constraints_dict.get("requires_entities", Array());
				if (entities_var.get_type() == Variant::ARRAY) {
					Array entities_array = entities_var;
					metadata.requires_entities.resize(entities_array.size());
					for (int i = 0; i < entities_array.size(); i++) {
						Dictionary entity_dict = entities_array[i];
						metadata.requires_entities[i] = PlannerEntityRequirement::from_dictionary(entity_dict);
					}
				}
			}
		}
	} else if (p_item.get_type() == Variant::ARRAY) {
		Array item_arr = p_item;
		// Check if last element is a dictionary with temporal_constraints
		if (item_arr.size() > 0) {
			Variant last = item_arr[item_arr.size() - 1];
			if (last.get_type() == Variant::DICTIONARY) {
				Dictionary last_dict = last;
				const Variant *temporal_constraints_var = last_dict.getptr("temporal_constraints");
				if (temporal_constraints_var && temporal_constraints_var->get_type() == Variant::DICTIONARY) {
					Dictionary constraints_dict = *temporal_constraints_var;
					metadata = PlannerMetadata::from_dictionary(constraints_dict);
				}
			}
		}
	}

	return metadata;
}

Variant PlannerPlan::_attach_temporal_constraints(const Variant &p_item, const Dictionary &p_temporal_constraints) {
	PlannerMetadata metadata = PlannerMetadata::from_dictionary(p_temporal_constraints);

	// Create a wrapper dictionary with the item and temporal_constraints
	Dictionary result;
	Dictionary constraints_dict = metadata.to_dictionary();

	if (p_item.get_type() == Variant::DICTIONARY) {
		Dictionary item_dict = p_item;
		// If already a dictionary, merge temporal constraints
		result = Dictionary(p_item);
		if (result.has("temporal_constraints")) {
			// Merge with existing temporal constraints
			Dictionary existing_temporal = result["temporal_constraints"];
			for (const Variant *key = constraints_dict.next(nullptr); key; key = constraints_dict.next(key)) {
				existing_temporal[*key] = constraints_dict[*key];
			}
			result["temporal_constraints"] = existing_temporal;
		} else {
			result["temporal_constraints"] = constraints_dict;
		}
	} else {
		// Wrap in dictionary with temporal_constraints
		result["item"] = p_item;
		result["temporal_constraints"] = constraints_dict;
	}

	return result;
}

Dictionary PlannerPlan::_get_temporal_constraints(const Variant &p_item) const {
	PlannerMetadata metadata = _extract_temporal_constraints(p_item);
	return metadata.to_dictionary();
}

bool PlannerPlan::_has_temporal_constraints(const Variant &p_item) const {
	PlannerMetadata metadata = _extract_temporal_constraints(p_item);
	return metadata.has_temporal();
}

Variant PlannerPlan::attach_metadata(const Variant &p_item, const Dictionary &p_temporal_constraints, const Dictionary &p_entity_constraints) {
	Dictionary result;

	// Extract the actual item if it's already wrapped
	Variant actual_item = p_item;
	if (p_item.get_type() == Variant::DICTIONARY) {
		Dictionary item_dict = p_item;
		const Variant *item_var = item_dict.getptr("item");
		if (item_var) {
			actual_item = *item_var;
		} else {
			actual_item = p_item; // Use as-is if no "item" key
		}
	}

	// Start with the item
	result["item"] = actual_item;

	// Add temporal constraints if provided
	if (!p_temporal_constraints.is_empty()) {
		Dictionary temporal_dict;
		const Variant *duration_var = p_temporal_constraints.getptr("duration");
		const Variant *start_time_var = p_temporal_constraints.getptr("start_time");
		const Variant *end_time_var = p_temporal_constraints.getptr("end_time");

		if (duration_var) {
			temporal_dict["duration"] = *duration_var;
		}
		if (start_time_var) {
			temporal_dict["start_time"] = *start_time_var;
		}
		if (end_time_var) {
			temporal_dict["end_time"] = *end_time_var;
		}
		if (!temporal_dict.is_empty()) {
			result["temporal_constraints"] = temporal_dict;
		}
	}

	// Add entity constraints if provided
	if (!p_entity_constraints.is_empty()) {
		Dictionary entity_dict;
		const Variant *requires_entities_var = p_entity_constraints.getptr("requires_entities");
		if (requires_entities_var) {
			// Full format: already has requires_entities
			entity_dict["requires_entities"] = *requires_entities_var;
		} else {
			// Convenience format: convert {type, capabilities} to requires_entities format
			const Variant *type_var = p_entity_constraints.getptr("type");
			const Variant *capabilities_var = p_entity_constraints.getptr("capabilities");
			if (type_var && capabilities_var) {
				Array entities_array;
				Dictionary entity_req;
				entity_req["type"] = *type_var;
				entity_req["capabilities"] = *capabilities_var;
				entities_array.push_back(entity_req);
				entity_dict["requires_entities"] = entities_array;
			}
		}
		if (!entity_dict.is_empty()) {
			result["constraints"] = entity_dict;
		}
	}

	return result;
}

bool PlannerPlan::_validate_entity_requirements(const Dictionary &p_state, const PlannerMetadata &p_metadata) const {
	// Check if metadata has entity requirements
	if (p_metadata.requires_entities.size() == 0) {
		return true; // No entity requirements, validation passes
	}

	// Match entities for all requirements
	Dictionary match_result = _match_entities(p_state, p_metadata.requires_entities);
	bool success = match_result["success"];

	if (!success && verbose >= 2) {
		String error = match_result["error"];
		print_line("Entity matching failed: " + error);
	}

	return success;
}

Dictionary PlannerPlan::_match_entities(const Dictionary &p_state, const LocalVector<PlannerEntityRequirement> &p_requirements) const {
	Dictionary result;
	result["success"] = false;
	result["matched_entities"] = Array();
	result["error"] = "";

	// Use internal HashMap/LocalVector for efficiency
	HashMap<String, String> entity_types; // entity_id -> type
	HashMap<String, LocalVector<String>> entity_capabilities; // entity_id -> capabilities

	// Extract entity data from state
	// State structure: entities are stored in a nested structure
	// We'll look for entity_capabilities or similar structure
	if (p_state.has("entity_capabilities")) {
		Dictionary entity_caps_dict = p_state["entity_capabilities"];
		Array entity_ids = entity_caps_dict.keys();

		for (int i = 0; i < entity_ids.size(); i++) {
			String entity_id = entity_ids[i];
			Dictionary entity_data = entity_caps_dict[entity_id];

			// Extract type (stored as "type" capability)
			if (entity_data.has("type")) {
				entity_types[entity_id] = entity_data["type"];
			}

			// Extract all capabilities (any non-type key that has a truthy value)
			LocalVector<String> caps;
			Array cap_keys = entity_data.keys();
			for (int j = 0; j < cap_keys.size(); j++) {
				String cap_key = cap_keys[j];
				if (cap_key != "type") {
					Variant cap_value = entity_data[cap_key];
					// Include capability if value is truthy (true, non-zero, non-empty)
					if (cap_value.operator bool()) {
						caps.push_back(cap_key);
					}
				}
			}
			entity_capabilities[entity_id] = caps;
		}
	}

	// Match entities to requirements
	Array matched_entities;

	// Match each requirement to an entity
	for (uint32_t req_idx = 0; req_idx < p_requirements.size(); req_idx++) {
		const PlannerEntityRequirement &req = p_requirements[req_idx];
		bool matched = false;

		// Try to find matching entity
		for (const KeyValue<String, String> &E : entity_types) {
			String entity_id = E.key;
			String entity_type = E.value;

			// Check type match
			if (entity_type != req.type) {
				continue;
			}

			// Check if entity has all required capabilities
			const LocalVector<String> *entity_caps = entity_capabilities.getptr(entity_id);
			if (entity_caps == nullptr) {
				continue;
			}

			bool has_all_caps = true;
			for (uint32_t cap_idx = 0; cap_idx < req.capabilities.size(); cap_idx++) {
				String required_cap = req.capabilities[cap_idx];
				bool found = false;
				for (uint32_t j = 0; j < entity_caps->size(); j++) {
					if ((*entity_caps)[j] == required_cap) {
						found = true;
						break;
					}
				}
				if (!found) {
					has_all_caps = false;
					break;
				}
			}

			if (has_all_caps) {
				// Found matching entity
				matched_entities.push_back(entity_id);
				matched = true;
				break;
			}
		}

		if (!matched) {
			result["success"] = false;
			// Convert capabilities to string for error message
			String caps_str = "[";
			for (uint32_t i = 0; i < req.capabilities.size(); i++) {
				if (i > 0) {
					caps_str += ", ";
				}
				caps_str += req.capabilities[i];
			}
			caps_str += "]";
			result["error"] = vformat("No entity found matching requirement: type=%s, capabilities=%s", req.type, caps_str);
			return result;
		}
	}

	result["success"] = true;
	result["matched_entities"] = matched_entities;
	return result;
}
