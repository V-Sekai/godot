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

#include "core/string/ustring.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/local_vector.h"
#include "core/variant/callable.h"
#include "core/variant/typed_array.h"

#include "backtracking.h"
#include "domain.h"
#include "graph_operations.h"
#include "multigoal.h"
#include "stn_constraints.h"

PlannerPlan::PlannerPlan() {
	// Initialize all state to defaults on construction
	// This ensures clean state when object is created
	reset();
}

PlannerPlan::~PlannerPlan() {
	// Clean up all state on destruction (C++ needs explicit cleanup, unlike functional Elixir)
	// This ensures no state leaks when the object is destroyed
	reset();
}

int PlannerPlan::get_verbose() const {
	return verbose;
}

void PlannerPlan::set_verbose(int p_verbose) {
	verbose = p_verbose;
}

Ref<PlannerDomain> PlannerPlan::get_current_domain() const {
	return current_domain;
}

Ref<PlannerResult> PlannerPlan::find_plan(Dictionary p_state, Array p_todo_list) {
	// Note: Array is a value type in Godot and cannot be null, so no null check needed

	if (verbose >= 1) {
		print_line("verbose=" + itos(verbose) + ":");
		print_line("    state = " + _item_to_string(p_state));
		print_line("    todo_list = " + _item_to_string(p_todo_list));
		if (verbose >= 2 && current_domain.is_valid()) {
			Dictionary command_dict = current_domain->command_dictionary;
			Array command_keys = command_dict.keys();
			print_line("    Available commands: " + _item_to_string(command_keys));
		}
	}

	// CRITICAL: Reset planning-specific state for each planning call
	// Note: We do NOT reset configuration (verbose, max_depth, domain) - those persist across planning calls
	// Only reset() resets everything including configuration

	// Reset solution graph (creates fresh graph with root node)
	solution_graph = PlannerSolutionGraph();

	// Reset command blacklist
	blacklisted_commands.clear();

	// Reset todo list tracking
	original_todo_list.clear();

	// Reset iteration counter
	iterations = 0;

	// Reset VSIDS activity tracking (all learning state)
	// VSIDS learns during backtracking within a single solve, not across solves
	method_activities.clear();
	activity_var_inc = 1.0;
	activity_bump_count = 0;
	rewarded_methods_this_solve.clear();

	// Reset STN solver and snapshot (temporal constraint state)
	stn.clear();
	stn_snapshot = PlannerSTNSolver::Snapshot(); // Reset snapshot to empty state

	// Lazy STN Validation: Only initialize STN if temporal constraints are present
	bool has_temporal = _todo_list_has_temporal_constraints(p_todo_list);
	if (has_temporal) {
		// Initialize STN solver with origin time point (planning-specific initialization)
		stn.add_time_point("origin");

		// Initialize time range if not already set
		if (time_range.get_start_time() == 0) {
			time_range.set_start_time(PlannerTimeRange::now_microseconds());
		}

		// Anchor origin to current absolute time
		PlannerSTNConstraints::anchor_to_origin(stn, "origin", time_range.get_start_time());
	} else if (verbose >= 2) {
		print_line("[LAZY_STN] No temporal constraints detected, skipping STN initialization");
	}

	// Guard: Domain must be set
	if (!current_domain.is_valid()) {
		if (verbose >= 1) {
			print_line("result = false (no domain set)");
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		// Optimized: Cache get_graph() result
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	// Guard: todo_list must not be empty
	if (p_todo_list.is_empty()) {
		if (verbose >= 1) {
			print_line("result = false (empty todo_list)");
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		// Optimized: Cache get_graph() result
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	// Store original todo_list to track completion of all tasks
	original_todo_list = p_todo_list.duplicate();

	// CRITICAL: Deep copy the state to prevent pollution from previous tests
	// Use duplicate(true) which works correctly for nested dictionaries in Godot
	Dictionary clean_state = p_state.duplicate(true);

	// Belief-immersed architecture: Merge allocentric facts and apply ego-centric perspective
	clean_state = _merge_allocentric_facts(clean_state);
	clean_state = _get_ego_centric_state(clean_state);

	if (verbose >= 3) {
		Array state_keys = clean_state.keys();
		print_line(vformat("[FIND_PLAN] Initial state has %d keys: %s", state_keys.size(), _item_to_string(state_keys)));
	}

	// Add initial tasks to the solution graph
	int parent_node_id = 0; // Root node
	if (verbose >= 2) {
		Array task_keys = current_domain->task_method_dictionary.keys();
		print_line(vformat("[FIND_PLAN] Using domain with %d task methods: %s", task_keys.size(), _item_to_string(task_keys)));
	}
	PlannerGraphOperations::add_nodes_and_edges(
			solution_graph,
			parent_node_id,
			p_todo_list,
			current_domain->command_dictionary,
			current_domain->task_method_dictionary,
			current_domain->unigoal_method_dictionary,
			current_domain->multigoal_method_list,
			verbose);

	// Guard: Root node must have successors (optimized: use internal structure)
	const PlannerNodeStruct *root_node_check = solution_graph.get_node_internal(0);
	if (root_node_check == nullptr || root_node_check->successors.is_empty()) {
		if (verbose >= 1) {
			ERR_PRINT("PlannerPlan::find_plan: Failed to add nodes to graph - root node missing successors");
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(clean_state);
		// Cache get_graph() result to avoid multiple calls
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	// Guard: Root must have successors if todo_list is not empty
	if (root_node_check->successors.is_empty() && !p_todo_list.is_empty()) {
		if (verbose >= 1) {
			ERR_PRINT("PlannerPlan::find_plan: Failed to add nodes to graph - root has no successors but todo_list is not empty");
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(clean_state);
		// Cache get_graph() result to avoid multiple calls
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	// Start planning loop (using iterative version to prevent stack overflow)
	Dictionary final_state = _planning_loop_iterative(parent_node_id, clean_state, 0);

	// Check if planning succeeded (if we got back to root with a valid state)
	// Planning succeeds if all nodes are closed and we're back at root
	// Note: Root node check is done in the success checking code below using internal structure

	// Check if all nodes reachable from root are closed (planning succeeded)
	// Only consider nodes that are reachable from root via closed nodes
	// This way, failed nodes that were removed from their parent's successors don't cause planning to fail
	bool planning_succeeded = true;
	// Get graph keys safely - use internal structure for fast access
	const HashMap<int, PlannerNodeStruct> &graph_internal = solution_graph.get_graph_internal();
	LocalVector<int> failed_nodes;
	LocalVector<int> open_nodes;
	LocalVector<int> reachable_nodes;
	LocalVector<int> to_visit;
	to_visit.push_back(0); // Start from root
	HashSet<int> visited; // O(1) lookups instead of O(n)

	// Find all nodes reachable from root via closed nodes (optimized: uses internal structure)
	while (!to_visit.is_empty()) {
		int node_id = to_visit[to_visit.size() - 1];
		to_visit.remove_at(to_visit.size() - 1);
		if (visited.has(node_id)) {
			continue;
		}
		visited.insert(node_id);
		reachable_nodes.push_back(node_id);

		// Use internal structure for fast access
		const PlannerNodeStruct *node = solution_graph.get_node_internal(node_id);
		if (node == nullptr) {
			if (verbose >= 2) {
				ERR_PRINT(vformat("Planning success check: Node %d not found, skipping", node_id));
			}
			continue; // Skip invalid node
		}

		// Only traverse through closed nodes (or root which is NA)
		if (node->status == PlannerNodeStatus::STATUS_CLOSED || node_id == 0) {
			// Use LocalVector successors directly (fast)
			for (uint32_t j = 0; j < node->successors.size(); j++) {
				int succ_id = node->successors[j];
				if (!visited.has(succ_id)) {
					to_visit.push_back(succ_id);
				}
			}
		}
	}

	// Check reachable nodes for failures
	bool has_reachable_closed_nodes = false;
	// Track FAILED VERIFY_GOAL nodes - they're acceptable if there's a CLOSED one for the same parent
	HashMap<int, LocalVector<int>> failed_verify_goals_by_parent; // parent_id -> array of failed verify goal node_ids
	LocalVector<int> closed_verify_goals;
	// Track FAILED VERIFY_MULTIGOAL nodes - they're acceptable if there's a CLOSED one for the same parent
	HashMap<int, LocalVector<int>> failed_verify_multigoals_by_parent; // parent_id -> array of failed verify multigoal node_ids
	LocalVector<int> closed_verify_multigoals;

	for (uint32_t i = 0; i < reachable_nodes.size(); i++) {
		int node_id = reachable_nodes[i];
		if (node_id == 0) {
			continue; // Skip root
		}

		// Use internal structure for fast access
		const PlannerNodeStruct *node = solution_graph.get_node_internal(node_id);
		if (node == nullptr) {
			if (verbose >= 2) {
				ERR_PRINT(vformat("Planning success check: Reachable node %d not found, skipping", node_id));
			}
			continue; // Skip invalid node
		}

		PlannerNodeStatus status = node->status;
		PlannerNodeType node_type = node->type;

		// Planning fails if any reachable node is open
		if (status == PlannerNodeStatus::STATUS_OPEN) {
			planning_succeeded = false;
			open_nodes.push_back(node_id);
		} else if (status == PlannerNodeStatus::STATUS_FAILED) {
			// FAILED VERIFY_GOAL and VERIFY_MULTIGOAL nodes are acceptable if there's a CLOSED one for the same parent
			if (node_type == PlannerNodeType::TYPE_VERIFY_GOAL ||
					node_type == PlannerNodeType::TYPE_VERIFY_MULTIGOAL) {
				// Get parent node ID by searching the graph (optimized: uses internal structure)
				int parent_id = -1;
				for (const KeyValue<int, PlannerNodeStruct> &kv : graph_internal) {
					int candidate_id = kv.key;
					if (candidate_id == node_id) {
						continue;
					}
					const PlannerNodeStruct &candidate = kv.value;
					// Check if this candidate has node_id as a successor
					for (uint32_t j = 0; j < candidate.successors.size(); j++) {
						if (candidate.successors[j] == node_id) {
							parent_id = candidate_id;
							break;
						}
					}
					if (parent_id >= 0) {
						break;
					}
				}
				if (parent_id >= 0) {
					if (node_type == PlannerNodeType::TYPE_VERIFY_GOAL) {
						LocalVector<int> &failed_list = failed_verify_goals_by_parent[parent_id];
						failed_list.push_back(node_id);
					} else { // TYPE_VERIFY_MULTIGOAL
						LocalVector<int> &failed_list = failed_verify_multigoals_by_parent[parent_id];
						failed_list.push_back(node_id);
					}
				}
				// Don't mark as failed yet - check if there's a CLOSED one
			} else {
				// Other FAILED nodes are real failures
				planning_succeeded = false;
				failed_nodes.push_back(node_id);
			}
		} else if (status == PlannerNodeStatus::STATUS_CLOSED) {
			has_reachable_closed_nodes = true;
			if (node_type == PlannerNodeType::TYPE_VERIFY_GOAL) {
				closed_verify_goals.push_back(node_id);
			} else if (node_type == PlannerNodeType::TYPE_VERIFY_MULTIGOAL) {
				closed_verify_multigoals.push_back(node_id);
			}
		}
	}

	// For each parent with failed verify goals, check if there's a closed one (optimized: direct iteration)
	for (const KeyValue<int, LocalVector<int>> &kv : failed_verify_goals_by_parent) {
		int parent_id = kv.key;
		bool has_closed_verify_goal = false;
		// Check if any closed verify goal has this parent (optimized: uses internal structure)
		for (uint32_t j = 0; j < closed_verify_goals.size(); j++) {
			int verify_goal_id = closed_verify_goals[j];
			// Find parent of this verify goal by searching graph_internal
			for (const KeyValue<int, PlannerNodeStruct> &kv2 : graph_internal) {
				int candidate_id = kv2.key;
				if (candidate_id == verify_goal_id) {
					continue;
				}
				const PlannerNodeStruct &candidate = kv2.value;
				// Check if this candidate has verify_goal_id as a successor
				for (uint32_t k = 0; k < candidate.successors.size(); k++) {
					if (candidate.successors[k] == verify_goal_id && candidate_id == parent_id) {
						has_closed_verify_goal = true;
						break;
					}
				}
				if (has_closed_verify_goal) {
					break;
				}
			}
			if (has_closed_verify_goal) {
				break;
			}
		}
		// If no closed verify goal for this parent, the failed ones are real failures
		if (!has_closed_verify_goal) {
			const LocalVector<int> &failed_list = failed_verify_goals_by_parent[parent_id];
			for (uint32_t j = 0; j < failed_list.size(); j++) {
				failed_nodes.push_back(failed_list[j]);
			}
			// Don't mark planning as failed just because of intermediate verify failures
			// The final CLOSED verify goal is what matters
		}
	}

	// For each parent with failed verify multigoals, check if there's a closed one (optimized: direct iteration)
	for (const KeyValue<int, LocalVector<int>> &kv : failed_verify_multigoals_by_parent) {
		int parent_id = kv.key;
		bool has_closed_verify_multigoal = false;
		// Check if any closed verify multigoal has this parent (optimized: uses internal structure)
		for (uint32_t j = 0; j < closed_verify_multigoals.size(); j++) {
			int verify_multigoal_id = closed_verify_multigoals[j];
			// Find parent of this verify multigoal by searching graph_internal
			for (const KeyValue<int, PlannerNodeStruct> &kv3 : graph_internal) {
				int candidate_id = kv3.key;
				if (candidate_id == verify_multigoal_id) {
					continue;
				}
				const PlannerNodeStruct &candidate = kv3.value;
				// Check if this candidate has verify_multigoal_id as a successor
				for (uint32_t k = 0; k < candidate.successors.size(); k++) {
					if (candidate.successors[k] == verify_multigoal_id && candidate_id == parent_id) {
						has_closed_verify_multigoal = true;
						break;
					}
				}
				if (has_closed_verify_multigoal) {
					break;
				}
			}
			if (has_closed_verify_multigoal) {
				break;
			}
		}
		// If no closed verify multigoal for this parent, the failed ones are real failures
		if (!has_closed_verify_multigoal) {
			const LocalVector<int> &failed_list = failed_verify_multigoals_by_parent[parent_id];
			for (uint32_t j = 0; j < failed_list.size(); j++) {
				failed_nodes.push_back(failed_list[j]);
			}
			// Don't mark planning as failed just because of intermediate verify failures
			// The final CLOSED verify multigoal is what matters
		}
	}

	// If no reachable closed nodes (besides root), planning failed
	if (!has_reachable_closed_nodes) {
		planning_succeeded = false;
	}

	// Create PlannerResult with final state and solution graph
	if (verbose >= 3) {
		print_line("[FIND_PLAN] Creating PlannerResult...");
		print_line(vformat("[FIND_PLAN] planning_succeeded=%s, final_state.is_empty()=%s",
				planning_succeeded ? "true" : "false",
				final_state.is_empty() ? "true" : "false"));
	}
	Ref<PlannerResult> result = memnew(PlannerResult);
	if (verbose >= 3) {
		print_line("[FIND_PLAN] PlannerResult created");
	}
	result->set_final_state(final_state);
	if (verbose >= 3) {
		print_line("[FIND_PLAN] Final state set");
	}
	// Optimized: Cache get_graph() result and avoid duplicate calls
	const Dictionary &graph_dict = solution_graph.get_graph();
	if (verbose >= 3) {
		print_line(vformat("[FIND_PLAN] Graph to set has %d keys", graph_dict.keys().size()));
	}
	// CRITICAL: Use deep duplicate to ensure the graph dictionary is not modified after setting
	// Only duplicate once - we'll reuse if we need to update
	Dictionary graph_copy = graph_dict.duplicate(true);
	result->set_solution_graph(graph_copy);
	if (verbose >= 3) {
		print_line("[FIND_PLAN] Solution graph set");
		// Verify the graph was set correctly
		Dictionary verify_graph = result->get_solution_graph();
		print_line(vformat("[FIND_PLAN] Verified: result graph has %d keys", verify_graph.keys().size()));
	}
	result->set_success(planning_succeeded && !final_state.is_empty());
	if (verbose >= 3) {
		print_line(vformat("[FIND_PLAN] Success set to %s", (planning_succeeded && !final_state.is_empty()) ? "true" : "false"));
	}

	if (planning_succeeded && !final_state.is_empty()) {
		// Mark root node as CLOSED when planning succeeds so extract_solution_plan can traverse from it
		if (verbose >= 3) {
			print_line("[FIND_PLAN] Planning succeeded, marking root as CLOSED");
		}
		// Optimized: Use internal structure to update root node status
		solution_graph.set_node_status(0, PlannerNodeStatus::STATUS_CLOSED);
		// Update the result's solution graph with the updated root node
		if (verbose >= 3) {
			print_line("[FIND_PLAN] Setting solution graph in result...");
		}
		// Optimized: Reuse cached graph_dict, but need to get fresh copy after update
		const Dictionary &updated_graph = solution_graph.get_graph();
		if (verbose >= 3) {
			print_line(vformat("[FIND_PLAN] Graph to store has %d keys", updated_graph.keys().size()));
		}
		// CRITICAL: Use deep duplicate to ensure the graph dictionary is not modified after setting
		Dictionary updated_graph_copy = updated_graph.duplicate(true);
		result->set_solution_graph(updated_graph_copy);
		if (verbose >= 3) {
			print_line("[FIND_PLAN] Solution graph set in result");
			// Verify the graph was set correctly
			Dictionary verify_graph = result->get_solution_graph();
			print_line(vformat("[FIND_PLAN] Verified after update: result graph has %d keys", verify_graph.keys().size()));
		}
		// Note: Methods are already rewarded immediately when they succeed during planning
		// No need for end-of-plan reward since VSIDS learns during backtracking

		if (verbose >= 1) {
			print_line("[FIND_PLAN] About to call extract_plan() for verbose output...");
			Array plan = result->extract_plan();
			print_line("result plan = " + _item_to_string(plan));
		}

		if (verbose >= 3) {
			print_line("[FIND_PLAN] Returning result (success case)");
		}
		return result;
	} else {
		if (verbose >= 2) {
			print_line("[FIND_PLAN] Planning failed or final_state empty, returning failure result");
		}
		if (verbose >= 1) {
			print_line("result = false (planning failed)");
			if (verbose >= 2 || !failed_nodes.is_empty() || !open_nodes.is_empty()) {
				// Print solution graph for debugging
				print_line("Solution graph structure:");
				const HashMap<int, PlannerNodeStruct> &graph_internal_debug = solution_graph.get_graph_internal();
				for (const KeyValue<int, PlannerNodeStruct> &kv : graph_internal_debug) {
					int node_id = kv.key;
					Dictionary node = solution_graph.get_node(node_id);
					int node_type = node["type"];
					int node_status = node["status"];
					Variant node_info = node["info"];
					TypedArray<int> successors = node["successors"];

					String type_str;
					switch (static_cast<PlannerNodeType>(node_type)) {
						case PlannerNodeType::TYPE_ROOT:
							type_str = "ROOT";
							break;
						case PlannerNodeType::TYPE_COMMAND:
							type_str = "COMMAND";
							break;
						case PlannerNodeType::TYPE_TASK:
							type_str = "TASK";
							break;
						case PlannerNodeType::TYPE_UNIGOAL:
							type_str = "UNIGOAL";
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
					// Convert LocalVector to Array for printing
					Array failed_nodes_array;
					for (uint32_t j = 0; j < failed_nodes.size(); j++) {
						failed_nodes_array.push_back(failed_nodes[j]);
					}
					print_line("Failed nodes: " + _item_to_string(failed_nodes_array));
				}
				if (!open_nodes.is_empty()) {
					// Convert LocalVector to Array for printing
					Array open_nodes_array;
					for (uint32_t j = 0; j < open_nodes.size(); j++) {
						open_nodes_array.push_back(open_nodes[j]);
					}
					print_line("Open nodes: " + _item_to_string(open_nodes_array));
				}
			}
		}
		return result;
	}
}

String PlannerPlan::_item_to_string(Variant p_item) {
	return String(p_item);
}

// VSIDS-style method activity tracking

String PlannerPlan::_method_to_id(Callable p_method) const {
	// Create unique ID from method's object and method name
	Object *obj = p_method.get_object();
	if (obj) {
		StringName method_name = p_method.get_method();
		// Convert pointer to string manually (vformat doesn't support %p)
		String obj_str = String::num_uint64((uint64_t)(uintptr_t)obj, 16);
		return vformat("%s_%s", obj_str, method_name);
	}
	// Fallback: use hash of callable itself
	return vformat("callable_%d", p_method.hash());
}

double PlannerPlan::_get_method_activity(Callable p_method) const {
	String method_id = _method_to_id(p_method);
	if (method_activities.has(method_id)) {
		return method_activities[method_id];
	}
	return 0.0; // Default activity for new methods
}

void PlannerPlan::_bump_method_activity(Callable p_method) {
	String method_id = _method_to_id(p_method);
	double current_activity = _get_method_activity(p_method);

	// Bump activity (like Chuffed's varBumpActivity)
	// Simple addition of current var_inc (proven approach from Chuffed)
	method_activities[method_id] = current_activity + activity_var_inc;
	activity_bump_count++;

	// Decay periodically (every N bumps, like Chuffed)
	// Chuffed calls varDecayActivity() once per conflict, we do it every N bumps
	if (activity_bump_count >= ACTIVITY_DECAY_INTERVAL) {
		_decay_method_activities();
		activity_bump_count = 0;
	}
}

void PlannerPlan::_decay_method_activities() {
	// Activity inflation (like Chuffed's varDecayActivity)
	// Instead of decaying activities directly, increase var_inc
	// This makes newer bumps worth more relative to older ones
	activity_var_inc *= ACTIVITY_INFLATION_FACTOR;

	// If var_inc gets too large, normalize by scaling down all activities
	// This prevents numerical overflow while maintaining relative ordering
	if (activity_var_inc > ACTIVITY_OVERFLOW_THRESHOLD) {
		// Optimized: Direct iteration over HashMap
		for (KeyValue<String, double> &kv : method_activities) {
			kv.value *= ACTIVITY_NORMALIZATION_FACTOR;
		}
		activity_var_inc *= ACTIVITY_NORMALIZATION_FACTOR;
	}
	// NOTE: We do NOT multiply activities by decay_factor here
	// Activity inflation (increasing var_inc) achieves the same effect more efficiently
}

void PlannerPlan::_bump_conflict_path_activities(int p_fail_node_id) {
	// Walk up the conflict path and bump activity of all methods
	// This learns from conflicts (VSIDS-style)
	// NOTE: For optimization, we primarily reward success, but still track failures
	// to avoid repeatedly trying methods that consistently fail
	int current_id = p_fail_node_id;

	while (current_id >= 0) {
		Dictionary node = solution_graph.get_node(current_id);
		if (node.has("selected_method")) {
			Variant method_variant = node["selected_method"];
			if (method_variant.get_type() == Variant::CALLABLE) {
				Callable method = method_variant;
				// Use smaller bump for failures (less impactful than success rewards)
				_bump_method_activity(method);
				if (verbose >= 3) {
					print_line(vformat("Bumped activity for method in conflict path (node %d)", current_id));
				}
			}
		}

		// Move to parent
		current_id = PlannerGraphOperations::find_predecessor(solution_graph, current_id);
		if (current_id < 0) {
			break;
		}
	}
}

void PlannerPlan::_reward_successful_methods(int p_plan_length) {
	// Reward all methods used in the successful plan
	// Shorter plans get higher rewards (optimization goal)
	// This is the primary learning mechanism for VSIDS optimization

	if (p_plan_length <= 0) {
		return; // Invalid plan length
	}

	// Calculate reward: inverse of plan length (shorter = better = higher reward)
	// Use much larger scale to ensure rewards dominate over subtask bonuses
	// For 30 actions: ~1000, for 300 actions: ~100 (10x difference)
	double base_reward = REWARD_BASE / (1.0 + p_plan_length);
	double reward = base_reward * activity_var_inc;

	// Walk through solution graph and reward all methods that were used
	Dictionary graph = solution_graph.get_graph();
	Array graph_keys = graph.keys();
	TypedArray<int> visited;
	Array to_visit;
	to_visit.push_back(0); // Start from root

	while (!to_visit.is_empty()) {
		int node_id = to_visit.pop_back();
		if (visited.has(node_id)) {
			continue;
		}
		visited.push_back(node_id);

		Dictionary node = graph[node_id];
		int node_status = node["status"];

		// Only reward methods in the successful path (closed nodes)
		if (node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
			// Reward the method that was selected for this node
			if (node.has("selected_method")) {
				Variant method_variant = node["selected_method"];
				if (method_variant.get_type() == Variant::CALLABLE) {
					Callable method = method_variant;
					String method_id = _method_to_id(method);
					double current_activity = _get_method_activity(method);

					// Reward successful method (add to activity)
					method_activities[method_id] = current_activity + reward;
					activity_bump_count++;

					if (verbose >= 3) {
						print_line(vformat("VSIDS: Rewarded method '%s' with %.6f (plan length: %d, new activity: %.6f)",
								method_id, reward, p_plan_length, current_activity + reward));
					}
				}
			}

			// Visit successors
			TypedArray<int> successors = node["successors"];
			for (int i = 0; i < successors.size(); i++) {
				int succ_id = successors[i];
				if (!visited.has(succ_id) && graph.has(succ_id)) {
					to_visit.push_back(succ_id);
				}
			}
		}
	}

	// Decay if needed
	if (activity_bump_count >= ACTIVITY_DECAY_INTERVAL) {
		_decay_method_activities();
		activity_bump_count = 0;
	}
}

int PlannerPlan::_count_closed_actions() {
	// Count closed action nodes without extracting full plan (safer during planning)
	int count = 0;
	Dictionary graph = solution_graph.get_graph();
	Array keys = graph.keys();

	for (int i = 0; i < keys.size(); i++) {
		Variant key = keys[i];
		if (key.get_type() != Variant::INT) {
			continue;
		}
		int node_id = key;
		if (!graph.has(node_id)) {
			continue;
		}
		Dictionary node = graph[node_id];
		if (!node.has("type") || !node.has("status")) {
			continue;
		}
		int node_type = node["type"];
		int node_status = node["status"];
		if (node_type == static_cast<int>(PlannerNodeType::TYPE_COMMAND) &&
				node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
			count++;
		}
	}
	return count;
}

void PlannerPlan::_reward_method_immediate(Callable p_method, int p_current_action_count) {
	// Reward method immediately when it succeeds
	// Use Chuffed's proven approach: simple fixed increment (activity_var_inc)
	// This allows VSIDS to learn during backtracking within a single solve

	if (p_current_action_count < 0) {
		return; // Invalid count
	}

	String method_id = _method_to_id(p_method);

	// CRITICAL: Only reward each method once per solve to prevent duplicate rewards
	// The same method may be called recursively, but we only want to reward it once
	// (Like Chuffed's seen[] array to prevent duplicate bumps)
	if (rewarded_methods_this_solve.has(method_id)) {
		if (verbose >= 3) {
			print_line(vformat("VSIDS: Method '%s' already rewarded this solve, skipping", method_id));
		}
		return; // Already rewarded
	}

	// Use Chuffed's approach: simple fixed increment
	// The activity_var_inc grows over time (activity inflation), so newer rewards
	// are worth more relative to older ones, achieving decay effect efficiently
	double current_activity = _get_method_activity(p_method);

	// Reward successful method (add var_inc, like Chuffed's varBumpActivity)
	method_activities[method_id] = current_activity + activity_var_inc;
	activity_bump_count++;
	rewarded_methods_this_solve.push_back(method_id); // Mark as rewarded

	if (verbose >= 3) {
		print_line(vformat("VSIDS: Rewarded method '%s' with var_inc=%.6f (current actions: %d, new activity: %.6f)",
				method_id, activity_var_inc, p_current_action_count, current_activity + activity_var_inc));
	}

	// Decay if needed (activity inflation: increase var_inc)
	if (activity_bump_count >= ACTIVITY_DECAY_INTERVAL) {
		_decay_method_activities();
		activity_bump_count = 0;
		if (verbose >= 3) {
			print_line(vformat("VSIDS: Activity inflation (new var_inc: %.6f)", activity_var_inc));
		}
	}
}

PlannerPlan::MethodCandidate PlannerPlan::_select_best_method(TypedArray<Callable> p_methods, Dictionary p_state, Variant p_node_info, Variant p_args, int p_node_type, bool p_track_alternatives, Array *p_alternatives) {
	MethodCandidate best_candidate;
	best_candidate.method = Callable();
	best_candidate.score = INITIAL_SCORE; // Very negative initial score
	best_candidate.method_id = "";
	best_candidate.activity = 0.0;
	best_candidate.reason = "";

	// Track all candidates for explanation/debugging
	Array all_candidates;

	// Evaluate all methods and collect candidates
	for (int i = 0; i < p_methods.size(); i++) {
		Callable method = p_methods[i];
		if (verbose >= 3) {
			bool is_valid_count = false;
			int method_arg_count = method.get_argument_count(&is_valid_count);
			if (is_valid_count) {
				String method_name = method.get_method();
				print_line(vformat("_select_best_method: Method[%d] '%s' expects %d arguments", i, method_name, method_arg_count));
			}
		}
		Variant result;

		// Guard: Handle different node types with early returns
		if (p_node_type == static_cast<int>(PlannerNodeType::TYPE_TASK)) {
			// Guard: p_args must be an Array
			if (p_args.get_type() != Variant::ARRAY) {
				if (verbose >= 2) {
					ERR_PRINT(vformat("_select_best_method: p_args is not an Array (type: %d), skipping", p_args.get_type()));
				}
				continue;
			}
			Array args = p_args;

			// Guard: args must not be empty (state must be provided)
			if (args.is_empty()) {
				if (verbose >= 2) {
					ERR_PRINT("_select_best_method: Task method args is empty - state must be provided");
				}
				continue;
			}

			// Guard: first arg must be a dictionary (state)
			Variant first_arg = args[0];
			if (first_arg.get_type() != Variant::DICTIONARY) {
				if (verbose >= 2) {
					ERR_PRINT("_select_best_method: Task method first arg is not a dictionary (state)");
				}
				continue;
			}

			// Guard: method must be valid
			if (!method.is_valid()) {
				if (verbose >= 2) {
					ERR_PRINT("_select_best_method: Task method is invalid");
				}
				continue;
			}

			// Debug logging
			if (verbose >= 3) {
				print_line(vformat("_select_best_method: Calling task method with %d args", args.size()));
				print_line(vformat("_select_best_method: First arg type: %d (DICT=%d), is_empty: %s",
						first_arg.get_type(), Variant::DICTIONARY, first_arg.get_type() == Variant::DICTIONARY ? "false" : "true"));
				if (first_arg.get_type() == Variant::DICTIONARY) {
					Dictionary state_dict = first_arg;
					Array keys = state_dict.keys();
					print_line(vformat("_select_best_method: State dict has %d keys: %s", state_dict.size(), _item_to_string(keys)));
				}
				String method_info = vformat("method[%d]", i);
				print_line(vformat("_select_best_method: About to call %s with %d args", method_info, args.size()));
			}

			// Prepare arguments for callp
			LocalVector<const Variant *> argptrs_vec;
			argptrs_vec.resize(args.size());
			for (int j = 0; j < args.size(); j++) {
				argptrs_vec[j] = &args[j];
			}
			const Variant **argptrs = argptrs_vec.ptr();

			// Call method
			Callable::CallError call_error;
			Variant return_value;
			bool is_valid_count = false;
			int method_arg_count = method.get_argument_count(&is_valid_count);
			if (verbose >= 3 && is_valid_count) {
				String method_name = method.get_method();
				print_line(vformat("_select_best_method: Method[%d] '%s' expects %d arguments, we're passing %d", i, method_name, method_arg_count, args.size()));
			}
			method.callp(argptrs, args.size(), return_value, call_error);

			// Guard: method call must succeed
			if (call_error.error != Callable::CallError::CALL_OK) {
				if (verbose >= 2) {
					ERR_PRINT(vformat("_select_best_method: Task method call failed with error %d (argument %d, expected %d)",
							call_error.error, call_error.argument, call_error.expected));
					if (is_valid_count) {
						ERR_PRINT(vformat("_select_best_method: Method signature expects %d arguments", method_arg_count));
					}
				}
				continue;
			}

			result = return_value;
			if (verbose >= 3) {
				print_line(vformat("_select_best_method: Task method returned type %d (ARRAY=%d)", result.get_type(), Variant::ARRAY));
				if (result.get_type() == Variant::ARRAY) {
					Array result_arr = result;
					print_line(vformat("_select_best_method: Task method returned array with %d elements", result_arr.size()));
				}
			}

			// Guard: result must not be NIL
			if (result.get_type() == Variant::NIL) {
				if (verbose >= 2) {
					ERR_PRINT("_select_best_method: Task method returned NIL (method may have explicitly returned NIL)");
				}
				continue;
			}
		} else if (p_node_type == static_cast<int>(PlannerNodeType::TYPE_UNIGOAL)) {
			Array unigoal_arr = p_node_info;
			// Guard: unigoal must have at least MIN_UNIGOAL_SIZE elements
			if (unigoal_arr.size() < MIN_UNIGOAL_SIZE) {
				continue; // Invalid unigoal
			}
			String subject = unigoal_arr[1];
			Variant value = unigoal_arr[2];
			result = method.call(p_state, subject, value);
		} else if (p_node_type == static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
			result = method.call(p_state, p_node_info);
		} else {
			continue; // Unknown node type
		}

		// Guard: result must be an ARRAY
		if (result.get_type() != Variant::ARRAY) {
			continue;
		}

		Array candidate_subtasks = result;
		if (verbose >= 3) {
			print_line(vformat("_select_best_method: Method returned array with %d elements: %s", candidate_subtasks.size(), _item_to_string(candidate_subtasks)));
		}

		// Guard: candidate_subtasks must not be blacklisted
		if (_is_command_blacklisted(candidate_subtasks)) {
			if (verbose >= 2) {
				print_line(vformat("Method returned blacklisted planner elements array (size %d), treating as failed method", candidate_subtasks.size()));
			}
			continue; // Treat as failed method
		}

		// Score this method using VSIDS activity
		// Optimized: Cache method ID to avoid repeated string conversions
		String method_id = _method_to_id(method);
		double activity = _get_method_activity(method);
		double score = activity * ACTIVITY_SCORE_MULTIPLIER; // Activity as primary score (VSIDS-style)

		// Add small bonus for methods with fewer subtasks (prefer direct methods)
		double subtask_bonus = 0.0;
		if (candidate_subtasks.size() > 0) {
			subtask_bonus = SUBTASK_BONUS_BASE / (1.0 + candidate_subtasks.size());
			score += subtask_bonus;
		}

		if (verbose >= 3) {
			double debug_scaled = activity * (ACTIVITY_SCORE_MULTIPLIER * 10.0); // For debug display only
			print_line(vformat("VSIDS: Evaluating method '%s' - activity: %.6f, scaled: %.6f, subtask bonus: %.2f, total score: %.6f",
					method_id, activity, debug_scaled, subtask_bonus, score));
		}

		// Track candidate for explanation/debugging
		if (p_track_alternatives && p_alternatives != nullptr) {
			Dictionary candidate_info;
			candidate_info["method_id"] = method_id;
			candidate_info["score"] = score;
			candidate_info["activity"] = activity;
			candidate_info["subtask_count"] = candidate_subtasks.size();
			candidate_info["subtask_bonus"] = subtask_bonus;
			candidate_info["subtasks"] = candidate_subtasks;
			all_candidates.push_back(candidate_info);
		}

		// Update best candidate if this score is better
		if (score > best_candidate.score) {
			best_candidate.method = method;
			best_candidate.subtasks = candidate_subtasks;
			best_candidate.score = score;
			best_candidate.method_id = method_id;
			best_candidate.activity = activity;
			best_candidate.reason = vformat("Selected due to highest score (%.2f) - activity: %.6f, subtask bonus: %.2f", score, activity, subtask_bonus);

			// Optimized: Early termination if we find a method with very high score
			// This indicates a method with high activity and few subtasks - likely optimal
			// Threshold: score > 1000 (very high activity * multiplier)
			if (score > 1000.0 && candidate_subtasks.size() <= 1) {
				if (verbose >= 3) {
					print_line(vformat("VSIDS: Early termination - found high-scoring method '%s' with score %.2f", method_id, score));
				}
				break; // Early exit - unlikely to find better
			}
		}
	}

	// Store alternatives if tracking enabled
	if (p_track_alternatives && p_alternatives != nullptr) {
		*p_alternatives = all_candidates;
	}

	// Set reason for best candidate if not already set
	if (best_candidate.method.is_valid() && best_candidate.reason.is_empty()) {
		best_candidate.reason = vformat("Selected method '%s' with score %.2f", best_candidate.method_id, best_candidate.score);
	}

	return best_candidate;
}

Ref<PlannerResult> PlannerPlan::run_lazy_lookahead(Dictionary p_state, Array p_todo_list, int p_max_tries) {
	// Note: Array is a value type in Godot and cannot be null, so no null check needed

	// Input validation: validate max_tries is positive
	if (p_max_tries <= 0) {
		if (verbose >= 1) {
			ERR_PRINT(vformat("PlannerPlan::run_lazy_lookahead: max_tries must be positive, got %d", p_max_tries));
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		// Optimized: Cache get_graph() result
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	// CRITICAL: Validate that domain is set before proceeding
	if (!current_domain.is_valid()) {
		ERR_PRINT("PlannerPlan::run_lazy_lookahead: current_domain is not set. Call set_current_domain() before run_lazy_lookahead().");
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		// Optimized: Cache get_graph() result
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	// Handle empty todo_list edge case
	if (p_todo_list.is_empty()) {
		if (verbose >= 1) {
			print_line("run_lazy_lookahead: Empty todo_list");
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		// Optimized: Cache get_graph() result
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
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

	Ref<PlannerResult> last_result; // Track the last successful result

	for (int tries = 1; tries <= p_max_tries; tries++) {
		if (verbose >= 1) {
			print_line(vformat("run_lazy_lookahead: %sth call to find_plan: %s", tries, ordinals.get(tries, "")));
		}

		Ref<PlannerResult> plan_result = find_plan(p_state, p_todo_list);
		if (!plan_result.is_valid() || !plan_result->get_success()) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("run_lazy_lookahead: find_plan has failed after %s calls.", tries));
			}
			// Return result with current state and last solution graph if available
			Ref<PlannerResult> result = memnew(PlannerResult);
			result->set_success(false);
			result->set_final_state(p_state);
			result->set_solution_graph(last_result.is_valid() ? last_result->get_solution_graph() : (plan_result.is_valid() ? plan_result->get_solution_graph() : Dictionary()));
			return result;
		}

		last_result = plan_result; // Track the last successful result

		Array plan = plan_result->extract_plan();
		if (plan.is_empty()) {
			if (verbose >= 1) {
				print_line(vformat("run_lazy_lookahead: Empty plan => success\nafter %s calls to find_plan.", tries));
			}
			if (verbose >= 2) {
				print_line(vformat("run_lazy_lookahead: final state %s", p_state));
			}
			// Return result with final state and solution graph
			Ref<PlannerResult> result = memnew(PlannerResult);
			result->set_success(true);
			result->set_final_state(p_state);
			result->set_solution_graph(plan_result->get_solution_graph());
			return result;
		}

		if (!plan.is_empty()) {
			Array action_list = plan;
			for (int i = 0; i < action_list.size(); i++) {
				Array action = action_list[i];
				// Validate action array is not empty before accessing first element
				if (action.is_empty() || action.size() < 1) {
					if (verbose >= 1) {
						ERR_PRINT("run_lazy_lookahead: Found empty action in plan, skipping");
					}
					continue;
				}
				// Safety: extract action name first to avoid multiple array accesses
				String command_name_str = action[0];
				if (!current_domain->command_dictionary.has(command_name_str)) {
					if (verbose >= 1) {
						ERR_PRINT(vformat("run_lazy_lookahead: Action '%s' not found in domain, skipping", command_name_str));
					}
					continue;
				}
				Callable command_name = current_domain->command_dictionary[command_name_str];
				if (verbose >= 1) {
					String action_arguments;
					Array actions = action.slice(1, action.size());
					for (Variant element : actions) {
						action_arguments += String(" ") + String(element);
					}
					print_line(vformat("run_lazy_lookahead: Task: %s, %s", command_name.get_method(), action_arguments));
				}

				// Inline _apply_task_and_continue (single use)
				Array argument;
				argument.push_back(p_state);
				argument.append_array(action.slice(1, action.size()));
				if (verbose >= 3) {
					print_line(vformat("_apply_task_and_continue %s, args = %s", command_name.get_method(), _item_to_string(action.slice(1, action.size()))));
				}
				Variant result = command_name.callv(argument);
				if (!result) {
					if (verbose >= 3) {
						print_line(vformat("Not applicable command %s", argument));
					}
					if (verbose >= 1) {
						ERR_PRINT(vformat("run_lazy_lookahead: WARNING: action %s failed; will call find_plan.", command_name));
					}
					break;
				}
				if (verbose >= 3) {
					print_line("Applied");
					print_line(result);
				}
				if (result.get_type() == Variant::DICTIONARY) {
					Dictionary new_state = result;
					if (verbose >= 2) {
						print_line(new_state);
					}
					p_state = new_state;
				} else {
					// Result is not a Dictionary, action failed
					if (verbose >= 1) {
						ERR_PRINT(vformat("run_lazy_lookahead: WARNING: action %s failed; will call find_plan.", command_name));
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

	// Return result with final state (planning failed due to too many tries)
	// Use the solution graph from the last successful find_plan call if available
	Ref<PlannerResult> result = memnew(PlannerResult);
	result->set_success(false);
	result->set_final_state(p_state);
	result->set_solution_graph(last_result.is_valid() ? last_result->get_solution_graph() : solution_graph.get_graph());
	return result;
}

void PlannerPlan::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_verbose"), &PlannerPlan::get_verbose);
	ClassDB::bind_method(D_METHOD("set_verbose", "level"), &PlannerPlan::set_verbose);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "verbose"), "set_verbose", "get_verbose");

	ClassDB::bind_method(D_METHOD("get_max_depth"), &PlannerPlan::get_max_depth);
	ClassDB::bind_method(D_METHOD("set_max_depth", "max_depth"), &PlannerPlan::set_max_depth);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_depth"), "set_max_depth", "get_max_depth");

	ClassDB::bind_method(D_METHOD("get_max_iterations"), &PlannerPlan::get_max_iterations);
	ClassDB::bind_method(D_METHOD("set_max_iterations", "max_iterations"), &PlannerPlan::set_max_iterations);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_iterations"), "set_max_iterations", "get_max_iterations");

	ClassDB::bind_method(D_METHOD("get_max_stack_size"), &PlannerPlan::get_max_stack_size);
	ClassDB::bind_method(D_METHOD("set_max_stack_size", "max_stack_size"), &PlannerPlan::set_max_stack_size);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_stack_size"), "set_max_stack_size", "get_max_stack_size");

	ClassDB::bind_method(D_METHOD("get_current_domain"), &PlannerPlan::get_current_domain);
	ClassDB::bind_method(D_METHOD("set_current_domain", "current_domain"), &PlannerPlan::set_current_domain);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "current_domain", PROPERTY_HINT_RESOURCE_TYPE, "Domain"), "set_current_domain", "get_current_domain");

	ClassDB::bind_method(D_METHOD("find_plan", "state", "todo_list"), &PlannerPlan::find_plan);
	ClassDB::bind_method(D_METHOD("run_lazy_lookahead", "state", "todo_list", "max_tries"), &PlannerPlan::run_lazy_lookahead, DEFVAL(10));
	ClassDB::bind_method(D_METHOD("run_lazy_refineahead", "state", "todo_list"), &PlannerPlan::run_lazy_refineahead);

	ClassDB::bind_method(D_METHOD("blacklist_command", "command"), &PlannerPlan::blacklist_command);
	ClassDB::bind_method(D_METHOD("get_iterations"), &PlannerPlan::get_iterations);
	ClassDB::bind_method(D_METHOD("get_method_activities"), &PlannerPlan::get_method_activities);
	ClassDB::bind_method(D_METHOD("reset_vsids_activity"), &PlannerPlan::reset_vsids_activity);
	ClassDB::bind_method(D_METHOD("reset"), &PlannerPlan::reset);
	ClassDB::bind_method(D_METHOD("simulate", "result", "state", "start_ind"), &PlannerPlan::simulate, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("replan", "result", "state", "fail_node_id"), &PlannerPlan::replan);
	ClassDB::bind_method(D_METHOD("load_solution_graph", "graph"), &PlannerPlan::load_solution_graph);

	// Belief-immersed architecture: Persona and belief management
	ClassDB::bind_method(D_METHOD("get_current_persona"), &PlannerPlan::get_current_persona);
	ClassDB::bind_method(D_METHOD("set_current_persona", "persona"), &PlannerPlan::set_current_persona);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "current_persona", PROPERTY_HINT_RESOURCE_TYPE, "PlannerPersona"), "set_current_persona", "get_current_persona");

	ClassDB::bind_method(D_METHOD("get_belief_manager"), &PlannerPlan::get_belief_manager);
	ClassDB::bind_method(D_METHOD("set_belief_manager", "manager"), &PlannerPlan::set_belief_manager);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "belief_manager", PROPERTY_HINT_RESOURCE_TYPE, "PlannerBeliefManager"), "set_belief_manager", "get_belief_manager");

	// Temporal planning methods
	ClassDB::bind_method(D_METHOD("get_time_range_dict"), &PlannerPlan::get_time_range_dict);
	ClassDB::bind_method(D_METHOD("set_time_range_dict", "time_range"), &PlannerPlan::set_time_range_dict);
	ClassDB::bind_method(D_METHOD("attach_metadata", "item", "temporal_constraints", "entity_constraints"), &PlannerPlan::attach_metadata, DEFVAL(Dictionary()), DEFVAL(Dictionary()));
}

// Temporal method implementations

int PlannerPlan::get_max_depth() const {
	return max_depth;
}

void PlannerPlan::set_max_depth(int p_max_depth) {
	max_depth = p_max_depth;
}

int PlannerPlan::get_max_iterations() const {
	return max_iterations;
}

void PlannerPlan::set_max_iterations(int p_max_iterations) {
	max_iterations = p_max_iterations;
}

int PlannerPlan::get_max_stack_size() const {
	return max_stack_size;
}

void PlannerPlan::set_max_stack_size(int p_max_stack_size) {
	max_stack_size = p_max_stack_size;
}

Dictionary PlannerPlan::get_method_activities() const {
	// Return a copy of method_activities dictionary for testing (converts HashMap to Dictionary)
	Dictionary result;
	for (const KeyValue<String, double> &kv : method_activities) {
		result[kv.key] = kv.value;
	}
	return result;
}

void PlannerPlan::reset_vsids_activity() {
	// Reset VSIDS activity tracking to initial state
	// This clears all learned activity scores, allowing a fresh start
	// Useful when switching problem types or domains
	method_activities.clear();
	activity_var_inc = 1.0;
	activity_bump_count = 0;
	if (verbose >= 1) {
		print_line("VSIDS activity tracking reset");
	}
}

void PlannerPlan::reset() {
	// Complete reset of ALL planner state for test isolation
	// This ensures no state pollution between tests (Elixir is functional/immutable, C++ needs explicit resets)

	// Reset solution graph (creates fresh graph with root node)
	solution_graph = PlannerSolutionGraph();

	// Reset command blacklist
	blacklisted_commands.clear();

	// Reset todo list tracking
	original_todo_list.clear();

	// Reset persona-related state (belief-immersed architecture)
	current_persona = Ref<PlannerPersona>();
	belief_manager = Ref<PlannerBeliefManager>();

	// Reset iteration counter
	iterations = 0;

	// Reset VSIDS activity tracking (all learning state)
	method_activities.clear();
	activity_var_inc = 1.0;
	activity_bump_count = 0;
	rewarded_methods_this_solve.clear();

	// Reset STN solver and snapshot (temporal constraint state)
	stn.clear();
	stn_snapshot = PlannerSTNSolver::Snapshot(); // Reset snapshot to empty state

	// Reset time range
	time_range.set_start_time(0);

	// CRITICAL: Clear current_domain to prevent domain pollution between tests
	// Tests must explicitly set the domain after calling reset()
	current_domain = Ref<PlannerDomain>();

	// Reset configuration to defaults
	max_depth = 10; // Default maximum recursion depth
	max_iterations = 50000; // Default maximum planning loop iterations
	max_stack_size = 10000; // Default maximum stack size
	verbose = 0; // Default verbosity level

	if (verbose >= 2) {
		print_line("PlannerPlan::reset() - All state cleared for test isolation (including domain, STN snapshot, max_depth, verbose, and all mutable state)");
	}
}

// Graph-based lazy refinement (Elixir-style)
Ref<PlannerResult> PlannerPlan::run_lazy_refineahead(Dictionary p_state, Array p_todo_list) {
	// Note: Array is a value type in Godot and cannot be null, so no null check needed

	if (verbose >= 1) {
		print_line("run_lazy_refineahead: Starting graph-based planning");
		print_line("Initial state keys: " + String(Variant(p_state.keys())));
		print_line("Todo list: " + _item_to_string(p_todo_list));
	}

	// CRITICAL: Reset planning-specific state for each planning call
	// Note: We do NOT reset configuration (verbose, max_depth, domain) - those persist across planning calls
	// Only reset() resets everything including configuration

	// Reset solution graph (creates fresh graph with root node)
	solution_graph = PlannerSolutionGraph();

	// Reset command blacklist
	blacklisted_commands.clear();

	// Reset todo list tracking
	original_todo_list.clear();

	// Reset iteration counter
	iterations = 0;

	// Reset VSIDS activity tracking (all learning state)
	method_activities.clear();
	activity_var_inc = 1.0;
	activity_bump_count = 0;
	rewarded_methods_this_solve.clear();

	// Reset STN solver and snapshot (temporal constraint state)
	stn.clear();
	stn_snapshot = PlannerSTNSolver::Snapshot(); // Reset snapshot to empty state

	// Lazy STN Validation: Only initialize STN if temporal constraints are present
	bool has_temporal = _todo_list_has_temporal_constraints(p_todo_list);
	if (has_temporal) {
		// Initialize STN solver with origin time point (planning-specific initialization)
		stn.add_time_point("origin"); // Origin time point (plan start)

		// Initialize time range if not already set
		if (time_range.get_start_time() == 0) {
			time_range.set_start_time(PlannerTimeRange::now_microseconds());
		}

		// Anchor origin to current absolute time
		PlannerSTNConstraints::anchor_to_origin(stn, "origin", time_range.get_start_time());
	} else if (verbose >= 2) {
		print_line("[LAZY_STN] No temporal constraints detected, skipping STN initialization");
	}

	// Validate that current_domain is set before accessing its members
	if (!current_domain.is_valid()) {
		if (verbose >= 1) {
			print_line("run_lazy_refineahead: Error - no domain set");
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		// Optimized: Cache get_graph() result
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	// Handle empty todo_list edge case
	if (p_todo_list.is_empty()) {
		if (verbose >= 1) {
			print_line("run_lazy_refineahead: Empty todo_list");
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		// Optimized: Cache get_graph() result
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	// Add initial tasks to the solution graph
	int parent_node_id = 0; // Root node
	if (verbose >= 2) {
		Array task_keys = current_domain->task_method_dictionary.keys();
		print_line(vformat("[RUN_LAZY_REFINEAHEAD] Using domain with %d task methods: %s", task_keys.size(), _item_to_string(task_keys)));
	}
	PlannerGraphOperations::add_nodes_and_edges(
			solution_graph,
			parent_node_id,
			p_todo_list,
			current_domain->command_dictionary,
			current_domain->task_method_dictionary,
			current_domain->unigoal_method_dictionary,
			current_domain->multigoal_method_list,
			verbose);

	// Start planning loop (using iterative version to prevent stack overflow)
	Dictionary final_state = _planning_loop_iterative(parent_node_id, p_state, 0);

	// Update time range with end time
	time_range.set_end_time(PlannerTimeRange::now_microseconds());
	time_range.calculate_duration();

	if (verbose >= 1) {
		print_line("run_lazy_refineahead: Completed graph-based planning");
		print_line("Duration: " + itos(time_range.get_duration()) + " microseconds");
	}

	// Create PlannerResult with final state and solution graph
	Ref<PlannerResult> result = memnew(PlannerResult);
	result->set_final_state(final_state);
	result->set_solution_graph(solution_graph.get_graph());
	// Check if planning succeeded (similar to find_plan logic)
	// For run_lazy_refineahead, we consider it successful if we got a non-empty final state
	result->set_success(!final_state.is_empty());

	return result;
}

// Iterative version to prevent stack overflow - uses a stack instead of recursion
// This is a full refactor that converts all recursive calls to stack operations
Dictionary PlannerPlan::_planning_loop_iterative(int p_parent_node_id, Dictionary p_state, int p_iter) {
	// PlanningFrame is now defined in plan.h

	LocalVector<PlanningFrame> stack;
	stack.push_back({ p_parent_node_id, p_state, p_iter });

	Dictionary final_state = p_state;

	// Main iterative loop - processes stack until empty
	// Safety: Limit maximum iterations to prevent infinite loops
	// Use max_depth * ITERATIONS_PER_DEPTH as a reasonable upper bound (allows ~ITERATIONS_PER_DEPTH nodes per depth level on average)
	// Also cap at max_iterations to prevent excessive memory usage even with very high max_depth
	const int MAX_ITERATIONS = MIN(max_depth * ITERATIONS_PER_DEPTH, max_iterations);
	int loop_count = 0;
	while (!stack.is_empty() && loop_count < MAX_ITERATIONS) {
		loop_count++;

		// Safety: Check stack size to prevent excessive memory usage
		if (static_cast<int>(stack.size()) > max_stack_size) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("Planning loop stack size (%d) exceeded maximum (%d), forcing exit", static_cast<int>(stack.size()), max_stack_size));
			}
			stack.clear();
			break;
		}
		PlanningFrame frame = stack[stack.size() - 1];
		stack.remove_at(stack.size() - 1);

		int parent_node_id = frame.parent_node_id;
		Dictionary state = frame.state;
		int iter = frame.iter;

		// Track maximum iteration reached
		if (iter > iterations) {
			iterations = iter;
		}

		// Check depth limit to prevent infinite loops
		if (iter >= max_depth) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("Planning depth limit (%d) exceeded, aborting", max_depth));
			}
			final_state = state;
			// Clear stack to force exit - don't continue processing
			stack.clear();
			break;
		}

		// Validate that current_domain is set
		if (!current_domain.is_valid()) {
			if (verbose >= 1) {
				ERR_PRINT("PlannerPlan::_planning_loop_iterative: current_domain is not set. Aborting planning loop.");
			}
			final_state = state;
			continue;
		}

		if (verbose >= 2) {
			print_line(vformat("_planning_loop_iterative: parent_node_id=%d, iter=%d", parent_node_id, iter));
		}

		// Find the first Open node
		Variant open_node_result = PlannerGraphOperations::find_open_node(solution_graph, parent_node_id);

		if (open_node_result.get_type() == Variant::NIL) {
			// No open node found, check if parent is root
			Dictionary parent_node = solution_graph.get_node(parent_node_id);
			int parent_type = parent_node["type"];

			if (parent_type == static_cast<int>(PlannerNodeType::TYPE_ROOT)) {
				// Check if all root children are CLOSED (all tasks completed)
				Dictionary root_node = solution_graph.get_node(0);
				TypedArray<int> root_successors = root_node["successors"];
				int closed_count = 0;
				bool all_closed = true;
				for (int i = 0; i < root_successors.size(); i++) {
					int child_id = root_successors[i];
					if (!solution_graph.get_graph().has(child_id)) {
						continue;
					}
					Dictionary child_node = solution_graph.get_node(child_id);
					int status = child_node["status"];
					if (status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
						closed_count++;
					} else {
						all_closed = false;
						if (verbose >= 3) {
							print_line(vformat("Planning at root: Found non-CLOSED child node %d (status=%d)", child_id, status));
						}
						break;
					}
				}
				// Check if we've completed all tasks from original todo_list
				if (all_closed) {
					if (closed_count >= original_todo_list.size()) {
						// Planning complete - all tasks are CLOSED
						if (verbose >= 1) {
							print_line("Planning complete, returning final state");
						}
						final_state = state;
						continue;
					} else {
						// Some tasks were removed (failed completely), recreate them
						if (verbose >= 2) {
							print_line(vformat("Planning at root: All remaining tasks CLOSED (%d/%d), recreating removed tasks...", closed_count, original_todo_list.size()));
						}
						Array tasks_to_recreate;
						TypedArray<int> failed_root_children_to_remove;

						if (root_successors.size() == 0 && original_todo_list.size() > 0) {
							if (verbose >= 2) {
								print_line("Planning at root: root_successors is empty but original_todo_list is not - tasks were never added, recreating all tasks");
							}
							tasks_to_recreate = original_todo_list.duplicate();
						} else {
							for (int i = 0; i < original_todo_list.size(); i++) {
								Variant task_item = original_todo_list[i];
								// Handle both strings (most frequent) and arrays
								String task_name;
								if (task_item.get_type() == Variant::STRING) {
									task_name = task_item;
								} else if (task_item.get_type() == Variant::ARRAY) {
									Array task_info = task_item;
									if (task_info.size() > 0) {
										task_name = task_info[0];
									} else {
										continue; // Skip empty arrays
									}
								} else {
									continue; // Skip invalid types
								}
								bool found_closed = false;
								for (int j = 0; j < root_successors.size(); j++) {
									int child_id = root_successors[j];
									if (!solution_graph.get_graph().has(child_id)) {
										continue;
									}
									Dictionary child_node = solution_graph.get_node(child_id);
									int child_status = child_node["status"];
									Variant child_info_variant = child_node["info"];
									// Handle both strings and arrays in child_info
									String child_task_name;
									if (child_info_variant.get_type() == Variant::STRING) {
										child_task_name = child_info_variant;
									} else if (child_info_variant.get_type() == Variant::ARRAY) {
										Array child_info = child_info_variant;
										if (child_info.size() > 0) {
											child_task_name = child_info[0];
										} else {
											continue;
										}
									} else {
										continue;
									}
									// Compare task names
									if (child_task_name == task_name) {
										if (child_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
											found_closed = true;
											break;
										} else if (child_status == static_cast<int>(PlannerNodeStatus::STATUS_FAILED)) {
											if (!failed_root_children_to_remove.has(child_id)) {
												failed_root_children_to_remove.push_back(child_id);
											}
										}
									}
								}
								if (!found_closed) {
									tasks_to_recreate.push_back(task_item);
								}
							}
						}
						if (failed_root_children_to_remove.size() > 0) {
							Dictionary root_node_iter = solution_graph.get_node(0);
							TypedArray<int> updated_successors;
							TypedArray<int> current_successors = root_node_iter["successors"];
							for (int i = 0; i < current_successors.size(); i++) {
								int child_id = current_successors[i];
								if (!failed_root_children_to_remove.has(child_id)) {
									updated_successors.push_back(child_id);
								}
							}
							root_node_iter["successors"] = updated_successors;
							solution_graph.update_node(0, root_node_iter);
							if (verbose >= 2) {
								print_line(vformat("Planning at root: Removed %d FAILED root children", failed_root_children_to_remove.size()));
							}
						}
						if (tasks_to_recreate.size() > 0) {
							blacklisted_commands.clear();
							if (verbose >= 2) {
								print_line(vformat("Planning at root: Cleared entire blacklist before recreating %d tasks (state has changed)", tasks_to_recreate.size()));
							}
							PlannerGraphOperations::add_nodes_and_edges(
									solution_graph,
									0,
									tasks_to_recreate,
									current_domain->command_dictionary,
									current_domain->task_method_dictionary,
									current_domain->unigoal_method_dictionary,
									current_domain->multigoal_method_list,
									verbose);
							if (verbose >= 2) {
								print_line(vformat("Planning at root: Recreated %d tasks, continuing...", tasks_to_recreate.size()));
							}
							// Push continuation to stack
							stack.push_back({ 0, state, iter + 1 });
							continue;
						}
						if (verbose >= 1) {
							print_line("Planning complete, returning final state");
						}
						final_state = state;
						continue;
					}
				} else {
					// Some tasks are not CLOSED, continue planning
					if (verbose >= 2) {
						print_line("Planning at root: Not all tasks are CLOSED, continuing...");
					}
					stack.push_back({ 0, state, iter + 1 });
					continue;
				}
			} else {
				// Move to predecessor
				int new_parent = PlannerGraphOperations::find_predecessor(solution_graph, parent_node_id);
				if (new_parent >= 0) {
					stack.push_back({ new_parent, state, iter + 1 });
					continue;
				}
				// If we can't find a predecessor and we're not at root, try going back to root
				// This can happen if the graph structure is incomplete or if we've backtracked too far
				if (parent_node_id != 0) {
					if (verbose >= 2) {
						print_line(vformat("Planning: Could not find predecessor for node %d, returning to root", parent_node_id));
					}
					stack.push_back({ 0, state, iter + 1 });
					continue;
				}
				// Already at root and can't find predecessor - this shouldn't happen, but handle gracefully
				if (verbose >= 1) {
					ERR_PRINT(vformat("Planning: At root but could not find predecessor - this should not happen"));
				}
				final_state = state;
				continue;
			}
		}

		int curr_node_id = open_node_result;
		Dictionary curr_node = solution_graph.get_node(curr_node_id);

		if (verbose >= 2) {
			print_line(vformat("Iteration %d: Refining node %d", iter, curr_node_id));
		}

		// Save current state if first visit (state is empty)
		Dictionary node_state = solution_graph.get_state_snapshot(curr_node_id);
		if (node_state.is_empty()) {
			// First visit - save state
			Array state_keys = state.keys();
			if (verbose >= 3) {
				print_line(vformat("Node %d: Saving state with %d keys: %s", curr_node_id, state_keys.size(), _item_to_string(state_keys)));
			}
			solution_graph.save_state_snapshot(curr_node_id, state.duplicate(true));
			PlannerSTNSolver::Snapshot snapshot = stn.create_snapshot();
			curr_node["stn_snapshot"] = snapshot.to_dictionary();
			solution_graph.update_node(curr_node_id, curr_node);
		} else {
			// Node has saved state - this means we're revisiting it
			// CRITICAL: For task and unigoal nodes, we should NOT restore state when reopening
			// The state should be preserved from successful actions
			// Only restore state for actions that failed and we're backtracking
			int node_status = curr_node.has("status") ? static_cast<int>(curr_node["status"]) : static_cast<int>(PlannerNodeStatus::STATUS_OPEN);
			int node_type = curr_node.has("type") ? static_cast<int>(curr_node["type"]) : -1;
			if (node_status == static_cast<int>(PlannerNodeStatus::STATUS_OPEN) &&
					(node_type == static_cast<int>(PlannerNodeType::TYPE_TASK) ||
							node_type == static_cast<int>(PlannerNodeType::TYPE_UNIGOAL))) {
				// Task or unigoal node was reopened - preserve current state (includes successful actions)
				// Don't restore old state, keep the accumulated progress
				if (verbose >= 3) {
					Array state_keys = state.keys();
					Array saved_keys = node_state.keys();
					String node_type_str = (node_type == static_cast<int>(PlannerNodeType::TYPE_TASK)) ? "TASK" : "UNIGOAL";
					print_line(vformat("Node %d (%s, OPEN): Preserving current state (%d keys: %s) instead of restoring saved state (%d keys: %s)",
							curr_node_id, node_type_str, state_keys.size(), _item_to_string(state_keys), saved_keys.size(), _item_to_string(saved_keys)));
				}
			} else {
				// Restore state for other cases (e.g., action nodes that failed)
				// CRITICAL: Use deep duplicate to ensure nested dictionaries are copied
				if (verbose >= 3) {
					Array saved_keys = node_state.keys();
					print_line(vformat("Node %d: Restoring saved state with %d keys: %s", curr_node_id, saved_keys.size(), _item_to_string(saved_keys)));
				}
				state = node_state.duplicate(true);
				// Also restore STN snapshot
				_restore_stn_from_node(curr_node_id);
			}
		}

		// Validate required dictionary keys exist
		if (!curr_node.has("type")) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("PlannerPlan::_planning_loop_iterative: Node %d missing 'type' field", curr_node_id));
			}
			final_state = state;
			continue;
		}
		if (!curr_node.has("info")) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("PlannerPlan::_planning_loop_iterative: Node %d missing 'info' field", curr_node_id));
			}
			final_state = state;
			continue;
		}

		int node_type = curr_node["type"];

		// Process node type - this will push new frames to stack or set final_state
		// We use a helper function to process each node type and determine next action
		// Pass curr_node by value (copy) to avoid stale reference issues after graph modifications
		Dictionary curr_node_copy = curr_node;
		bool should_continue = _process_node_iterative(parent_node_id, curr_node_id, curr_node_copy, node_type, state, iter, stack, final_state);
		if (!should_continue) {
			break; // Final state set, exit loop
		}
		// Otherwise continue loop to process next frame from stack
	}

	// Safety check: Warn if we hit the iteration limit
	if (loop_count >= MAX_ITERATIONS && !stack.is_empty()) {
		if (verbose >= 1) {
			ERR_PRINT(vformat("Planning loop exceeded maximum iterations (%d, based on max_depth=%d and max_iterations=%d), forcing exit. Stack size: %d", MAX_ITERATIONS, max_depth, max_iterations, stack.size()));
		}
		stack.clear();
	}

	return final_state;
}

// Helper function to process a single node iteratively - converts recursive calls to stack pushes
// Note: p_curr_node is passed by value (copy) to avoid stale reference issues after graph modifications
bool PlannerPlan::_process_node_iterative(int p_parent_node_id, int p_curr_node_id, Dictionary p_curr_node, int p_node_type, Dictionary &p_state, int p_iter, LocalVector<PlanningFrame> &p_stack, Dictionary &p_final_state) {
	// Full iterative implementation - duplicates switch statement logic and converts all recursive calls to stack pushes
	switch (static_cast<PlannerNodeType>(p_node_type)) {
		case PlannerNodeType::TYPE_ROOT: {
			// Root node: check if all successors are closed
			TypedArray<int> successors = p_curr_node["successors"];
			bool all_successors_closed = true;

			for (int i = 0; i < successors.size(); i++) {
				int succ_id = successors[i];
				Dictionary succ_node = solution_graph.get_node(succ_id);
				int succ_status = succ_node["status"];
				if (succ_status != static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
					all_successors_closed = false;
					break;
				}
			}

			if (all_successors_closed) {
				// All successors are closed, planning succeeded
				p_final_state = p_state;
				return false; // Return false to stop planning (success)
			} else {
				// Not all successors are closed yet, continue processing
				// Find the first open successor and push it to stack
				for (int i = 0; i < successors.size(); i++) {
					int succ_id = successors[i];
					Dictionary succ_node = solution_graph.get_node(succ_id);
					int succ_status = succ_node["status"];
					if (succ_status == static_cast<int>(PlannerNodeStatus::STATUS_OPEN)) {
						p_stack.push_back({ succ_id, p_state, p_iter + 1 });
						return true;
					}
				}
				// This shouldn't happen - if not all are closed, there should be at least one open
				p_final_state = p_state;
				return false;
			}
		}

		case PlannerNodeType::TYPE_TASK: {
			Variant task_info = p_curr_node["info"];

			// Guard: Validate entity requirements
			PlannerMetadata item_metadata = _extract_metadata(task_info);
			if (!_validate_entity_requirements(p_state, item_metadata)) {
				if (verbose >= 2) {
					print_line("Task entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			// Unwrap task_info if it's in dictionary format
			Variant actual_task_info = task_info;
			if (task_info.get_type() == Variant::DICTIONARY) {
				Dictionary dict = task_info;
				if (dict.has("item")) {
					actual_task_info = dict["item"];
				}
			}

			// Extract task name (handle both strings and arrays)
			String task_name;
			if (actual_task_info.get_type() == Variant::STRING) {
				task_name = actual_task_info;
			} else if (actual_task_info.get_type() == Variant::ARRAY) {
				Array task_arr = actual_task_info;
				if (!task_arr.is_empty() && task_arr.size() >= 1) {
					task_name = String(task_arr[0]);
				}
			}

			// Guard: Task must have available methods
			TypedArray<Callable> available_methods;
			if (current_domain.is_valid() && current_domain->task_method_dictionary.has(task_name)) {
				Variant methods_var = current_domain->task_method_dictionary[task_name];
				available_methods = TypedArray<Callable>(methods_var);
			}
			if (available_methods.is_empty()) {
				if (verbose >= 1) {
					ERR_PRINT(vformat("PlannerPlan::_process_node_iterative: Task '%s' has no available methods in current domain", task_name));
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			// Guard: Task must not be blacklisted
			if (_is_command_blacklisted(actual_task_info)) {
				if (verbose >= 2) {
					print_line("Task is blacklisted, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			if (!p_curr_node.has("state") || p_curr_node["state"].get_type() == Variant::NIL) {
				p_curr_node["state"] = p_state;
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				if (verbose >= 3) {
					print_line(vformat("Saved state in node %d (first visit)", p_curr_node_id));
				}
			}

			// Build args array: state is first argument, then additional arguments from task (if any)
			// Task format: string "task_name" or array [task_name, arg1, arg2, ...]
			Array args;
			args.push_back(p_state);
			int additional_args_count = 0;
			if (actual_task_info.get_type() == Variant::ARRAY) {
				Array task_arr = actual_task_info;
				if (verbose >= 3) {
					print_line(vformat("Task refinement: task_arr = %s (size %d)", _item_to_string(task_arr), task_arr.size()));
				}
				// CRITICAL: slice(1) preserves nested arrays/fluents (e.g., [5, 5] in ["move_task", "r1", [5, 5]])
				// This ensures fluent arguments are passed correctly to task methods
				if (task_arr.size() > 1) {
					args.append_array(task_arr.slice(1));
					additional_args_count = task_arr.size() - 1;
				}
			} else if (verbose >= 3) {
				print_line(vformat("Task refinement: task = %s (string, no args)", String(actual_task_info)));
			}
			if (verbose >= 3) {
				Array state_keys = p_state.keys();
				print_line(vformat("Task refinement: args = [state with %d keys: %s] + %d additional args from task",
						state_keys.size(), _item_to_string(state_keys), additional_args_count));
			}

			// Track alternatives for explanation/debugging if verbose enough
			Array alternatives;
			bool track_alternatives = (verbose >= 2);
			MethodCandidate best = _select_best_method(available_methods, p_state, actual_task_info, args, static_cast<int>(PlannerNodeType::TYPE_TASK), track_alternatives, &alternatives);

			// Guard: Must find a working method
			if (!best.method.is_valid()) {
				if (verbose >= 2) {
					print_line("Task refinement failed, backtracking");
				}
				_bump_conflict_path_activities(p_curr_node_id);
				// Only blacklist the task if it's a subtask (not a root-level task)
				if (p_parent_node_id > 0) {
					_blacklist_command(actual_task_info);
					if (verbose >= 2) {
						print_line("Blacklisted subtask info since all methods failed");
					}
				} else {
					if (verbose >= 2) {
						print_line("Root-level task failed - will retry with different methods after backtracking");
					}
				}
				if (p_parent_node_id >= 0) {
					// Note: created_subtasks is not in PlannerNodeStruct, so we use Dictionary
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
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands, verbose);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				Variant open_node_result = PlannerGraphOperations::find_open_node(solution_graph, 0);
				if (open_node_result.get_type() != Variant::NIL) {
					p_stack.push_back({ 0, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			// Successfully refined - process subtasks
			Callable selected_method = best.method;
			Array subtasks = best.subtasks;
			if (verbose >= 2) {
				print_line(vformat("Selected method with activity score %.2f", best.score));
			}
			if (verbose >= 3) {
				double activity = _get_method_activity(best.method);
				String method_id = _method_to_id(best.method);
				print_line(vformat("VSIDS: Selected task method '%s' with activity %.6f (score %.2f, subtasks: %d)",
						method_id, activity, best.score, subtasks.size()));
			}

			// Store decision info for explanation/debugging
			Dictionary decision_info;
			decision_info["selected_method_id"] = best.method_id;
			decision_info["selected_score"] = best.score;
			decision_info["selected_activity"] = best.activity;
			decision_info["selected_reason"] = best.reason;
			if (track_alternatives && alternatives.size() > 0) {
				decision_info["alternatives"] = alternatives;
				decision_info["total_candidates"] = alternatives.size();
			}
			decision_info["available_methods_count"] = available_methods.size();

			p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
			p_curr_node["selected_method"] = selected_method;
			p_curr_node["created_subtasks"] = subtasks;
			p_curr_node["decision_info"] = decision_info;
			solution_graph.update_node(p_curr_node_id, p_curr_node);

			if (verbose >= 2) {
				Array task_keys = current_domain->task_method_dictionary.keys();
				print_line(vformat("[PLANNING_LOOP] Adding subtasks using domain with %d task methods: %s", task_keys.size(), _item_to_string(task_keys)));
			}
			PlannerGraphOperations::add_nodes_and_edges(
					solution_graph,
					p_curr_node_id,
					subtasks,
					current_domain->command_dictionary,
					current_domain->task_method_dictionary,
					current_domain->unigoal_method_dictionary,
					current_domain->multigoal_method_list,
					verbose);

			int action_count = _count_closed_actions();
			_reward_method_immediate(selected_method, action_count);

			p_stack.push_back({ p_curr_node_id, p_state, p_iter + 1 });
			return true;
		}

		case PlannerNodeType::TYPE_COMMAND: {
			Variant action_info = p_curr_node["info"];

			if (_is_command_blacklisted(action_info)) {
				if (verbose >= 2) {
					print_line("Command is blacklisted (unexpected - individual actions should not be blacklisted), backtracking");
				}
				if (p_parent_node_id >= 0) {
					// Note: created_subtasks is not in PlannerNodeStruct, so we use Dictionary
					Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
					if (parent_node.has("created_subtasks")) {
						Array parent_subtasks = parent_node["created_subtasks"];
						_blacklist_command(parent_subtasks);
						if (verbose >= 2) {
							print_line("Blacklisted parent method array that contained blacklisted action");
						}
					}
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			stn_snapshot = stn.create_snapshot();
			p_curr_node["stn_snapshot"] = stn_snapshot.to_dictionary();
			solution_graph.update_node(p_curr_node_id, p_curr_node);

			PlannerMetadata item_metadata = _extract_metadata(action_info);
			Dictionary temporal_metadata;
			if (_has_temporal_constraints(action_info)) {
				temporal_metadata = _get_temporal_constraints(action_info);
			}

			if (!_validate_entity_requirements(p_state, item_metadata)) {
				if (verbose >= 2) {
					print_line("Action entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			if (!p_curr_node.has("command")) {
				if (verbose >= 1) {
					ERR_PRINT(vformat("PlannerPlan::_process_node_iterative: Command node %d missing 'command' field", p_curr_node_id));
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}
			Callable command = p_curr_node["command"];
			Variant actual_action_info = action_info;
			if (action_info.get_type() == Variant::DICTIONARY) {
				Dictionary dict = action_info;
				if (dict.has("item")) {
					actual_action_info = dict["item"];
				}
			}
			Array action_arr = actual_action_info;

			// Log temporal metadata if present
			if (!temporal_metadata.is_empty() && verbose >= 1) {
				String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
				String temporal_info = "";
				if (temporal_metadata.has("start_time")) {
					int64_t start_time = temporal_metadata.get("start_time", 0);
					temporal_info += " start_time=" + String::num_int64(start_time);
				}
				if (temporal_metadata.has("end_time")) {
					int64_t end_time = temporal_metadata.get("end_time", 0);
					temporal_info += " end_time=" + String::num_int64(end_time);
				}
				if (temporal_metadata.has("duration")) {
					int64_t duration = temporal_metadata.get("duration", 0);
					temporal_info += " duration=" + String::num_int64(duration);
				}
				print_line(vformat("[METADATA] Action '%s' has temporal constraints:%s", command_name, temporal_info));
			}

			// Log entity requirements if present
			if (item_metadata.requires_entities.size() > 0 && verbose >= 1) {
				String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
				String entity_info = "";
				for (uint32_t i = 0; i < item_metadata.requires_entities.size(); i++) {
					const PlannerEntityRequirement &req = item_metadata.requires_entities[i];
					if (i > 0) {
						entity_info += ", ";
					}
					entity_info += vformat("type=%s capabilities=[", req.type);
					for (uint32_t j = 0; j < req.capabilities.size(); j++) {
						if (j > 0) {
							entity_info += ", ";
						}
						entity_info += req.capabilities[j];
					}
					entity_info += "]";
				}
				print_line(vformat("[METADATA] Action '%s' has entity requirements: %s", command_name, entity_info));
			}

			if (action_arr.is_empty()) {
				if (verbose >= 1) {
					ERR_PRINT("PlannerPlan::_process_node_iterative: Action array is empty");
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			if (!command.is_valid() || command.is_null()) {
				if (verbose >= 1) {
					String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
					print_line(vformat("Command '%s' not found in domain, marking as failed", command_name));
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			Array args;
			args.push_back(p_state);
			args.append_array(action_arr.slice(1));

			int64_t action_start_time;
			if (temporal_metadata.has("start_time")) {
				action_start_time = temporal_metadata["start_time"];
			} else {
				action_start_time = PlannerTimeRange::now_microseconds();
			}

			if (verbose >= 2) {
				String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
				print_line(vformat("Executing command '%s' with args: %s", command_name, _item_to_string(args.slice(1))));
			}

			Variant result = command.callv(args);

			if (result.get_type() == Variant::NIL) {
				if (verbose >= 2) {
					String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
					print_line(vformat("Command '%s' failed (returned NIL), backtracking", command_name));
				}
				_bump_conflict_path_activities(p_curr_node_id);
				if (p_parent_node_id >= 0) {
					// Note: created_subtasks is not in PlannerNodeStruct, so we use Dictionary
					Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
					if (parent_node.has("created_subtasks")) {
						Array created_subtasks = parent_node["created_subtasks"];
						_blacklist_command(created_subtasks);
						if (verbose >= 2) {
							print_line("Blacklisted parent method array that contained failing action");
						}
					}
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}
			if (result.get_type() == Variant::BOOL && result == Variant(false)) {
				if (verbose >= 2) {
					String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
					print_line(vformat("Action '%s' failed (returned false), backtracking", command_name));
				}
				_bump_conflict_path_activities(p_curr_node_id);
				if (p_parent_node_id >= 0) {
					// Note: created_subtasks is not in PlannerNodeStruct, so we use Dictionary
					Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
					if (parent_node.has("created_subtasks")) {
						Array created_subtasks = parent_node["created_subtasks"];
						_blacklist_command(created_subtasks);
						if (verbose >= 2) {
							print_line("Blacklisted parent method array that contained failing action");
						}
					}
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}
			if (result.get_type() != Variant::DICTIONARY) {
				if (verbose >= 1) {
					String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
					ERR_PRINT(vformat("PlannerPlan::_process_node_iterative: Action '%s' returned non-Dictionary result (type: %d), marking as failed",
							command_name, result.get_type()));
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}
			Dictionary new_state = result;
			if (new_state.is_empty()) {
				if (verbose >= 2) {
					String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
					print_line(vformat("Action '%s' failed (returned empty dictionary), backtracking", command_name));
				}
				_bump_conflict_path_activities(p_curr_node_id);
				if (p_parent_node_id >= 0) {
					// Note: created_subtasks is not in PlannerNodeStruct, so we use Dictionary
					Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
					if (parent_node.has("created_subtasks")) {
						Array created_subtasks = parent_node["created_subtasks"];
						_blacklist_command(created_subtasks);
						if (verbose >= 2) {
							print_line("Blacklisted parent method array that contained failing action");
						}
					}
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			int64_t action_end_time;
			if (temporal_metadata.has("end_time")) {
				action_end_time = temporal_metadata["end_time"];
			} else {
				action_end_time = PlannerTimeRange::now_microseconds();
			}

			int64_t action_duration;
			if (temporal_metadata.has("duration")) {
				action_duration = temporal_metadata["duration"];
			} else {
				action_duration = action_end_time - action_start_time;
			}

			if (result.get_type() == Variant::DICTIONARY) {
				if (verbose >= 2) {
					String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
					print_line(vformat("Action '%s' succeeded, new state keys: %s", command_name, String(Variant(new_state.keys()))));
				}

				bool has_temporal = temporal_metadata.has("start_time") || temporal_metadata.has("end_time") || temporal_metadata.has("duration");

				if (has_temporal) {
					if (action_arr.is_empty()) {
						if (verbose >= 2) {
							ERR_PRINT("Action array is empty, cannot create STN interval");
						}
						p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
						solution_graph.update_node(p_curr_node_id, p_curr_node);
						PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
								solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
						solution_graph = backtrack_result.graph;
						if (backtrack_result.parent_node_id >= 0) {
							_restore_stn_from_node(backtrack_result.parent_node_id);
							p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
							return true;
						}
						p_final_state = p_state;
						return false;
					}
					if (action_arr.is_empty() || action_arr.size() < 1) {
						if (verbose >= 1) {
							ERR_PRINT("PlannerPlan::_process_node_iterative: Action array is empty when processing temporal metadata");
						}
						p_final_state = p_state;
						return false;
					}
					String action_id = action_arr[0];
					int64_t metadata_start = temporal_metadata.get("start_time", 0);
					int64_t metadata_end = temporal_metadata.get("end_time", 0);
					int64_t metadata_duration = temporal_metadata.get("duration", action_duration);

					if (verbose >= 1) {
						String start_str = String::num_int64(metadata_start);
						String end_str = String::num_int64(metadata_end);
						String duration_str = String::num_int64(metadata_duration);
						print_line(vformat("[METADATA] Adding STN interval for action '%s': start=%s end=%s duration=%s",
								action_id, start_str, end_str, duration_str));
					}

					bool stn_success = PlannerSTNConstraints::add_interval(
							stn, action_id, metadata_start, metadata_end, metadata_duration);

					if (!stn_success) {
						if (verbose >= 2) {
							print_line("Failed to add interval to STN, backtracking");
						}
						_blacklist_command(action_info);
						stn.restore_snapshot(stn_snapshot);
						PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
								solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
						solution_graph = backtrack_result.graph;
						if (backtrack_result.parent_node_id >= 0) {
							_restore_stn_from_node(backtrack_result.parent_node_id);
							p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
							return true;
						}
						p_final_state = p_state;
						return false;
					}

					stn.check_consistency();
					if (!stn.is_consistent()) {
						if (verbose >= 2) {
							print_line("STN inconsistent after action, backtracking");
						}
						_blacklist_command(action_info);
						PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
								solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
						solution_graph = backtrack_result.graph;
						if (backtrack_result.parent_node_id >= 0) {
							_restore_stn_from_node(backtrack_result.parent_node_id);
							p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
							return true;
						}
						p_final_state = p_state;
						return false;
					}
				} else {
					if (verbose >= 3) {
						String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
						print_line(vformat("Action '%s' has no temporal constraints, skipping STN addition", command_name));
					}
				}

				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				p_curr_node["start_time"] = action_start_time;
				p_curr_node["end_time"] = action_end_time;
				p_curr_node["duration"] = action_duration;
				solution_graph.update_node(p_curr_node_id, p_curr_node);

				time_range.set_end_time(action_end_time);
				time_range.calculate_duration();

				p_stack.push_back({ p_parent_node_id, new_state, p_iter + 1 });
				return true;
			} else {
				String command_name = action_arr.is_empty() ? "unknown" : String(action_arr[0]);
				if (verbose >= 1) {
					print_line(vformat("Action '%s' failed (returned %s, expected Dictionary), backtracking",
							command_name, Variant::get_type_name(result.get_type())));
					if (verbose >= 2) {
						print_line(vformat("  Action args: %s", _item_to_string(args.slice(1))));
						print_line(vformat("  Current state: %s", _item_to_string(p_state)));
					}
				}
				_bump_conflict_path_activities(p_curr_node_id);
				if (p_parent_node_id >= 0) {
					// Note: created_subtasks is not in PlannerNodeStruct, so we use Dictionary
					// But we can still optimize by caching the lookup
					Dictionary parent_node = solution_graph.get_node(p_parent_node_id);
					if (parent_node.has("created_subtasks")) {
						Array parent_subtasks = parent_node["created_subtasks"];
						_blacklist_command(parent_subtasks);
						if (verbose >= 2) {
							print_line("Blacklisted parent method array that contained failing action");
						}
					}
				}
				stn.restore_snapshot(stn_snapshot);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				Variant open_node_result = PlannerGraphOperations::find_open_node(solution_graph, 0);
				if (open_node_result.get_type() != Variant::NIL) {
					p_stack.push_back({ 0, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}
		}

		case PlannerNodeType::TYPE_UNIGOAL: {
			Variant unigoal_info = p_curr_node["info"];

			Variant actual_unigoal_info = unigoal_info;
			if (unigoal_info.get_type() == Variant::DICTIONARY) {
				Dictionary dict = unigoal_info;
				if (dict.has("item")) {
					actual_unigoal_info = dict["item"];
				}
			}

			if (_is_command_blacklisted(actual_unigoal_info)) {
				if (verbose >= 2) {
					print_line("Unigoal is blacklisted, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			Array unigoal_arr = actual_unigoal_info;
			if (unigoal_arr.size() < 3) {
				p_final_state = p_state;
				return false;
			}

			String predicate = unigoal_arr[0];
			String subject = unigoal_arr[1];
			Variant value = unigoal_arr[2];

			PlannerMetadata item_metadata = _extract_metadata(unigoal_info);
			if (!_validate_entity_requirements(p_state, item_metadata)) {
				if (verbose >= 2) {
					print_line("Unigoal entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			if (p_state.has(predicate)) {
				Dictionary predicate_dict = p_state[predicate];
				if (predicate_dict.has(subject) && predicate_dict[subject] == value) {
					p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
					solution_graph.update_node(p_curr_node_id, p_curr_node);
					p_stack.push_back({ p_curr_node_id, p_state, p_iter + 1 });
					return true;
				}
			}

			TypedArray<Callable> available_methods;
			if (current_domain.is_valid() && current_domain->unigoal_method_dictionary.has(predicate)) {
				Variant methods_var = current_domain->unigoal_method_dictionary[predicate];
				available_methods = TypedArray<Callable>(methods_var);
			}

			if (available_methods.is_empty()) {
				if (verbose >= 1) {
					ERR_PRINT(vformat("PlannerPlan::_process_node_iterative: Unigoal predicate '%s' has no available methods in current domain", predicate));
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			// Track alternatives for explanation/debugging if verbose enough
			Array alternatives;
			bool track_alternatives = (verbose >= 2);
			MethodCandidate best = _select_best_method(available_methods, p_state, actual_unigoal_info, Variant(), static_cast<int>(PlannerNodeType::TYPE_UNIGOAL), track_alternatives, &alternatives);

			Callable selected_method;
			Array subtasks;
			bool found_working_method = false;

			if (best.method.is_valid()) {
				selected_method = best.method;
				subtasks = best.subtasks;
				found_working_method = true;
				if (verbose >= 2) {
					print_line(vformat("Selected method with activity score %.2f", best.score));
				}
				if (verbose >= 3) {
					double activity = _get_method_activity(best.method);
					String method_id = _method_to_id(best.method);
					print_line(vformat("VSIDS: Selected unigoal method '%s' with activity %.6f (score %.2f, subtasks: %d)",
							method_id, activity, best.score, subtasks.size()));
				}
			}

			if (found_working_method) {
				// Store decision info for explanation/debugging
				Dictionary decision_info;
				decision_info["selected_method_id"] = best.method_id;
				decision_info["selected_score"] = best.score;
				decision_info["selected_activity"] = best.activity;
				decision_info["selected_reason"] = best.reason;
				if (track_alternatives && alternatives.size() > 0) {
					decision_info["alternatives"] = alternatives;
					decision_info["total_candidates"] = alternatives.size();
				}
				decision_info["available_methods_count"] = available_methods.size();

				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				p_curr_node["selected_method"] = selected_method;
				p_curr_node["created_subtasks"] = subtasks.duplicate(true);
				p_curr_node["decision_info"] = decision_info;
				solution_graph.update_node(p_curr_node_id, p_curr_node);

				PlannerGraphOperations::add_nodes_and_edges(
						solution_graph,
						p_curr_node_id,
						subtasks,
						current_domain->command_dictionary,
						current_domain->task_method_dictionary,
						current_domain->unigoal_method_dictionary,
						current_domain->multigoal_method_list,
						verbose);

				int action_count = _count_closed_actions();
				_reward_method_immediate(selected_method, action_count);

				p_stack.push_back({ p_curr_node_id, p_state, p_iter + 1 });
				return true;
			}

			if (verbose >= 2) {
				print_line("Unigoal refinement failed, backtracking");
			}
			_bump_conflict_path_activities(p_curr_node_id);
			_blacklist_command(actual_unigoal_info);
			if (verbose >= 2) {
				print_line("Blacklisted unigoal info since all methods failed");
			}
			if (p_parent_node_id >= 0) {
				// Note: created_subtasks is not in PlannerNodeStruct, so we use Dictionary
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
					solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands, verbose);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				_restore_stn_from_node(backtrack_result.parent_node_id);
				p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
				return true;
			}
			p_final_state = p_state;
			return false;
		}

		case PlannerNodeType::TYPE_MULTIGOAL: {
			Variant multigoal_variant = p_curr_node["info"];

			if (multigoal_variant.get_type() == Variant::DICTIONARY) {
				Dictionary dict = multigoal_variant;
				if (dict.has("item")) {
					multigoal_variant = dict["item"];
				}
			}

			if (!PlannerMultigoal::is_multigoal_array(multigoal_variant)) {
				p_final_state = p_state;
				return false;
			}
			Array multigoal = multigoal_variant;

			if (multigoal_variant.get_type() == Variant::ARRAY) {
				if (_is_command_blacklisted(multigoal_variant)) {
					if (verbose >= 2) {
						print_line("MultiGoal is blacklisted, backtracking");
					}
					PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
							solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
					solution_graph = backtrack_result.graph;
					if (backtrack_result.parent_node_id >= 0) {
						_restore_stn_from_node(backtrack_result.parent_node_id);
						p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
						return true;
					}
					p_final_state = p_state;
					return false;
				}
			}

			PlannerMetadata item_metadata = _extract_metadata(multigoal_variant);
			if (!_validate_entity_requirements(p_state, item_metadata)) {
				if (verbose >= 2) {
					print_line("MultiGoal entity requirements not met, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			Array goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, multigoal);
			if (goals_not_achieved.is_empty()) {
				if (verbose >= 1) {
					print_line("MultiGoal already achieved, marking as closed");
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				Array empty_subgoals;
				PlannerGraphOperations::add_nodes_and_edges(
						solution_graph,
						p_curr_node_id,
						empty_subgoals,
						current_domain->command_dictionary,
						current_domain->task_method_dictionary,
						current_domain->unigoal_method_dictionary,
						current_domain->multigoal_method_list,
						verbose);
				p_stack.push_back({ p_curr_node_id, p_state, p_iter + 1 });
				return true;
			}

			TypedArray<int> successors = p_curr_node["successors"];
			if (successors.size() > 0) {
				for (int i = 0; i < successors.size(); i++) {
					int succ_id = successors[i];
					Dictionary succ_node = solution_graph.get_node(succ_id);
					int succ_status = succ_node["status"];
					if (succ_status == static_cast<int>(PlannerNodeStatus::STATUS_OPEN)) {
						if (verbose >= 2) {
							print_line(vformat("MultiGoal node %d already has successors, continuing from open successor %d", p_curr_node_id, succ_id));
						}
						p_stack.push_back({ succ_id, p_state, p_iter + 1 });
						return true;
					}
				}
				Array goals_not_achieved_check = PlannerMultigoal::method_goals_not_achieved(p_state, multigoal);
				if (goals_not_achieved_check.is_empty()) {
					p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
					solution_graph.update_node(p_curr_node_id, p_curr_node);
					p_stack.push_back({ p_parent_node_id, p_state, p_iter + 1 });
					return true;
				}
			}

			TypedArray<Callable> available_methods;
			if (current_domain.is_valid()) {
				available_methods = current_domain->multigoal_method_list;
			}

			if (available_methods.is_empty()) {
				if (verbose >= 1) {
					ERR_PRINT("PlannerPlan::_process_node_iterative: MultiGoal has no available methods in current domain");
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}

			// Track alternatives for explanation/debugging if verbose enough
			Array alternatives;
			bool track_alternatives = (verbose >= 2);
			MethodCandidate best = _select_best_method(available_methods, p_state, multigoal_variant, Variant(), static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL), track_alternatives, &alternatives);

			Callable selected_method;
			Array subgoals;
			bool found_working_method = false;

			if (best.method.is_valid()) {
				selected_method = best.method;
				subgoals = best.subtasks;
				found_working_method = true;
				if (verbose >= 2) {
					print_line(vformat("Selected method with activity score %.2f", best.score));
				}
				if (verbose >= 3) {
					double activity = _get_method_activity(best.method);
					String method_id = _method_to_id(best.method);
					print_line(vformat("VSIDS: Selected multigoal method '%s' with activity %.6f (score %.2f, subgoals: %d)",
							method_id, activity, best.score, subgoals.size()));
				}
			}

			if (found_working_method) {
				// Store decision info for explanation/debugging
				Dictionary decision_info;
				decision_info["selected_method_id"] = best.method_id;
				decision_info["selected_score"] = best.score;
				decision_info["selected_activity"] = best.activity;
				decision_info["selected_reason"] = best.reason;
				if (track_alternatives && alternatives.size() > 0) {
					decision_info["alternatives"] = alternatives;
					decision_info["total_candidates"] = alternatives.size();
				}
				decision_info["available_methods_count"] = available_methods.size();

				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				p_curr_node["selected_method"] = selected_method;
				p_curr_node["created_subtasks"] = subgoals;
				p_curr_node["decision_info"] = decision_info;
				solution_graph.update_node(p_curr_node_id, p_curr_node);

				PlannerGraphOperations::add_nodes_and_edges(
						solution_graph,
						p_curr_node_id,
						subgoals,
						current_domain->command_dictionary,
						current_domain->task_method_dictionary,
						current_domain->unigoal_method_dictionary,
						current_domain->multigoal_method_list,
						verbose);

				int action_count = _count_closed_actions();
				_reward_method_immediate(selected_method, action_count);

				// Get fresh copy of node after add_nodes_and_edges (which may have modified it)
				Dictionary updated_node = solution_graph.get_node(p_curr_node_id);
				TypedArray<int> new_successors = updated_node["successors"];
				if (new_successors.size() > 0) {
					p_stack.push_back({ new_successors[0], p_state, p_iter + 1 });
					return true;
				}
				p_stack.push_back({ p_parent_node_id, p_state, p_iter + 1 });
				return true;
			}

			if (verbose >= 2) {
				print_line("MultiGoal refinement failed, backtracking");
			}
			_bump_conflict_path_activities(p_curr_node_id);
			if (multigoal_variant.get_type() == Variant::ARRAY) {
				_blacklist_command(multigoal_variant);
				if (verbose >= 2) {
					print_line("Blacklisted multigoal info since all methods failed");
				}
			}
			if (p_parent_node_id >= 0) {
				// Note: created_subtasks is not in PlannerNodeStruct, so we use Dictionary
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
					solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands, verbose);
			solution_graph = backtrack_result.graph;
			if (backtrack_result.parent_node_id >= 0) {
				_restore_stn_from_node(backtrack_result.parent_node_id);
				p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
				return true;
			}
			p_final_state = p_state;
			return false;
		}

		case PlannerNodeType::TYPE_VERIFY_GOAL: {
			// Optimized: Use internal structure to avoid Dictionary conversion
			const PlannerNodeStruct *parent_node = solution_graph.get_node_internal(p_parent_node_id);
			if (parent_node == nullptr) {
				p_final_state = p_state;
				return false;
			}
			Variant unigoal_variant = parent_node->info;
			Array unigoal_arr = unigoal_variant;
			if (unigoal_arr.size() >= 3) {
				String predicate = unigoal_arr[0];
				String subject = unigoal_arr[1];
				Variant value = unigoal_arr[2];

				if (p_state.has(predicate)) {
					Dictionary predicate_dict = p_state[predicate];
					if (predicate_dict.has(subject) && predicate_dict[subject] == value) {
						if (verbose >= 2) {
							print_line(vformat("Unigoal verified: %s[%s] == %s", predicate, subject, value));
						}
						p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
						solution_graph.update_node(p_curr_node_id, p_curr_node);
						p_stack.push_back({ p_parent_node_id, p_state, p_iter + 1 });
						return true;
					}
				}
			}

			if (verbose >= 2) {
				if (unigoal_arr.size() >= 3) {
					String predicate = unigoal_arr[0];
					String subject = unigoal_arr[1];
					Variant value = unigoal_arr[2];
					Variant current_value;
					if (p_state.has(predicate)) {
						Dictionary predicate_dict = p_state[predicate];
						if (predicate_dict.has(subject)) {
							current_value = predicate_dict[subject];
						}
					}
					print_line(vformat("Unigoal verification failed: %s[%s] = %s (need %s), re-refining parent unigoal",
							predicate, subject, current_value, value));
				} else {
					print_line("Unigoal verification failed, re-refining parent unigoal");
				}
			}

			// Get Dictionary version for update (since we need to modify status)
			Dictionary parent_node_dict = solution_graph.get_node(p_parent_node_id);
			parent_node_dict["status"] = static_cast<int>(PlannerNodeStatus::STATUS_OPEN);
			solution_graph.update_node(p_parent_node_id, parent_node_dict);

			p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
			solution_graph.update_node(p_curr_node_id, p_curr_node);

			p_stack.push_back({ p_parent_node_id, p_state, p_iter + 1 });
			return true;
		}

		case PlannerNodeType::TYPE_VERIFY_MULTIGOAL: {
			// Optimized: Use internal structure to avoid Dictionary conversion
			const PlannerNodeStruct *parent_node = solution_graph.get_node_internal(p_parent_node_id);
			if (parent_node == nullptr) {
				p_final_state = p_state;
				return false;
			}
			Variant multigoal_variant = parent_node->info;

			if (multigoal_variant.get_type() == Variant::DICTIONARY) {
				Dictionary dict = multigoal_variant;
				if (dict.has("item")) {
					multigoal_variant = dict["item"];
				}
			}

			if (!PlannerMultigoal::is_multigoal_array(multigoal_variant)) {
				if (verbose >= 2) {
					print_line("MultiGoal verification failed: invalid parent multigoal, backtracking");
				}
				PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
						solution_graph, p_parent_node_id, p_curr_node_id, p_state, blacklisted_commands);
				solution_graph = backtrack_result.graph;
				blacklisted_commands = backtrack_result.blacklisted_commands;
				if (backtrack_result.parent_node_id >= 0) {
					_restore_stn_from_node(backtrack_result.parent_node_id);
					p_stack.push_back({ backtrack_result.parent_node_id, backtrack_result.state, p_iter + 1 });
					return true;
				}
				p_final_state = p_state;
				return false;
			}
			Array multigoal = multigoal_variant;

			Array goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, multigoal);
			if (goals_not_achieved.is_empty()) {
				if (verbose >= 1) {
					print_line("MultiGoal verified successfully");
				}
				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);
				p_stack.push_back({ p_parent_node_id, p_state, p_iter + 1 });
				return true;
			} else {
				if (verbose >= 2) {
					print_line(vformat("MultiGoal verification failed: %d goals not achieved, re-refining parent multigoal", goals_not_achieved.size()));
				}

				// Get Dictionary version for update (since we need to modify status)
				Dictionary parent_node_dict = solution_graph.get_node(p_parent_node_id);
				parent_node_dict["status"] = static_cast<int>(PlannerNodeStatus::STATUS_OPEN);
				solution_graph.update_node(p_parent_node_id, parent_node_dict);

				p_curr_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_FAILED);
				solution_graph.update_node(p_curr_node_id, p_curr_node);

				p_stack.push_back({ p_parent_node_id, p_state, p_iter + 1 });
				return true;
			}
		}

		default:
			p_final_state = p_state;
			return false;
	}
}

void PlannerPlan::_restore_stn_from_node(int p_node_id) {
	// Guard: node_id must be valid
	if (p_node_id < 0) {
		return;
	}

	Dictionary node = solution_graph.get_node(p_node_id);

	// Guard: node must have stn_snapshot
	if (!node.has("stn_snapshot") || node["stn_snapshot"].get_type() == Variant::NIL) {
		return; // No snapshot exists, keep current STN state
	}

	Dictionary snapshot_dict = node["stn_snapshot"];

	// Guard: snapshot_dict must not be empty
	if (snapshot_dict.is_empty()) {
		return; // Empty snapshot, keep current STN state
	}

	PlannerSTNSolver::Snapshot snapshot = PlannerSTNSolver::Snapshot::from_dictionary(snapshot_dict);
	stn.restore_snapshot(snapshot);
	if (verbose >= 3) {
		print_line("Restored STN snapshot from node " + itos(p_node_id));
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
			Variant action_elem = action_arr[j];
			Variant blacklisted_elem = blacklisted_arr[j];
			// For nested arrays, we need to compare element by element
			if (action_elem.get_type() == Variant::ARRAY && blacklisted_elem.get_type() == Variant::ARRAY) {
				Array action_elem_arr = action_elem;
				Array blacklisted_elem_arr = blacklisted_elem;
				if (action_elem_arr.size() != blacklisted_elem_arr.size()) {
					match = false;
					break;
				}
				for (int k = 0; k < action_elem_arr.size(); k++) {
					if (action_elem_arr[k] != blacklisted_elem_arr[k]) {
						match = false;
						break;
					}
				}
				if (!match) {
					break;
				}
			} else if (action_elem != blacklisted_elem) {
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

void PlannerPlan::_blacklist_command(Variant p_command) {
	if (!_is_command_blacklisted(p_command)) {
		blacklisted_commands.push_back(p_command);
	}
}

void PlannerPlan::blacklist_command(Variant p_command) {
	_blacklist_command(p_command);
}

Array PlannerPlan::simulate(Ref<PlannerResult> p_result, Dictionary p_state, int p_start_ind) {
	Array state_list;
	state_list.push_back(p_state.duplicate()); // Initial state

	// Guard: result must be valid
	if (!p_result.is_valid()) {
		if (verbose >= 1) {
			ERR_PRINT("PlannerPlan::simulate: p_result is invalid");
		}
		return state_list;
	}

	// Guard: domain must be set
	if (!current_domain.is_valid()) {
		if (verbose >= 1) {
			ERR_PRINT("PlannerPlan::simulate: current_domain is not set");
		}
		return state_list;
	}

	// Load the solution graph from the result
	load_solution_graph(p_result->get_solution_graph());

	// Extract plan from solution graph
	Array plan = PlannerGraphOperations::extract_solution_plan(solution_graph, verbose);

	// Guard: start_ind must be valid
	if (p_start_ind < 0 || p_start_ind >= plan.size()) {
		if (verbose >= 1) {
			ERR_PRINT(vformat("PlannerPlan::simulate: start_ind %d is out of range (plan size: %d)", p_start_ind, plan.size()));
		}
		return state_list;
	}

	Dictionary state_copy = p_state.duplicate();
	Dictionary command_dict = current_domain->command_dictionary;

	// Execute actions starting from p_start_ind
	for (int i = p_start_ind; i < plan.size(); i++) {
		Variant action_variant = plan[i];
		Array action;

		// Unwrap if dictionary-wrapped
		if (action_variant.get_type() == Variant::DICTIONARY) {
			Dictionary dict = action_variant;
			if (dict.has("item")) {
				action_variant = dict["item"];
			}
		}

		// Guard: action must be an Array
		if (action_variant.get_type() != Variant::ARRAY) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("PlannerPlan::simulate: action at index %d is not an Array", i));
			}
			break;
		}

		action = action_variant;

		// Guard: action must not be empty
		if (action.is_empty() || action.size() < 1) {
			continue;
		}

		String command_name = action[0];

		// Guard: action must exist in domain
		if (!command_dict.has(command_name)) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("PlannerPlan::simulate: action '%s' not found in domain", command_name));
			}
			break;
		}

		Callable command_callable = command_dict[command_name];
		Array args;
		args.push_back(state_copy);
		args.append_array(action.slice(1, action.size()));

		Variant result = command_callable.callv(args);

		// Guard: action must return a Dictionary (success)
		if (result.get_type() != Variant::DICTIONARY) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("PlannerPlan::simulate: action '%s' failed at index %d", command_name, i));
			}
			break;
		}

		state_copy = result;
		state_list.push_back(state_copy.duplicate());
	}

	return state_list;
}

int PlannerPlan::_post_failure_modify(int p_fail_node_id, Dictionary p_state) {
	// Mark nodes from root to failure point as OPEN and "old"
	// Remove descendants of reopened nodes
	// Similar to IPyHOP's _post_failure_modify

	// Get all nodes in reverse DFS order (from root)
	Array all_nodes;
	Array to_visit;
	to_visit.push_back(0); // Start from root
	TypedArray<int> visited;

	// Collect all nodes in DFS preorder
	while (!to_visit.is_empty()) {
		int node_id = to_visit.pop_back();
		if (visited.has(node_id)) {
			continue;
		}
		visited.push_back(node_id);
		all_nodes.push_back(node_id);

		Dictionary node = solution_graph.get_node(node_id);
		if (!node.has("successors")) {
			continue; // Skip nodes without successors
		}
		TypedArray<int> successors = node["successors"];
		// Add successors in reverse order
		for (int i = successors.size() - 1; i >= 0; i--) {
			int succ_id = successors[i];
			if (!visited.has(succ_id) && solution_graph.get_graph().has(succ_id)) {
				to_visit.push_back(succ_id);
			}
		}
	}

	// Process nodes in reverse order (from leaves to root)
	int max_id = -1;
	for (int i = all_nodes.size() - 1; i >= 0; i--) {
		int node_id = all_nodes[i];
		if (node_id > max_id) {
			max_id = node_id;
		}

		Dictionary node = solution_graph.get_node(node_id);
		if (!node.has("type") || !node.has("status")) {
			if (verbose >= 1) {
				ERR_PRINT(vformat("PlannerPlan::_post_failure_modify: Node %d missing 'type' or 'status' field", node_id));
			}
			continue; // Skip malformed nodes
		}
		int node_type = node["type"];
		int node_status = node["status"];

		// Set all nodes to OPEN
		solution_graph.set_node_status(node_id, PlannerNodeStatus::STATUS_OPEN);

		// Stop when we reach the failure node
		if (node_id == p_fail_node_id) {
			break;
		}

		// For T/G/M nodes, clear state, selected_method, and reset available_methods
		if (node_type == static_cast<int>(PlannerNodeType::TYPE_TASK) ||
				node_type == static_cast<int>(PlannerNodeType::TYPE_UNIGOAL) ||
				node_type == static_cast<int>(PlannerNodeType::TYPE_MULTIGOAL)) {
			node["state"] = Dictionary();
			node["selected_method"] = Variant();
			// Reset available_methods iterator (keep the methods list)
			if (node.has("available_methods")) {
				TypedArray<Callable> methods = node["available_methods"];
				node["available_methods"] = methods; // Reset to full list
			}
			solution_graph.update_node(node_id, node);

			// Remove descendants
			PlannerGraphOperations::remove_descendants(solution_graph, node_id, false);
		}

		// Mark CLOSED nodes as "old"
		if (node_status == static_cast<int>(PlannerNodeStatus::STATUS_CLOSED)) {
			solution_graph.set_node_tag(node_id, "old");
			// Update state snapshot for CLOSED nodes
			node["state"] = p_state.duplicate();
			solution_graph.update_node(node_id, node);
		} else {
			// Clear state for non-CLOSED nodes
			node["state"] = Dictionary();
			solution_graph.update_node(node_id, node);
		}
	}

	return max_id + 1; // Return next available node ID
}

Ref<PlannerResult> PlannerPlan::replan(Ref<PlannerResult> p_result, Dictionary p_state, int p_fail_node_id) {
	if (!p_result.is_valid()) {
		if (verbose >= 1) {
			ERR_PRINT("PlannerPlan::replan: p_result is invalid");
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		result->set_solution_graph(Dictionary());
		return result;
	}

	if (!current_domain.is_valid()) {
		if (verbose >= 1) {
			ERR_PRINT("PlannerPlan::replan: current_domain is not set");
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		result->set_solution_graph(p_result->get_solution_graph());
		return result;
	}

	// Load the solution graph from the result
	load_solution_graph(p_result->get_solution_graph());

	if (!solution_graph.get_graph().has(p_fail_node_id)) {
		if (verbose >= 1) {
			ERR_PRINT(vformat("PlannerPlan::replan: fail_node_id %d not found in solution graph", p_fail_node_id));
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		// Optimized: Cache get_graph() result
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	if (verbose >= 1) {
		print_line(vformat("replan: Starting replanning from failure node %d", p_fail_node_id));
	}

	// Reset iteration counter
	iterations = 0;

	// Modify graph: mark nodes as OPEN/old, remove descendants
	int max_id = _post_failure_modify(p_fail_node_id, p_state);

	// Find parent of failure node
	int parent_id = PlannerGraphOperations::find_predecessor(solution_graph, p_fail_node_id);
	if (parent_id < 0) {
		if (verbose >= 1) {
			ERR_PRINT(vformat("PlannerPlan::replan: Could not find parent of fail_node_id %d", p_fail_node_id));
		}
		Ref<PlannerResult> result = memnew(PlannerResult);
		result->set_success(false);
		result->set_final_state(p_state);
		// Optimized: Cache get_graph() result
		const Dictionary &graph_dict = solution_graph.get_graph();
		result->set_solution_graph(graph_dict);
		return result;
	}

	// Backtrack from failure point
	PlannerBacktracking::BacktrackResult backtrack_result = PlannerBacktracking::backtrack(
			solution_graph, parent_id, p_fail_node_id, p_state, blacklisted_commands, verbose);
	solution_graph = backtrack_result.graph;
	blacklisted_commands = backtrack_result.blacklisted_commands;

	// Update next_node_id to max_id
	solution_graph.set_next_node_id(max_id);

	// Continue planning from backtrack point
	Dictionary final_state = _planning_loop_iterative(backtrack_result.parent_node_id, backtrack_result.state, 0);

	// Extract only "new" actions
	Array new_plan = PlannerGraphOperations::extract_new_commands(solution_graph);

	// Create result
	Ref<PlannerResult> result = memnew(PlannerResult);
	result->set_final_state(final_state);
	result->set_solution_graph(solution_graph.get_graph());
	result->set_success(!final_state.is_empty() && new_plan.size() > 0);

	if (verbose >= 1) {
		print_line(vformat("replan: Completed, extracted %d new actions", new_plan.size()));
	}

	return result;
}

void PlannerPlan::load_solution_graph(Dictionary p_graph) {
	solution_graph = PlannerSolutionGraph();

	// Load graph from Dictionary - convert to internal structure
	solution_graph.load_from_dictionary(p_graph);
}

PlannerMetadata PlannerPlan::_extract_metadata(const Variant &p_item) const {
	PlannerMetadata extracted_metadata;

	// Check if item has temporal_constraints field
	if (p_item.get_type() == Variant::DICTIONARY) {
		Dictionary item_dict = p_item;
		const Variant *temporal_constraints_var = item_dict.getptr("temporal_constraints");
		if (temporal_constraints_var && temporal_constraints_var->get_type() == Variant::DICTIONARY) {
			Dictionary constraints_dict = *temporal_constraints_var;
			extracted_metadata = PlannerMetadata::from_dictionary(constraints_dict);
		}
		// Also check for entity requirements in constraints field (for combined format)
		const Variant *constraints_var = item_dict.getptr("constraints");
		if (constraints_var && constraints_var->get_type() == Variant::DICTIONARY) {
			Dictionary constraints_dict = *constraints_var;
			if (constraints_dict.has("requires_entities")) {
				Variant entities_var = constraints_dict.get("requires_entities", Array());
				if (entities_var.get_type() == Variant::ARRAY) {
					Array entities_array = entities_var;
					extracted_metadata.requires_entities.resize(entities_array.size());
					for (int i = 0; i < entities_array.size(); i++) {
						Dictionary entity_dict = entities_array[i];
						extracted_metadata.requires_entities[i] = PlannerEntityRequirement::from_dictionary(entity_dict);
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
					extracted_metadata = PlannerMetadata::from_dictionary(constraints_dict);
				}
			}
		}
	}

	return extracted_metadata;
}

Variant PlannerPlan::_attach_temporal_constraints(const Variant &p_item, const Dictionary &p_temporal_constraints) {
	PlannerMetadata temporal_metadata_obj = PlannerMetadata::from_dictionary(p_temporal_constraints);

	// Create a wrapper dictionary with the item and temporal_constraints
	Dictionary result;
	Dictionary constraints_dict = temporal_metadata_obj.to_dictionary();

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
	// Extract only temporal constraints (excludes entity requirements)
	PlannerMetadata extracted_metadata = _extract_metadata(p_item);
	extracted_metadata.requires_entities.clear();
	return extracted_metadata.to_dictionary();
}

bool PlannerPlan::_has_temporal_constraints(const Variant &p_item) const {
	// Extract only temporal constraints (excludes entity requirements)
	PlannerMetadata extracted_metadata = _extract_metadata(p_item);
	extracted_metadata.requires_entities.clear();
	return extracted_metadata.has_temporal();
}

bool PlannerPlan::_todo_list_has_temporal_constraints(const Array &p_todo_list) const {
	// Check if any element in the todo_list has temporal constraints
	for (int i = 0; i < p_todo_list.size(); i++) {
		Variant item = p_todo_list[i];
		if (_has_temporal_constraints(item)) {
			return true;
		}
		// Also check if it's a multigoal array (array of arrays)
		if (item.get_type() == Variant::ARRAY) {
			Array item_arr = item;
			for (int j = 0; j < item_arr.size(); j++) {
				Variant sub_item = item_arr[j];
				if (_has_temporal_constraints(sub_item)) {
					return true;
				}
			}
		}
	}
	return false;
}

// Belief-immersed architecture integration
Dictionary PlannerPlan::_get_ego_centric_state(const Dictionary &p_state) const {
	Dictionary ego_state = p_state.duplicate(true);

	// If we have a current persona, merge their beliefs into the state
	if (current_persona.is_valid()) {
		Dictionary beliefs = current_persona->get_beliefs_about_others();

		// Merge beliefs into state (persona's perspective)
		// This allows planning from the persona's ego-centric view
		Array belief_keys = beliefs.keys();
		for (int i = 0; i < belief_keys.size(); i++) {
			String target_persona_id = belief_keys[i];
			Dictionary target_beliefs = beliefs[target_persona_id];

			// Add beliefs as state predicates (persona's beliefs about others)
			// Format: "belief_{target_persona_id}_{belief_key}" = belief_value
			Array target_belief_keys = target_beliefs.keys();
			for (int j = 0; j < target_belief_keys.size(); j++) {
				String belief_key = target_belief_keys[j];
				Variant belief_value = target_beliefs[belief_key];
				String state_key = vformat("belief_%s_%s", target_persona_id, belief_key);
				ego_state[state_key] = belief_value;
			}
		}
	}

	return ego_state;
}

Dictionary PlannerPlan::_merge_allocentric_facts(const Dictionary &p_state) const {
	return p_state; // TODO: merge allocentric triples from belief_manager
}

void PlannerPlan::_update_beliefs_from_action(const Variant &p_action, const Dictionary &p_state_before, const Dictionary &p_state_after) {
	// Guard: persona and belief_manager must be valid
	if (!current_persona.is_valid() || !belief_manager.is_valid()) {
		return;
	}

	// Guard: action must be an array with at least one element
	Array action_arr = p_action;
	if (action_arr.size() < 1) {
		return;
	}

	String command_name = action_arr[0];

	// Create observation dictionary from action execution
	Dictionary observation;
	observation["action"] = command_name;
	observation["state_before"] = p_state_before;
	observation["state_after"] = p_state_after;
	observation["time"] = PlannerTimeRange::now_microseconds();
	observation["confidence"] = 1.0; // Direct observation has full confidence

	// Process observation through belief manager
	String persona_id = current_persona->get_persona_id();
	belief_manager->process_observation_for_persona(persona_id, observation);
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
	if (p_entity_constraints.is_empty()) {
		return result;
	}

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

	return result;
}

bool PlannerPlan::_validate_entity_requirements(const Dictionary &p_state, const PlannerMetadata &p_metadata) const {
	// Guard: No entity requirements means validation passes
	if (p_metadata.requires_entities.size() == 0) {
		return true;
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

			// Guard: Type must match
			if (entity_type != req.type) {
				continue;
			}

			// Guard: Entity must have capabilities
			const LocalVector<String> *entity_caps = entity_capabilities.getptr(entity_id);
			if (entity_caps == nullptr) {
				continue;
			}

			// Check if entity has all required capabilities
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

		// Guard: Each requirement must be matched
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

// GDScript-compatible wrappers for time range
Dictionary PlannerPlan::get_time_range_dict() const {
	Dictionary result;
	result["start_time"] = time_range.get_start_time();
	result["end_time"] = time_range.get_end_time();
	result["duration"] = time_range.get_duration();
	return result;
}

void PlannerPlan::set_time_range_dict(const Dictionary &p_time_range) {
	const Variant *start_time_var = p_time_range.getptr("start_time");
	const Variant *end_time_var = p_time_range.getptr("end_time");
	const Variant *duration_var = p_time_range.getptr("duration");

	if (start_time_var) {
		time_range.set_start_time(*start_time_var);
	}
	if (end_time_var) {
		time_range.set_end_time(*end_time_var);
	}
	if (duration_var) {
		time_range.set_duration(*duration_var);
	}
}
