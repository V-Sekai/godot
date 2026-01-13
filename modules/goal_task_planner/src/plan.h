/**************************************************************************/
/*  plan.h                                                                */
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

#pragma once

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

#include "core/io/resource.h"
#include "core/templates/hash_map.h"
#include "core/variant/typed_array.h"

#include "planner_belief_manager.h"
#include "planner_metadata.h"
#include "planner_persona.h"
#include "planner_result.h"
#include "planner_time_range.h"
#include "solution_graph.h"
#include "stn_solver.h"

class PlannerDomain;
struct PlannerTimeRange;

class PlannerPlan : public Resource {
	GDCLASS(PlannerPlan, Resource);

	int verbose = 0;
	Ref<PlannerDomain> current_domain;
	PlannerTimeRange time_range; // Added for temporal
	PlannerSolutionGraph solution_graph; // Solution graph for explicit backtracking
	TypedArray<Variant> blacklisted_commands; // Blacklisted commands/actions
	PlannerSTNSolver stn; // STN solver for temporal constraint validation
	PlannerSTNSolver::Snapshot stn_snapshot; // STN snapshot for backtracking
	Array original_todo_list; // Store original todo_list to check if all tasks completed

	// Belief-immersed architecture support
	Ref<PlannerPersona> current_persona; // Current persona for ego-centric planning
	Ref<PlannerBeliefManager> belief_manager; // Belief manager for multi-persona interactions

	int max_depth = 10; // Maximum recursion depth to prevent infinite loops
	int max_iterations = 50000; // Maximum planning loop iterations (safety limit to prevent infinite loops)
	int max_stack_size = 10000; // Maximum stack size to prevent excessive memory usage
	int iterations = 0; // Track number of planning iterations

	// VSIDS-style method activity tracking (following Chuffed's proven approach)
	// Optimized: Use HashMap instead of Dictionary for faster lookups
	HashMap<String, double> method_activities; // Track activity scores: method_id -> double
	double activity_var_inc = 1.0; // Increment value (grows over time via activity inflation)
	int activity_bump_count = 0; // Track bumps to trigger decay
	static const int ACTIVITY_DECAY_INTERVAL = 100; // Decay every N bumps
	inline static constexpr double ACTIVITY_INFLATION_FACTOR = 1.05; // Increase increment by 5% (matches Chuffed)
	inline static constexpr double ACTIVITY_OVERFLOW_THRESHOLD = 1e100; // Threshold for activity overflow normalization
	inline static constexpr double ACTIVITY_NORMALIZATION_FACTOR = 1e-100; // Factor to normalize activities when overflow occurs
	inline static constexpr double ACTIVITY_SCORE_MULTIPLIER = 10.0; // Multiplier for activity in method scoring
	inline static constexpr double SUBTASK_BONUS_BASE = 10.0; // Base value for subtask count bonus in method scoring
	inline static constexpr double REWARD_BASE = 10000.0; // Base reward value for successful method rewards
	inline static constexpr double INITIAL_SCORE = -1e100; // Very negative initial score for method selection
	static const int ITERATIONS_PER_DEPTH = 1000; // Multiplier for max_depth in iteration limit calculation
	static const int MIN_UNIGOAL_SIZE = 3; // Minimum number of elements in unigoal array
	// Note: No activity_decay_factor - we use activity inflation (var_inc *= ACTIVITY_INFLATION_FACTOR) instead
	TypedArray<String> rewarded_methods_this_solve; // Track which methods already rewarded this solve

	static String _item_to_string(Variant p_item);

	// VSIDS activity management
	String _method_to_id(Callable p_method) const;
	double _get_method_activity(Callable p_method) const;
	void _bump_method_activity(Callable p_method);
	void _decay_method_activities();
	void _bump_conflict_path_activities(int p_fail_node_id);
	void _reward_successful_methods(int p_plan_length);
	void _reward_method_immediate(Callable p_method, int p_current_action_count);
	int _count_closed_actions(); // Count closed action nodes in solution graph

	// Method selection with activity scoring
	struct MethodCandidate {
		Callable method;
		Array subtasks;
		double score;
		String method_id; // For explanation/debugging
		double activity; // For explanation/debugging
		String reason; // Why this method was chosen/rejected
	};
	MethodCandidate _select_best_method(TypedArray<Callable> p_methods, Dictionary p_state, Variant p_node_info, Variant p_args, int p_node_type, bool p_track_alternatives = false, Array *p_alternatives = nullptr);
	// Graph-based planning methods
	Dictionary _planning_loop_iterative(int p_parent_node_id, Dictionary p_state, int p_iter);
	// Helper for iterative planning: processes a single node and pushes frames to stack or sets final_state
	// Returns true to continue loop, false if final_state is set
	// PlanningFrame is defined inside _planning_loop_iterative, so we use a forward declaration approach
	struct PlanningFrame {
		int parent_node_id;
		Dictionary state;
		int iter;
	};
	// Note: p_curr_node is passed by value (copy) to avoid stale reference issues after graph modifications
	bool _process_node_iterative(int p_parent_node_id, int p_curr_node_id, Dictionary p_curr_node, int p_node_type, Dictionary &p_state, int p_iter, LocalVector<PlanningFrame> &p_stack, Dictionary &p_final_state);
	bool _is_command_blacklisted(Variant p_command) const;
	void _blacklist_command(Variant p_command);
	void _restore_stn_from_node(int p_node_id);
	int _post_failure_modify(int p_fail_node_id, Dictionary p_state);

	PlannerMetadata _extract_metadata(const Variant &p_item) const; // Extract full PlannerMetadata (temporal + entity requirements)

	// Entity matching helper (used during planning when PlannerMetadata has entity requirements)
	Dictionary _match_entities(const Dictionary &p_state, const LocalVector<PlannerEntityRequirement> &p_requirements) const;
	bool _validate_entity_requirements(const Dictionary &p_state, const PlannerMetadata &p_metadata) const;

	// Temporal constraint methods (internal use only - use attach_metadata() as public API)
	Variant _attach_temporal_constraints(const Variant &p_item, const Dictionary &p_temporal_constraints);
	Dictionary _get_temporal_constraints(const Variant &p_item) const;
	bool _has_temporal_constraints(const Variant &p_item) const;
	// Check if any element in todo_list has temporal constraints (for lazy STN initialization)
	bool _todo_list_has_temporal_constraints(const Array &p_todo_list) const;

	// Belief-immersed architecture helpers
	Dictionary _get_ego_centric_state(const Dictionary &p_state) const;
	Dictionary _merge_allocentric_facts(const Dictionary &p_state) const;
	void _update_beliefs_from_action(const Variant &p_action, const Dictionary &p_state_before, const Dictionary &p_state_after);

public:
	// Constructor: Initialize all state to defaults
	PlannerPlan();
	// Destructor: Clean up all state on object destruction (C++ needs explicit cleanup, unlike functional Elixir)
	~PlannerPlan();

	// Unified metadata attachment method (public API)
	// Attach temporal and/or entity constraints to any planner element (action, task, goal, multigoal)
	// p_temporal: Dictionary with optional keys: "duration", "start_time", "end_time" (all int64_t in microseconds)
	// p_entity: Dictionary with either:
	//   - {"type": String, "capabilities": Array} (convenience format)
	//   - {"requires_entities": Array} (full format with PlannerEntityRequirement dictionaries)
	Variant attach_metadata(const Variant &p_item, const Dictionary &p_temporal_constraints = Dictionary(), const Dictionary &p_entity_constraints = Dictionary());
	int get_verbose() const;
	void set_verbose(int p_level);
	Ref<PlannerDomain> get_current_domain() const;
	void set_current_domain(Ref<PlannerDomain> p_current_domain) { current_domain = p_current_domain; }
	void set_max_depth(int p_max_depth);
	int get_max_depth() const;
	void set_max_iterations(int p_max_iterations);
	int get_max_iterations() const;
	void set_max_stack_size(int p_max_stack_size);
	int get_max_stack_size() const;
	Ref<PlannerResult> find_plan(Dictionary p_state, Array p_todo_list);
	Ref<PlannerResult> run_lazy_lookahead(Dictionary p_state, Array p_todo_list, int p_max_tries = 10);
	// Graph-based lazy refinement (Elixir-style)
	Ref<PlannerResult> run_lazy_refineahead(Dictionary p_state, Array p_todo_list);
	// Temporal methods
	PlannerTimeRange get_time_range() const { return time_range; }
	void set_time_range(PlannerTimeRange p_time_range) { time_range = p_time_range; }
	// GDScript-compatible wrappers for time range (structs not directly usable in GDScript)
	Dictionary get_time_range_dict() const;
	void set_time_range_dict(const Dictionary &p_time_range);

	// Public API methods
	void blacklist_command(Variant p_command);
	int get_iterations() const { return iterations; }
	Dictionary get_method_activities() const; // Get VSIDS activity scores for testing
	void reset_vsids_activity(); // Reset VSIDS activity tracking (clears all activity scores)
	void reset(); // Reset all planner state for complete test isolation
	Array simulate(Ref<PlannerResult> p_result, Dictionary p_state, int p_start_ind = 0);
	Ref<PlannerResult> replan(Ref<PlannerResult> p_result, Dictionary p_state, int p_fail_node_id);
	void load_solution_graph(Dictionary p_graph);

	// Belief-immersed architecture: Persona and belief management
	Ref<PlannerPersona> get_current_persona() const { return current_persona; }
	void set_current_persona(Ref<PlannerPersona> p_persona) { current_persona = p_persona; }
	Ref<PlannerBeliefManager> get_belief_manager() const { return belief_manager; }
	void set_belief_manager(Ref<PlannerBeliefManager> p_manager) { belief_manager = p_manager; }

protected:
	static void _bind_methods();
};
