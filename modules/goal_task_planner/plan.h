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
#include "core/variant/typed_array.h"

#include "modules/goal_task_planner/multigoal.h"
#include "modules/goal_task_planner/planner_metadata.h"
#include "modules/goal_task_planner/planner_result.h"
#include "modules/goal_task_planner/planner_time_range.h"
#include "modules/goal_task_planner/solution_graph.h"
#include "modules/goal_task_planner/stn_solver.h"

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

	int max_depth = 10; // Maximum recursion depth to prevent infinite loops
	static String _item_to_string(Variant p_item);
	Variant _apply_task_and_continue(Dictionary p_state, Callable p_command, Array p_arguments);
	// Graph-based planning methods
	Dictionary _planning_loop_recursive(int p_parent_node_id, Dictionary p_state, int p_iter);
	bool _is_command_blacklisted(Variant p_command) const;
	bool _contains_blacklisted_action(Array p_subtasks) const;
	void _blacklist_command(Variant p_command);
	void _restore_stn_from_node(int p_node_id);

	PlannerMetadata _extract_temporal_constraints(const Variant &p_item) const;
	PlannerMetadata _extract_metadata(const Variant &p_item) const; // Extract full PlannerMetadata (temporal + entity requirements)

	// Entity matching helper (used during planning when PlannerMetadata has entity requirements)
	Dictionary _match_entities(const Dictionary &p_state, const LocalVector<PlannerEntityRequirement> &p_requirements) const;
	bool _validate_entity_requirements(const Dictionary &p_state, const PlannerMetadata &p_metadata) const;

public:
	// Temporal constraint methods (public for testing)
	Variant _attach_temporal_constraints(const Variant &p_item, const Dictionary &p_temporal_constraints);
	Dictionary _get_temporal_constraints(const Variant &p_item) const;
	bool _has_temporal_constraints(const Variant &p_item) const;

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
	Ref<PlannerResult> find_plan(Dictionary p_state, Array p_todo_list);
	Ref<PlannerResult> run_lazy_lookahead(Dictionary p_state, Array p_todo_list, int p_max_tries = 10);
	// Graph-based lazy refinement (Elixir-style)
	Ref<PlannerResult> run_lazy_refineahead(Dictionary p_state, Array p_todo_list);
	// Temporal methods
	PlannerTimeRange get_time_range() const { return time_range; }
	void set_time_range(PlannerTimeRange p_time_range) { time_range = p_time_range; }

protected:
	static void _bind_methods();
};
