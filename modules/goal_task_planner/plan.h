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
#include "modules/goal_task_planner/planner_time_range.h"
#include "modules/goal_task_planner/solution_graph.h"
#include "modules/goal_task_planner/stn_solver.h"

class PlannerDomain;
struct PlannerTimeRange;

class PlannerPlan : public Resource {
	GDCLASS(PlannerPlan, Resource);

	int verbose = 0;
	TypedArray<PlannerDomain> domains;
	Ref<PlannerDomain> current_domain;
	PlannerTimeRange time_range; // Added for temporal
	PlannerSolutionGraph solution_graph; // Solution graph for explicit backtracking
	TypedArray<Variant> blacklisted_commands; // Blacklisted commands/actions
	PlannerSTNSolver stn; // STN solver for temporal constraint validation
	PlannerSTNSolver::Snapshot stn_snapshot; // STN snapshot for backtracking

	// If verify_goals is True, then whenever the planner uses a method m to refine
	// unigoal or multigoal, it will insert a "verification" task into the
	// current partial plan. If verify_goals is False, the planner won't insert any
	// verification tasks into the plan.
	//
	// The purpose of the verification task is to raise an exception if the
	// refinement produced by m doesn't achieve the goal or multigoal that it is
	// supposed to achieve. The verification task won't insert anything into the
	// final plan; it just will verify whether m did what it was supposed to do.
	bool verify_goals = true;
	int max_depth = 10; // Maximum recursion depth to prevent infinite loops
	static String _item_to_string(Variant p_item);
	Variant _apply_task_and_continue(Dictionary p_state, Callable p_command, Array p_arguments);
	// Graph-based planning methods
	Dictionary _planning_loop_recursive(int p_parent_node_id, Dictionary p_state, int p_iter);
	bool _is_command_blacklisted(Variant p_command) const;
	void _blacklist_command(Variant p_command);
	void _restore_stn_from_node(int p_node_id);

	// Goal solver methods (moved from PlannerGoalSolver)
	// Constraining factor for a goal/task - two optimization strategies:
	// 1. Method count: fewer total methods = more constraining
	// 2. Applicable method count: fewer applicable methods in current state = more constraining
	struct ConstrainingFactor {
		int total_method_count; // Total methods available for this unigoal
		int applicable_method_count; // Methods actually applicable in current state
		bool has_temporal_constraints;

		ConstrainingFactor() :
				total_method_count(0), applicable_method_count(0), has_temporal_constraints(false) {}
		ConstrainingFactor(int p_total, int p_applicable, bool p_temporal) :
				total_method_count(p_total), applicable_method_count(p_applicable), has_temporal_constraints(p_temporal) {}

		// Compare: more constraining = fewer applicable methods, or has temporal constraints
		// Use applicable_method_count as primary factor (more accurate optimization)
		bool operator<(const ConstrainingFactor &p_other) const {
			if (has_temporal_constraints != p_other.has_temporal_constraints) {
				return has_temporal_constraints; // Temporal constraints make it more constraining
			}
			return applicable_method_count < p_other.applicable_method_count; // Fewer applicable methods = more constraining
		}
	};

	// Internal storage for goal ordering
	struct GoalWithFactor {
		Variant goal;
		ConstrainingFactor factor;

		GoalWithFactor() :
				goal(), factor() {}
		GoalWithFactor(const Variant &p_goal, const ConstrainingFactor &p_factor) :
				goal(p_goal), factor(p_factor) {}
	};

	ConstrainingFactor _calculate_constraining_factor(const Variant &p_goal, const Dictionary &p_state, const Dictionary &p_unigoal_method_dict) const;
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
	Array _optimize_unigoal_order(const Array &p_unigoals, const Dictionary &p_state, const Dictionary &p_unigoal_method_dict);
	int get_verbose() const;
	void set_verbose(int p_level);
	TypedArray<PlannerDomain> get_domains() const;
	void set_domains(TypedArray<PlannerDomain> p_domain);
	Ref<PlannerDomain> get_current_domain() const;
	void set_current_domain(Ref<PlannerDomain> p_current_domain) { current_domain = p_current_domain; }
	void set_verify_goals(bool p_value);
	bool get_verify_goals() const;
	void set_max_depth(int p_max_depth);
	int get_max_depth() const;
	Variant find_plan(Dictionary p_state, Array p_todo_list);
	Dictionary run_lazy_lookahead(Dictionary p_state, Array p_todo_list, int p_max_tries = 10);
	// Graph-based lazy refinement (Elixir-style)
	Dictionary run_lazy_refineahead(Dictionary p_state, Array p_todo_list);
	// Temporal methods
	String generate_plan_id();
	PlannerTimeRange get_time_range() const { return time_range; }
	void set_time_range(PlannerTimeRange p_time_range) { time_range = p_time_range; }
	Dictionary submit_operation(Dictionary p_operation);
	Dictionary get_global_state();

protected:
	static void _bind_methods();
};
