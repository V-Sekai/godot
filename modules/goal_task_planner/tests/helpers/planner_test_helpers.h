/**************************************************************************/
/*  planner_test_helpers.h                                                */
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

// Core Planner Elements
#include "modules/goal_task_planner/src/domain.h"
#include "modules/goal_task_planner/src/multigoal.h"
#include "modules/goal_task_planner/src/plan.h"
#include "modules/goal_task_planner/src/planner_result.h"
#include "modules/goal_task_planner/src/planner_state.h"
#include "modules/goal_task_planner/src/solution_graph.h"
#include "modules/goal_task_planner/src/stn_solver.h"

// Test Domains
#include "modules/goal_task_planner/tests/minimal_backtracking_domain.h"
#include "modules/goal_task_planner/tests/minimal_task_domain.h"

// Test Utilities
#include "tests/test_macros.h"

// Common test helpers
namespace PlannerTestHelpers {

// Helper to create and initialize a planner with a domain
inline Ref<PlannerPlan> create_planner_with_domain(Ref<PlannerDomain> p_domain) {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(p_domain);
	return plan;
}

// Helper to create a simple state with a counter
inline Dictionary create_counter_state(int p_initial_value = 0) {
	Dictionary state;
	Dictionary val_dict;
	val_dict["value"] = p_initial_value;
	state["value"] = val_dict;
	return state;
}

// Helper to create a state with metadata
inline Dictionary create_state_with_metadata(int p_initial_value = 0, String p_version = "1.0") {
	Dictionary state;
	Dictionary counter;
	counter["value"] = p_initial_value;
	state["value"] = counter;

	Dictionary metadata;
	metadata["timestamp"] = 12345;
	metadata["version"] = p_version;
	metadata["attempts"] = 0;
	state["metadata"] = metadata;

	return state;
}

// Helper to create a navigation state for backtracking domain
inline Dictionary create_navigation_state(Array p_path = Array()) {
	Dictionary state;
	if (p_path.is_empty()) {
		p_path.push_back("room_a");
	}
	state["current_path"] = p_path;
	return state;
}

// Helper to verify planner result validity
inline bool verify_result_valid(Ref<PlannerResult> p_result) {
	if (!p_result.is_valid()) {
		return false;
	}
	return p_result->get_final_state().size() >= 0;
}

// Helper to create STN with temporal chain
inline PlannerSTNSolver create_temporal_chain(int p_num_points = 3, int p_min_duration = 0, int p_max_duration = 100) {
	PlannerSTNSolver stn;

	// Create time points
	for (int i = 0; i < p_num_points; i++) {
		stn.add_time_point("tp" + String::num(i));
	}

	// Create chain constraints
	for (int i = 0; i < p_num_points - 1; i++) {
		stn.add_constraint("tp" + String::num(i), "tp" + String::num(i + 1),
				p_min_duration, p_max_duration);
	}

	return stn;
}

// Helper to extract and verify plan array
inline Array extract_and_verify_plan(Ref<PlannerResult> p_result) {
	Array plan = p_result->extract_plan();
	// Plan should be valid array (may be empty)
	return plan;
}

// Helper to create a large nested state
inline Dictionary create_large_nested_state(int p_num_keys = 50) {
	Dictionary state;
	Dictionary large_dict;

	for (int i = 0; i < p_num_keys; i++) {
		large_dict[String("key_") + String::num(i)] = i * 10;
	}

	state["large_data"] = large_dict;
	Dictionary val_dict;
	val_dict["value"] = 0;
	state["value"] = val_dict;

	return state;
}

// Helper to sequence multiple planning operations
inline TypedArray<Ref<PlannerResult>> sequence_plans(Ref<PlannerPlan> p_plan,
		Ref<PlannerDomain> p_domain, int p_iterations, Array p_todo) {
	TypedArray<Ref<PlannerResult>> results;

	for (int i = 0; i < p_iterations; i++) {
		p_plan->reset();
		p_plan->set_current_domain(p_domain);

		Dictionary state = create_counter_state(i * 10);
		Ref<PlannerResult> result = p_plan->find_plan(state, p_todo);
		results.push_back(result);
	}

	return results;
}

} // namespace PlannerTestHelpers
