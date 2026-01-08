/**************************************************************************/
/*  test_backtracking_combinations.h                                      */
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

#include "../src/domain.h"
#include "../src/plan.h"
#include "../src/planner_result.h"
#include "../src/stn_solver.h"
#include "minimal_backtracking_domain.h"
#include "minimal_task_domain.h"
#include "tests/test_macros.h"

namespace TestBacktrackingCombinations {

TEST_CASE("[Modules][Planner] Backtracking with sequential tasks") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalBacktrackingDomain::create_minimal_backtracking_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Array path;
	path.push_back("room_a");
	state["current_path"] = path;

	Array todo;
	todo.push_back("navigate_to_room");
	todo.push_back("navigate_to_room");

	Ref<PlannerResult> result = plan->find_plan(state, todo);
	CHECK(result.is_valid());
}

TEST_CASE("[Modules][Planner] Backtracking with state constraints") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state1;
	Dictionary val1;
	val1["value"] = 0;
	state1["value"] = val1;

	Array todo1;
	todo1.push_back("increment");

	Ref<PlannerResult> result1 = plan->find_plan(state1, todo1);
	CHECK(result1.is_valid());

	plan->reset();

	Dictionary state2;
	Dictionary val2;
	val2["value"] = 100;
	state2["value"] = val2;

	Array todo2;
	todo2.push_back("increment");

	Ref<PlannerResult> result2 = plan->find_plan(state2, todo2);
	CHECK(result2.is_valid());
}

TEST_CASE("[Modules][Planner] Backtracking with temporal constraints") {
	PlannerSTNSolver stn;

	stn.add_time_point("start");
	stn.add_time_point("action1");
	stn.add_time_point("action2");
	stn.add_time_point("end");

	// First attempt: valid constraints
	stn.add_constraint("start", "action1", 0, 100);
	stn.add_constraint("action1", "action2", 50, 150);
	stn.add_constraint("action2", "end", 0, 100);

	CHECK(stn.is_consistent());

	// Check transitive distances
	int64_t start_to_action1 = stn.get_distance("start", "action1");
	int64_t action1_to_action2 = stn.get_distance("action1", "action2");
	int64_t action2_to_end = stn.get_distance("action2", "end");

	CHECK(start_to_action1 >= 0 || start_to_action1 == PlannerSTNSolver::STN_INFINITY);
	CHECK(action1_to_action2 >= 0 || action1_to_action2 == PlannerSTNSolver::STN_INFINITY);
	CHECK(action2_to_end >= 0 || action2_to_end == PlannerSTNSolver::STN_INFINITY);
}

TEST_CASE("[Modules][Planner] Backtracking with multiple method paths") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalBacktrackingDomain::create_minimal_backtracking_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Array path;
	path.push_back("room_a");
	path.push_back("room_b");
	state["current_path"] = path;

	Array todo;
	todo.push_back("navigate_to_room");

	Ref<PlannerResult> result = plan->find_plan(state, todo);
	CHECK(result.is_valid());
}

TEST_CASE("[Modules][Planner] Backtracking with state preservation") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary original_state;
	Dictionary val;
	val["value"] = 42;
	original_state["value"] = val;

	// Store original value
	int original_value = int(val["value"]);

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(original_state, todo);

	// Verify input state not mutated
	Dictionary unchanged_val = original_state["value"];
	CHECK(int(unchanged_val["value"]) == original_value);
}

TEST_CASE("[Modules][Planner] Backtracking across plan resets") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();

	for (int i = 0; i < 3; i++) {
		plan->reset();
		plan->set_current_domain(domain);

		Dictionary state;
		Dictionary val;
		val["value"] = i;
		state["value"] = val;

		Array todo;
		todo.push_back("increment");

		Ref<PlannerResult> result = plan->find_plan(state, todo);
		CHECK(result.is_valid());
	}
}

TEST_CASE("[Modules][Planner] Backtracking with empty then full todo") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary val;
	val["value"] = 0;
	state["value"] = val;

	// First plan with empty todo
	Array empty_todo;
	Ref<PlannerResult> result1 = plan->find_plan(state, empty_todo);
	CHECK(result1.is_valid());

	// Plan again with tasks
	plan->reset();
	Array full_todo;
	full_todo.push_back("increment");

	Ref<PlannerResult> result2 = plan->find_plan(state, full_todo);
	CHECK(result2.is_valid());
}

TEST_CASE("[Modules][Planner] Backtracking with metadata changes") {
	Ref<PlannerPlan> plan1 = memnew(PlannerPlan);
	plan1->reset();
	plan1->set_max_stack_size(5000);

	Ref<PlannerPlan> plan2 = memnew(PlannerPlan);
	plan2->reset();
	plan2->set_max_stack_size(20000);

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();

	Dictionary state;
	Dictionary val;
	val["value"] = 0;
	state["value"] = val;

	Array todo;
	todo.push_back("increment");

	plan1->set_current_domain(domain);
	Ref<PlannerResult> result1 = plan1->find_plan(state, todo);
	CHECK(result1.is_valid());

	plan2->set_current_domain(domain);
	Ref<PlannerResult> result2 = plan2->find_plan(state, todo);
	CHECK(result2.is_valid());
}

TEST_CASE("[Modules][Planner] Backtracking with complex state structure") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary counter;
	counter["value"] = 0;
	state["value"] = counter;

	Dictionary metadata;
	metadata["timestamp"] = 12345;
	metadata["version"] = "1.0";
	metadata["attempts"] = 0;
	state["metadata"] = metadata;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);
	CHECK(result.is_valid());
}

TEST_CASE("[Modules][Planner] Backtracking with domain switching") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	Ref<PlannerDomain> domain1 = MinimalTaskDomain::create_minimal_domain();
	plan->reset();
	plan->set_current_domain(domain1);

	Dictionary state1;
	Dictionary val1;
	val1["value"] = 0;
	state1["value"] = val1;

	Array todo1;
	todo1.push_back("increment");

	Ref<PlannerResult> result1 = plan->find_plan(state1, todo1);
	CHECK(result1.is_valid());

	// Switch domain
	Ref<PlannerDomain> domain2 = MinimalBacktrackingDomain::create_minimal_backtracking_domain();
	plan->reset();
	plan->set_current_domain(domain2);

	Dictionary state2;
	Array path;
	path.push_back("start");
	state2["current_path"] = path;

	Array todo2;
	todo2.push_back("navigate_to_room");

	Ref<PlannerResult> result2 = plan->find_plan(state2, todo2);
	CHECK(result2.is_valid());

	// Back to first domain
	plan->reset();
	plan->set_current_domain(domain1);

	Ref<PlannerResult> result3 = plan->find_plan(state1, todo1);
	CHECK(result3.is_valid());
}

TEST_CASE("[Modules][Planner] Backtracking with STN snapshots") {
	PlannerSTNSolver stn;

	stn.add_time_point("start");
	stn.add_time_point("mid");
	stn.add_time_point("end");

	stn.add_constraint("start", "mid", 0, 100);
	stn.add_constraint("mid", "end", 0, 100);

	// Create snapshot before additional constraints
	PlannerSTNSolver::Snapshot snapshot1 = stn.create_snapshot();
	CHECK(snapshot1.consistent);

	// Add more constraints
	stn.add_constraint("start", "end", 50, 200);

	// Snapshot should still be valid
	PlannerSTNSolver::Snapshot snapshot2 = stn.create_snapshot();
	CHECK(snapshot2.consistent);
}

TEST_CASE("[Modules][Planner] Backtracking with result chain") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();

	Array results;
	for (int i = 0; i < 3; i++) {
		plan->reset();
		plan->set_current_domain(domain);

		Dictionary state;
		Dictionary val;
		val["value"] = i * 10;
		state["value"] = val;

		Array todo;
		todo.push_back("increment");

		Ref<PlannerResult> result = plan->find_plan(state, todo);
		results.push_back(result);
	}

	// Verify all results are valid
	for (int i = 0; i < results.size(); i++) {
		Ref<PlannerResult> r = results[i];
		CHECK(r.is_valid());
	}
}

} // namespace TestBacktrackingCombinations
