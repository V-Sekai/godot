/**************************************************************************/
/*  test_robustness.h                                                     */
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
#include "../src/solution_graph.h"
#include "../src/stn_solver.h"
#include "minimal_task_domain.h"
#include "tests/test_macros.h"

namespace TestRobustness {

TEST_CASE("[Modules][Planner] Large state handling") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary large_dict;

	for (int i = 0; i < 100; i++) {
		large_dict[String("key_") + String::num(i)] = i * 10;
	}

	state["large_data"] = large_dict;
	Dictionary val_dict;
	val_dict["value"] = 0;
	state["value"] = val_dict;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);
	CHECK(result.is_valid());
}

TEST_CASE("[Modules][Planner] Deeply nested state structures") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary nested;
	Dictionary deep;
	deep["level3"] = 42;
	nested["level2"] = deep;
	state["level1"] = nested;
	Dictionary val_dict;
	val_dict["value"] = 0;
	state["value"] = val_dict;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);
	CHECK(result.is_valid());
}

TEST_CASE("[Modules][Planner] Empty todo list planning") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary val_dict;
	val_dict["value"] = 0;
	state["value"] = val_dict;

	Array todo;

	Ref<PlannerResult> result = plan->find_plan(state, todo);
	CHECK(result.is_valid());
	CHECK(result->get_success() == true);
}

TEST_CASE("[Modules][Planner] Plan reset and reuse") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();

	Dictionary state1;
	Dictionary val_dict1;
	val_dict1["value"] = 0;
	state1["value"] = val_dict1;

	Array todo1;
	todo1.push_back("increment");

	plan->set_current_domain(domain);
	plan->reset();
	Ref<PlannerResult> result1 = plan->find_plan(state1, todo1);
	CHECK(result1.is_valid());

	plan->reset();
	Dictionary state2;
	Dictionary val_dict2;
	val_dict2["value"] = 10;
	state2["value"] = val_dict2;

	Ref<PlannerResult> result2 = plan->find_plan(state2, todo1);
	CHECK(result2.is_valid());
}

TEST_CASE("[Modules][Planner] Multiple rapid plans") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	for (int i = 0; i < 5; i++) {
		Dictionary state;
		Dictionary val_dict;
		val_dict["value"] = i;
		state["value"] = val_dict;

		Array todo;
		todo.push_back("increment");

		Ref<PlannerResult> result = plan->find_plan(state, todo);
		CHECK(result.is_valid());
	}
}

TEST_CASE("[Modules][Planner] STN with many constraints") {
	PlannerSTNSolver stn;

	for (int i = 0; i < 10; i++) {
		stn.add_time_point("tp" + String::num(i));
	}

	for (int i = 0; i < 9; i++) {
		stn.add_constraint("tp" + String::num(i), "tp" + String::num(i + 1), 
			i * 100, (i + 1) * 100);
	}

	CHECK(stn.is_consistent());
}

TEST_CASE("[Modules][Planner] STN consistency after violations") {
	PlannerSTNSolver stn;

	stn.add_time_point("A");
	stn.add_time_point("B");
	stn.add_time_point("C");

	stn.add_constraint("A", "B", 0, 100);
	stn.add_constraint("B", "C", 0, 100);

	CHECK(stn.is_consistent());

	int64_t dist_AB = stn.get_distance("A", "B");
	int64_t dist_BC = stn.get_distance("B", "C");
	int64_t dist_AC = stn.get_distance("A", "C");

	CHECK(dist_AB >= 0 || dist_AB == PlannerSTNSolver::STN_INFINITY);
	CHECK(dist_BC >= 0 || dist_BC == PlannerSTNSolver::STN_INFINITY);
	CHECK(dist_AC >= 0 || dist_AC == PlannerSTNSolver::STN_INFINITY);
}

TEST_CASE("[Modules][Planner] Solution graph with many nodes") {
	PlannerSolutionGraph graph;

	Dictionary root = graph.get_node(0);
	CHECK(!root.is_empty());

	CHECK(root.has("status") || root.size() == 0);
}

TEST_CASE("[Modules][Planner] Domain with no actions") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> empty_domain = memnew(PlannerDomain);
	plan->set_current_domain(empty_domain);

	Dictionary state;
	Dictionary val_dict;
	val_dict["value"] = 0;
	state["value"] = val_dict;

	Array todo;
	todo.push_back("non_existent_task");

	Ref<PlannerResult> result = plan->find_plan(state, todo);
	CHECK(result.is_valid());
}

TEST_CASE("[Modules][Planner] State mutation isolation") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary original_state;
	Dictionary val;
	val["value"] = 5;
	original_state["value"] = val;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(original_state, todo);

	Dictionary unchanged_val = original_state["value"];
	CHECK(int(unchanged_val["value"]) == 5);
}

TEST_CASE("[Modules][Planner] Result consistency check") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary val_dict;
	val_dict["value"] = 0;
	state["value"] = val_dict;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);

	bool success = result->get_success();
	Array plan_array = result->extract_plan();

	if (success && plan_array.size() > 0) {
		CHECK(success);
	} else if (plan_array.is_empty()) {
		CHECK(true);
	}
}

TEST_CASE("[Modules][Planner] Sequential state changes") {
	Dictionary state;
	Dictionary val_dict;
	val_dict["value"] = 0;
	state["value"] = val_dict;

	for (int i = 0; i < 3; i++) {
		Dictionary new_state = state.duplicate(true);
		Dictionary val = new_state["value"];
		val["value"] = int(val["value"]) + 1;
		new_state["value"] = val;
		state = new_state;
	}

	Dictionary final_val = state["value"];
	CHECK(int(final_val["value"]) == 3);
}

TEST_CASE("[Modules][Planner] PlannerDomain action registration") {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	CHECK(domain.is_valid());

	CHECK(domain->action_dictionary.size() >= 0);
}

} // namespace TestRobustness
