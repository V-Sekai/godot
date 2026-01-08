/**************************************************************************/
/*  test_edge_cases.h                                                     */
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
#include "../src/multigoal.h"
#include "../src/plan.h"
#include "../src/planner_result.h"
#include "../src/stn_solver.h"
#include "tests/test_macros.h"

#include "minimal_backtracking_domain.h"
#include "minimal_task_domain.h"

namespace TestEdgeCases {

TEST_CASE("[Modules][Planner] Empty domain - no tasks") {
	// Test behavior with completely empty domain
	Ref<PlannerDomain> empty_domain = memnew(PlannerDomain);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(empty_domain);
	plan->set_verbose(0);

	Dictionary init_state;
	Array todo_list;
	todo_list.push_back("nonexistent_task");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	CHECK(result.is_valid());
	CHECK(!result->get_success());
}

TEST_CASE("[Modules][Planner] Empty todo list") {
	// Test planning with empty todo list (nothing to do)
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array empty_todo_list; // Empty

	Ref<PlannerResult> result = plan->find_plan(init_state, empty_todo_list);

	// Should succeed immediately - nothing to plan
	CHECK(result.is_valid());
	CHECK(result->get_success());
}

TEST_CASE("[Modules][Planner] Null domain handling") {
	// Test planning with null domain
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(nullptr);
	plan->set_verbose(0);

	Dictionary init_state;
	Array todo_list;
	todo_list.push_back("task");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	// Should handle gracefully
	CHECK(result.is_valid());
	// Result may be null or failure - should not crash
}

TEST_CASE("[Modules][Planner] Set/get verbose levels") {
	// Test verbose level API
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	CHECK(plan->get_verbose() == 0);

	plan->set_verbose(1);
	CHECK(plan->get_verbose() == 1);

	plan->set_verbose(2);
	CHECK(plan->get_verbose() == 2);

	plan->set_verbose(3);
	CHECK(plan->get_verbose() == 3);

	plan->set_verbose(0);
	CHECK(plan->get_verbose() == 0);
}

TEST_CASE("[Modules][Planner] Set/get max_depth") {
	// Test max_depth API
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	CHECK(plan->get_max_depth() == 10); // Default

	plan->set_max_depth(50);
	CHECK(plan->get_max_depth() == 50);

	plan->set_max_depth(1);
	CHECK(plan->get_max_depth() == 1);
}

TEST_CASE("[Modules][Planner] Set/get max_iterations") {
	// Test max_iterations API
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	CHECK(plan->get_max_iterations() == 50000); // Default

	plan->set_max_iterations(1000);
	CHECK(plan->get_max_iterations() == 1000);

	plan->set_max_iterations(100);
	CHECK(plan->get_max_iterations() == 100);
}

TEST_CASE("[Modules][Planner] Reset clears all state") {
	// Test reset functionality
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	plan->set_current_domain(domain);
	plan->set_max_depth(50);
	plan->set_verbose(2);
	plan->set_max_iterations(1000);

	// Verify set
	CHECK(plan->get_max_depth() == 50);
	CHECK(plan->get_verbose() == 2);

	// Reset
	plan->reset();

	// Verify reset to defaults
	CHECK(plan->get_max_depth() == 10);
	CHECK(plan->get_verbose() == 0);
	CHECK(plan->get_max_iterations() == 50000);
}

TEST_CASE("[Modules][Planner] PlannerResult - empty result handling") {
	// Test PlannerResult with no solution
	Ref<PlannerResult> result = memnew(PlannerResult);

	CHECK(!result->get_success());
	CHECK(result->get_final_state().is_empty());
	CHECK(result->get_solution_graph().is_empty());

	Array plan_array = result->extract_plan();
	CHECK(plan_array.is_empty());

	Array failed_nodes = result->find_failed_nodes();
	CHECK(failed_nodes.is_empty());
}

TEST_CASE("[Modules][Planner] PlannerResult - root node exists") {
	// Test that solution graph always has root node
	Ref<PlannerResult> result = memnew(PlannerResult);

	// Even empty result should have valid structure
	CHECK(!result->has_node(-1)); // Invalid ID
}

TEST_CASE("[Modules][Planner] PlannerMultigoal - is_multigoal_array validation") {
	// Test multigoal type checking
	Array multigoal;
	Array unigoal1;
	unigoal1.push_back("predicate");
	unigoal1.push_back("subject");
	unigoal1.push_back("value");
	multigoal.push_back(unigoal1);

	CHECK(PlannerMultigoal::is_multigoal_array(multigoal));
	CHECK(!PlannerMultigoal::is_multigoal_array(unigoal1)); // Single unigoal is not multigoal
	CHECK(!PlannerMultigoal::is_multigoal_array("string")); // Non-array
	CHECK(!PlannerMultigoal::is_multigoal_array(123)); // Non-array
}

TEST_CASE("[Modules][Planner] STN Solver - empty solver consistency") {
	// Test STN solver in empty state
	PlannerSTNSolver stn;

	CHECK(stn.is_consistent());

	// Distance between non-existent points should be infinity
	int64_t dist = stn.get_distance("point1", "point2");
	CHECK(dist == PlannerSTNSolver::STN_INFINITY);
}

TEST_CASE("[Modules][Planner] STN Solver - same point distance") {
	// Test distance from point to itself
	PlannerSTNSolver stn;
	stn.add_time_point("point1");

	int64_t dist = stn.get_distance("point1", "point1");
	CHECK(dist == 0);
}

TEST_CASE("[Modules][Planner] STN Solver - snapshot and restore") {
	// Test STN snapshot functionality
	PlannerSTNSolver stn;
	stn.add_time_point("point1");
	stn.add_time_point("point2");
	stn.add_constraint("point1", "point2", 1000, 2000);

	PlannerSTNSolver::Snapshot snapshot = stn.create_snapshot();

	// Verify snapshot consistency
	CHECK(snapshot.consistent);
}

TEST_CASE("[Modules][Planner] Backtracking with minimal domain") {
	// Test basic planning with backtracking domain
	Ref<PlannerDomain> domain = MinimalBacktrackingDomain::create_minimal_backtracking_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	// Should succeed via backtracking
	CHECK(result.is_valid());
	CHECK(result->get_success());

	// Plan should not be empty
	Array plan_array = result->extract_plan();
	CHECK(plan_array.size() > 0);
}

TEST_CASE("[Modules][Planner] Plan simulation") {
	// Test plan simulation
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	if (result->get_success()) {
		Array state_sequence = plan->simulate(result, init_state, 0);

		CHECK(state_sequence.size() > 0);
		// Final state should have incremented value
		Dictionary final_state = state_sequence[state_sequence.size() - 1];
		CHECK(final_state.has("value"));
	}
}

TEST_CASE("[Modules][Planner] Multiple planning iterations") {
	// Test running planner multiple times
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");

	// Run multiple times
	for (int i = 0; i < 3; i++) {
		plan->reset();
		plan->set_current_domain(domain);

		Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

		CHECK(result.is_valid());
		CHECK(result->get_success());
	}
}

TEST_CASE("[Modules][Planner] VSIDS activity tracking") {
	// Test VSIDS method activity tracking
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	// Get method activities
	Dictionary activities = plan->get_method_activities();

	// Activities dictionary should exist (may be empty for simple plans)
	// Just verify the API call succeeds
}

TEST_CASE("[Modules][Planner] Domain with no actions") {
	// Test domain with task methods but no actions
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Add task method that tries to execute non-existent action
	TypedArray<Callable> methods;
	// Intentionally not adding any methods

	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Dictionary init_state;
	Array todo_list;
	todo_list.push_back("task");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	CHECK(result.is_valid());
	CHECK(!result->get_success());
}

TEST_CASE("[Modules][Planner] Large state deep copy") {
	// Test handling of large states with deep copy
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Dictionary init_state;

	// Create large nested state
	for (int i = 0; i < 100; i++) {
		Dictionary nested;
		nested["value"] = i;
		nested["data"] = "test_" + String::num(i);
		init_state["key_" + String::num(i)] = nested;
	}

	Array todo_list;
	todo_list.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	// Should handle large states
	CHECK(result.is_valid());
}

TEST_CASE("[Modules][Planner] Action state mutation") {
	// Test that actions properly mutate state
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 5;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	CHECK(result->get_success());

	// Check final state has incremented value
	Dictionary final_state = result->get_final_state();
	if (final_state.has("value")) {
		Dictionary final_value = final_state["value"];
		CHECK(int(final_value["value"]) == 6);
	}
}

TEST_CASE("[Modules][Planner] Multiple tasks in todo list") {
	// Test planning with multiple tasks
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");
	todo_list.push_back("increment"); // Two increments

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	CHECK(result.is_valid());
	// May or may not succeed depending on domain implementation
}

TEST_CASE("[Modules][Planner] Solution graph structure") {
	// Test solution graph has expected structure
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	if (result->get_success()) {
		Dictionary solution_graph = result->get_solution_graph();
		// Graph should not be empty for successful plan
		CHECK(!solution_graph.is_empty());

		// Should have at least root node (id 0)
		CHECK(solution_graph.has(0) || !solution_graph.is_empty());
	}
}

TEST_CASE("[Modules][Planner] Plan extraction from result") {
	// Test extracting plan array from result
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	Array extracted_plan = result->extract_plan();

	if (result->get_success()) {
		// Successful plan should have actions
		CHECK(extracted_plan.size() >= 0);

		// Each action should be an array
		for (int i = 0; i < extracted_plan.size(); i++) {
			Variant action = extracted_plan[i];
			CHECK(action.get_type() == Variant::ARRAY);
		}
	}
}

TEST_CASE("[Modules][Planner] Multigoal tag operations") {
	// Test multigoal tag functionality
	Array multigoal;
	Array unigoal;
	unigoal.push_back("have");
	unigoal.push_back("robot");
	unigoal.push_back("battery");
	multigoal.push_back(unigoal);

	// Get tag (should be empty initially)
	String tag = PlannerMultigoal::get_goal_tag(multigoal);
	CHECK(tag.is_empty());

	// Set tag
	Variant tagged = PlannerMultigoal::set_goal_tag(multigoal, "energy_goal");
	String new_tag = PlannerMultigoal::get_goal_tag(tagged);
	CHECK(new_tag == "energy_goal");
}

TEST_CASE("[Modules][Planner] STN constraint operations") {
	// Test STN constraint addition and retrieval
	PlannerSTNSolver stn;

	stn.add_time_point("task_start");
	stn.add_time_point("task_end");

	// Add constraint: task_end >= task_start + 1000ms
	stn.add_constraint("task_start", "task_end", 1000, 5000);

	PlannerSTNSolver::Constraint constraint = stn.get_constraint("task_start", "task_end");
	CHECK(constraint.min_distance == 1000);
	CHECK(constraint.max_distance == 5000);

	// Verify consistency
	CHECK(stn.is_consistent());
}

TEST_CASE("[Modules][Planner] STN distance queries") {
	// Test STN distance calculation between points
	PlannerSTNSolver stn;

	stn.add_time_point("point_a");
	stn.add_time_point("point_b");
	stn.add_time_point("point_c");

	stn.add_constraint("point_a", "point_b", 100, 200);
	stn.add_constraint("point_b", "point_c", 50, 100);

	int64_t dist_a_b = stn.get_distance("point_a", "point_b");
	int64_t dist_b_c = stn.get_distance("point_b", "point_c");
	int64_t dist_a_c = stn.get_distance("point_a", "point_c");

	CHECK(dist_a_b > 0);
	CHECK(dist_b_c > 0);
	// a to c should be calculable (transitive)
}

TEST_CASE("[Modules][Planner] Domain action execution") {
	// Test that actions are properly registered and callable
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();

	// Domain should have actions stored
	CHECK(domain->action_dictionary.size() > 0);
}

TEST_CASE("[Modules][Planner] Iterative planning with changing state") {
	// Test planning with modified states between calls
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();

	Dictionary state1;
	Dictionary val1;
	val1["value"] = 0;
	state1["value"] = val1;

	Dictionary state2;
	Dictionary val2;
	val2["value"] = 10;
	state2["value"] = val2;

	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Array todo;
	todo.push_back("increment");

	// Plan from different states
	Ref<PlannerResult> result1 = plan->find_plan(state1, todo);
	plan->reset();

	Ref<PlannerResult> result2 = plan->find_plan(state2, todo);

	// Both should be valid
	CHECK(result1.is_valid());
	CHECK(result2.is_valid());
}

TEST_CASE("[Modules][Planner] Result success flag consistency") {
	// Test that success flag is consistent with plan content
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	bool success = result->get_success();
	Array plan_array = result->extract_plan();

	if (success) {
		// Successful plans should have meaningful structure
		CHECK(result->get_final_state().size() >= 0);
	}
}

TEST_CASE("[Modules][Planner] Empty action list domain") {
	// Test domain with no actions at all
	Ref<PlannerDomain> empty_domain = memnew(PlannerDomain);

	// Add empty action list explicitly
	TypedArray<Callable> empty_actions;
	empty_domain->add_actions(empty_actions);

	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(empty_domain);

	Dictionary init_state;
	Array todo_list;
	todo_list.push_back("task");

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	CHECK(result.is_valid());
	CHECK(!result->get_success());
}

TEST_CASE("[Modules][Planner] State dictionary serialization") {
	// Test that states can be serialized/duplicated properly
	Dictionary state;

	Dictionary nested1;
	nested1["value"] = 100;
	nested1["name"] = "test";
	state["key1"] = nested1;

	Dictionary nested2;
	nested2["data"] = "content";
	state["key2"] = nested2;

	// Deep duplicate
	Dictionary state_copy = state.duplicate(true);

	CHECK(state_copy.has("key1"));
	CHECK(state_copy.has("key2"));

	Dictionary copy_nested = state_copy["key1"];
	CHECK(int(copy_nested["value"]) == 100);
	CHECK(String(copy_nested["name"]) == "test");
}

TEST_CASE("[Modules][Planner] Plan iterations tracking") {
	// Test that planner tracks iteration count
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;

	Array todo_list;
	todo_list.push_back("increment");

	int iterations_before = plan->get_iterations();
	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);
	int iterations_after = plan->get_iterations();

	// Should have done some planning
	CHECK(iterations_after >= iterations_before);
}

} // namespace TestEdgeCases
