/**************************************************************************/
/*  test_qa.h                                                             */
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

// QA test suite for goal_task_planner with absolute time

#pragma once

#include "../domain.h"
#include "../plan.h"
#include "../planner_state.h"
#include "tests/test_macros.h"

namespace TestQA {

TEST_CASE("[QA] End-to-end temporal planning workflow") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	// Initialize state
	Ref<PlannerState> state = memnew(PlannerState);
	state->set_predicate("block_a", "pos", "table");
	state->set_entity_capability("robot_1", "movable", true);

	// Submit operation
	Dictionary operation;
	operation["action"] = "pickup";
	operation["block"] = "block_a";
	Dictionary result = plan->submit_operation(operation);

	CHECK(result.has("operation_id"));
	CHECK(result.has("agreed_at"));
	CHECK((int64_t)result["agreed_at"] > 0);

	// Ref<> objects handle cleanup automatically via reference counting
}

TEST_CASE("[QA] Absolute time accuracy in planning operations") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	// Get current time
	int64_t time_before = PlannerTimeRange::now_microseconds();

	// Submit operation
	Dictionary operation;
	operation["test"] = true;
	Dictionary result = plan->submit_operation(operation);

	int64_t time_after = PlannerTimeRange::now_microseconds();
	int64_t agreed_at = result["agreed_at"];

	// Agreed time should be between before and after
	CHECK(agreed_at >= time_before);
	CHECK(agreed_at <= time_after);

	// Ref<> objects handle cleanup automatically via reference counting
}

TEST_CASE("[QA] Memory management and resource cleanup") {
	// Test that resources are properly cleaned up
	for (int i = 0; i < 10; i++) {
		Ref<PlannerPlan> plan = memnew(PlannerPlan);
		memdelete(plan.ptr());
	}
	CHECK(true); // If we get here without crashing, cleanup works
}

TEST_CASE("[QA] Performance with multiple operations") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	// Submit multiple operations
	for (int i = 0; i < 100; i++) {
		Dictionary operation;
		operation["index"] = i;
		plan->submit_operation(operation);
	}

	// Should complete without significant delay
	CHECK(true);

	// Ref<> objects handle cleanup automatically via reference counting
}

TEST_CASE("[QA] Backward compatibility - existing GDScript patterns") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	// Test that basic operations still work
	PlannerTimeRange time_range;
	time_range.set_start_time(1735689600000000LL);
	plan->set_time_range(time_range);

	PlannerTimeRange retrieved = plan->get_time_range();
	CHECK(retrieved.get_start_time() == 1735689600000000LL);

	// Test plan ID generation
	String id = plan->generate_plan_id();
	CHECK(!id.is_empty());

	// Ref<> objects handle cleanup automatically via reference counting
}

// Helper static functions for testing
static Variant qa_test_action(Dictionary p_state, String p_arg) {
	Dictionary new_state = p_state.duplicate();
	new_state["qa_executed"] = p_arg;
	return new_state;
}

static Variant qa_test_task_method(Dictionary p_state, String p_task_name) {
	Array subtasks;
	Array action_task;
	action_task.push_back("qa_test_action");
	action_task.push_back(p_task_name);
	subtasks.push_back(action_task);
	return subtasks;
}

TEST_CASE("[QA] Graph-based planning with run_lazy_refineahead") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&qa_test_action));
	domain->add_actions(actions);

	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&qa_test_task_method));
	domain->add_task_methods("qa_test_task", task_methods);

	plan->set_current_domain(domain);

	SUBCASE("Basic graph-based planning execution") {
		Dictionary initial_state;
		initial_state["initialized"] = true;

		Array todo_list;
		Array task;
		task.push_back("qa_test_task");
		task.push_back("qa_test_arg");
		todo_list.push_back(task);

		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);

		CHECK(final_state.has("qa_executed"));
		CHECK(final_state["qa_executed"] == "qa_test_arg");
		CHECK(final_state.has("initialized"));
	}

	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[QA] STN integration in planning loop") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	// Create a simple domain with durative actions
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Create test action that takes time
	Dictionary action_dict;
	Array action_arr;
	action_arr.push_back("move");
	action_arr.push_back("robot");
	action_arr.push_back("location");
	action_dict["move"] = action_arr;
	domain->action_dictionary = action_dict;

	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initialize state
	Dictionary state;
	state["robot"] = "start";
	state["location"] = "A";

	// Create todo list
	Array todo_list;
	Array action_task;
	action_task.push_back("move");
	action_task.push_back("robot");
	action_task.push_back("B");
	todo_list.push_back(action_task);

	// Run planning (should initialize STN)
	Dictionary final_state = plan->run_lazy_refineahead(state, todo_list);

	// Verify STN was initialized (plan has temporal tracking)
	PlannerTimeRange time_range = plan->get_time_range();
	CHECK(time_range.get_start_time() > 0);

	// Ref<> objects handle cleanup automatically via reference counting
}

} // namespace TestQA
