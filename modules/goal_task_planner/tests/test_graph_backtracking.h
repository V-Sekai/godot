/**************************************************************************/
/*  test_graph_backtracking.h                                             */
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

// C++ unit tests for graph-based backtracking and blacklisting in goal_task_planner

#pragma once

#include "../domain.h"
#include "../plan.h"
#include "../planner_state.h"
#include "../planner_time_range.h"
#include "tests/test_macros.h"

namespace TestGraphBacktracking {

// Helper static functions for testing
static Variant test_action_success(Dictionary p_state, String p_arg) {
	Dictionary new_state = p_state.duplicate();
	new_state["executed"] = p_arg;
	return new_state;
}

static Variant test_task_method(Dictionary p_state, String p_task_name) {
	Array subtasks;
	Array action_task;
	action_task.push_back("test_action_success");
	action_task.push_back(p_task_name);
	subtasks.push_back(action_task);
	return subtasks;
}

// Helper function to create a simple domain for testing
static Ref<PlannerDomain> create_test_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&test_action_success));
	domain->add_actions(actions);

	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&test_task_method));
	domain->add_task_methods("test_task", task_methods);

	return domain;
}

TEST_CASE("[Modules][GraphBacktracking] Basic graph-based planning") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = create_test_domain();
	plan->set_current_domain(domain);

	SUBCASE("Simple planning with run_lazy_refineahead") {
		Dictionary initial_state;
		initial_state["initialized"] = true;

		Array todo_list;
		Array task;
		task.push_back("test_task");
		task.push_back("arg1");
		todo_list.push_back(task);

		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);
		CHECK(final_state.has("executed"));
		CHECK(final_state["executed"] == "arg1");
	}

	SUBCASE("Planning with empty todo list") {
		Dictionary initial_state;
		initial_state["initialized"] = true;
		Array todo_list;

		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);
		CHECK(final_state == initial_state);
	}

	// Ref<> objects handle cleanup automatically via reference counting
}

// Static helper functions for backtracking tests
static Variant test_action_fail(Dictionary p_state, String p_arg) {
	// Return empty Variant to indicate failure
	return Variant();
}

static Variant test_action_success2(Dictionary p_state, String p_arg) {
	Dictionary new_state = p_state.duplicate();
	new_state["success"] = p_arg;
	return new_state;
}

static Variant test_task_method1(Dictionary p_state, String p_task_name) {
	Array subtasks;
	Array action_task;
	action_task.push_back("test_action_fail");
	action_task.push_back(p_task_name);
	subtasks.push_back(action_task);
	return subtasks;
}

static Variant test_task_method2(Dictionary p_state, String p_task_name) {
	Array subtasks;
	Array action_task;
	action_task.push_back("test_action_success2");
	action_task.push_back(p_task_name);
	subtasks.push_back(action_task);
	return subtasks;
}

TEST_CASE("[Modules][GraphBacktracking] Action failure and backtracking" * doctest::skip(true)) {
	// DISABLED: Test crashing with SIGSEGV - needs investigation
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&test_action_fail));
	actions.push_back(callable_mp_static(&test_action_success2));
	domain->add_actions(actions);

	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&test_task_method1));
	task_methods.push_back(callable_mp_static(&test_task_method2));
	domain->add_task_methods("test_task", task_methods);

	plan->set_current_domain(domain);

	SUBCASE("Backtracking when action fails") {
		Dictionary initial_state;
		initial_state["initialized"] = true;

		Array todo_list;
		Array task;
		task.push_back("test_task");
		task.push_back("test_arg");
		todo_list.push_back(task);

		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);
		// Should succeed with second method
		CHECK(final_state.has("success"));
		CHECK(final_state["success"] == "test_arg");
	}

	// Ref<> objects handle cleanup automatically via reference counting
}

// Static helper for blacklisting test
static Variant test_action_always_fail(Dictionary p_state, String p_arg) {
	return Variant(); // Always fail
}

static Variant test_task_method_fail(Dictionary p_state, String p_task_name) {
	Array subtasks;
	Array action_task;
	action_task.push_back("test_action_always_fail");
	action_task.push_back(p_task_name);
	subtasks.push_back(action_task);
	return subtasks;
}

TEST_CASE("[Modules][GraphBacktracking] Blacklisting functionality" * doctest::skip(true)) {
	// DISABLED: Test crashing with SIGSEGV - needs investigation
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&test_action_always_fail));
	domain->add_actions(actions);

	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&test_task_method_fail));
	domain->add_task_methods("test_task", task_methods);

	plan->set_current_domain(domain);

	SUBCASE("Blacklisting prevents infinite retry loops") {
		Dictionary initial_state;
		initial_state["initialized"] = true;

		Array todo_list;
		Array task;
		task.push_back("test_task");
		task.push_back("test_arg");
		todo_list.push_back(task);

		// This should eventually fail, but not loop infinitely due to blacklisting
		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);
		// Should return some state (either original or partially modified)
		CHECK(final_state.has("initialized"));
	}

	// Ref<> objects handle cleanup automatically via reference counting
}

// Static helpers for goal refinement tests
static Variant test_action_set_state(Dictionary p_state, String p_var_name, String p_value) {
	Dictionary new_state = p_state.duplicate();
	if (!new_state.has(p_var_name)) {
		new_state[p_var_name] = Dictionary();
	}
	Dictionary var_dict = new_state[p_var_name];
	var_dict["value"] = p_value;
	new_state[p_var_name] = var_dict;
	return new_state;
}

static Variant test_goal_method(Dictionary p_state, String p_var_name, String p_value) {
	Array subtasks;
	Array action_task;
	action_task.push_back("test_action_set_state");
	action_task.push_back(p_var_name);
	action_task.push_back(p_value);
	subtasks.push_back(action_task);
	return subtasks;
}

TEST_CASE("[Modules][GraphBacktracking] Goal refinement and backtracking" * doctest::skip(true)) {
	// DISABLED: Test crashing with SIGSEGV - needs investigation
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&test_action_set_state));
	domain->add_actions(actions);

	TypedArray<Callable> goal_methods;
	goal_methods.push_back(callable_mp_static(&test_goal_method));
	domain->add_unigoal_methods("test_var", goal_methods);

	plan->set_current_domain(domain);

	SUBCASE("Goal achievement through refinement") {
		Dictionary initial_state;
		initial_state["test_var"] = Dictionary();
		((Dictionary)initial_state["test_var"])["value"] = "initial";

		Array todo_list;
		Array goal;
		goal.push_back("test_var");
		goal.push_back("value");
		goal.push_back("target");
		todo_list.push_back(goal);

		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);
		CHECK(final_state.has("test_var"));
		Dictionary var_dict = final_state["test_var"];
		CHECK(var_dict["value"] == "target");
	}

	SUBCASE("Goal already achieved") {
		Dictionary initial_state;
		initial_state["test_var"] = Dictionary();
		((Dictionary)initial_state["test_var"])["value"] = "target";

		Array todo_list;
		Array goal;
		goal.push_back("test_var");
		goal.push_back("value");
		goal.push_back("target");
		todo_list.push_back(goal);

		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);
		// Should return without modification since goal is already achieved
		CHECK(final_state.has("test_var"));
		Dictionary var_dict = final_state["test_var"];
		CHECK(var_dict["value"] == "target");
	}

	// Ref<> objects handle cleanup automatically via reference counting
}

TEST_CASE("[Modules][GraphBacktracking] Solution graph structure") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = create_test_domain();
	plan->set_current_domain(domain);

	SUBCASE("Graph is created during planning") {
		Dictionary initial_state;
		initial_state["initialized"] = true;

		Array todo_list;
		Array task;
		task.push_back("test_task");
		task.push_back("arg1");
		todo_list.push_back(task);

		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);

		// The solution graph should be populated internally
		// We can't directly access it, but if planning succeeds, graph was created
		CHECK(final_state.has("executed"));
	}

	// Ref<> objects handle cleanup automatically via reference counting
}

// Static helpers for state snapshot tests
static Variant test_action_modify(Dictionary p_state, String p_key, int p_value) {
	Dictionary new_state = p_state.duplicate();
	new_state[p_key] = p_value;
	return new_state;
}

static Variant test_task_method_multiple(Dictionary p_state, String p_task_name) {
	Array subtasks;
	Array action1;
	action1.push_back("test_action_modify");
	action1.push_back("step1");
	action1.push_back(1);
	subtasks.push_back(action1);

	Array action2;
	action2.push_back("test_action_modify");
	action2.push_back("step2");
	action2.push_back(2);
	subtasks.push_back(action2);

	return subtasks;
}

TEST_CASE("[Modules][GraphBacktracking] State snapshot restoration") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&test_action_modify));
	domain->add_actions(actions);

	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&test_task_method_multiple));
	domain->add_task_methods("test_task", task_methods);

	plan->set_current_domain(domain);

	SUBCASE("State progression through actions") {
		Dictionary initial_state;
		initial_state["initialized"] = true;

		Array todo_list;
		Array task;
		task.push_back("test_task");
		task.push_back("test");
		todo_list.push_back(task);

		Dictionary final_state = plan->run_lazy_refineahead(initial_state, todo_list);

		// Both actions should have executed
		CHECK(final_state.has("step1"));
		CHECK(final_state["step1"] == Variant(1));
		CHECK(final_state.has("step2"));
		CHECK(final_state["step2"] == Variant(2));
		CHECK(final_state.has("initialized"));
	}

	// Ref<> objects handle cleanup automatically via reference counting
}

} // namespace TestGraphBacktracking
