/**************************************************************************/
/*  test_ipyhop_compatibility.h                                           */
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

#include "../domains/temporal_entity_test_domain.h"
#include "../../src/domain.h"
#include "../../src/plan.h"
#include "../../src/planner_result.h"
#include "../../src/planner_time_range.h"
#include "core/variant/callable.h"
#include "../helpers/ipyhop_test_helpers.h"
#include "tests/test_macros.h"

#include "core/io/json.h"
#include "core/io/file_access.h"

namespace TestIPyHOPCompatibility {

// Disabled: Test failing - needs investigation
#if 1
TEST_CASE("[Modules][Planner] IPyHOP Compatibility - Sample Test 1") {
	// Create completely fresh domain and planner for this test
	Ref<PlannerDomain> domain = create_ipyhop_test_domain();

	// Add task methods for task_method_1
	TypedArray<Callable> task_method_1_methods;
	task_method_1_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_1_1));
	task_method_1_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_1_2));
	task_method_1_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_1_3));
	domain->add_task_methods("task_method_1", task_method_1_methods);

	// Add task methods for task_method_2
	TypedArray<Callable> task_method_2_methods;
	task_method_2_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_2_1));
	task_method_2_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_2_2));
	domain->add_task_methods("task_method_2", task_method_2_methods);

	// Create fresh planner instance - ensures no state from previous tests
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_max_depth(100);
	plan->set_verbose(0); // Disable verbose for clean test output

	Dictionary init_state = create_init_state_1();

	Array todo_list;
	Array task1;
	task1.push_back("task_method_1");
	Array task2;
	task2.push_back("task_method_2");
	todo_list.push_back(task1);
	todo_list.push_back(task2);

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	// Load expected plan from JSON fixture
	String fixture_path = "modules/goal_task_planner/tests/fixtures/ipyhop_sample_test_1.json";
	Ref<FileAccess> f = FileAccess::open(fixture_path, FileAccess::READ);
	bool file_opened = f.is_valid();
	CHECK_MESSAGE(file_opened, ("Could not open fixture file: " + fixture_path).utf8().get_data());

	Array expected_plan;
	if (file_opened) {
		String json_string = f->get_as_text();
		JSON json;
		Error err = json.parse(json_string);
		CHECK_MESSAGE(err == OK, ("JSON parse error: " + json.get_error_message()).utf8().get_data());
		expected_plan = json.get_data();
	} else {
        // Fallback for when file is not found (e.g. CI or different CWD) so we see what fails but don't crash hard?
        // Actually, if file fails, the test should fail.
        // We can keep the manual construction as a fallback OR just fail. 
        // Failing is better to enforce fixture usage.
	}

	CHECK(result.is_valid());
	CHECK(result->get_success());
	CHECK(validate_plan_result(result, expected_plan));
}
#endif


// Disabled: Test failing - needs investigation
#if 1
TEST_CASE("[Modules][Planner] IPyHOP Compatibility - Backtracking Test 1") {
	Ref<PlannerDomain> domain = create_ipyhop_test_domain();

	// Add task methods for put_it
	TypedArray<Callable> put_it_methods;
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_err));
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m0));
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m1));
	domain->add_task_methods("put_it", put_it_methods);

	// Add task methods for need0
	TypedArray<Callable> need0_methods;
	need0_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_need0));
	domain->add_task_methods("need0", need0_methods);

	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_max_depth(100);
	// Remove verbose - test is passing

	Dictionary init_state;
	init_state["flag"] = -1;

	Array todo_list;
	Array task1;
	task1.push_back("put_it");
	Array task2;
	task2.push_back("need0");
	todo_list.push_back(task1);
	todo_list.push_back(task2);

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	// Expected plan: [('action_putv', 0), ('action_getv', 0), ('action_getv', 0)]
	Array expected_plan;
	Array action1;
	action1.push_back("action_putv");
	action1.push_back(0);
	Array action2;
	action2.push_back("action_getv");
	action2.push_back(0);
	Array action3;
	action3.push_back("action_getv");
	action3.push_back(0);
	expected_plan.push_back(action1);
	expected_plan.push_back(action2);
	expected_plan.push_back(action3);

	CHECK(result.is_valid());
	CHECK(result->get_success());
	CHECK(validate_plan_result(result, expected_plan));
}
#endif

// Disabled: Test failing - needs investigation
#if 1
TEST_CASE("[Modules][Planner] IPyHOP Compatibility - Backtracking Test 2") {
	Ref<PlannerDomain> domain = create_ipyhop_test_domain();

	// Add task methods for put_it
	TypedArray<Callable> put_it_methods;
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_err));
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m0));
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m1));
	domain->add_task_methods("put_it", put_it_methods);

	// Add task methods for need01 (need0 then need1)
	TypedArray<Callable> need01_methods;
	need01_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_need0));
	need01_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_need1));
	domain->add_task_methods("need01", need01_methods);

	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_max_depth(100);
	// Remove verbose - test is passing

	Dictionary init_state;
	init_state["flag"] = -1;

	Array todo_list;
	Array task1;
	task1.push_back("put_it");
	Array task2;
	task2.push_back("need01");
	todo_list.push_back(task1);
	todo_list.push_back(task2);

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	// Expected plan: [('action_putv', 0), ('action_getv', 0), ('action_getv', 0)]
	Array expected_plan;
	Array action1;
	action1.push_back("action_putv");
	action1.push_back(0);
	Array action2;
	action2.push_back("action_getv");
	action2.push_back(0);
	Array action3;
	action3.push_back("action_getv");
	action3.push_back(0);
	expected_plan.push_back(action1);
	expected_plan.push_back(action2);
	expected_plan.push_back(action3);

	CHECK(result.is_valid());
	CHECK(result->get_success());
	CHECK(validate_plan_result(result, expected_plan));
}
#endif

// Disabled: Test failing - needs investigation
#if 1
TEST_CASE("[Modules][Planner] IPyHOP Compatibility - Backtracking Test 3") {
	Ref<PlannerDomain> domain = create_ipyhop_test_domain();

	// Add task methods for put_it
	TypedArray<Callable> put_it_methods;
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_err));
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m0));
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m1));
	domain->add_task_methods("put_it", put_it_methods);

	// Add task methods for need10 (need1 then need0)
	TypedArray<Callable> need10_methods;
	need10_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_need1));
	need10_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_need0));
	domain->add_task_methods("need10", need10_methods);

	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_max_depth(100);
	// Remove verbose - test is passing

	Dictionary init_state;
	init_state["flag"] = -1;

	Array todo_list;
	Array task1;
	task1.push_back("put_it");
	Array task2;
	task2.push_back("need10");
	todo_list.push_back(task1);
	todo_list.push_back(task2);

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	// Expected plan: [('action_putv', 0), ('action_getv', 0), ('action_getv', 0)]
	Array expected_plan;
	Array action1;
	action1.push_back("action_putv");
	action1.push_back(0);
	Array action2;
	action2.push_back("action_getv");
	action2.push_back(0);
	Array action3;
	action3.push_back("action_getv");
	action3.push_back(0);
	expected_plan.push_back(action1);
	expected_plan.push_back(action2);
	expected_plan.push_back(action3);

	CHECK(result.is_valid());
	CHECK(result->get_success());
	CHECK(validate_plan_result(result, expected_plan));
}
#endif

// Disabled: Test failing - needs investigation
#if 1
TEST_CASE("[Modules][Planner] IPyHOP Compatibility - Backtracking Test 4") {
	// Create completely fresh domain and planner for this test
	Ref<PlannerDomain> domain = create_ipyhop_test_domain();

	// Add task methods for put_it
	TypedArray<Callable> put_it_methods;
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_err));
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m0));
	put_it_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m1));
	domain->add_task_methods("put_it", put_it_methods);

	// Add task methods for need1
	TypedArray<Callable> need1_methods;
	need1_methods.push_back(callable_mp_static(&IPyHOPTestDomainCallable::task_method_m_need1));
	domain->add_task_methods("need1", need1_methods);

	// Create fresh planner instance - ensures no state from previous tests
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_max_depth(100);
	plan->set_verbose(0); // Disable verbose for clean test output

	Dictionary init_state;
	init_state["flag"] = -1;

	Array todo_list;
	Array task1;
	task1.push_back("put_it");
	Array task2;
	task2.push_back("need1");
	todo_list.push_back(task1);
	todo_list.push_back(task2);

	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);

	// Expected plan: [('action_putv', 1), ('action_getv', 1), ('action_getv', 1)]
	Array expected_plan;
	Array action1;
	action1.push_back("action_putv");
	action1.push_back(1);
	Array action2;
	action2.push_back("action_getv");
	action2.push_back(1);
	Array action3;
	action3.push_back("action_getv");
	action3.push_back(1);
	expected_plan.push_back(action1);
	expected_plan.push_back(action2);
	expected_plan.push_back(action3);

	CHECK(result.is_valid());
	CHECK(result->get_success());
	CHECK(validate_plan_result(result, expected_plan));
}
#endif

TEST_CASE("[Modules][Planner] IPyHOP Compatibility - Temporal and Entity Constraints Test") {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Register actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&TemporalEntityTestDomainCallable::action_work_task));
	actions.push_back(callable_mp_static(&TemporalEntityTestDomainCallable::action_use_tool));
	domain->add_actions(actions);

	// Register task methods
	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&TemporalEntityTestDomainCallable::task_complete_work));
	domain->add_task_methods("complete_work", task_methods);

	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_max_depth(50);

	// Setup initial state with entity capabilities
	Dictionary state_dict;
	Dictionary worker;
	Dictionary capabilities;
	capabilities["using_tools"] = true; // Worker has capability to use tools
	worker["capabilities"] = capabilities;
	state_dict["worker1"] = worker;

	// Setup todo list
	Array todo_list;
	Array task1;
	task1.push_back("complete_work");
	task1.push_back("worker1");
	task1.push_back("build_shelf");
	todo_list.push_back(task1);

	// Set temporal constraints - plan starts at time 0
	PlannerTimeRange time_range;
	time_range.set_start_time(0LL);
	plan->set_time_range(time_range);

	// Attach temporal constraint to action_work_task (takes 10 seconds)
	Dictionary temporal_constraint;
	temporal_constraint["duration"] = static_cast<int64_t>(10000000LL); // 10 seconds in microseconds
	Array work_task_action;
	work_task_action.push_back("action_work_task");
	work_task_action.push_back("worker1");
	work_task_action.push_back("build_shelf");
	plan->attach_metadata(work_task_action, temporal_constraint);

	// Attach entity capability requirement to task
	Dictionary entity_constraints;
	entity_constraints["type"] = "worker1";
	Array required_capabilities;
	required_capabilities.push_back("using_tools");
	entity_constraints["capabilities"] = required_capabilities;
	plan->attach_metadata(task1, Dictionary(), entity_constraints);

	// Find plan
	Ref<PlannerResult> result = plan->find_plan(state_dict, todo_list);

	// Verify result is valid
	CHECK(result.is_valid());
	CHECK(result->get_success());
	if (result.is_valid() && result->get_success()) {
		Array plan_result = result->extract_plan();
		// Plan should contain at least 2 actions: use_tool and work_task
		CHECK(plan_result.size() >= 2);

		// Verify plan contains expected actions
		bool found_use_tool = false;
		bool found_work_task = false;
		for (int i = 0; i < plan_result.size(); i++) {
			Array action = plan_result[i];
			if (action.size() > 0) {
				String action_name = action[0];
				if (action_name == "action_use_tool") {
					found_use_tool = true;
				} else if (action_name == "action_work_task") {
					found_work_task = true;
				}
			}
		}
		CHECK(found_use_tool);
		CHECK(found_work_task);
	}
}

} // namespace TestIPyHOPCompatibility
