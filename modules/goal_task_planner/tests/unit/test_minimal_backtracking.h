/**************************************************************************/
/*  test_minimal_backtracking.h                                           */
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

#include "../../src/domain.h"
#include "../../src/plan.h"
#include "../../src/planner_result.h"
#include "../domains/minimal_backtracking_domain.h"
#include "core/variant/callable.h"
#include "tests/test_macros.h"

namespace TestMinimalBacktracking {

TEST_CASE("[Modules][Planner][MinimalBacktracking] Task with backtracking - first method fails, second succeeds") {
	// Minimal backtracking test: Task has two methods
	// First method returns Variant() (fails), second method returns action (succeeds)
	// This verifies backtracking works correctly

	Ref<PlannerDomain> domain = MinimalBacktrackingDomain::create_minimal_backtracking_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);
	plan->set_max_depth(10);
	plan->set_verbose(0);

	// Create initial state with value = 0
	Dictionary init_state;
	Dictionary value_dict;
	value_dict["value"] = 0;
	init_state["value"] = value_dict;
	Dictionary clean_init_state = init_state.duplicate(true);

	// Create todo list with increment task
	Array todo_list;
	todo_list.push_back("increment");

	// Plan should succeed after backtracking from first method to second
	Ref<PlannerResult> result = plan->find_plan(clean_init_state, todo_list);

	CHECK(result.is_valid());
	CHECK(result->get_success()); // Should succeed after backtracking

	// Extract plan and verify it contains the increment action
	Array plan_result = result->extract_plan();
	CHECK(plan_result.size() > 0); // Should have at least one action

	// Verify the action is correct
	if (plan_result.size() > 0) {
		Array first_action = plan_result[0];
		CHECK(first_action.size() == 2);
		CHECK(first_action[0] == "action_increment");
		CHECK(int(first_action[1]) == 1); // increment amount
	}

	// Verify final state
	Dictionary final_state = result->get_final_state();
	CHECK(final_state.has("value"));
	Dictionary final_value_dict = final_state["value"];
	CHECK(final_value_dict.has("value"));
	CHECK(int(final_value_dict["value"]) == 1); // Value should be incremented to 1
}

} // namespace TestMinimalBacktracking
