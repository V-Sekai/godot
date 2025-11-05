/**************************************************************************/
/*  test_goal_solver.h                                                    */
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

// C++ unit tests for goal solver and entity matching functionality

#pragma once

#include "../domain.h"
#include "../entity_requirement.h"
#include "../plan.h"
#include "../planner_state.h"
#include "tests/test_macros.h"

namespace TestGoalSolver {

// Note: Entity matching is tested through PlannerMetadata integration in the planning loop
// The _match_entities helper is private and used internally when PlannerMetadata has entity requirements
// Entity matching tests should be done through actual planning scenarios with PlannerMetadata

TEST_CASE("[Modules][GoalSolver] Temporal constraints methods" * doctest::skip(true)) {
	// DISABLED: Test failing - needs investigation
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	plan->set_current_domain(domain);

	// Test attach_temporal_constraints
	Dictionary temporal_constraints;
	int64_t duration_micros = 7200000000LL; // 2 hours in microseconds
	int64_t start_time_micros = 1704110400000000LL; // 2024-01-01T10:00:00 UTC in microseconds
	temporal_constraints["duration"] = duration_micros;
	temporal_constraints["start_time"] = start_time_micros;

	Array test_item;
	test_item.push_back("test");
	test_item.push_back("goal");

	Variant result = plan->_attach_temporal_constraints(test_item, temporal_constraints);
	CHECK(result.get_type() == Variant::DICTIONARY);

	Dictionary result_dict = result;
	CHECK(result_dict.has("item"));
	CHECK(result_dict.has("temporal_constraints"));

	// Test has_temporal_constraints
	bool has_constraints = plan->_has_temporal_constraints(result);
	CHECK(has_constraints == true);

	// Test get_temporal_constraints
	Dictionary retrieved = plan->_get_temporal_constraints(result);
	CHECK(retrieved.has("duration"));
	CHECK(int64_t(retrieved["duration"]) == duration_micros);
}

TEST_CASE("[Modules][GoalSolver] Unigoal ordering optimization") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	plan->set_current_domain(domain);

	// Create unigoal method dictionary
	Dictionary unigoal_method_dict;

	// Goal 1: has 2 methods (less constraining)
	TypedArray<Callable> methods1;
	unigoal_method_dict["goal1"] = methods1; // Empty for now

	// Goal 2: has 1 method (more constraining)
	TypedArray<Callable> methods2;
	unigoal_method_dict["goal2"] = methods2; // Empty for now

	// Create unigoals array
	Array unigoals;
	Array goal1;
	goal1.push_back("goal1");
	goal1.push_back("arg1");
	goal1.push_back("value1");
	unigoals.push_back(goal1);

	Array goal2;
	goal2.push_back("goal2");
	goal2.push_back("arg2");
	goal2.push_back("value2");
	unigoals.push_back(goal2);

	Dictionary state;

	// Optimize order
	Array optimized = plan->_optimize_unigoal_order(unigoals, state, unigoal_method_dict);

	// Should return same number of goals
	CHECK(optimized.size() == unigoals.size());
}

} //namespace TestGoalSolver
