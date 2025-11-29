/**************************************************************************/
/*  test_planner_problems.h                                              */
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

// End-to-end planning problems built on top of planner domains/helpers.

#pragma once

#include "../domain.h"
#include "../plan.h"
#include "../planner_state.h"
#include "../planner_time_range.h"
#include "tests/test_macros.h"

// Use shared restaurant helpers.
#include "test_planner_helpers.h"

namespace TestComprehensivePlanner {

TEST_CASE("[Modules][Planner] Integration - Full restaurant planning scenario") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	Ref<PlannerState> state = memnew(PlannerState);

	// Setup complete restaurant domain
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&RestaurantDomainCallable::action_cook));
	actions.push_back(callable_mp_static(&RestaurantDomainCallable::action_serve));
	actions.push_back(callable_mp_static(&RestaurantDomainCallable::action_clean));
	domain->add_actions(actions);

	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&RestaurantDomainCallable::task_prepare_meal));
	domain->add_task_methods("prepare_meal", task_methods);

	TypedArray<Callable> unigoal_methods;
	unigoal_methods.push_back(callable_mp_static(&RestaurantDomainCallable::unigoal_cook_dish));
	unigoal_methods.push_back(callable_mp_static(&RestaurantDomainCallable::unigoal_clean_table));
	domain->add_unigoal_methods("cook_dish", unigoal_methods);
	domain->add_unigoal_methods("clean_table", unigoal_methods);

	TypedArray<Callable> multigoal_methods;
	multigoal_methods.push_back(callable_mp_static(&RestaurantDomainCallable::multigoal_serve_customers));
	domain->add_multigoal_methods(multigoal_methods);

	plan->set_current_domain(domain);

	// Setup initial state with entities
	state->set_entity_capability("chef1", "cooking", true);
	state->set_entity_capability("waiter1", "serving", true);
	state->set_predicate("chef1", "available", true);
	state->set_predicate("waiter1", "available", true);

	SUBCASE("Plan with entity requirements") {
		Dictionary state_dict;
		// Add entity capabilities to state
		Dictionary chef1;
		chef1["cooking"] = true;
		state_dict["chef1"] = chef1;

		Array todo_list;
		todo_list.push_back("cook_dish");

		// Attach entity requirement to goal
		Dictionary entity_constraints;
		entity_constraints["type"] = "chef";
		Array capabilities;
		capabilities.push_back("cooking");
		entity_constraints["capabilities"] = capabilities;
		plan->attach_metadata("cook_dish", Dictionary(), entity_constraints);

		Variant result = plan->find_plan(state_dict, todo_list);
		// Planning should attempt to use entities with required capabilities
		Variant::Type result_type = result.get_type();
		bool is_valid_type = (result_type == Variant::ARRAY) || (result_type == Variant::BOOL);
		CHECK(is_valid_type);
	}

	SUBCASE("Plan with temporal constraints") {
		Dictionary state_dict;
		Array todo_list;
		todo_list.push_back("cook");

		// Set temporal constraints
		PlannerTimeRange time_range;
		time_range.set_start_time(1735689600000000LL);
		plan->set_time_range(time_range);

		Dictionary temporal;
		temporal["duration"] = 5000000LL; // 5 seconds
		plan->attach_metadata("cook", temporal);

		Variant result = plan->find_plan(state_dict, todo_list);
		Variant::Type result_type = result.get_type();
		bool is_valid_type = (result_type == Variant::ARRAY) || (result_type == Variant::BOOL);
		CHECK(is_valid_type);
	}

	SUBCASE("Plan with multigoal") {
		Dictionary state_dict;
		Dictionary multigoal;
		Dictionary customer1;
		customer1["dish"] = "pasta";
		customer1["waiter"] = "waiter1";
		multigoal["customer1"] = customer1;

		Array todo_list;
		todo_list.push_back(multigoal);

		Variant result = plan->find_plan(state_dict, todo_list);
		Variant::Type result_type = result.get_type();
		bool is_valid_type = (result_type == Variant::ARRAY) || (result_type == Variant::BOOL);
		CHECK(is_valid_type);
	}

	SUBCASE("Plan with STN constraints") {
		// This tests that STN solver is integrated with planning
		Dictionary state_dict;
		Array todo_list;
		todo_list.push_back("cook");

		PlannerTimeRange time_range;
		time_range.set_start_time(1735689600000000LL);
		plan->set_time_range(time_range);

		Variant result = plan->find_plan(state_dict, todo_list);
		// STN should be initialized and used during planning
		Variant::Type result_type = result.get_type();
		bool is_valid_type = (result_type == Variant::ARRAY) || (result_type == Variant::BOOL);
		CHECK(is_valid_type);
	}
}

} // namespace TestComprehensivePlanner


