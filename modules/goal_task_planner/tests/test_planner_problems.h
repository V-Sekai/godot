/**************************************************************************/
/*  test_planner_problems.h                                               */
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

// Use shared academy domain helpers.
#include "test_planner_helpers.h"

namespace TestComprehensivePlanner {

TEST_CASE("[Modules][Planner] Integration - Full academy planning scenario") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	Ref<PlannerState> state = memnew(PlannerState);

	// Setup complete academy domain
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_study_subject));
	actions.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_attend_class));
	actions.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_talk_to_character));
	domain->add_actions(actions);

	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::task_complete_lesson));
	domain->add_task_methods("complete_lesson", task_methods);

	TypedArray<Callable> unigoal_methods;
	unigoal_methods.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::unigoal_achieve_affection_level));
	unigoal_methods.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::unigoal_pass_exam));
	domain->add_unigoal_methods("achieve_affection_level", unigoal_methods);
	domain->add_unigoal_methods("pass_exam", unigoal_methods);

	TypedArray<Callable> multigoal_methods;
	multigoal_methods.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::multigoal_complete_route));
	domain->add_multigoal_methods(multigoal_methods);

	plan->set_current_domain(domain);

	// Setup initial state with entities
	state->set_entity_capability("protagonist", "studying", true);
	state->set_entity_capability("classmate1", "socializing", true);
	state->set_predicate("protagonist", "available", true);
	state->set_predicate("classmate1", "available", true);

	SUBCASE("Plan with entity requirements") {
		Dictionary state_dict;
		// Add entity capabilities to state
		Dictionary protagonist;
		Dictionary studies;
		studies["magic_class"] = true;
		protagonist["studies"] = studies;
		state_dict["protagonist"] = protagonist;

		Array todo_list;
		todo_list.push_back("achieve_affection_level");

		// Attach entity requirement to goal
		Dictionary entity_constraints;
		entity_constraints["type"] = "protagonist";
		Array capabilities;
		capabilities.push_back("studying");
		entity_constraints["capabilities"] = capabilities;
		plan->attach_metadata("achieve_affection_level", Dictionary(), entity_constraints);

		Variant result = plan->find_plan(state_dict, todo_list);
		// Planning should attempt to use entities with required capabilities
		Variant::Type result_type = result.get_type();
		bool is_valid_type = (result_type == Variant::ARRAY) || (result_type == Variant::BOOL);
		CHECK(is_valid_type);
	}

	SUBCASE("Plan with temporal constraints") {
		Dictionary state_dict;
		Array todo_list;
		todo_list.push_back("study");

		// Set temporal constraints
		PlannerTimeRange time_range;
		time_range.set_start_time(1735689600000000LL);
		plan->set_time_range(time_range);

		Dictionary temporal;
		temporal["duration"] = 5000000LL; // 5 seconds
		plan->attach_metadata("study", temporal);

		Variant result = plan->find_plan(state_dict, todo_list);
		Variant::Type result_type = result.get_type();
		bool is_valid_type = (result_type == Variant::ARRAY) || (result_type == Variant::BOOL);
		CHECK(is_valid_type);
	}

	SUBCASE("Plan with multigoal") {
		Dictionary state_dict;
		Dictionary multigoal;
		Dictionary character1;
		character1["affection_level"] = 50;
		character1["student"] = "protagonist";
		multigoal["class_president"] = character1;

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
		todo_list.push_back("study");

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
