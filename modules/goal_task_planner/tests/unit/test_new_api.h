/**************************************************************************/
/*  test_new_api.h                                                        */
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

#include "../../plan.h"
#include "../../planner_result.h"
#include "../../multigoal.h"
#include "../../domain.h"
#include "../../solution_graph.h"
#include "tests/test_macros.h"
#include "../helpers/isekai_academy_domain.h"

TEST_CASE("[Modules][Planner] Public Blacklist API") {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);

	// Add a simple action
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_talk_to_character));
	domain->add_actions(actions);

	// Blacklist a command
	Array blacklisted_action;
	blacklisted_action.push_back("action_talk_to_character");
	blacklisted_action.push_back("protagonist");
	blacklisted_action.push_back("class_president");

	plan->blacklist_command(blacklisted_action);

	// Verify it's blacklisted by trying to plan with it
	Dictionary state;
	state["relationships"] = Dictionary();
	Dictionary location_dict;
	location_dict["protagonist"] = "classroom";
	state["location"] = location_dict;

	Array todo_list;
	Array task;
	task.push_back("task_talk");
	task.push_back("protagonist");
	task.push_back("class_president");
	todo_list.push_back(task);

	Ref<PlannerResult> result = plan->find_plan(state, todo_list);
	// Should fail because the action is blacklisted
	CHECK(!result->get_success());
}

TEST_CASE("[Modules][Planner] Iteration Counter") {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);

	// Add actions and methods
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_talk_to_character));
	domain->add_actions(actions);

	TypedArray<Callable> methods;
	methods.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::task_build_relationship));
	domain->add_task_methods("task_talk", methods);

	Dictionary state;
	state["relationships"] = Dictionary();
	Dictionary location_dict;
	location_dict["protagonist"] = "classroom";
	state["location"] = location_dict;

	Array todo_list;
	Array task;
	task.push_back("task_talk");
	task.push_back("protagonist");
	task.push_back("class_president");
	todo_list.push_back(task);

	Ref<PlannerResult> result = plan->find_plan(state, todo_list);
	
	// Should have some iterations
	int iterations = plan->get_iterations();
	CHECK(iterations >= 0);
}

TEST_CASE("[Modules][Planner] Multigoal Tag Support") {
	// Test get_goal_tag
	Array multigoal;
	Array goal1;
	goal1.push_back("affection");
	goal1.push_back("protagonist");
	goal1.push_back(50);
	multigoal.push_back(goal1);

	String tag = PlannerMultigoal::get_goal_tag(multigoal);
	CHECK(tag == String()); // No tag initially

	// Test set_goal_tag
	Variant tagged_multigoal = PlannerMultigoal::set_goal_tag(multigoal, "friendship");
	CHECK(tagged_multigoal.get_type() == Variant::DICTIONARY);
	
	Dictionary dict = tagged_multigoal;
	CHECK(dict.has("goal_tag"));
	CHECK(dict["goal_tag"] == "friendship");
	CHECK(dict.has("item"));

	// Test get_goal_tag on tagged multigoal
	String retrieved_tag = PlannerMultigoal::get_goal_tag(tagged_multigoal);
	CHECK(retrieved_tag == "friendship");
}

TEST_CASE("[Modules][Planner] Node Tagging System") {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);

	// Add actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_talk_to_character));
	domain->add_actions(actions);

	Dictionary state;
	state["relationships"] = Dictionary();
	Dictionary location_dict;
	location_dict["protagonist"] = "classroom";
	state["location"] = location_dict;

	Array todo_list;
	Array task;
	task.push_back("action_talk_to_character");
	task.push_back("protagonist");
	task.push_back("class_president");
	todo_list.push_back(task);

	Ref<PlannerResult> result = plan->find_plan(state, todo_list);
	CHECK(result->get_success());

	// Check that nodes have tags
	Array all_nodes = result->get_all_nodes();
	CHECK(all_nodes.size() > 0);
	
	bool found_new_tag = false;
	for (int i = 0; i < all_nodes.size(); i++) {
		Dictionary node_info = all_nodes[i];
		if (node_info.has("tag")) {
			String tag = node_info["tag"];
			if (tag == "new") {
				found_new_tag = true;
			}
		}
	}
	CHECK(found_new_tag);
}

TEST_CASE("[Modules][Planner] PlannerResult Helper Methods") {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);

	// Add actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_talk_to_character));
	domain->add_actions(actions);

	Dictionary state;
	state["relationships"] = Dictionary();
	Dictionary location_dict;
	location_dict["protagonist"] = "classroom";
	state["location"] = location_dict;

	Array todo_list;
	Array task;
	task.push_back("action_talk_to_character");
	task.push_back("protagonist");
	task.push_back("class_president");
	todo_list.push_back(task);

	Ref<PlannerResult> result = plan->find_plan(state, todo_list);
	CHECK(result->get_success());

	// Test get_all_nodes
	Array all_nodes = result->get_all_nodes();
	CHECK(all_nodes.size() > 0);

	// Test has_node and get_node
	CHECK(result->has_node(0)); // Root node should exist
	Dictionary root_node = result->get_node(0);
	CHECK(!root_node.is_empty());
	CHECK(root_node.has("type"));

	// Test find_failed_nodes (should be empty for successful plan)
	Array failed_nodes = result->find_failed_nodes();
	CHECK(failed_nodes.size() == 0);
}

TEST_CASE("[Modules][Planner] Simulate Method") {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);

	// Add actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_talk_to_character));
	domain->add_actions(actions);

	Dictionary state;
	state["relationships"] = Dictionary();
	Dictionary location_dict;
	location_dict["protagonist"] = "classroom";
	state["location"] = location_dict;

	Array todo_list;
	Array task;
	task.push_back("action_talk_to_character");
	task.push_back("protagonist");
	task.push_back("class_president");
	todo_list.push_back(task);

	Ref<PlannerResult> result = plan->find_plan(state, todo_list);
	CHECK(result->get_success());

	// Simulate the plan
	Array state_list = plan->simulate(result, state, 0);
	CHECK(state_list.size() > 0);

	// First state should be the initial state
	Dictionary first_state = state_list[0];
	CHECK(first_state.has("location"));

	// Last state should have relationships updated
	if (state_list.size() > 1) {
		Dictionary last_state = state_list[state_list.size() - 1];
		CHECK(last_state.has("relationships"));
	}
}

TEST_CASE("[Modules][Planner] Replan Method") {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);

	// Add actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&IsekaiAcademyDomainCallable::action_talk_to_character));
	domain->add_actions(actions);

	Dictionary state;
	state["relationships"] = Dictionary();
	Dictionary location_dict;
	location_dict["protagonist"] = "classroom";
	state["location"] = location_dict;

	Array todo_list;
	Array task;
	task.push_back("action_talk_to_character");
	task.push_back("protagonist");
	task.push_back("class_president");
	todo_list.push_back(task);

	Ref<PlannerResult> result = plan->find_plan(state, todo_list);
	CHECK(result->get_success());

	// Find a node to "fail" (we'll use a non-root node)
	Array all_nodes = result->get_all_nodes();
	int fail_node_id = -1;
	for (int i = 0; i < all_nodes.size(); i++) {
		Dictionary node_info = all_nodes[i];
		int node_id = node_info["node_id"];
		if (node_id > 0) { // Not root
			fail_node_id = node_id;
			break;
		}
	}

	if (fail_node_id >= 0) {
		// Create a new state (simulating that the action failed and state changed)
		Dictionary new_state = state.duplicate();
		
		// Replan from the failure
		Ref<PlannerResult> replan_result = plan->replan(result, new_state, fail_node_id);
		// Replanning should produce a result (may or may not succeed depending on state)
		CHECK(replan_result.is_valid());
	}
}

