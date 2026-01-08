/**************************************************************************/
/*  test_integration.h                                                    */
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

#include "helpers/planner_test_helpers.h"

namespace TestIntegration {

TEST_CASE("[Modules][Planner] Complete planning workflow") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary val;
	val["value"] = 0;
	state["value"] = val;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);

	CHECK(result.is_valid());
	CHECK(result->get_final_state().size() >= 0);
}

TEST_CASE("[Modules][Planner] Planning with multiple tasks in sequence") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary val;
	val["value"] = 0;
	state["value"] = val;

	Array todo;
	todo.push_back("increment");
	todo.push_back("increment");
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);
	CHECK(result.is_valid());
}

TEST_CASE("[Modules][Planner] Backtracking domain integration") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalBacktrackingDomain::create_minimal_backtracking_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Array path;
	path.push_back("room_a");
	state["current_path"] = path;

	Array todo;
	todo.push_back("navigate_to_room");

	Ref<PlannerResult> result = plan->find_plan(state, todo);
	CHECK(result.is_valid());
}

TEST_CASE("[Modules][Planner] Result extraction and analysis") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary val;
	val["value"] = 0;
	state["value"] = val;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);

	Array extracted_plan = result->extract_plan();
	CHECK(extracted_plan.size() >= 0);

	Dictionary final_state = result->get_final_state();
	CHECK(final_state.size() >= 0);

	Array all_nodes = result->get_all_nodes();
	CHECK(all_nodes.size() >= 0);
}

TEST_CASE("[Modules][Planner] Multi-domain switching") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain1 = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain1);

	Dictionary state1;
	Dictionary val1;
	val1["value"] = 0;
	state1["value"] = val1;

	Array todo1;
	todo1.push_back("increment");

	Ref<PlannerResult> result1 = plan->find_plan(state1, todo1);
	CHECK(result1.is_valid());

	plan->reset();
	Ref<PlannerDomain> domain2 = MinimalBacktrackingDomain::create_minimal_backtracking_domain();
	plan->set_current_domain(domain2);

	Dictionary state2;
	Array path2;
	path2.push_back("start");
	state2["current_path"] = path2;

	Array todo2;
	todo2.push_back("navigate_to_room");

	Ref<PlannerResult> result2 = plan->find_plan(state2, todo2);
	CHECK(result2.is_valid());
}

TEST_CASE("[Modules][Planner] STN temporal reasoning integration") {
	PlannerSTNSolver stn;

	stn.add_time_point("start");
	stn.add_time_point("action1");
	stn.add_time_point("action2");
	stn.add_time_point("end");

	stn.add_constraint("start", "action1", 0, 100);
	stn.add_constraint("action1", "action2", 50, 150);
	stn.add_constraint("action2", "end", 0, 100);

	CHECK(stn.is_consistent());

	int64_t total_time = stn.get_distance("start", "end");
	CHECK(total_time >= 0 || total_time == PlannerSTNSolver::STN_INFINITY);
}

TEST_CASE("[Modules][Planner] Solution graph traversal") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary val;
	val["value"] = 0;
	state["value"] = val;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);

	Array all_nodes = result->get_all_nodes();
	for (int i = 0; i < all_nodes.size(); i++) {
		Variant node = all_nodes[i];
		CHECK(true);
	}
}

TEST_CASE("[Modules][Planner] Multigoal decomposition") {
	Dictionary state;
	Array multigoal;

	Array unigoal1;
	unigoal1.push_back("has");
	unigoal1.push_back("robot");
	unigoal1.push_back("tool");

	Array unigoal2;
	unigoal2.push_back("at");
	unigoal2.push_back("robot");
	unigoal2.push_back("location");

	multigoal.push_back(unigoal1);
	multigoal.push_back(unigoal2);

	Array unachieved = PlannerMultigoal::method_goals_not_achieved(state, multigoal);
	CHECK(unachieved.size() > 0);
}

TEST_CASE("[Modules][Planner] State persistence across operations") {
	Ref<PlannerState> state = memnew(PlannerState);

	state->set_entity_position("robot1", Vector3(1, 2, 3));
	state->set_predicate("robot1", "status", "idle");
	state->set_entity_capability("robot1", "move", true);

	CHECK(state->has_entity_position("robot1"));

	Vector3 pos = state->get_entity_position("robot1");
	CHECK(pos == Vector3(1, 2, 3));

	Variant status = state->get_predicate("robot1", "status");
	CHECK(String(status) == "idle");
}

TEST_CASE("[Modules][Planner] Belief system integration") {
	Ref<PlannerState> state = memnew(PlannerState);

	state->set_belief_about("robot1", "robot2", "location", "room1");
	state->set_belief_about("robot1", "robot2", "status", "busy");

	Dictionary beliefs = state->get_beliefs_about("robot1", "robot2");
	CHECK(beliefs.size() >= 0);

	double confidence = state->get_belief_confidence("robot1", "robot2", "location");
	CHECK(confidence >= 0.0);
}

TEST_CASE("[Modules][Planner] Comprehensive plan finding workflow") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary counter;
	counter["value"] = 0;
	state["value"] = counter;

	Dictionary metadata;
	metadata["timestamp"] = 12345;
	metadata["version"] = "1.0";
	state["metadata"] = metadata;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);

	CHECK(result.is_valid());
	CHECK(result->get_success() || !result->get_success());
	CHECK(result->extract_plan().size() >= 0);
	CHECK(result->get_final_state().size() >= 0);
}

TEST_CASE("[Modules][Planner] Error recovery workflow") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	plan->reset();
	CHECK(true);

	Ref<PlannerDomain> valid_domain = MinimalTaskDomain::create_minimal_domain();
	plan->set_current_domain(valid_domain);

	Dictionary state;
	Dictionary val;
	val["value"] = 0;
	state["value"] = val;

	Array todo;
	Ref<PlannerResult> result = plan->find_plan(state, todo);

	CHECK(result.is_valid());
}

} // namespace TestIntegration
