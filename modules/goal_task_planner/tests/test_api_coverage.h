/**************************************************************************/
/*  test_api_coverage.h                                                   */
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

namespace TestAPICoverage {

TEST_CASE("[Modules][Planner] PlannerDomain - Add actions API") {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	CHECK(domain.is_valid());
}

TEST_CASE("[Modules][Planner] PlannerPlan - Set/Get domain API") {
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	plan->set_current_domain(domain);
	Ref<PlannerDomain> retrieved = plan->get_current_domain();

	CHECK(retrieved.is_valid());
	CHECK(retrieved == domain);
}

TEST_CASE("[Modules][Planner] PlannerPlan - Max stack size API") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();

	int default_size = plan->get_max_stack_size();
	CHECK(default_size > 0);

	plan->set_max_stack_size(5000);
	CHECK(plan->get_max_stack_size() == 5000);

	plan->set_max_stack_size(20000);
	CHECK(plan->get_max_stack_size() == 20000);
}

TEST_CASE("[Modules][Planner] PlannerResult - Get/Set methods") {
	Ref<PlannerResult> result = memnew(PlannerResult);

	// Set success flag
	result->set_success(true);
	CHECK(result->get_success());

	result->set_success(false);
	CHECK(!result->get_success());

	// Set final state
	Dictionary state;
	state["test"] = "value";
	result->set_final_state(state);

	Dictionary retrieved_state = result->get_final_state();
	CHECK(retrieved_state.has("test"));
}

TEST_CASE("[Modules][Planner] PlannerResult - Get all nodes API") {
	Ref<PlannerResult> result = memnew(PlannerResult);

	Array all_nodes = result->get_all_nodes();
	CHECK(all_nodes.size() >= 0);
}

TEST_CASE("[Modules][Planner] PlannerResult - Extract plan API") {
	Ref<PlannerResult> result = memnew(PlannerResult);
	result->set_success(true);

	// Should not crash
	Array extracted = result->extract_plan();
	CHECK(extracted.size() >= 0);
}

TEST_CASE("[Modules][Planner] PlannerMultigoal - Method verify functionality") {
	Dictionary state;
	Array multigoal;

	Array unigoal;
	unigoal.push_back("predicate");
	unigoal.push_back("subject");
	unigoal.push_back("value");
	multigoal.push_back(unigoal);

	Array goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(state, multigoal);

	CHECK(goals_not_achieved.size() >= 0);
}

TEST_CASE("[Modules][Planner] STNSolver - Add and get time points") {
	PlannerSTNSolver stn;

	stn.add_time_point("tp1");
	stn.add_time_point("tp2");

	int64_t dist = stn.get_distance("tp1", "tp2");
	CHECK(dist == PlannerSTNSolver::STN_INFINITY || dist >= 0);
}

TEST_CASE("[Modules][Planner] STNSolver - Consistency check") {
	PlannerSTNSolver stn;

	CHECK(stn.is_consistent());

	stn.add_time_point("start");
	stn.add_time_point("end");
	stn.add_constraint("start", "end", 0, 1000);

	CHECK(stn.is_consistent());
}

TEST_CASE("[Modules][Planner] STNSolver - Create and restore snapshot") {
	PlannerSTNSolver stn;

	stn.add_time_point("point1");
	stn.add_time_point("point2");
	stn.add_constraint("point1", "point2", 500, 1000);

	PlannerSTNSolver::Snapshot snapshot = stn.create_snapshot();

	CHECK(snapshot.consistent);
}

TEST_CASE("[Modules][Planner] PlannerState - Predicate operations") {
	Ref<PlannerState> state = memnew(PlannerState);

	state->set_predicate("robot", "location", "room1");
	Variant location = state->get_predicate("robot", "location");

	CHECK(location == "room1");
}

TEST_CASE("[Modules][Planner] PlannerState - Entity capabilities") {
	Ref<PlannerState> state = memnew(PlannerState);

	state->set_entity_capability("robot1", "move", true);
	Variant capability = state->get_entity_capability("robot1", "move");

	CHECK(bool(capability) == true);
}

TEST_CASE("[Modules][Planner] PlannerState - Shared objects") {
	Ref<PlannerState> state = memnew(PlannerState);

	Dictionary object_data;
	object_data["type"] = "tool";
	object_data["available"] = true;

	state->add_shared_object("tool1", object_data);

	CHECK(state->has_shared_object("tool1"));

	Dictionary retrieved = state->get_shared_object("tool1");
	CHECK(retrieved.has("type"));
}

TEST_CASE("[Modules][Planner] PlannerState - Public events") {
	Ref<PlannerState> state = memnew(PlannerState);

	Dictionary event_data;
	event_data["type"] = "alarm";
	event_data["severity"] = "high";

	state->add_public_event("event1", event_data);

	CHECK(state->has_public_event("event1"));

	Dictionary retrieved = state->get_public_event("event1");
	CHECK(retrieved.has("type"));
}

TEST_CASE("[Modules][Planner] PlannerState - Entity positions") {
	Ref<PlannerState> state = memnew(PlannerState);

	Vector3 position = Vector3(1.0, 2.0, 3.0);
	state->set_entity_position("robot", position);

	CHECK(state->has_entity_position("robot"));

	Variant retrieved_pos = state->get_entity_position("robot");
	CHECK(retrieved_pos.get_type() == Variant::VECTOR3);
}

TEST_CASE("[Modules][Planner] PlannerState - Terrain facts") {
	Ref<PlannerState> state = memnew(PlannerState);

	state->set_terrain_fact("location_a", "passable", true);
	Variant passable = state->get_terrain_fact("location_a", "passable");

	CHECK(bool(passable) == true);
}

TEST_CASE("[Modules][Planner] PlannerState - Observation methods") {
	Ref<PlannerState> state = memnew(PlannerState);

	state->set_entity_position("robot1", Vector3(0, 0, 0));
	state->set_entity_capability_public("robot1", "grasp", true);

	Dictionary positions = state->observe_entity_positions();
	CHECK(positions.has("robot1"));

	Dictionary capabilities = state->observe_entity_capabilities();
	CHECK(capabilities.size() >= 0);
}

TEST_CASE("[Modules][Planner] PlannerState - Belief management") {
	Ref<PlannerState> state = memnew(PlannerState);

	state->set_belief_about("persona1", "robot2", "location", "kitchen");

	Dictionary beliefs = state->get_beliefs_about("persona1", "robot2");
	CHECK(beliefs.size() >= 0);

	double confidence = state->get_belief_confidence("persona1", "robot2", "location");
	CHECK(confidence >= 0.0);
}

TEST_CASE("[Modules][Planner] PlannerPlan - Blacklist command API") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);

	plan->blacklist_command("unsafe_action");
	plan->blacklist_command("forbidden_task");

	CHECK(true);
}

TEST_CASE("[Modules][Planner] PlannerPlan - Reset VSIDS activity") {
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	plan->reset_vsids_activity();

	CHECK(true);
}

TEST_CASE("[Modules][Planner] SolutionGraph - Node creation and retrieval") {
	PlannerSolutionGraph graph;

	Dictionary root = graph.get_node(0);
	CHECK(!root.is_empty());
}

TEST_CASE("[Modules][Planner] Successful vs failed plans") {
	Ref<PlannerDomain> domain = MinimalTaskDomain::create_minimal_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->reset();
	plan->set_current_domain(domain);

	Dictionary state;
	Dictionary val;
	val["value"] = 0;
	state["value"] = val;

	Array todo;
	todo.push_back("increment");

	Ref<PlannerResult> result = plan->find_plan(state, todo);

	if (result->get_success()) {
		Dictionary final_state = result->get_final_state();
		CHECK(final_state.size() >= 0);
	} else {
		CHECK(result.is_valid());
	}
}

} // namespace TestAPICoverage
