/**************************************************************************/
/*  test_bt_planning.h                                                    */
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

/**
 * test_bt_planning.h
 * Unit tests for BT planning nodes and get_debug_plan_detail / BehaviorTreeData debug_plan_detail.
 *
 * =============================================================================
 * Copyright (c) 2023-present Serhii Snitsaruk and the LimboAI contributors.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#include "limbo_test.h"

#include "modules/limboai/blackboard/blackboard.h"
#include "modules/limboai/bt/tasks/bt_action.h"
#include "modules/limboai/bt/tasks/planning/bt_run_planner.h"
#include "modules/limboai/planning/src/domain.h"
#include "modules/limboai/planning/src/plan.h"
#include "modules/limboai/planning/src/planner_state.h"
#include "modules/limboai/tests/planning_test_domains.h"

#ifdef TOOLS_ENABLED
#include "modules/limboai/editor/debugger/behavior_tree_data.h"
#endif

namespace TestBTPlanning {

TEST_CASE("[Modules][LimboAI][Planner] get_debug_plan_detail") {
	SUBCASE("BTAction returns empty when not initialized") {
		Ref<BTAction> action = memnew(BTAction);
		Dictionary d = action->get_debug_plan_detail();
		CHECK(d.is_empty());
	}
}

#ifdef TOOLS_ENABLED
TEST_CASE("[Modules][LimboAI][Planner] BehaviorTreeData serialize/deserialize with debug_plan_detail") {
	// Build array in serialize format: [bt_instance_id, node_owner_path, source_bt_path, per-task: id, name, is_custom, num_children, status, elapsed, type_name, script_path, debug_plan_detail]
	Array arr;
	arr.push_back(12345); // INT for deserialize
	arr.push_back(NodePath());
	arr.push_back(String("res://test.tres"));
	arr.push_back(1); // task id
	arr.push_back(String("Run Planner"));
	arr.push_back(false);
	arr.push_back(0);
	arr.push_back(int(BTTask::SUCCESS));
	arr.push_back(0.5);
	arr.push_back(String("BTRunPlanner"));
	arr.push_back(String());
	Dictionary debug_detail;
	debug_detail["plan"] = Array();
	debug_detail["plan_index"] = 0;
	debug_detail["solution_graph"] = Dictionary();
	arr.push_back(debug_detail);

	Ref<BehaviorTreeData> data = BehaviorTreeData::deserialize(arr);
	REQUIRE(data.is_valid());
	CHECK_EQ(data->tasks.size(), 1);
	const BehaviorTreeData::TaskData &task0 = data->tasks.get(0);
	CHECK(task0.debug_plan_detail.has("plan"));
	CHECK(task0.debug_plan_detail.has("plan_index"));
	CHECK(task0.debug_plan_detail.has("solution_graph"));
	CHECK_EQ(int(task0.debug_plan_detail["plan_index"]), 0);
}
#endif

TEST_CASE("[Modules][LimboAI][Planner] BTRunPlanner goal planning") {
	Node *dummy = memnew(Node);
	Ref<Blackboard> bb = memnew(Blackboard);
	Ref<PlannerState> state = memnew(PlannerState);
	state->set_blackboard(bb);
	// Unigoal: predicate "value", subject "value", desired value 1
	Array todo_list;
	todo_list.push_back("value");
	todo_list.push_back("value");
	todo_list.push_back(1);
	bb->set_var(StringName("todo_list"), todo_list);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(PlanningTestDomains::create_minimal_goal_domain());
	Ref<BTRunPlanner> run_planner = memnew(BTRunPlanner);
	run_planner->set_planner_plan(plan);
	run_planner->set_planner_state(state);
	run_planner->initialize(dummy, bb, dummy);
	BTTask::Status status = run_planner->execute(0.0);
	CHECK(status == BTTask::SUCCESS);
	Array plan_arr = bb->get_var(StringName("plan"), Array(), false);
	CHECK(plan_arr.size() >= 1);
	CHECK_EQ(int(bb->get_var(StringName("plan_index"), 0, false)), 0);
	memdelete(dummy);
}

TEST_CASE("[Modules][LimboAI][Planner] BTRunPlanner task HTN") {
	Node *dummy = memnew(Node);
	Ref<Blackboard> bb = memnew(Blackboard);
	Ref<PlannerState> state = memnew(PlannerState);
	state->set_blackboard(bb);
	Array todo_list;
	todo_list.push_back("increment");
	bb->set_var(StringName("todo_list"), todo_list);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(PlanningTestDomains::create_minimal_htn_domain());
	Ref<BTRunPlanner> run_planner = memnew(BTRunPlanner);
	run_planner->set_planner_plan(plan);
	run_planner->set_planner_state(state);
	run_planner->initialize(dummy, bb, dummy);
	BTTask::Status status = run_planner->execute(0.0);
	CHECK(status == BTTask::SUCCESS);
	Array plan_arr = bb->get_var(StringName("plan"), Array(), false);
	CHECK(plan_arr.size() >= 1);
	memdelete(dummy);
}

TEST_CASE("[Modules][LimboAI][Planner] BTRunPlanner backtracking") {
	Node *dummy = memnew(Node);
	Ref<Blackboard> bb = memnew(Blackboard);
	Ref<PlannerState> state = memnew(PlannerState);
	state->set_blackboard(bb);
	Array todo_list;
	todo_list.push_back("increment");
	bb->set_var(StringName("todo_list"), todo_list);
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(PlanningTestDomains::create_minimal_backtracking_domain());
	Ref<BTRunPlanner> run_planner = memnew(BTRunPlanner);
	run_planner->set_planner_plan(plan);
	run_planner->set_planner_state(state);
	run_planner->initialize(dummy, bb, dummy);
	BTTask::Status status = run_planner->execute(0.0);
	CHECK(status == BTTask::SUCCESS);
	Array plan_arr = bb->get_var(StringName("plan"), Array(), false);
	CHECK(plan_arr.size() >= 1);
	// Plan should be from second method: action_increment with arg 1
	if (plan_arr.size() >= 1) {
		Variant first = plan_arr[0];
		if (first.get_type() == Variant::ARRAY) {
			Array cmd = first;
			if (cmd.size() >= 1) {
				CHECK(String(cmd[0]) == "action_increment");
			}
		}
	}
	memdelete(dummy);
}

TEST_CASE("[Modules][LimboAI][Planner] BTRunPlanner metadata and entity capabilities with goal, HTN, backtracking") {
	Node *dummy = memnew(Node);
	Ref<Blackboard> bb = memnew(Blackboard);
	Ref<PlannerState> state = memnew(PlannerState);
	state->set_blackboard(bb);
	state->set_terrain_fact("loc1", "fact_key", 100);
	state->set_entity_capability("ent2", "health", 50);
	state->set_entity_capability_public("ent1", "speed", 5.0);

	SUBCASE("Goal + metadata/entity_caps") {
		Array todo_list;
		todo_list.push_back("value");
		todo_list.push_back("value");
		todo_list.push_back(1);
		bb->set_var(StringName("todo_list"), todo_list);
		Ref<PlannerPlan> plan = memnew(PlannerPlan);
		plan->set_current_domain(PlanningTestDomains::create_minimal_goal_domain());
		Ref<BTRunPlanner> run_planner = memnew(BTRunPlanner);
		run_planner->set_planner_plan(plan);
		run_planner->set_planner_state(state);
		run_planner->initialize(dummy, bb, dummy);
		CHECK(run_planner->execute(0.0) == BTTask::SUCCESS);
		CHECK(state->has_terrain_fact("loc1", "fact_key"));
		CHECK_EQ(int(state->get_terrain_fact("loc1", "fact_key")), 100);
		CHECK(state->has_entity("ent2"));
		CHECK_EQ(int(state->get_entity_capability("ent2", "health")), 50);
		CHECK(state->has_entity_capability_public("ent1", "speed"));
		CHECK(bb->get_var(StringName("plan"), Array(), false).get_type() == Variant::ARRAY);
	}
	SUBCASE("HTN + metadata/entity_caps") {
		Array todo_list;
		todo_list.push_back("increment");
		bb->set_var(StringName("todo_list"), todo_list);
		Ref<PlannerPlan> plan = memnew(PlannerPlan);
		plan->set_current_domain(PlanningTestDomains::create_minimal_htn_domain());
		Ref<BTRunPlanner> run_planner = memnew(BTRunPlanner);
		run_planner->set_planner_plan(plan);
		run_planner->set_planner_state(state);
		run_planner->initialize(dummy, bb, dummy);
		CHECK(run_planner->execute(0.0) == BTTask::SUCCESS);
		CHECK(state->has_terrain_fact("loc1", "fact_key"));
		CHECK(state->has_entity("ent2"));
		CHECK(state->has_entity_capability_public("ent1", "speed"));
	}
	SUBCASE("Backtracking + metadata/entity_caps") {
		Array todo_list;
		todo_list.push_back("increment");
		bb->set_var(StringName("todo_list"), todo_list);
		Ref<PlannerPlan> plan = memnew(PlannerPlan);
		plan->set_current_domain(PlanningTestDomains::create_minimal_backtracking_domain());
		Ref<BTRunPlanner> run_planner = memnew(BTRunPlanner);
		run_planner->set_planner_plan(plan);
		run_planner->set_planner_state(state);
		run_planner->initialize(dummy, bb, dummy);
		CHECK(run_planner->execute(0.0) == BTTask::SUCCESS);
		CHECK(state->has_terrain_fact("loc1", "fact_key"));
		CHECK(state->has_entity("ent2"));
		CHECK(state->has_entity_capability_public("ent1", "speed"));
	}
	memdelete(dummy);
}

} // namespace TestBTPlanning
