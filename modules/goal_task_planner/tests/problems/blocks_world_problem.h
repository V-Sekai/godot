/**************************************************************************/
/*  blocks_world_problem.h                                                 */
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

#include "../../domain.h"
#include "../../plan.h"
#include "../../planner_result.h"
#include "../../solution_graph.h"
#include "../domains/blocks_world_domain.h"
#include "core/variant/callable.h"
#include "tests/test_macros.h"

namespace TestBlocksWorld {

// Helper: Verify that a plan achieves the goal state
bool verify_plan_achieves_goal(Ref<PlannerPlan> plan, Dictionary init_state, Dictionary goal, Array plan_result) {
	// Simulate plan execution
	Ref<PlannerResult> temp_result = memnew(PlannerResult);
	temp_result->set_success(true);
	temp_result->set_final_state(init_state);
	
	// Create a minimal solution graph for simulation with proper successors links
	Dictionary graph;
	Dictionary root_node;
	root_node["type"] = static_cast<int>(PlannerNodeType::TYPE_ROOT);
	root_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
	
	// Link action nodes in sequence: root -> action1 -> action2 -> ... -> actionN
	Array root_successors;
	if (plan_result.size() > 0) {
		root_successors.push_back(1); // First action node
	}
	root_node["successors"] = root_successors;
	graph[0] = root_node;
	
	// Add action nodes with proper successors links
	for (int i = 0; i < plan_result.size(); i++) {
		Dictionary action_node;
		action_node["type"] = static_cast<int>(PlannerNodeType::TYPE_ACTION);
		action_node["status"] = static_cast<int>(PlannerNodeStatus::STATUS_CLOSED);
		action_node["info"] = plan_result[i];
		
		// Link to next action (if not last)
		Array action_successors;
		if (i < plan_result.size() - 1) {
			action_successors.push_back(i + 2); // Next action node (i+1+1 because root is 0)
		}
		action_node["successors"] = action_successors;
		
		graph[i + 1] = action_node;
	}
	
	temp_result->set_solution_graph(graph);
	
	// Simulate plan execution
	Array state_sequence = plan->simulate(temp_result, init_state, 0);
	if (state_sequence.is_empty()) {
		return false;
	}
	
	// Get final state
	Dictionary final_state = state_sequence[state_sequence.size() - 1];
	
	// Check if goal is achieved
	Dictionary goal_pos = goal.get("pos", Dictionary());
	Dictionary goal_clear = goal.get("clear", Dictionary());
	Dictionary goal_holding = goal.get("holding", Dictionary());
	
	Dictionary final_pos = final_state.get("pos", Dictionary());
	Dictionary final_clear = final_state.get("clear", Dictionary());
	Dictionary final_holding = final_state.get("holding", Dictionary());
	
	// Check all goal positions
	Array goal_pos_keys = goal_pos.keys();
	for (int i = 0; i < goal_pos_keys.size(); i++) {
		Variant block = goal_pos_keys[i];
		Variant goal_location = goal_pos[block];
		if (!final_pos.has(block) || final_pos[block] != goal_location) {
			return false;
		}
	}
	
	// Check goal clear states (if specified)
	if (!goal_clear.is_empty()) {
		Array goal_clear_keys = goal_clear.keys();
		for (int i = 0; i < goal_clear_keys.size(); i++) {
			Variant block = goal_clear_keys[i];
			bool goal_clear_val = goal_clear[block];
			if (final_clear.has(block) && (bool)final_clear[block] != goal_clear_val) {
				return false;
			}
		}
	}
	
	// Check holding state (if specified)
	if (!goal_holding.is_empty() && goal_holding.has("hand")) {
		bool goal_hand = goal_holding["hand"];
		if (final_holding.has("hand") && (bool)final_holding["hand"] != goal_hand) {
			return false;
		}
	}
	
	return true;
}

// Helper: Create initial state 1 (small problem)
Dictionary create_init_state_1() {
	Dictionary state;
	
	Dictionary pos;
	pos["a"] = "b";
	pos["b"] = "table";
	pos["c"] = "table";
	state["pos"] = pos;
	
	Dictionary clear;
	clear["c"] = true;
	clear["b"] = false;
	clear["a"] = true;
	state["clear"] = clear;
	
	Dictionary holding;
	holding["hand"] = false;
	state["holding"] = holding;
	
	return state;
}

// Helper: Create goal 1a
Dictionary create_goal1a() {
	Dictionary goal;
	
	Dictionary pos;
	pos["c"] = "b";
	pos["b"] = "a";
	pos["a"] = "table";
	goal["pos"] = pos;
	
	Dictionary clear;
	clear["c"] = true;
	clear["b"] = false;
	clear["a"] = false;
	goal["clear"] = clear;
	
	Dictionary holding;
	holding["hand"] = false;
	goal["holding"] = holding;
	
	return goal;
}

// Helper: Create initial state 3 (large benchmark problem)
Dictionary create_init_state_3() {
	Dictionary state;
	
	Dictionary pos;
	// Stack 1: 1->12->13->table
	pos[1] = 12;
	pos[12] = 13;
	pos[13] = "table";
	// Stack 2: 11->10->5->4->14->15->table
	pos[11] = 10;
	pos[10] = 5;
	pos[5] = 4;
	pos[4] = 14;
	pos[14] = 15;
	pos[15] = "table";
	// Stack 3: 9->8->7->6->table
	pos[9] = 8;
	pos[8] = 7;
	pos[7] = 6;
	pos[6] = "table";
	// Stack 4: 19->18->17->16->3->2->table
	pos[19] = 18;
	pos[18] = 17;
	pos[17] = 16;
	pos[16] = 3;
	pos[3] = 2;
	pos[2] = "table";
	
	state["pos"] = pos;
	
	Dictionary clear;
	for (int i = 1; i < 20; i++) {
		clear[i] = false;
	}
	clear[1] = true;
	clear[11] = true;
	clear[9] = true;
	clear[19] = true;
	state["clear"] = clear;
	
	Dictionary holding;
	holding["hand"] = false;
	state["holding"] = holding;
	
	return state;
}

// Helper: Create goal 3 (large benchmark problem)
Dictionary create_goal3() {
	Dictionary goal;
	
	Dictionary pos;
	pos[15] = 13;
	pos[13] = 8;
	pos[8] = 9;
	pos[9] = 4;
	pos[4] = "table";
	pos[12] = 2;
	pos[2] = 3;
	pos[3] = 16;
	pos[16] = 11;
	pos[11] = 7;
	pos[7] = 6;
	pos[6] = "table";
	goal["pos"] = pos;
	
	Dictionary clear;
	clear[17] = true;
	clear[15] = true;
	clear[12] = true;
	goal["clear"] = clear;
	
	return goal;
}

// Helper: Setup blocks world domain
Ref<PlannerDomain> create_blocks_world_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	
	// Register actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&BlocksWorldDomainCallable::action_pickup));
	actions.push_back(callable_mp_static(&BlocksWorldDomainCallable::action_unstack));
	actions.push_back(callable_mp_static(&BlocksWorldDomainCallable::action_putdown));
	actions.push_back(callable_mp_static(&BlocksWorldDomainCallable::action_stack));
	domain->add_actions(actions);
	
	// Register task methods
	TypedArray<Callable> move_blocks_methods;
	move_blocks_methods.push_back(callable_mp_static(&BlocksWorldDomainCallable::task_move_blocks));
	domain->add_task_methods("move_blocks", move_blocks_methods);
	
	TypedArray<Callable> move_one_methods;
	move_one_methods.push_back(callable_mp_static(&BlocksWorldDomainCallable::task_move_one));
	domain->add_task_methods("move_one", move_one_methods);
	
	TypedArray<Callable> get_methods;
	get_methods.push_back(callable_mp_static(&BlocksWorldDomainCallable::task_get));
	domain->add_task_methods("get", get_methods);
	
	TypedArray<Callable> put_methods;
	put_methods.push_back(callable_mp_static(&BlocksWorldDomainCallable::task_put));
	domain->add_task_methods("put", put_methods);
	
	return domain;
}

TEST_CASE("[Modules][Planner] Blocks World - Small Problem (init_state_1, goal1a)") {
	Ref<PlannerDomain> domain = create_blocks_world_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_max_depth(500); // Increased depth limit for small problem
	plan->set_verbose(2); // Enable verbose output to debug inefficiency
	
	Dictionary init_state = create_init_state_1();
	Dictionary goal = create_goal1a();
	
	Array todo_list;
	Array task;
	task.push_back("move_blocks");
	task.push_back(goal);
	todo_list.push_back(task);
	
	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);
	
	CHECK(result.is_valid());
	CHECK(result->get_success());
	
	Array plan_result = result->extract_plan();
	// IPyHOP finds 6 actions, but our planner may find a different valid plan
	// Accept any plan that achieves the goal (at least 6 actions, reasonable upper bound)
	CHECK(plan_result.size() >= 6);
	CHECK(plan_result.size() <= 50); // Reasonable upper bound for small problem
	
	// Verify plan actually achieves the goal
	bool plan_correct = verify_plan_achieves_goal(plan, init_state, goal, plan_result);
	CHECK(plan_correct);
	
	// Verify plan contains expected action types
	bool has_unstack = false;
	bool has_putdown = false;
	bool has_pickup = false;
	bool has_stack = false;
	
	for (int i = 0; i < plan_result.size(); i++) {
		Array action = plan_result[i];
		if (action.size() > 0) {
			String action_name = action[0];
			if (action_name == "action_unstack") {
				has_unstack = true;
			} else if (action_name == "action_putdown") {
				has_putdown = true;
			} else if (action_name == "action_pickup") {
				has_pickup = true;
			} else if (action_name == "action_stack") {
				has_stack = true;
			}
		}
	}
	
	// Should use at least one pickup/unstack action
	bool has_pickup_or_unstack = has_unstack || has_pickup;
	CHECK(has_pickup_or_unstack);
	// Should use at least one putdown/stack action
	bool has_putdown_or_stack = has_putdown || has_stack;
	CHECK(has_putdown_or_stack);
}

TEST_CASE("[Modules][Planner] Blocks World - Large Benchmark (init_state_3, goal3)") {
	Ref<PlannerDomain> domain = create_blocks_world_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_max_depth(300); // Higher depth for large problem
	plan->set_verbose(0); // Set to 2 for debugging
	
	Dictionary init_state = create_init_state_3();
	Dictionary goal = create_goal3();
	
	Array todo_list;
	Array task;
	task.push_back("move_blocks");
	task.push_back(goal);
	todo_list.push_back(task);
	
	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);
	
	CHECK(result.is_valid());
	CHECK(result->get_success());
	
	Array plan_result = result->extract_plan();
	// IPyHOP finds 33 actions for bw_large_d, but our planner may find a different valid plan
	// Accept any plan that achieves the goal (at least 20 actions, reasonable upper bound)
	CHECK(plan_result.size() >= 20); // Minimum reasonable plan size
	CHECK(plan_result.size() <= 150); // Reasonable upper bound for large problem
	
	// Verify plan actually achieves the goal
	bool plan_correct = verify_plan_achieves_goal(plan, init_state, goal, plan_result);
	CHECK(plan_correct);
	
	// Verify plan contains expected action types
	bool has_unstack = false;
	bool has_putdown = false;
	bool has_pickup = false;
	bool has_stack = false;
	
	for (int i = 0; i < plan_result.size(); i++) {
		Array action = plan_result[i];
		if (action.size() > 0) {
			String action_name = action[0];
			if (action_name == "action_unstack") {
				has_unstack = true;
			} else if (action_name == "action_putdown") {
				has_putdown = true;
			} else if (action_name == "action_pickup") {
				has_pickup = true;
			} else if (action_name == "action_stack") {
				has_stack = true;
			}
		}
	}
	
	CHECK(has_unstack);
	CHECK(has_putdown);
	CHECK(has_pickup);
	CHECK(has_stack);
}

TEST_CASE("[Modules][Planner] Blocks World - Performance Test") {
	Ref<PlannerDomain> domain = create_blocks_world_domain();
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	plan->set_current_domain(domain);
	plan->set_max_depth(200);
	plan->set_verbose(0);
	
	Dictionary init_state = create_init_state_3();
	Dictionary goal = create_goal3();
	
	Array todo_list;
	Array task;
	task.push_back("move_blocks");
	task.push_back(goal);
	todo_list.push_back(task);
	
	// Measure iterations
	Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);
	
	CHECK(result.is_valid());
	CHECK(result->get_success());
	
	int iterations = plan->get_iterations();
	CHECK(iterations > 0);
	
	// Large problem should complete in reasonable iterations
	// (exact number depends on implementation, but should be finite)
	CHECK(iterations < 10000); // Sanity check
	
	Array plan_result = result->extract_plan();
	CHECK(plan_result.size() > 0);
}

} // namespace TestBlocksWorld

