/**************************************************************************/
/*  test_blocks_domain.h                                                  */
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

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

// This file is based on the blocks-world examples included with IPyHOP:
//	modules/goal_task_planner/thirdparty/IPyHOP/examples/blocks_world/
//
// For reference, see:
//	- blocks_world_actions.py: Actions (a_pickup, a_unstack, a_putdown, a_stack)
//	- blocks_world_methods_1.py: Methods (tm_move_blocks, tm_move_one, tm_get, tm_put)
//	- blocks_world_problem.py: Initial states and goals
//	- blocks_world_example.py: Example test cases

#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "tests/test_macros.h"

#include "modules/goal_task_planner/domain.h"
#include "modules/goal_task_planner/multigoal.h"
#include "modules/goal_task_planner/plan.h"
#include "modules/goal_task_planner/planner_state.h"

#ifdef TOOLS_ENABLED
namespace TestBlocksDomain {

// Blocks World Commands (Allocentric World Changes)
// Commands change the allocentric (shared/global) world state.
// These are the atomic operations that actually modify the physical world.
// State variables:
// - pos[b] = block b's position, which may be 'table', 'hand', or another block.
// - clear[b] = False if a block is on b or the hand is holding b, else True.
// - holding['hand'] = name of the block being held by hand, or False if the hand is empty.

static Variant c_pickup(Dictionary p_state, String p_b) {
	Dictionary pos = p_state["pos"];
	Dictionary clear = p_state["clear"];
	Dictionary holding = p_state["holding"];
	
	if (pos[p_b] == "table" && clear[p_b] == true && holding["hand"] == false) {
		Dictionary new_state = p_state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary new_holding = holding.duplicate();
		
		new_pos[p_b] = "hand";
		new_clear[p_b] = false;
		new_holding["hand"] = p_b;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		return new_state;
	}
	return false; // Return false if not applicable
}

static Variant c_unstack(Dictionary p_state, String p_b, String p_c) {
	Dictionary pos = p_state["pos"];
	Dictionary clear = p_state["clear"];
	Dictionary holding = p_state["holding"];
	
	if (pos[p_b] == p_c && p_c != "table" && clear[p_b] == true && holding["hand"] == false) {
		Dictionary new_state = p_state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary new_holding = holding.duplicate();
		
		new_pos[p_b] = "hand";
		new_clear[p_b] = false;
		new_holding["hand"] = p_b;
		new_clear[p_c] = true;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		return new_state;
	}
	return false; // Return false if not applicable
}

static Variant c_putdown(Dictionary p_state, String p_b) {
	Dictionary pos = p_state["pos"];
	
	if (pos[p_b] == "hand") {
		Dictionary new_state = p_state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary clear = p_state["clear"];
		Dictionary new_clear = clear.duplicate();
		Dictionary holding = p_state["holding"];
		Dictionary new_holding = holding.duplicate();
		
		new_pos[p_b] = "table";
		new_clear[p_b] = true;
		new_holding["hand"] = false;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		return new_state;
	}
	return false; // Return false if not applicable
}

static Variant c_stack(Dictionary p_state, String p_b, String p_c) {
	Dictionary pos = p_state["pos"];
	Dictionary clear = p_state["clear"];
	
	if (pos[p_b] == "hand" && clear[p_c] == true) {
		Dictionary new_state = p_state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary holding = p_state["holding"];
		Dictionary new_holding = holding.duplicate();
		
		new_pos[p_b] = p_c;
		new_clear[p_b] = true;
		new_holding["hand"] = false;
		new_clear[p_c] = false;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		return new_state;
	}
	return false; // Return false if not applicable
}

// Blocks World Actions (Egocentric Planner State)
// Actions are egocentric - they affect only the planner's internal state representation,
// not the allocentric world.

static Variant a_pickup(Dictionary p_state, String p_b) {
	Dictionary pos = p_state["pos"];
	Dictionary clear = p_state["clear"];
	Dictionary holding = p_state["holding"];
	
	if (pos[p_b] == "table" && clear[p_b] == true && holding["hand"] == false) {
		Dictionary new_state = p_state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary new_holding = holding.duplicate();
		
		new_pos[p_b] = "hand";
		new_clear[p_b] = false;
		new_holding["hand"] = p_b;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		return new_state;
	}
	return false; // Return false if not applicable
}

static Variant a_unstack(Dictionary p_state, String p_b, String p_c) {
	Dictionary pos = p_state["pos"];
	Dictionary clear = p_state["clear"];
	Dictionary holding = p_state["holding"];
	
	if (pos[p_b] == p_c && p_c != "table" && clear[p_b] == true && holding["hand"] == false) {
		Dictionary new_state = p_state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary new_holding = holding.duplicate();
		
		new_pos[p_b] = "hand";
		new_clear[p_b] = false;
		new_holding["hand"] = p_b;
		new_clear[p_c] = true;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		return new_state;
	}
	return false; // Return false if not applicable
}

static Variant a_putdown(Dictionary p_state, String p_b) {
	Dictionary pos = p_state["pos"];
	
	if (pos[p_b] == "hand") {
		Dictionary new_state = p_state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary clear = p_state["clear"];
		Dictionary new_clear = clear.duplicate();
		Dictionary holding = p_state["holding"];
		Dictionary new_holding = holding.duplicate();
		
		new_pos[p_b] = "table";
		new_clear[p_b] = true;
		new_holding["hand"] = false;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		return new_state;
	}
	return false; // Return false if not applicable
}

static Variant a_stack(Dictionary p_state, String p_b, String p_c) {
	Dictionary pos = p_state["pos"];
	Dictionary clear = p_state["clear"];
	
	if (pos[p_b] == "hand" && clear[p_c] == true) {
		Dictionary new_state = p_state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary holding = p_state["holding"];
		Dictionary new_holding = holding.duplicate();
		
		new_pos[p_b] = p_c;
		new_clear[p_b] = true;
		new_holding["hand"] = false;
		new_clear[p_c] = false;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		return new_state;
	}
	return false; // Return false if not applicable
}

// Helper function: is_done - check if block b1 is in its final position
static bool is_done(String p_b1, Dictionary p_state, Ref<PlannerMultigoal> p_goal) {
	if (p_b1 == "table") {
		return true;
	}
	
	Dictionary goal_pos = p_goal->get_goal_conditions_for_variable("pos");
	Dictionary state_pos = p_state["pos"];
	
	if (goal_pos.has(p_b1) && goal_pos[p_b1] != state_pos[p_b1]) {
		return false;
	}
	if (state_pos[p_b1] == "table") {
		return true;
	}
	return is_done(state_pos[p_b1], p_state, p_goal);
}

// Helper function: status - determine status of block b1
static String status(String p_b1, Dictionary p_state, Ref<PlannerMultigoal> p_goal) {
	if (is_done(p_b1, p_state, p_goal)) {
		return "done";
	}
	
	Dictionary clear = p_state["clear"];
	if (clear[p_b1] == false) {
		return "inaccessible";
	}
	
	Dictionary goal_pos = p_goal->get_goal_conditions_for_variable("pos");
	if (!goal_pos.has(p_b1) || goal_pos[p_b1] == "table") {
		return "move-to-table";
	}
	
	String goal_dest = goal_pos[p_b1];
	if (is_done(goal_dest, p_state, p_goal)) {
		Dictionary state_clear = p_state["clear"];
		if (state_clear[goal_dest] == true) {
			return "move-to-block";
		}
	}
	return "waiting";
}

// Helper function: all_blocks - get all blocks from state
static Array all_blocks(Dictionary p_state) {
	Dictionary clear = p_state["clear"];
	return clear.keys();
}

// Helper function: find_if - find first block matching condition
static Variant find_if(Dictionary p_state, Ref<PlannerMultigoal> p_goal, String p_status) {
	Array blocks = all_blocks(p_state);
	for (int i = 0; i < blocks.size(); i++) {
		String b = blocks[i];
		if (status(b, p_state, p_goal) == p_status) {
			return b;
		}
	}
	return Variant(); // Return null if not found
}

// Blocks World Methods

// Method: tm_move_blocks - move blocks to achieve goal
static Variant tm_move_blocks(Dictionary p_state, Ref<PlannerMultigoal> p_goal) {
	Array blocks = all_blocks(p_state);
	
	// Try to find a block that can be moved to its final position
	for (int i = 0; i < blocks.size(); i++) {
		String b1 = blocks[i];
		String s = status(b1, p_state, p_goal);
		if (s == "move-to-table") {
			Array subtasks;
			Array move_task;
			move_task.push_back("move_one");
			move_task.push_back(b1);
			move_task.push_back("table");
			subtasks.push_back(move_task);
			
			Array move_blocks_task;
			move_blocks_task.push_back("move_blocks");
			move_blocks_task.push_back(p_goal);
			subtasks.push_back(move_blocks_task);
			return subtasks;
		} else if (s == "move-to-block") {
			Dictionary goal_pos = p_goal->get_goal_conditions_for_variable("pos");
			String dest = goal_pos[b1];
			Array subtasks;
			Array move_task;
			move_task.push_back("move_one");
			move_task.push_back(b1);
			move_task.push_back(dest);
			subtasks.push_back(move_task);
			
			Array move_blocks_task;
			move_blocks_task.push_back("move_blocks");
			move_blocks_task.push_back(p_goal);
			subtasks.push_back(move_blocks_task);
			return subtasks;
		}
	}
	
	// Try to find a waiting block and move it to table
	Variant b1 = find_if(p_state, p_goal, "waiting");
	if (b1.get_type() != Variant::NIL) {
		Array subtasks;
		Array move_task;
		move_task.push_back("move_one");
		move_task.push_back(b1);
		move_task.push_back("table");
		subtasks.push_back(move_task);
		
		Array move_blocks_task;
		move_blocks_task.push_back("move_blocks");
		move_blocks_task.push_back(p_goal);
		subtasks.push_back(move_blocks_task);
		return subtasks;
	}
	
	// No blocks need moving
	return Array(); // Empty array means task is done
}

// Method: tm_move_one - move one block to destination
static Variant tm_move_one(Dictionary p_state, String p_b1, String p_dest) {
	Array subtasks;
	
	Array get_task;
	get_task.push_back("get");
	get_task.push_back(p_b1);
	subtasks.push_back(get_task);
	
	Array put_task;
	put_task.push_back("put");
	put_task.push_back(p_b1);
	put_task.push_back(p_dest);
	subtasks.push_back(put_task);
	
	return subtasks;
}

// Method: tm_get - get a block (pickup or unstack)
static Variant tm_get(Dictionary p_state, String p_b1) {
	Dictionary clear = p_state["clear"];
	Dictionary pos = p_state["pos"];
	
	if (clear[p_b1] == true) {
		Array subtask;
		if (pos[p_b1] == "table") {
			subtask.push_back("a_pickup");
			subtask.push_back(p_b1);
		} else {
			subtask.push_back("a_unstack");
			subtask.push_back(p_b1);
			subtask.push_back(pos[p_b1]);
		}
		return subtask;
	}
	return false; // Not applicable
}

// Method: tm_put - put a block (putdown or stack)
static Variant tm_put(Dictionary p_state, String p_b1, String p_b2) {
	Dictionary holding = p_state["holding"];
	Dictionary clear = p_state["clear"];
	
	if (holding["hand"] == p_b1) {
		Array subtask;
		if (p_b2 == "table") {
			subtask.push_back("a_putdown");
			subtask.push_back(p_b1);
		} else {
			if (clear[p_b2] == true) {
				subtask.push_back("a_stack");
				subtask.push_back(p_b1);
				subtask.push_back(p_b2);
			} else {
				return false; // Not applicable
			}
		}
		return subtask;
	}
	return false; // Not applicable
}

// Helper function to set up blocks domain - goals only (unigoal methods)
static Ref<PlannerDomain> setup_blocks_domain_goals_only() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	
	// Add commands (allocentric world changes)
	TypedArray<Callable> commands;
	commands.push_back(callable_mp_static(&c_pickup));
	commands.push_back(callable_mp_static(&c_unstack));
	commands.push_back(callable_mp_static(&c_putdown));
	commands.push_back(callable_mp_static(&c_stack));
	domain->add_actions(commands);
	
	// Add actions (egocentric planner state)
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&a_pickup));
	actions.push_back(callable_mp_static(&a_unstack));
	actions.push_back(callable_mp_static(&a_putdown));
	actions.push_back(callable_mp_static(&a_stack));
	domain->add_actions(actions);
	
	// Add unigoal methods (goal-based decomposition)
	TypedArray<Callable> get_methods;
	get_methods.push_back(callable_mp_static(&tm_get));
	domain->add_unigoal_methods("get", get_methods);
	
	TypedArray<Callable> put_methods;
	put_methods.push_back(callable_mp_static(&tm_put));
	domain->add_unigoal_methods("put", put_methods);
	
	return domain;
}

// Helper function to set up blocks domain - tasks only (task methods)
static Ref<PlannerDomain> setup_blocks_domain_tasks_only() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);
	
	// Add commands (allocentric world changes)
	TypedArray<Callable> commands;
	commands.push_back(callable_mp_static(&c_pickup));
	commands.push_back(callable_mp_static(&c_unstack));
	commands.push_back(callable_mp_static(&c_putdown));
	commands.push_back(callable_mp_static(&c_stack));
	domain->add_actions(commands);
	
	// Add actions (egocentric planner state)
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&a_pickup));
	actions.push_back(callable_mp_static(&a_unstack));
	actions.push_back(callable_mp_static(&a_putdown));
	actions.push_back(callable_mp_static(&a_stack));
	domain->add_actions(actions);
	
	// Add task methods (task-based decomposition)
	TypedArray<Callable> move_blocks_methods;
	move_blocks_methods.push_back(callable_mp_static(&tm_move_blocks));
	domain->add_task_methods("move_blocks", move_blocks_methods);
	
	TypedArray<Callable> move_one_methods;
	move_one_methods.push_back(callable_mp_static(&tm_move_one));
	domain->add_task_methods("move_one", move_one_methods);
	
	return domain;
}

// Helper function to set up blocks domain (for backward compatibility)
static Ref<PlannerDomain> setup_blocks_domain() {
	return setup_blocks_domain_tasks_only();
}

// Helper function to create initial state 1 (from IPyHOP example)
static Dictionary create_init_state_1() {
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

// Helper function to create multigoal 1a (from IPyHOP example)
static Ref<PlannerMultigoal> create_multigoal_1a() {
	Dictionary goal_state;
	
	Dictionary pos;
	pos["c"] = "b";
	pos["b"] = "a";
	pos["a"] = "table";
	goal_state["pos"] = pos;
	
	Dictionary clear;
	clear["c"] = true;
	clear["b"] = false;
	clear["a"] = false;
	goal_state["clear"] = clear;
	
	Dictionary holding;
	holding["hand"] = false;
	goal_state["holding"] = holding;
	
	Ref<PlannerMultigoal> goal = memnew(PlannerMultigoal("goal1a", goal_state));
	return goal;
}

// Helper function to create goal 1a (for backward compatibility)
static Ref<PlannerMultigoal> create_goal_1a() {
	return create_multigoal_1a();
}

// Helper function to create state with entity capabilities
static Dictionary create_state_with_entities(const Dictionary &p_base_state, const String &p_entity_id, const String &p_capability, Variant p_value) {
	Dictionary state = p_base_state.duplicate();
	
	// Add entity capabilities to state
	Dictionary entity_capabilities;
	if (state.has("entity_capabilities")) {
		entity_capabilities = state["entity_capabilities"];
	}
	
	Dictionary entity_caps;
	if (entity_capabilities.has(p_entity_id)) {
		entity_caps = entity_capabilities[p_entity_id];
	}
	entity_caps[p_capability] = p_value;
	entity_capabilities[p_entity_id] = entity_caps;
	state["entity_capabilities"] = entity_capabilities;
	
	return state;
}

// Helper function to attach PlannerMetadata with entity requirements
static Dictionary attach_entity_metadata(const Array &p_action_array, const String &p_entity_type, const Array &p_capabilities) {
	Dictionary metadata_dict;
	Array entities_array;
	Dictionary entity_req;
	entity_req["type"] = p_entity_type;
	entity_req["capabilities"] = p_capabilities;
	entities_array.push_back(entity_req);
	metadata_dict["requires_entities"] = entities_array;
	
	Dictionary result;
	result["item"] = p_action_array;
	result["metadata"] = metadata_dict;
	return result;
}

// Helper function to attach PlannerMetadata with temporal constraints
static Dictionary attach_temporal_metadata(const Array &p_action_array, int64_t p_start_time_micros, int64_t p_end_time_micros, int64_t p_duration_micros) {
	Dictionary temporal_dict;
	temporal_dict["duration"] = String::num_int64(p_duration_micros);
	temporal_dict["start_time"] = String::num_int64(p_start_time_micros);
	temporal_dict["end_time"] = String::num_int64(p_end_time_micros);
	
	Dictionary result;
	result["item"] = p_action_array;
	result["temporal_constraints"] = temporal_dict;
	return result;
}

TEST_CASE("[Modules][BlocksDomain] Basic actions") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain();
	plan->set_current_domain(domain);
	
	Dictionary state = create_init_state_1();
	
	SUBCASE("a_pickup should fail for block a (not on table)") {
		Array todo_list;
		Array action;
		action.push_back("a_pickup");
		action.push_back("a");
		todo_list.push_back(action);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == false); // Should fail
	}
	
	SUBCASE("a_pickup should fail for block b (not clear)") {
		Array todo_list;
		Array action;
		action.push_back("a_pickup");
		action.push_back("b");
		todo_list.push_back(action);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == false); // Should fail
	}
	
	SUBCASE("a_pickup should succeed for block c") {
		Array todo_list;
		Array action;
		action.push_back("a_pickup");
		action.push_back("c");
		todo_list.push_back(action);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		Array plan_array = result;
		CHECK(plan_array.size() == 1);
		Array first_action = plan_array[0];
		CHECK(first_action[0] == "a_pickup");
		CHECK(first_action[1] == "c");
	}
	
	SUBCASE("get should succeed for block a (unstack)") {
		Array todo_list;
		Array goal;
		goal.push_back("get");
		goal.push_back("a");
		goal.push_back(true); // desired value (any non-false)
		todo_list.push_back(goal);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		Array plan_array = result;
		CHECK(plan_array.size() == 1);
		Array first_action = plan_array[0];
		CHECK(first_action[0] == "a_unstack");
		CHECK(first_action[1] == "a");
		CHECK(first_action[2] == "b");
	}
	
	SUBCASE("get should succeed for block c (pickup)") {
		Array todo_list;
		Array goal;
		goal.push_back("get");
		goal.push_back("c");
		goal.push_back(true); // desired value
		todo_list.push_back(goal);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		Array plan_array = result;
		CHECK(plan_array.size() == 1);
		Array first_action = plan_array[0];
		CHECK(first_action[0] == "a_pickup");
		CHECK(first_action[1] == "c");
	}
	
	SUBCASE("move_one should succeed") {
		Array todo_list;
		Array task;
		task.push_back("move_one");
		task.push_back("a");
		task.push_back("table");
		todo_list.push_back(task);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		Array plan_array = result;
		CHECK(plan_array.size() == 2);
		// First action should be unstack
		Array first_action = plan_array[0];
		CHECK(first_action[0] == "a_unstack");
		// Second action should be putdown
		Array second_action = plan_array[1];
		CHECK(second_action[0] == "a_putdown");
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Block stacking problem") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	Ref<PlannerMultigoal> goal = create_goal_1a();
	
	SUBCASE("move_blocks task should achieve goal 1a") {
		Array todo_list;
		Array task;
		task.push_back("move_blocks");
		task.push_back(goal);
		todo_list.push_back(task);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		Array plan_array = result;
		
		// Expected plan from IPyHOP example:
		// [('a_unstack', 'a', 'b'), ('a_putdown', 'a'), ('a_pickup', 'b'), ('a_stack', 'b', 'a'),
		//  ('a_pickup', 'c'), ('a_stack', 'c', 'b')]
		CHECK(plan_array.size() >= 4); // Should have at least 4 actions
		
		// Verify first action
		Array first_action = plan_array[0];
		CHECK(first_action[0] == "a_unstack");
		CHECK(first_action[1] == "a");
		
		// Verify plan contains required actions
		bool found_unstack_a = false;
		bool found_putdown_a = false;
		bool found_pickup_b = false;
		bool found_stack_b = false;
		bool found_pickup_c = false;
		bool found_stack_c = false;
		
		for (int i = 0; i < plan_array.size(); i++) {
			Array action = plan_array[i];
			String action_name = action[0];
			if (action_name == "a_unstack" && action[1] == "a") {
				found_unstack_a = true;
			} else if (action_name == "a_putdown" && action[1] == "a") {
				found_putdown_a = true;
			} else if (action_name == "a_pickup" && action[1] == "b") {
				found_pickup_b = true;
			} else if (action_name == "a_stack" && action[1] == "b" && action[2] == "a") {
				found_stack_b = true;
			} else if (action_name == "a_pickup" && action[1] == "c") {
				found_pickup_c = true;
			} else if (action_name == "a_stack" && action[1] == "c" && action[2] == "b") {
				found_stack_c = true;
			}
		}
		
		CHECK(found_unstack_a == true);
		CHECK(found_putdown_a == true);
		CHECK(found_pickup_b == true);
		CHECK(found_stack_b == true);
		CHECK(found_pickup_c == true);
		CHECK(found_stack_c == true);
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Graph-based planning with run_lazy_refineahead") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	Ref<PlannerMultigoal> goal = create_goal_1a();
	
	SUBCASE("run_lazy_refineahead should solve blocks domain") {
		Array todo_list;
		Array task;
		task.push_back("move_blocks");
		task.push_back(goal);
		todo_list.push_back(task);
		
		Dictionary final_state = plan->run_lazy_refineahead(state, todo_list);
		
		// Verify final state matches goal
		Dictionary goal_pos = goal->get_goal_conditions_for_variable("pos");
		Dictionary final_pos = final_state["pos"];
		
		Array goal_pos_keys = goal_pos.keys();
		for (int i = 0; i < goal_pos_keys.size(); i++) {
			String block = goal_pos_keys[i];
			if (goal_pos.has(block)) {
				CHECK(final_pos.has(block));
				CHECK(final_pos[block] == goal_pos[block]);
			}
		}
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Basic commands") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_tasks_only();
	plan->set_current_domain(domain);
	
	Dictionary state = create_init_state_1();
	
	SUBCASE("c_pickup should fail for block a (not on table)") {
		Array todo_list;
		Array command;
		command.push_back("c_pickup");
		command.push_back("a");
		todo_list.push_back(command);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == false); // Should fail
	}
	
	SUBCASE("c_pickup should succeed for block c") {
		Array todo_list;
		Array command;
		command.push_back("c_pickup");
		command.push_back("c");
		todo_list.push_back(command);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		Array plan_array = result;
		CHECK(plan_array.size() == 1);
		Array first_command = plan_array[0];
		CHECK(first_command[0] == "c_pickup");
		CHECK(first_command[1] == "c");
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Goal-based planning (unigoal methods only)") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_goals_only();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	Ref<PlannerMultigoal> multigoal = create_multigoal_1a();
	
	SUBCASE("Unigoal methods should solve blocks problem") {
		Array todo_list;
		Array goal;
		goal.push_back("get");
		goal.push_back("a");
		goal.push_back(true); // desired value
		todo_list.push_back(goal);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		Array plan_array = result;
		CHECK(plan_array.size() >= 1);
		// Should use unigoal decomposition (get -> a_unstack)
		Array first_action = plan_array[0];
		CHECK(first_action[0] == "a_unstack");
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Task-based planning (task methods only)") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_tasks_only();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	Ref<PlannerMultigoal> multigoal = create_multigoal_1a();
	
	SUBCASE("Task methods should solve blocks problem") {
		Array todo_list;
		Array task;
		task.push_back("move_blocks");
		task.push_back(multigoal);
		todo_list.push_back(task);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		Array plan_array = result;
		CHECK(plan_array.size() >= 4); // Should have multiple actions
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Graph-based planning with run_lazy_refineahead (goal methods)") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_goals_only();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	Ref<PlannerMultigoal> multigoal = create_multigoal_1a();
	
	SUBCASE("run_lazy_refineahead should solve blocks domain with goal methods") {
		Array todo_list;
		Array goal;
		goal.push_back("get");
		goal.push_back("a");
		goal.push_back(true);
		todo_list.push_back(goal);
		
		Dictionary final_state = plan->run_lazy_refineahead(state, todo_list);
		CHECK(final_state.has("pos"));
		// Verify final state has been modified
		Dictionary final_pos = final_state["pos"];
		CHECK(final_pos.has("a"));
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Graph-based planning with run_lazy_refineahead (task methods)") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_tasks_only();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	Ref<PlannerMultigoal> multigoal = create_multigoal_1a();
	
	SUBCASE("run_lazy_refineahead should solve blocks domain with task methods") {
		Array todo_list;
		Array task;
		task.push_back("move_blocks");
		task.push_back(multigoal);
		todo_list.push_back(task);
		
		Dictionary final_state = plan->run_lazy_refineahead(state, todo_list);
		
		// Verify final state matches multigoal
		Dictionary multigoal_pos = multigoal->get_goal_conditions_for_variable("pos");
		Dictionary final_pos = final_state["pos"];
		
		Array multigoal_pos_keys = multigoal_pos.keys();
		for (int i = 0; i < multigoal_pos_keys.size(); i++) {
			String block = multigoal_pos_keys[i];
			if (multigoal_pos.has(block)) {
				CHECK(final_pos.has(block));
				CHECK(final_pos[block] == multigoal_pos[block]);
			}
		}
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Entity requirements in commands") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_tasks_only();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	
	SUBCASE("Command with entity requirements should fail when entities unavailable") {
		Array command;
		command.push_back("c_pickup");
		command.push_back("c");
		Array capabilities;
		capabilities.push_back("gripper");
		Dictionary command_with_metadata = attach_entity_metadata(command, "robot", capabilities);
		
		Array todo_list;
		todo_list.push_back(command_with_metadata);
		
		// State doesn't have entity_capabilities, so should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == false);
	}
	
	SUBCASE("Command with entity requirements should succeed when entities available") {
		// Add entity capabilities to state
		Dictionary state_dict = state;
		Dictionary entity_capabilities;
		Dictionary robot1_caps;
		robot1_caps["gripper"] = true;
		entity_capabilities["robot1"] = robot1_caps;
		state_dict["entity_capabilities"] = entity_capabilities;
		
		Array command;
		command.push_back("c_pickup");
		command.push_back("c");
		Array capabilities;
		capabilities.push_back("gripper");
		Dictionary command_with_metadata = attach_entity_metadata(command, "robot", capabilities);
		
		Array todo_list;
		todo_list.push_back(command_with_metadata);
		
		Variant result = plan->find_plan(state_dict, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Temporal constraints in commands") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_tasks_only();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	
	SUBCASE("Command with temporal constraints should succeed") {
		// Use absolute time in microseconds
		int64_t start_time_micros = 1735689600000000LL; // 2025-01-01 00:00:00 UTC
		int64_t duration_micros = 1800000000LL; // 30 minutes
		int64_t end_time_micros = start_time_micros + duration_micros;
		
		Array command;
		command.push_back("c_pickup");
		command.push_back("c");
		Dictionary command_with_temporal = attach_temporal_metadata(command, start_time_micros, end_time_micros, duration_micros);
		
		Array todo_list;
		todo_list.push_back(command_with_temporal);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Entity requirements in tasks") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_tasks_only();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	Ref<PlannerMultigoal> multigoal = create_multigoal_1a();
	
	SUBCASE("Task with entity requirements should fail when entities unavailable") {
		Array task;
		task.push_back("move_blocks");
		task.push_back(multigoal);
		Array capabilities;
		capabilities.push_back("gripper");
		Dictionary task_with_metadata = attach_entity_metadata(task, "robot", capabilities);
		
		Array todo_list;
		todo_list.push_back(task_with_metadata);
		
		// State doesn't have entity_capabilities, so should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == false);
	}
	
	SUBCASE("Task with entity requirements should succeed when entities available") {
		// Add entity capabilities to state
		Dictionary state_dict = state;
		Dictionary entity_capabilities;
		Dictionary robot1_caps;
		robot1_caps["gripper"] = true;
		entity_capabilities["robot1"] = robot1_caps;
		state_dict["entity_capabilities"] = entity_capabilities;
		
		Array task;
		task.push_back("move_blocks");
		task.push_back(multigoal);
		Array capabilities;
		capabilities.push_back("gripper");
		Dictionary task_with_metadata = attach_entity_metadata(task, "robot", capabilities);
		
		Array todo_list;
		todo_list.push_back(task_with_metadata);
		
		Variant result = plan->find_plan(state_dict, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Temporal constraints in tasks") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_tasks_only();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	Ref<PlannerMultigoal> multigoal = create_multigoal_1a();
	
	SUBCASE("Task with temporal constraints should succeed") {
		// Use absolute time in microseconds
		int64_t start_time_micros = 1735689600000000LL; // 2025-01-01 00:00:00 UTC
		int64_t duration_micros = 1800000000LL; // 30 minutes
		int64_t end_time_micros = start_time_micros + duration_micros;
		
		Array task;
		task.push_back("move_blocks");
		task.push_back(multigoal);
		Dictionary task_with_temporal = attach_temporal_metadata(task, start_time_micros, end_time_micros, duration_micros);
		
		Array todo_list;
		todo_list.push_back(task_with_temporal);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

TEST_CASE("[Modules][BlocksDomain] Combined temporal and entity requirements") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_blocks_domain_tasks_only();
	plan->set_current_domain(domain);
	plan->set_verbose(0);
	
	Dictionary state = create_init_state_1();
	
	// Add entity capabilities to state
	Dictionary entity_capabilities;
	Dictionary robot1_caps;
	robot1_caps["gripper"] = true;
	entity_capabilities["robot1"] = robot1_caps;
	state["entity_capabilities"] = entity_capabilities;
	
	SUBCASE("Command with both temporal and entity requirements should succeed") {
		int64_t start_time_micros = 1735689600000000LL;
		int64_t duration_micros = 1800000000LL;
		int64_t end_time_micros = start_time_micros + duration_micros;
		
		Array command;
		command.push_back("c_pickup");
		command.push_back("c");
		
		// Attach both temporal and entity metadata
		Dictionary temporal_dict = attach_temporal_metadata(command, start_time_micros, end_time_micros, duration_micros);
		Array capabilities;
		capabilities.push_back("gripper");
		Dictionary entity_dict = attach_entity_metadata(command, "robot", capabilities);
		
		// Combine both metadata
		Dictionary combined_metadata;
		combined_metadata["item"] = command;
		combined_metadata["temporal_constraints"] = temporal_dict["temporal_constraints"];
		combined_metadata["metadata"] = entity_dict["metadata"];
		
		Array todo_list;
		todo_list.push_back(combined_metadata);
		
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
	}
	
	memdelete(domain.ptr());
	memdelete(plan.ptr());
}

}
#endif // TOOLS_ENABLED

