/**************************************************************************/
/*  blocks_world_domain.h                                                 */
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

#include "../../planner_state.h"
#include "core/variant/callable.h"
#include <climits>

// Wrapper class used to construct Callables for free functions in BlocksWorldDomain.
class BlocksWorldDomainCallable {
public:
	// Actions
	static Dictionary action_pickup(Dictionary p_state, Variant p_b);
	static Dictionary action_unstack(Dictionary p_state, Variant p_b, Variant p_c);
	static Dictionary action_putdown(Dictionary p_state, Variant p_b);
	static Dictionary action_stack(Dictionary p_state, Variant p_b, Variant p_c);
	
	// Task methods
	static Array task_move_blocks(Dictionary p_state, Dictionary p_goal);
	static Array task_move_one(Dictionary p_state, Variant p_b1, Variant p_dest);
	static Array task_get(Dictionary p_state, Variant p_b1);
	static Array task_put(Dictionary p_state, Variant p_b1, Variant p_b2);
};

// Helper functions for blocks world domain
namespace BlocksWorldDomain {

// Helper: Check if a block is done (in correct position)
bool is_done(Variant p_b1, Dictionary p_state, Dictionary p_goal) {
	if (p_b1.get_type() == Variant::STRING && String(p_b1) == "table") {
		return true;
	}
	
	Dictionary pos = p_state.get("pos", Dictionary());
	Dictionary goal_pos = p_goal.get("pos", Dictionary());
	
	if (goal_pos.has(p_b1)) {
		Variant goal_location = goal_pos[p_b1];
		Variant current_location = pos.get(p_b1, Variant());
		if (current_location != goal_location) {
			return false;
		}
	}
	
	Variant current_location = pos.get(p_b1, Variant());
	if (current_location.get_type() == Variant::STRING && String(current_location) == "table") {
		return true;
	}
	
	return is_done(current_location, p_state, p_goal);
}

// Helper: Get status of a block
String status(Variant p_b1, Dictionary p_state, Dictionary p_goal) {
	if (is_done(p_b1, p_state, p_goal)) {
		return "done";
	}
	
	Dictionary clear = p_state.get("clear", Dictionary());
	if (!clear.get(p_b1, false)) {
		return "inaccessible";
	}
	
	Dictionary goal_pos = p_goal.get("pos", Dictionary());
	if (!goal_pos.has(p_b1) || goal_pos[p_b1].get_type() == Variant::STRING) {
		return "move-to-table";
	}
	
	Variant goal_location = goal_pos[p_b1];
	if (is_done(goal_location, p_state, p_goal)) {
		Dictionary clear_dict = p_state.get("clear", Dictionary());
		if (clear_dict.get(goal_location, false)) {
			return "move-to-block";
		}
	}
	
	return "waiting";
}

// Helper: Get all blocks
Array all_blocks(Dictionary p_state) {
	Dictionary clear = p_state.get("clear", Dictionary());
	Array blocks;
	Array keys = clear.keys();
	for (int i = 0; i < keys.size(); i++) {
		Variant key = keys[i];
		if (key.get_type() != Variant::STRING || String(key) != "hand") {
			blocks.push_back(key);
		}
	}
	return blocks;
}

// Actions
Dictionary action_pickup(Dictionary state, Variant b) {
	Dictionary pos = state.get("pos", Dictionary());
	Dictionary clear = state.get("clear", Dictionary());
	Dictionary holding = state.get("holding", Dictionary());
	
	Variant b_pos = pos.get(b, Variant());
	bool b_clear = clear.get(b, false);
	bool hand_empty = !holding.get("hand", false);
	
	if (b_pos.get_type() == Variant::STRING && String(b_pos) == "table" && b_clear && hand_empty) {
		Dictionary new_state = state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary new_holding = holding.duplicate();
		
		new_pos[b] = "hand";
		new_clear[b] = false;
		new_holding["hand"] = b;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		
		return new_state;
	}
	
	return Dictionary(); // Return empty dict on failure
}

Dictionary action_unstack(Dictionary state, Variant b, Variant c) {
	Dictionary pos = state.get("pos", Dictionary());
	Dictionary clear = state.get("clear", Dictionary());
	Dictionary holding = state.get("holding", Dictionary());
	
	Variant b_pos = pos.get(b, Variant());
	bool b_clear = clear.get(b, false);
	bool hand_empty = !holding.get("hand", false);
	
	if (b_pos == c && c.get_type() != Variant::STRING && b_clear && hand_empty) {
		Dictionary new_state = state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary new_holding = holding.duplicate();
		
		new_pos[b] = "hand";
		new_clear[b] = false;
		new_holding["hand"] = b;
		new_clear[c] = true;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		
		return new_state;
	}
	
	return Dictionary();
}

Dictionary action_putdown(Dictionary state, Variant b) {
	Dictionary pos = state.get("pos", Dictionary());
	Dictionary clear = state.get("clear", Dictionary());
	Dictionary holding = state.get("holding", Dictionary());
	
	Variant b_pos = pos.get(b, Variant());
	
	if (b_pos.get_type() == Variant::STRING && String(b_pos) == "hand") {
		Dictionary new_state = state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary new_holding = holding.duplicate();
		
		new_pos[b] = "table";
		new_clear[b] = true;
		new_holding["hand"] = false;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		
		return new_state;
	}
	
	return Dictionary();
}

Dictionary action_stack(Dictionary state, Variant b, Variant c) {
	Dictionary pos = state.get("pos", Dictionary());
	Dictionary clear = state.get("clear", Dictionary());
	Dictionary holding = state.get("holding", Dictionary());
	
	Variant b_pos = pos.get(b, Variant());
	bool c_clear = clear.get(c, false);
	
	if (b_pos.get_type() == Variant::STRING && String(b_pos) == "hand" && c_clear) {
		Dictionary new_state = state.duplicate();
		Dictionary new_pos = pos.duplicate();
		Dictionary new_clear = clear.duplicate();
		Dictionary new_holding = holding.duplicate();
		
		new_pos[b] = c;
		new_clear[b] = true;
		new_holding["hand"] = false;
		new_clear[c] = false;
		
		new_state["pos"] = new_pos;
		new_state["clear"] = new_clear;
		new_state["holding"] = new_holding;
		
		return new_state;
	}
	
	return Dictionary();
}

// Task methods
// Helper: Count how many blocks need to be moved before this block can reach its goal
int count_blocks_to_move(Variant p_b1, Dictionary p_state, Dictionary p_goal) {
	Dictionary goal_pos = p_goal.get("pos", Dictionary());
	if (!goal_pos.has(p_b1)) {
		return 0; // Block should go to table, no dependencies
	}
	
	Variant goal_location = goal_pos[p_b1];
	if (goal_location.get_type() == Variant::STRING && String(goal_location) == "table") {
		return 0; // Going to table, no dependencies
	}
	
	// Count blocks that need to be moved before goal_location is ready
	Dictionary pos = p_state.get("pos", Dictionary());
	Variant current_location = pos.get(goal_location, Variant());
	Variant goal_location_goal = goal_pos.get(goal_location, Variant());
	
	int count = 0;
	if (current_location != goal_location_goal) {
		count = 1; // The goal location itself needs to be moved
		count += count_blocks_to_move(goal_location, p_state, p_goal);
	}
	
	return count;
}

Array task_move_blocks(Dictionary state, Dictionary goal) {
	Array blocks = all_blocks(state);
	
	// Priority 1: Try to move a block to its final position (move-to-block)
	// Prefer blocks with fewer dependencies (blocks that are closer to their goal)
	Variant best_block_move_to_block;
	int best_dependency_count = INT_MAX;
	
	for (int i = 0; i < blocks.size(); i++) {
		Variant b1 = blocks[i];
		String s = status(b1, state, goal);
		
		if (s == "move-to-block") {
			int deps = count_blocks_to_move(b1, state, goal);
			if (deps < best_dependency_count) {
				best_dependency_count = deps;
				best_block_move_to_block = b1;
			}
		}
	}
	
	if (best_block_move_to_block.get_type() != Variant::NIL) {
		Dictionary goal_pos = goal.get("pos", Dictionary());
		Variant dest = goal_pos[best_block_move_to_block];
		
		Array result;
		Array move_task;
		move_task.push_back("move_one");
		move_task.push_back(best_block_move_to_block);
		move_task.push_back(dest);
		result.push_back(move_task);
		
		Array recurse_task;
		recurse_task.push_back("move_blocks");
		recurse_task.push_back(goal);
		result.push_back(recurse_task);
		
		return result;
	}
	
	// Priority 2: Try to move a block to table (move-to-table)
	// Prefer blocks that are blocking other blocks
	for (int i = 0; i < blocks.size(); i++) {
		Variant b1 = blocks[i];
		String s = status(b1, state, goal);
		
		if (s == "move-to-table") {
			Array result;
			Array move_task;
			move_task.push_back("move_one");
			move_task.push_back(b1);
			move_task.push_back("table");
			result.push_back(move_task);
			
			Array recurse_task;
			recurse_task.push_back("move_blocks");
			recurse_task.push_back(goal);
			result.push_back(recurse_task);
			
			return result;
		}
	}
	
	// Priority 3: Try to move a waiting block to table
	for (int i = 0; i < blocks.size(); i++) {
		Variant b1 = blocks[i];
		if (status(b1, state, goal) == "waiting") {
			Array result;
			Array move_task;
			move_task.push_back("move_one");
			move_task.push_back(b1);
			move_task.push_back("table");
			result.push_back(move_task);
			
			Array recurse_task;
			recurse_task.push_back("move_blocks");
			recurse_task.push_back(goal);
			result.push_back(recurse_task);
			
			return result;
		}
	}
	
	// No blocks need moving
	return Array();
}

Array task_move_one(Dictionary state, Variant b1, Variant dest) {
	Array result;
	
	Array get_task;
	get_task.push_back("get");
	get_task.push_back(b1);
	result.push_back(get_task);
	
	Array put_task;
	put_task.push_back("put");
	put_task.push_back(b1);
	put_task.push_back(dest);
	result.push_back(put_task);
	
	return result;
}

Array task_get(Dictionary state, Variant b1) {
	Dictionary clear = state.get("clear", Dictionary());
	Dictionary pos = state.get("pos", Dictionary());
	
	if (clear.get(b1, false)) {
		Variant b1_pos = pos.get(b1, Variant());
		
		if (b1_pos.get_type() == Variant::STRING && String(b1_pos) == "table") {
			Array result;
			Array action;
			action.push_back("action_pickup");
			action.push_back(b1);
			result.push_back(action);
			return result;
		} else {
			Array result;
			Array action;
			action.push_back("action_unstack");
			action.push_back(b1);
			action.push_back(b1_pos);
			result.push_back(action);
			return result;
		}
	}
	
	return Array(); // Return empty array if not applicable
}

Array task_put(Dictionary state, Variant b1, Variant b2) {
	Dictionary holding = state.get("holding", Dictionary());
	
		if (holding.get("hand", Variant()) == b1) {
		if (b2.get_type() == Variant::STRING && String(b2) == "table") {
			Array result;
			Array action;
			action.push_back("action_putdown");
			action.push_back(b1);
			result.push_back(action);
			return result;
		} else {
			Array result;
			Array action;
			action.push_back("action_stack");
			action.push_back(b1);
			action.push_back(b2);
			result.push_back(action);
			return result;
		}
	}
	
	return Array();
}

} // namespace BlocksWorldDomain

// Callable wrapper implementations
Dictionary BlocksWorldDomainCallable::action_pickup(Dictionary p_state, Variant p_b) {
	return BlocksWorldDomain::action_pickup(p_state, p_b);
}

Dictionary BlocksWorldDomainCallable::action_unstack(Dictionary p_state, Variant p_b, Variant p_c) {
	return BlocksWorldDomain::action_unstack(p_state, p_b, p_c);
}

Dictionary BlocksWorldDomainCallable::action_putdown(Dictionary p_state, Variant p_b) {
	return BlocksWorldDomain::action_putdown(p_state, p_b);
}

Dictionary BlocksWorldDomainCallable::action_stack(Dictionary p_state, Variant p_b, Variant p_c) {
	return BlocksWorldDomain::action_stack(p_state, p_b, p_c);
}

Array BlocksWorldDomainCallable::task_move_blocks(Dictionary p_state, Dictionary p_goal) {
	return BlocksWorldDomain::task_move_blocks(p_state, p_goal);
}

Array BlocksWorldDomainCallable::task_move_one(Dictionary p_state, Variant p_b1, Variant p_dest) {
	return BlocksWorldDomain::task_move_one(p_state, p_b1, p_dest);
}

Array BlocksWorldDomainCallable::task_get(Dictionary p_state, Variant p_b1) {
	return BlocksWorldDomain::task_get(p_state, p_b1);
}

Array BlocksWorldDomainCallable::task_put(Dictionary p_state, Variant p_b1, Variant p_b2) {
	return BlocksWorldDomain::task_put(p_state, p_b1, p_b2);
}

