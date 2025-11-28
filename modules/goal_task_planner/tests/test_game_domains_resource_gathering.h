/**************************************************************************/
/*  test_game_domains_resource_gathering.h                                */
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

// Resource Gathering domain for game-inspired planner tests

#pragma once

#include "../domain.h"
#include "../multigoal.h"
#include "test_game_domains_helpers.h"

#ifdef TOOLS_ENABLED
namespace TestGameDomainsBacktracking {

// ============================================================================
// RESOURCE GATHERING DOMAIN
// ============================================================================

// Actions
static Variant action_get_axe(Dictionary p_state, String p_location) {
	Dictionary inventory = p_state["inventory"];
	Dictionary tools = p_state["tools"];

	// Check if axe is available at location
	Dictionary locations = p_state["locations"];
	if (!locations.has(p_location)) {
		return false;
	}
	Dictionary location = locations[p_location];
	if (!location.has("axe") || !location["axe"].operator bool()) {
		return false;
	}

	// Get the axe
	Dictionary new_state = p_state.duplicate();
	Dictionary new_inventory = inventory.duplicate();
	Dictionary new_tools = tools.duplicate();
	Dictionary new_locations = locations.duplicate();
	Dictionary new_location = location.duplicate();

	new_tools["axe"] = true;
	new_location["axe"] = false;
	new_locations[p_location] = new_location;
	new_inventory["axe"] = true;

	new_state["inventory"] = new_inventory;
	new_state["tools"] = new_tools;
	new_state["locations"] = new_locations;
	return new_state;
}

static Variant action_chop_tree(Dictionary p_state, String p_tree_id) {
	Dictionary tools = p_state["tools"];
	Dictionary resources = p_state["resources"];
	Dictionary trees = p_state["trees"];

	// Need axe to chop tree
	if (!tools.has("axe") || !tools["axe"].operator bool()) {
		return false;
	}

	// Check if tree exists and is available
	if (!trees.has(p_tree_id) || !trees[p_tree_id].operator bool()) {
		return false;
	}

	// Chop the tree
	Dictionary new_state = p_state.duplicate();
	Dictionary new_resources = resources.duplicate();
	Dictionary new_trees = trees.duplicate();

	int wood_count = resources.has("wood") ? resources["wood"].operator int() : 0;
	new_resources["wood"] = wood_count + 1;
	new_trees[p_tree_id] = false; // Tree is now chopped

	new_state["resources"] = new_resources;
	new_state["trees"] = new_trees;
	return new_state;
}

static Variant action_mine_ore(Dictionary p_state, String p_ore_location) {
	Dictionary tools = p_state["tools"];
	Dictionary resources = p_state["resources"];
	Dictionary locations = p_state["locations"];

	// Need pickaxe to mine or
	if (!tools.has("pickaxe") || !tools["pickaxe"].operator bool()) {
		return false;
	}

	// Check if or location exists and has or
	if (!locations.has(p_ore_location)) {
		return false;
	}
	Dictionary location = locations[p_ore_location];
	if (!location.has("or") || !location["or"].operator bool()) {
		return false;
	}

	// Mine the or
	Dictionary new_state = p_state.duplicate();
	Dictionary new_resources = resources.duplicate();
	Dictionary new_locations = locations.duplicate();
	Dictionary new_location = location.duplicate();

	int ore_count = resources.has("or") ? resources["or"].operator int() : 0;
	new_resources["or"] = ore_count + 1;
	new_location["or"] = false; // Or is now mined

	new_locations[p_ore_location] = new_location;
	new_state["resources"] = new_resources;
	new_state["locations"] = new_locations;
	return new_state;
}

static Variant action_craft_pickaxe(Dictionary p_state) {
	Dictionary resources = p_state["resources"];
	Dictionary tools = p_state["tools"];

	// Need 2 wood and 1 iron to craft pickaxe
	int wood = resources.has("wood") ? resources["wood"].operator int() : 0;
	int iron = resources.has("iron") ? resources["iron"].operator int() : 0;

	if (wood < 2 || iron < 1) {
		return false;
	}

	// Craft pickaxe
	Dictionary new_state = p_state.duplicate();
	Dictionary new_resources = resources.duplicate();
	Dictionary new_tools = tools.duplicate();

	new_resources["wood"] = wood - 2;
	new_resources["iron"] = iron - 1;
	new_tools["pickaxe"] = true;

	new_state["resources"] = new_resources;
	new_state["tools"] = new_tools;
	return new_state;
}

// Task methods
static Variant task_method_collect_resources(Dictionary p_state, Dictionary p_requirements) {
	// Requirements: {"wood": 3, "or": 2}
	Dictionary resources = p_state["resources"];
	Array subtasks;

	Array resource_keys = p_requirements.keys();
	for (int i = 0; i < resource_keys.size(); i++) {
		String resource_type = resource_keys[i];
		int needed = p_requirements[resource_type].operator int();
		int current = resources.has(resource_type) ? resources[resource_type].operator int() : 0;

		if (current < needed) {
			// Need to gather this resource
			if (resource_type == "wood") {
				Array task;
				task.push_back("gather_wood");
				task.push_back(needed - current);
				subtasks.push_back(task);
			} else if (resource_type == "or") {
				Array task;
				task.push_back("gather_ore");
				task.push_back(needed - current);
				subtasks.push_back(task);
			}
		}
	}

	if (subtasks.is_empty()) {
		return Array(); // Already have all resources
	}
	return subtasks;
}

static Variant task_method_prepare_tools(Dictionary p_state, Array p_tool_list) {
	Dictionary tools = p_state["tools"];
	Array subtasks;

	for (int i = 0; i < p_tool_list.size(); i++) {
		String tool = p_tool_list[i];
		if (!tools.has(tool) || !tools[tool].operator bool()) {
			// Need to get this tool
			if (tool == "axe") {
				Array task;
				task.push_back("get_axe_task");
				subtasks.push_back(task);
			} else if (tool == "pickaxe") {
				Array task;
				task.push_back("craft_pickaxe_task");
				subtasks.push_back(task);
			}
		}
	}

	if (subtasks.is_empty()) {
		return Array(); // Already have all tools
	}
	return subtasks;
}

static Variant task_method_gather_wood(Dictionary p_state, int p_amount) {
	Dictionary resources = p_state["resources"];
	Dictionary tools = p_state["tools"];
	int current_wood = resources.has("wood") ? resources["wood"].operator int() : 0;

	if (current_wood >= p_amount) {
		return Array(); // Already have enough wood
	}

	// Need axe to gather wood
	if (!tools.has("axe") || !tools["axe"].operator bool()) {
		// First get the axe
		Array subtasks;
		Array get_axe_task;
		get_axe_task.push_back("get_axe_task");
		subtasks.push_back(get_axe_task);

		Array gather_task;
		gather_task.push_back("gather_wood");
		gather_task.push_back(p_amount);
		subtasks.push_back(gather_task);
		return subtasks;
	}

	// Have axe, can chop trees
	Array subtasks;
	Array action;
	action.push_back("action_chop_tree");
	action.push_back("tree1");
	subtasks.push_back(action);

	if (p_amount > 1) {
		Array gather_task;
		gather_task.push_back("gather_wood");
		gather_task.push_back(p_amount - 1);
		subtasks.push_back(gather_task);
	}
	return subtasks;
}

static Variant task_method_gather_ore(Dictionary p_state, int p_amount) {
	Dictionary resources = p_state["resources"];
	Dictionary tools = p_state["tools"];
	int current_ore = resources.has("or") ? resources["or"].operator int() : 0;

	if (current_ore >= p_amount) {
		return Array(); // Already have enough or
	}

	// Need pickaxe to mine or
	if (!tools.has("pickaxe") || !tools["pickaxe"].operator bool()) {
		// First craft the pickaxe
		Array subtasks;
		Array craft_task;
		craft_task.push_back("craft_pickaxe_task");
		subtasks.push_back(craft_task);

		Array gather_task;
		gather_task.push_back("gather_ore");
		gather_task.push_back(p_amount);
		subtasks.push_back(gather_task);
		return subtasks;
	}

	// Have pickaxe, can mine or
	Array subtasks;
	Array action;
	action.push_back("action_mine_ore");
	action.push_back("mine1");
	subtasks.push_back(action);

	if (p_amount > 1) {
		Array gather_task;
		gather_task.push_back("gather_ore");
		gather_task.push_back(p_amount - 1);
		subtasks.push_back(gather_task);
	}
	return subtasks;
}

static Variant task_method_get_axe_task(Dictionary p_state) {
	Dictionary tools = p_state["tools"];
	if (tools.has("axe") && tools["axe"].operator bool()) {
		return Array(); // Already have axe
	}

	Array subtasks;
	Array action;
	action.push_back("action_get_axe");
	action.push_back("shop");
	subtasks.push_back(action);
	return subtasks;
}

static Variant task_method_craft_pickaxe_task(Dictionary p_state) {
	Dictionary tools = p_state["tools"];
	if (tools.has("pickaxe") && tools["pickaxe"].operator bool()) {
		return Array(); // Already have pickaxe
	}

	Array subtasks;
	Array action;
	action.push_back("action_craft_pickaxe");
	subtasks.push_back(action);
	return subtasks;
}

// Goal methods
static Variant goal_method_has_axe(Dictionary p_state, String p_arg, Variant p_desired_value) {
	Dictionary tools = p_state["tools"];
	bool has_axe = tools.has("axe") && tools["axe"].operator bool();

	if (has_axe == p_desired_value.operator bool()) {
		return Array(); // Goal already achieved
	}

	// Need to get axe
	Array subtasks;
	Array task;
	task.push_back("get_axe_task");
	subtasks.push_back(task);
	return subtasks;
}

static Variant goal_method_has_axe_alternative(Dictionary p_state, String p_arg, Variant p_desired_value) {
	// Alternative method that might fail
	Dictionary locations = p_state["locations"];
	if (!locations.has("shop")) {
		return false; // Shop doesn't exist, this method fails
	}

	// Try to get axe from shop
	Array subtasks;
	Array action;
	action.push_back("action_get_axe");
	action.push_back("shop");
	subtasks.push_back(action);
	return subtasks;
}

// Multigoal methods
static Variant multigoal_method_collect_resources_multigoal(Dictionary p_state, Dictionary p_multigoal) {
	// Multigoal format: {"resources": {"wood": 3, "or": 2}}
	Dictionary goal_resources = PlannerMultigoal::get_goal_conditions_for_variable(p_multigoal, "resources");
	Dictionary current_resources = p_state["resources"];

	Dictionary goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, p_multigoal);
	if (goals_not_achieved.is_empty()) {
		return Array(); // All goals achieved
	}

	// Need to collect resources
	Array subtasks;
	Array task;
	task.push_back("collect_resources");
	task.push_back(goal_resources);
	subtasks.push_back(task);
	return subtasks;
}

// Setup resource gathering domain
static Ref<PlannerDomain> setup_resource_gathering_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Add actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&action_get_axe));
	actions.push_back(callable_mp_static(&action_chop_tree));
	actions.push_back(callable_mp_static(&action_mine_ore));
	actions.push_back(callable_mp_static(&action_craft_pickaxe));
	domain->add_actions(actions);

	// Add task methods
	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&task_method_collect_resources));
	domain->add_task_methods("collect_resources", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_prepare_tools));
	domain->add_task_methods("prepare_tools", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_gather_wood));
	domain->add_task_methods("gather_wood", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_gather_ore));
	domain->add_task_methods("gather_ore", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_get_axe_task));
	domain->add_task_methods("get_axe_task", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_craft_pickaxe_task));
	domain->add_task_methods("craft_pickaxe_task", task_methods);

	// Add goal methods (with multiple methods for backtracking tests)
	TypedArray<Callable> goal_methods;
	goal_methods.push_back(callable_mp_static(&goal_method_has_axe));
	goal_methods.push_back(callable_mp_static(&goal_method_has_axe_alternative));
	domain->add_unigoal_methods("has_axe", goal_methods);

	// Add multigoal methods
	TypedArray<Callable> multigoal_methods;
	multigoal_methods.push_back(callable_mp_static(&multigoal_method_collect_resources_multigoal));
	domain->add_multigoal_methods(multigoal_methods);

	// Set up action dictionary
	Dictionary action_dict;
	action_dict["action_get_axe"] = callable_mp_static(&action_get_axe);
	action_dict["action_chop_tree"] = callable_mp_static(&action_chop_tree);
	action_dict["action_mine_ore"] = callable_mp_static(&action_mine_ore);
	action_dict["action_craft_pickaxe"] = callable_mp_static(&action_craft_pickaxe);
	domain->action_dictionary = action_dict;

	return domain;
}

} // namespace TestGameDomainsBacktracking
#endif // TOOLS_ENABLED
