/**************************************************************************/
/*  test_game_domains_inventory_crafting.h                                */
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

// Inventory/Crafting domain for game-inspired planner tests

#pragma once

#include "../domain.h"
#include "../multigoal.h"
#include "test_game_domains_helpers.h"

#ifdef TOOLS_ENABLED
namespace TestGameDomainsBacktracking {

// ============================================================================
// INVENTORY/CRAFTING DOMAIN
// ============================================================================

// Actions
static Variant action_craft_item(Dictionary p_state, String p_item_name, Dictionary p_recipe) {
	Dictionary inventory = p_state["inventory"];
	Dictionary recipes = p_state["recipes"];

	// Check if recipe exists
	if (!recipes.has(p_item_name)) {
		return false;
	}
	Dictionary required_recipe = recipes[p_item_name];

	// Check if we have all ingredients
	Array ingredient_keys = required_recipe.keys();
	for (int i = 0; i < ingredient_keys.size(); i++) {
		String ingredient = ingredient_keys[i];
		int needed = required_recipe[ingredient].operator int();
		int available = inventory.has(ingredient) ? inventory[ingredient].operator int() : 0;
		if (available < needed) {
			return false; // Missing ingredient
		}
	}

	// Check inventory space
	int inventory_size = inventory.has("_size") ? inventory["_size"].operator int() : 0;
	int max_size = inventory.has("_max_size") ? inventory["_max_size"].operator int() : 10;
	if (inventory_size >= max_size && !inventory.has(p_item_name)) {
		return false; // Inventory full
	}

	// Craft the item
	Dictionary new_state = p_state.duplicate();
	Dictionary new_inventory = inventory.duplicate();

	// Consume ingredients
	for (int i = 0; i < ingredient_keys.size(); i++) {
		String ingredient = ingredient_keys[i];
		int needed = required_recipe[ingredient].operator int();
		int current = new_inventory[ingredient].operator int();
		new_inventory[ingredient] = current - needed;
		if (new_inventory[ingredient].operator int() == 0) {
			new_inventory.erase(ingredient);
		}
	}

	// Add crafted item
	int current_count = new_inventory.has(p_item_name) ? new_inventory[p_item_name].operator int() : 0;
	new_inventory[p_item_name] = current_count + 1;

	if (!new_inventory.has("_size")) {
		new_inventory["_size"] = 0;
	}
	int new_size = new_inventory["_size"].operator int() + 1;
	new_inventory["_size"] = new_size;

	new_state["inventory"] = new_inventory;
	return new_state;
}

static Variant action_store_item(Dictionary p_state, String p_item_name) {
	Dictionary inventory = p_state["inventory"];
	Dictionary storage = p_state["storage"];

	if (!inventory.has(p_item_name) || inventory[p_item_name].operator int() <= 0) {
		return false; // Don't have item
	}

	// Store the item
	Dictionary new_state = p_state.duplicate();
	Dictionary new_inventory = inventory.duplicate();
	Dictionary new_storage = storage.duplicate();

	int current_inv = new_inventory[p_item_name].operator int();
	new_inventory[p_item_name] = current_inv - 1;
	if (new_inventory[p_item_name].operator int() == 0) {
		new_inventory.erase(p_item_name);
	}

	int current_storage = new_storage.has(p_item_name) ? new_storage[p_item_name].operator int() : 0;
	new_storage[p_item_name] = current_storage + 1;

	if (new_inventory.has("_size")) {
		new_inventory["_size"] = new_inventory["_size"].operator int() - 1;
	}

	new_state["inventory"] = new_inventory;
	new_state["storage"] = new_storage;
	return new_state;
}

// Task methods
static Variant task_method_create_weapon(Dictionary p_state, String p_weapon_type) {
	Dictionary inventory = p_state["inventory"];
	Dictionary recipes = p_state["recipes"];

	// Check if already have weapon
	if (inventory.has(p_weapon_type) && inventory[p_weapon_type].operator int() > 0) {
		return Array(); // Already have weapon
	}

	// Check recipe
	if (!recipes.has(p_weapon_type)) {
		return false; // No recipe for this weapon
	}
	Dictionary recipe = recipes[p_weapon_type];

	// Check if we have ingredients
	Array ingredient_keys = recipe.keys();
	Array missing_ingredients;
	for (int i = 0; i < ingredient_keys.size(); i++) {
		String ingredient = ingredient_keys[i];
		int needed = recipe[ingredient].operator int();
		int available = inventory.has(ingredient) ? inventory[ingredient].operator int() : 0;
		if (available < needed) {
			missing_ingredients.push_back(ingredient);
			missing_ingredients.push_back(needed - available);
		}
	}

	if (!missing_ingredients.is_empty()) {
		// Need to gather missing ingredients
		Array subtasks;
		Array gather_task;
		gather_task.push_back("gather_ingredients");
		gather_task.push_back(missing_ingredients);
		subtasks.push_back(gather_task);

		Array craft_task;
		craft_task.push_back("create_weapon");
		craft_task.push_back(p_weapon_type);
		subtasks.push_back(craft_task);
		return subtasks;
	}

	// Have all ingredients, can craft
	Array subtasks;
	Array action;
	action.push_back("action_craft_item");
	action.push_back(p_weapon_type);
	action.push_back(recipe);
	subtasks.push_back(action);
	return subtasks;
}

static Variant task_method_gather_ingredients(Dictionary p_state, Array p_ingredients) {
	// p_ingredients: [ingredient1, amount1, ingredient2, amount2, ...]
	Dictionary inventory = p_state["inventory"];
	Array subtasks;

	for (int i = 0; i < p_ingredients.size(); i += 2) {
		String ingredient = p_ingredients[i];
		int needed = p_ingredients[i + 1].operator int();
		int current = inventory.has(ingredient) ? inventory[ingredient].operator int() : 0;

		if (current < needed) {
			// Need to gather this ingredient (simplified - just add to inventory)
			Array action;
			action.push_back("action_gather_ingredient");
			action.push_back(ingredient);
			action.push_back(needed - current);
			subtasks.push_back(action);
		}
	}

	if (subtasks.is_empty()) {
		return Array(); // Already have all ingredients
	}
	return subtasks;
}

static Variant action_gather_ingredient(Dictionary p_state, String p_ingredient, int p_amount) {
	// Simplified gathering action
	Dictionary inventory = p_state["inventory"];
	Dictionary new_state = p_state.duplicate();
	Dictionary new_inventory = inventory.duplicate();

	int current = new_inventory.has(p_ingredient) ? new_inventory[p_ingredient].operator int() : 0;
	new_inventory[p_ingredient] = current + p_amount;

	if (!new_inventory.has("_size")) {
		new_inventory["_size"] = 0;
	}
	new_inventory["_size"] = new_inventory["_size"].operator int() + p_amount;

	new_state["inventory"] = new_inventory;
	return new_state;
}

// Goal methods
static Variant goal_method_inventory_has(Dictionary p_state, String p_item_name, Variant p_desired_value) {
	Dictionary inventory = p_state["inventory"];
	bool has_item = inventory.has(p_item_name) && inventory[p_item_name].operator int() > 0;

	if (has_item == p_desired_value.operator bool()) {
		return Array(); // Goal already achieved
	}

	// Need to get item - try crafting
	Array subtasks;
	Array task;
	task.push_back("create_weapon");
	task.push_back(p_item_name);
	subtasks.push_back(task);
	return subtasks;
}

static Variant goal_method_inventory_has_alternative(Dictionary p_state, String p_item_name, Variant p_desired_value) {
	// Alternative method that might fail
	Dictionary recipes = p_state["recipes"];
	if (!recipes.has(p_item_name)) {
		return false; // No recipe, this method fails
	}

	// Try to craft
	Array subtasks;
	Dictionary recipe = recipes[p_item_name];
	Array action;
	action.push_back("action_craft_item");
	action.push_back(p_item_name);
	action.push_back(recipe);
	subtasks.push_back(action);
	return subtasks;
}

// Multigoal methods
static Variant multigoal_method_recipe_requirements(Dictionary p_state, Dictionary p_multigoal) {
	// Multigoal format: {"inventory": {"iron": 2, "wood": 3}}
	Dictionary goal_inventory = PlannerMultigoal::get_goal_conditions_for_variable(p_multigoal, "inventory");
	Dictionary current_inventory = p_state["inventory"];

	Dictionary goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, p_multigoal);
	if (goals_not_achieved.is_empty()) {
		return Array(); // All goals achieved
	}

	// Need to gather ingredients
	Array subtasks;
	Array task;
	task.push_back("gather_ingredients");
	Array ingredients;
	Array goal_keys = goal_inventory.keys();
	for (int i = 0; i < goal_keys.size(); i++) {
		String item = goal_keys[i];
		int needed = goal_inventory[item].operator int();
		int current = current_inventory.has(item) ? current_inventory[item].operator int() : 0;
		if (current < needed) {
			ingredients.push_back(item);
			ingredients.push_back(needed - current);
		}
	}
	task.push_back(ingredients);
	subtasks.push_back(task);
	return subtasks;
}

// Setup inventory/crafting domain
static Ref<PlannerDomain> setup_inventory_crafting_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Add actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&action_craft_item));
	actions.push_back(callable_mp_static(&action_store_item));
	actions.push_back(callable_mp_static(&action_gather_ingredient));
	domain->add_actions(actions);

	// Add task methods
	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&task_method_create_weapon));
	domain->add_task_methods("create_weapon", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_gather_ingredients));
	domain->add_task_methods("gather_ingredients", task_methods);

	// Add goal methods (with multiple methods for backtracking)
	TypedArray<Callable> goal_methods;
	goal_methods.push_back(callable_mp_static(&goal_method_inventory_has));
	goal_methods.push_back(callable_mp_static(&goal_method_inventory_has_alternative));
	domain->add_unigoal_methods("inventory_has", goal_methods);

	// Add multigoal methods
	TypedArray<Callable> multigoal_methods;
	multigoal_methods.push_back(callable_mp_static(&multigoal_method_recipe_requirements));
	domain->add_multigoal_methods(multigoal_methods);

	// Set up action dictionary
	Dictionary action_dict;
	action_dict["action_craft_item"] = callable_mp_static(&action_craft_item);
	action_dict["action_store_item"] = callable_mp_static(&action_store_item);
	action_dict["action_gather_ingredient"] = callable_mp_static(&action_gather_ingredient);
	domain->action_dictionary = action_dict;

	return domain;
}

} // namespace TestGameDomainsBacktracking
#endif // TOOLS_ENABLED
