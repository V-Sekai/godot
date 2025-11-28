/**************************************************************************/
/*  test_game_domains_quest_system.h                                      */
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

// Quest System domain for game-inspired planner tests

#pragma once

#include "../domain.h"
#include "../multigoal.h"
#include "test_game_domains_helpers.h"

#ifdef TOOLS_ENABLED
namespace TestGameDomainsBacktracking {

// ============================================================================
// QUEST SYSTEM DOMAIN
// ============================================================================

// Actions
static Variant action_accept_quest(Dictionary p_state, String p_quest_id) {
	Dictionary quests = p_state["quests"];
	Dictionary active_quests = p_state["active_quests"];

	// Check if quest exists and not already active
	if (!quests.has(p_quest_id)) {
		return false;
	}
	if (active_quests.has(p_quest_id)) {
		return false; // Already active
	}

	// Accept quest
	Dictionary new_state = p_state.duplicate();
	Dictionary new_active_quests = active_quests.duplicate();
	Dictionary quest = quests[p_quest_id];

	new_active_quests[p_quest_id] = quest;
	new_state["active_quests"] = new_active_quests;
	return new_state;
}

static Variant action_complete_objective(Dictionary p_state, String p_objective_id) {
	Dictionary active_quests = p_state["active_quests"];
	Dictionary objectives = p_state["objectives"];

	// Check if objective exists and is part of active quest
	if (!objectives.has(p_objective_id)) {
		return false;
	}
	Dictionary objective = objectives[p_objective_id];
	String quest_id = objective["quest_id"];

	if (!active_quests.has(quest_id)) {
		return false; // Quest not active
	}

	// Complete objective
	Dictionary new_state = p_state.duplicate();
	Dictionary new_objectives = objectives.duplicate();
	Dictionary new_objective = objective.duplicate();

	new_objective["completed"] = true;
	new_objectives[p_objective_id] = new_objective;
	new_state["objectives"] = new_objectives;
	return new_state;
}

static Variant action_talk_to_npc(Dictionary p_state, String p_npc_id) {
	Dictionary npcs = p_state["npcs"];
	Dictionary inventory = p_state["inventory"];

	// Check if NPC exists and is available
	if (!npcs.has(p_npc_id)) {
		return false;
	}
	Dictionary npc = npcs[p_npc_id];
	if (!npc.has("available") || !npc["available"].operator bool()) {
		return false; // NPC not available
	}

	// Talk to NPC (might give item)
	Dictionary new_state = p_state.duplicate();
	Dictionary new_inventory = inventory.duplicate();

	if (npc.has("gives_item")) {
		String item = npc["gives_item"];
		int count = npc.has("gives_count") ? npc["gives_count"].operator int() : 1;
		int current = new_inventory.has(item) ? new_inventory[item].operator int() : 0;
		new_inventory[item] = current + count;
	}

	new_state["inventory"] = new_inventory;
	return new_state;
}

static Variant action_defeat_enemy(Dictionary p_state, String p_enemy_id) {
	Dictionary enemies = p_state["enemies"];
	Dictionary inventory = p_state["inventory"];

	// Check if enemy exists
	if (!enemies.has(p_enemy_id)) {
		return false;
	}
	Dictionary enemy = enemies[p_enemy_id];

	// Need weapon to defeat enemy
	if (!inventory.has("weapon") || inventory["weapon"].operator int() <= 0) {
		return false; // No weapon
	}

	// Defeat enemy
	Dictionary new_state = p_state.duplicate();
	Dictionary new_enemies = enemies.duplicate();
	Dictionary new_enemy = enemy.duplicate();

	new_enemy["defeated"] = true;
	new_enemies[p_enemy_id] = new_enemy;
	new_state["enemies"] = new_enemies;
	return new_state;
}

// Task methods
static Variant task_method_complete_quest(Dictionary p_state, String p_quest_id) {
	Dictionary active_quests = p_state["active_quests"];
	Dictionary objectives = p_state["objectives"];
	Dictionary quests = p_state["quests"];

	// Check if quest is already completed
	if (!active_quests.has(p_quest_id)) {
		// Check if quest needs to be accepted first
		if (quests.has(p_quest_id)) {
			Array subtasks;
			Array accept_task;
			accept_task.push_back("accept_quest_task");
			accept_task.push_back(p_quest_id);
			subtasks.push_back(accept_task);

			Array complete_task;
			complete_task.push_back("complete_quest");
			complete_task.push_back(p_quest_id);
			subtasks.push_back(complete_task);
			return subtasks;
		}
		return false; // Quest doesn't exist
	}

	// Check prerequisites
	Dictionary quest = quests[p_quest_id];
	if (quest.has("prerequisites")) {
		Array prerequisites = quest["prerequisites"];
		Array incomplete_prereqs;
		for (int i = 0; i < prerequisites.size(); i++) {
			String prereq_id = prerequisites[i];
			if (!active_quests.has(prereq_id)) {
				incomplete_prereqs.push_back(prereq_id);
			}
		}

		if (!incomplete_prereqs.is_empty()) {
			// Need to complete prerequisites first
			Array subtasks;
			for (int i = 0; i < incomplete_prereqs.size(); i++) {
				Array prereq_task;
				prereq_task.push_back("complete_quest");
				prereq_task.push_back(incomplete_prereqs[i]);
				subtasks.push_back(prereq_task);
			}

			Array complete_task;
			complete_task.push_back("complete_quest");
			complete_task.push_back(p_quest_id);
			subtasks.push_back(complete_task);
			return subtasks;
		}
	}

	// Get all objectives for this quest
	Array objective_keys = objectives.keys();
	Array quest_objectives;
	for (int i = 0; i < objective_keys.size(); i++) {
		String obj_id = objective_keys[i];
		Dictionary obj = objectives[obj_id];
		if (obj.has("quest_id") && obj["quest_id"] == p_quest_id) {
			if (!obj.has("completed") || !obj["completed"].operator bool()) {
				quest_objectives.push_back(obj_id);
			}
		}
	}

	if (quest_objectives.is_empty()) {
		return Array(); // Quest already completed
	}

	// Complete all objectives
	Array subtasks;
	for (int i = 0; i < quest_objectives.size(); i++) {
		Array obj_task;
		obj_task.push_back("achieve_objective");
		obj_task.push_back(quest_objectives[i]);
		subtasks.push_back(obj_task);
	}
	return subtasks;
}

static Variant task_method_achieve_objective(Dictionary p_state, String p_objective_id) {
	Dictionary objectives = p_state["objectives"];

	if (!objectives.has(p_objective_id)) {
		return false;
	}
	Dictionary objective = objectives[p_objective_id];

	if (objective.has("completed") && objective["completed"].operator bool()) {
		return Array(); // Already completed
	}

	String obj_type = objective.has("type") ? objective["type"] : "unknown";
	Array subtasks;

	if (obj_type == "talk_to_npc") {
		String npc_id = objective["npc_id"];
		Array action;
		action.push_back("action_talk_to_npc");
		action.push_back(npc_id);
		subtasks.push_back(action);

		Array complete_action;
		complete_action.push_back("action_complete_objective");
		complete_action.push_back(p_objective_id);
		subtasks.push_back(complete_action);
	} else if (obj_type == "defeat_enemy") {
		Dictionary inventory = p_state["inventory"];
		if (!inventory.has("weapon") || inventory["weapon"].operator int() <= 0) {
			// Need weapon first
			Array get_weapon_task;
			get_weapon_task.push_back("get_weapon_task");
			subtasks.push_back(get_weapon_task);
		}

		String enemy_id = objective["enemy_id"];
		Array action;
		action.push_back("action_defeat_enemy");
		action.push_back(enemy_id);
		subtasks.push_back(action);

		Array complete_action;
		complete_action.push_back("action_complete_objective");
		complete_action.push_back(p_objective_id);
		subtasks.push_back(complete_action);
	}

	return subtasks;
}

static Variant task_method_accept_quest_task(Dictionary p_state, String p_quest_id) {
	Dictionary active_quests = p_state["active_quests"];
	if (active_quests.has(p_quest_id)) {
		return Array(); // Already accepted
	}

	Array subtasks;
	Array action;
	action.push_back("action_accept_quest");
	action.push_back(p_quest_id);
	subtasks.push_back(action);
	return subtasks;
}

static Variant task_method_get_weapon_task(Dictionary p_state) {
	Dictionary inventory = p_state["inventory"];
	if (inventory.has("weapon") && inventory["weapon"].operator int() > 0) {
		return Array(); // Already have weapon
	}

	// Try to get weapon from NPC
	Array subtasks;
	Array action;
	action.push_back("action_talk_to_npc");
	action.push_back("merchant");
	subtasks.push_back(action);
	return subtasks;
}

// Goal methods
static Variant goal_method_quest_completed(Dictionary p_state, String p_quest_id, Variant p_desired_value) {
	Dictionary active_quests = p_state["active_quests"];
	Dictionary objectives = p_state["objectives"];

	// Check if quest is completed (all objectives done)
	bool is_completed = true;
	Array objective_keys = objectives.keys();
	for (int i = 0; i < objective_keys.size(); i++) {
		String obj_id = objective_keys[i];
		Dictionary obj = objectives[obj_id];
		if (obj.has("quest_id") && obj["quest_id"] == p_quest_id) {
			if (!obj.has("completed") || !obj["completed"].operator bool()) {
				is_completed = false;
				break;
			}
		}
	}

	if (is_completed == p_desired_value.operator bool()) {
		return Array(); // Goal already achieved
	}

	// Need to complete quest
	Array subtasks;
	Array task;
	task.push_back("complete_quest");
	task.push_back(p_quest_id);
	subtasks.push_back(task);
	return subtasks;
}

static Variant goal_method_quest_completed_alternative(Dictionary p_state, String p_quest_id, Variant p_desired_value) {
	// Alternative method that might fail
	Dictionary quests = p_state["quests"];
	if (!quests.has(p_quest_id)) {
		return false; // Quest doesn't exist, this method fails
	}

	// Try to complete quest
	Array subtasks;
	Array task;
	task.push_back("complete_quest");
	task.push_back(p_quest_id);
	subtasks.push_back(task);
	return subtasks;
}

// Multigoal methods
static Variant multigoal_method_quest_chain(Dictionary p_state, Dictionary p_multigoal) {
	// Multigoal format: {"quests": {"quest1": true, "quest2": true}}
	Dictionary goal_quests = PlannerMultigoal::get_goal_conditions_for_variable(p_multigoal, "quests");

	Dictionary goals_not_achieved = PlannerMultigoal::method_goals_not_achieved(p_state, p_multigoal);
	if (goals_not_achieved.is_empty()) {
		return Array(); // All goals achieved
	}

	// Need to complete quests
	Array subtasks;
	Array goal_keys = goal_quests.keys();
	for (int i = 0; i < goal_keys.size(); i++) {
		String quest_id = goal_keys[i];
		Array task;
		task.push_back("complete_quest");
		task.push_back(quest_id);
		subtasks.push_back(task);
	}
	return subtasks;
}

// Setup quest system domain
static Ref<PlannerDomain> setup_quest_system_domain() {
	Ref<PlannerDomain> domain = memnew(PlannerDomain);

	// Add actions
	TypedArray<Callable> actions;
	actions.push_back(callable_mp_static(&action_accept_quest));
	actions.push_back(callable_mp_static(&action_complete_objective));
	actions.push_back(callable_mp_static(&action_talk_to_npc));
	actions.push_back(callable_mp_static(&action_defeat_enemy));
	domain->add_actions(actions);

	// Add task methods
	TypedArray<Callable> task_methods;
	task_methods.push_back(callable_mp_static(&task_method_complete_quest));
	domain->add_task_methods("complete_quest", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_achieve_objective));
	domain->add_task_methods("achieve_objective", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_accept_quest_task));
	domain->add_task_methods("accept_quest_task", task_methods);

	task_methods.clear();
	task_methods.push_back(callable_mp_static(&task_method_get_weapon_task));
	domain->add_task_methods("get_weapon_task", task_methods);

	// Add goal methods (with multiple methods for backtracking)
	TypedArray<Callable> goal_methods;
	goal_methods.push_back(callable_mp_static(&goal_method_quest_completed));
	goal_methods.push_back(callable_mp_static(&goal_method_quest_completed_alternative));
	domain->add_unigoal_methods("quest_completed", goal_methods);

	// Add multigoal methods
	TypedArray<Callable> multigoal_methods;
	multigoal_methods.push_back(callable_mp_static(&multigoal_method_quest_chain));
	domain->add_multigoal_methods(multigoal_methods);

	// Set up action dictionary
	Dictionary action_dict;
	action_dict["action_accept_quest"] = callable_mp_static(&action_accept_quest);
	action_dict["action_complete_objective"] = callable_mp_static(&action_complete_objective);
	action_dict["action_talk_to_npc"] = callable_mp_static(&action_talk_to_npc);
	action_dict["action_defeat_enemy"] = callable_mp_static(&action_defeat_enemy);
	domain->action_dictionary = action_dict;

	return domain;
}

} // namespace TestGameDomainsBacktracking
#endif // TOOLS_ENABLED
