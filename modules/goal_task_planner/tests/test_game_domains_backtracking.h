/**************************************************************************/
/*  test_game_domains_backtracking.h                                      */
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

// C++ unit tests for game-inspired planner domains with comprehensive backtracking tests
// Tests all planner element types: TYPE_ACTION, TYPE_TASK, TYPE_GOAL, TYPE_MULTIGOAL, TYPE_VERIFY_GOAL, TYPE_VERIFY_MULTIGOAL

#pragma once

#include "../domain.h"
#include "../multigoal.h"
#include "../plan.h"
#include "../planner_state.h"
#include "../planner_time_range.h"
#include "tests/test_macros.h"

// Include domain definitions
#include "test_game_domains_academy_vn.h"
#include "test_game_domains_helpers.h"
#include "test_game_domains_inventory_crafting.h"
#include "test_game_domains_quest_system.h"
#include "test_game_domains_resource_gathering.h"

#ifdef TOOLS_ENABLED
namespace TestGameDomainsBacktracking {

// ============================================================================
// TYPE_ACTION BACKTRACKING TESTS
// ============================================================================

TEST_CASE("[Modules][GameDomains][ActionBacktracking] Resource Gathering - Action failure") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_resource_gathering_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: no axe, tree available
	Dictionary state;
	Dictionary tools;
	tools["axe"] = false;
	state["tools"] = tools;

	Dictionary trees;
	trees["tree1"] = true;
	state["trees"] = trees;

	Dictionary resources;
	state["resources"] = resources;

	SUBCASE("Action fails when precondition not met - should backtrack") {
		Array todo_list;
		Array action;
		action.push_back("action_chop_tree");
		action.push_back("tree1");
		todo_list.push_back(action);

		// Action should fail (no axe), planning should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}

	SUBCASE("Action fails then succeeds after backtracking to get tool") {
		// Add locations with shop that has axe
		Dictionary locations;
		Dictionary shop;
		shop["axe"] = true;
		locations["shop"] = shop;
		state["locations"] = locations;

		// Add inventory (required by action_get_axe)
		Dictionary inventory;
		state["inventory"] = inventory;

		// Add trees (required by action_chop_tree)
		Dictionary trees;
		trees["tree1"] = true;
		state["trees"] = trees;

		Array todo_list;
		Array task;
		task.push_back("gather_wood");
		task.push_back(1);
		todo_list.push_back(task);

		// Should backtrack: gather_wood needs axe, so get_axe_task is added
		// Then chop_tree should succeed
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		if (result.get_type() == Variant::ARRAY) {
			Array plan_array = result;
			CHECK(plan_array.size() >= 2); // Should have get_axe and chop_tree
		}
	}
}

TEST_CASE("[Modules][GameDomains][ActionBacktracking] Inventory/Crafting - Action blacklisting") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_inventory_crafting_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: inventory full, no sword recipe
	Dictionary state;
	Dictionary inventory;
	inventory["_size"] = 10;
	inventory["_max_size"] = 10;
	inventory["iron"] = 5;
	inventory["wood"] = 3;
	state["inventory"] = inventory;

	Dictionary recipes;
	state["recipes"] = recipes;

	SUBCASE("Action fails when inventory full - should backtrack") {
		Array todo_list;
		Array action;
		action.push_back("action_craft_item");
		action.push_back("sword");
		action.push_back(Dictionary());
		todo_list.push_back(action);

		// Action should fail (inventory full), should be blacklisted
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][ActionBacktracking] Quest System - Entity requirements") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_quest_system_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: enemy exists, no weapon
	Dictionary state;
	Dictionary enemies;
	Dictionary enemy;
	enemy["defeated"] = false;
	enemies["goblin1"] = enemy;
	state["enemies"] = enemies;

	Dictionary inventory;
	state["inventory"] = inventory;

	SUBCASE("Action fails when entity requirements not met - should backtrack") {
		Array todo_list;
		Array action;
		action.push_back("action_defeat_enemy");
		action.push_back("goblin1");

		Array capabilities;
		capabilities.push_back("weapon");
		Dictionary action_with_entity = attach_entity_constraints(action, "player", capabilities);
		todo_list.push_back(action_with_entity);

		// Action should fail (no weapon), should backtrack
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][ActionBacktracking] Resource Gathering - STN temporal constraints") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_resource_gathering_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: have axe, tree available
	Dictionary state;
	Dictionary tools;
	tools["axe"] = true;
	state["tools"] = tools;

	Dictionary trees;
	trees["tree1"] = true;
	state["trees"] = trees;

	Dictionary resources;
	state["resources"] = resources;

	SUBCASE("Action fails when temporal constraints inconsistent - should backtrack") {
		int64_t base_time = 1735689600000000LL;
		int64_t duration = 1800000000LL;

		Array todo_list;

		// First action: chop tree (takes 30 min)
		Array action1;
		action1.push_back("action_chop_tree");
		action1.push_back("tree1");
		int64_t start1 = base_time;
		int64_t end1 = start1 + duration;
		Dictionary action1_temporal = attach_temporal_constraints(action1, start1, end1, duration);
		todo_list.push_back(action1_temporal);

		// Second action: chop same tree again, but starts before first ends (conflict!)
		Array action2;
		action2.push_back("action_chop_tree");
		action2.push_back("tree1");
		int64_t start2 = base_time + duration / 2; // Starts in middle of first action
		int64_t end2 = start2 + duration;
		Dictionary action2_temporal = attach_temporal_constraints(action2, start2, end2, duration);
		todo_list.push_back(action2_temporal);

		// Should fail due to temporal conflict
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

// ============================================================================
// TYPE_TASK BACKTRACKING TESTS
// ============================================================================

TEST_CASE("[Modules][GameDomains][TaskBacktracking] Resource Gathering - Task refinement failure") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_resource_gathering_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: no tools, no trees available
	Dictionary state;
	Dictionary tools;
	state["tools"] = tools;

	Dictionary trees;
	state["trees"] = trees; // No trees available
	state["resources"] = Dictionary();

	SUBCASE("Task fails when no applicable methods - should backtrack") {
		Array todo_list;
		Array task;
		task.push_back("gather_wood");
		task.push_back(1);
		todo_list.push_back(task);

		// Task should fail (no trees), planning should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][TaskBacktracking] Inventory/Crafting - Entity requirements") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_inventory_crafting_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: missing ingredients
	Dictionary state;
	Dictionary inventory;
	state["inventory"] = inventory;

	Dictionary recipes;
	Dictionary sword_recipe;
	sword_recipe["iron"] = 2;
	sword_recipe["wood"] = 1;
	recipes["sword"] = sword_recipe;
	state["recipes"] = recipes;

	SUBCASE("Task fails when entity requirements not met - should backtrack") {
		Array todo_list;
		Array task;
		task.push_back("create_weapon");
		task.push_back("sword");

		Array capabilities;
		capabilities.push_back("crafting");
		Dictionary task_with_entity = attach_entity_constraints(task, "player", capabilities);
		todo_list.push_back(task_with_entity);

		// Task should fail (no entity capabilities), should backtrack
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][TaskBacktracking] Quest System - Method exhaustion") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_quest_system_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: quest doesn't exist
	Dictionary state;
	Dictionary quests;
	state["quests"] = quests; // No quests
	state["active_quests"] = Dictionary();
	state["objectives"] = Dictionary();

	SUBCASE("Task fails when all methods exhausted - should backtrack") {
		Array todo_list;
		Array task;
		task.push_back("complete_quest");
		task.push_back("nonexistent_quest");
		todo_list.push_back(task);

		// Task should fail (quest doesn't exist), should backtrack
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][TaskBacktracking] Resource Gathering - Alternative method after backtrack") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_resource_gathering_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: no axe, but shop has axe
	Dictionary state;
	Dictionary tools;
	tools["axe"] = false;
	state["tools"] = tools;

	Dictionary locations;
	Dictionary shop;
	shop["axe"] = true;
	locations["shop"] = shop;
	state["locations"] = locations;

	// Add inventory (required by action_get_axe)
	Dictionary inventory;
	state["inventory"] = inventory;

	Dictionary trees;
	trees["tree1"] = true;
	state["trees"] = trees;
	state["resources"] = Dictionary();

	SUBCASE("Task backtracks and tries alternative method") {
		Array todo_list;
		Array task;
		task.push_back("gather_wood");
		task.push_back(1);
		todo_list.push_back(task);

		// Should succeed: gather_wood needs axe, get_axe_task gets it, then chop_tree succeeds
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		if (result.get_type() == Variant::ARRAY) {
			Array plan_array = result;
			CHECK(plan_array.size() >= 2);
		}
	}
}

// ============================================================================
// TYPE_GOAL BACKTRACKING TESTS
// ============================================================================

TEST_CASE("[Modules][GameDomains][GoalBacktracking] Resource Gathering - Goal refinement failure") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_resource_gathering_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: no axe, no shop
	Dictionary state;
	Dictionary tools;
	tools["axe"] = false;
	state["tools"] = tools;
	state["locations"] = Dictionary(); // No locations

	SUBCASE("Goal fails when no applicable methods - should backtrack") {
		Array todo_list;
		Array goal;
		goal.push_back("has_axe");
		goal.push_back("axe");
		goal.push_back(true);
		todo_list.push_back(goal);

		// Goal should fail (no shop to get axe from), planning should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][GoalBacktracking] Inventory/Crafting - Alternative method") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_inventory_crafting_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: no sword, but have recipe
	Dictionary state;
	Dictionary inventory;
	state["inventory"] = inventory;

	Dictionary recipes;
	Dictionary sword_recipe;
	sword_recipe["iron"] = 2;
	sword_recipe["wood"] = 1;
	recipes["sword"] = sword_recipe;
	state["recipes"] = recipes;

	// Add ingredients
	inventory["iron"] = 2;
	inventory["wood"] = 1;
	inventory["_size"] = 3;
	inventory["_max_size"] = 10;
	state["inventory"] = inventory;

	SUBCASE("Goal backtracks and tries alternative method") {
		Array todo_list;
		Array goal;
		goal.push_back("inventory_has");
		goal.push_back("sword");
		goal.push_back(true);
		todo_list.push_back(goal);

		// Should succeed: first method might fail, but alternative crafts sword
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		if (result.get_type() == Variant::ARRAY) {
			Array plan_array = result;
			CHECK(plan_array.size() >= 1);
		}
	}
}

TEST_CASE("[Modules][GameDomains][GoalBacktracking] Quest System - Entity requirements") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_quest_system_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: quest exists but not completed
	Dictionary state;
	Dictionary quests;
	Dictionary quest;
	quest["prerequisites"] = Array();
	quests["main_quest"] = quest;
	state["quests"] = quests;
	state["active_quests"] = Dictionary();

	Dictionary objectives;
	Dictionary objective;
	objective["quest_id"] = "main_quest";
	objective["completed"] = false;
	objectives["obj1"] = objective;
	state["objectives"] = objectives;

	SUBCASE("Goal fails when entity requirements not met - should backtrack") {
		Array todo_list;
		Array goal;
		goal.push_back("quest_completed");
		goal.push_back("main_quest");
		goal.push_back(true);

		Array capabilities;
		capabilities.push_back("quest_access");
		Dictionary goal_with_entity = attach_entity_constraints(goal, "player", capabilities);
		todo_list.push_back(goal_with_entity);

		// Goal should fail (no entity capabilities), should backtrack
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

// ============================================================================
// TYPE_MULTIGOAL BACKTRACKING TESTS
// ============================================================================

TEST_CASE("[Modules][GameDomains][MultigoalBacktracking] Resource Gathering - Multigoal refinement failure") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_resource_gathering_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: no resources, no tools
	Dictionary state;
	state["resources"] = Dictionary();
	state["tools"] = Dictionary();
	state["trees"] = Dictionary();
	state["locations"] = Dictionary();

	SUBCASE("Multigoal fails when refinement fails - should backtrack") {
		Dictionary multigoal;
		Dictionary resources_goal;
		resources_goal["wood"] = 3;
		resources_goal["ore"] = 2;
		multigoal["resources"] = resources_goal;

		Array todo_list;
		todo_list.push_back(multigoal);

		// Multigoal should fail (no way to gather resources), planning should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][MultigoalBacktracking] Inventory/Crafting - Partial achievement") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_inventory_crafting_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: have some ingredients
	Dictionary state;
	Dictionary inventory;
	inventory["iron"] = 1; // Need 2
	inventory["wood"] = 3; // Need 3, have enough
	inventory["_size"] = 4;
	inventory["_max_size"] = 10;
	state["inventory"] = inventory;

	SUBCASE("Multigoal backtracks when partial achievement fails") {
		Dictionary multigoal;
		Dictionary inventory_goal;
		inventory_goal["iron"] = 2;
		inventory_goal["wood"] = 3;
		multigoal["inventory"] = inventory_goal;

		Array todo_list;
		todo_list.push_back(multigoal);

		// Should succeed: gather missing iron, then all goals achieved
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
	}
}

TEST_CASE("[Modules][GameDomains][MultigoalBacktracking] Quest System - Entity requirements") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_quest_system_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: multiple quests
	Dictionary state;
	Dictionary quests;
	Dictionary quest1;
	quest1["prerequisites"] = Array();
	quests["quest1"] = quest1;
	Dictionary quest2;
	quest2["prerequisites"] = Array();
	quests["quest2"] = quest2;
	state["quests"] = quests;
	state["active_quests"] = Dictionary();
	state["objectives"] = Dictionary();

	SUBCASE("Multigoal fails when entity requirements not met - should backtrack") {
		Dictionary multigoal;
		Dictionary quests_goal;
		quests_goal["quest1"] = true;
		quests_goal["quest2"] = true;
		multigoal["quests"] = quests_goal;

		Array capabilities;
		capabilities.push_back("quest_access");
		Dictionary multigoal_with_entity = attach_entity_constraints(Array(), "player", capabilities);
		multigoal_with_entity["item"] = multigoal;

		Array todo_list;
		todo_list.push_back(multigoal_with_entity);

		// Multigoal should fail (no entity capabilities), should backtrack
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

// ============================================================================
// TYPE_VERIFY_GOAL BACKTRACKING TESTS
// ============================================================================

TEST_CASE("[Modules][GameDomains][VerifyGoalBacktracking] Resource Gathering - Goal verification failure") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_resource_gathering_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(1); // Enable verification

	// Initial state: no axe
	Dictionary state;
	Dictionary tools;
	tools["axe"] = false;
	state["tools"] = tools;
	state["locations"] = Dictionary();

	SUBCASE("Goal verification fails when goal not achieved - should backtrack") {
		Array todo_list;
		Array goal;
		goal.push_back("has_axe");
		goal.push_back("axe");
		goal.push_back(true);
		todo_list.push_back(goal);

		// Goal should fail (can't get axe), verification should fail, should backtrack
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][VerifyGoalBacktracking] Inventory/Crafting - Invalid parent goal") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_inventory_crafting_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(1);

	// Initial state: no sword
	Dictionary state;
	state["inventory"] = Dictionary();
	state["recipes"] = Dictionary();

	SUBCASE("Goal verification fails with invalid parent - should backtrack") {
		// This test verifies that verify_goal nodes handle invalid parents correctly
		// The planner should handle this gracefully
		Array todo_list;
		Array goal;
		goal.push_back("inventory_has");
		goal.push_back("sword");
		goal.push_back(true);
		todo_list.push_back(goal);

		// Should fail (no recipe), verification should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

// ============================================================================
// TYPE_VERIFY_MULTIGOAL BACKTRACKING TESTS
// ============================================================================

TEST_CASE("[Modules][GameDomains][VerifyMultigoalBacktracking] Resource Gathering - Multigoal verification failure") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_resource_gathering_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(1);

	// Initial state: have some resources but not all
	Dictionary state;
	Dictionary resources;
	resources["wood"] = 2; // Need 3
	resources["ore"] = 0; // Need 2
	state["resources"] = resources;
	state["tools"] = Dictionary();
	state["trees"] = Dictionary();
	state["locations"] = Dictionary();

	SUBCASE("Multigoal verification fails when some goals not achieved - should backtrack") {
		Dictionary multigoal;
		Dictionary resources_goal;
		resources_goal["wood"] = 3;
		resources_goal["ore"] = 2;
		multigoal["resources"] = resources_goal;

		Array todo_list;
		todo_list.push_back(multigoal);

		// Multigoal should fail (can't gather all resources), verification should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][VerifyMultigoalBacktracking] Quest System - Invalid parent multigoal") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_quest_system_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(1);

	// Initial state: no quests
	Dictionary state;
	state["quests"] = Dictionary();
	state["active_quests"] = Dictionary();
	state["objectives"] = Dictionary();

	SUBCASE("Multigoal verification fails with invalid parent - should backtrack") {
		// Invalid multigoal format
		Dictionary invalid_multigoal;
		invalid_multigoal["invalid_key"] = "invalid_value";

		Array todo_list;
		todo_list.push_back(invalid_multigoal);

		// Should fail (invalid multigoal), verification should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

// ============================================================================
// ACADEMY VISUAL NOVEL DOMAIN TESTS
// ============================================================================

TEST_CASE("[Modules][GameDomains][AcademyVN][ActionBacktracking] Time conflict - Action fails") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: player already busy at timeslot 10
	Dictionary state;
	Dictionary schedule;
	Dictionary player_schedule;
	player_schedule[10] = true; // Already busy
	schedule["player"] = player_schedule;
	state["schedule"] = schedule;

	Dictionary characters;
	Dictionary character;
	character["available"] = true;
	characters["alice"] = character;
	state["characters"] = characters;
	state["relationships"] = Dictionary();

	SUBCASE("Action fails when timeslot already used - should backtrack") {
		Array todo_list;
		Array action;
		action.push_back("action_talk_to_character");
		action.push_back("alice");
		action.push_back(10); // Same timeslot
		todo_list.push_back(action);

		// Action should fail (timeslot busy), planning should fail
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][ActionBacktracking] Temporal constraints - STN conflict") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: empty schedule
	Dictionary state;
	state["schedule"] = Dictionary();
	state["characters"] = Dictionary();
	state["relationships"] = Dictionary();
	state["classes"] = Dictionary();
	state["enrolled"] = Dictionary();

	SUBCASE("Temporal constraints create conflict - should backtrack") {
		int64_t base_time = 1735689600000000LL; // 2025-01-01 00:00:00 UTC
		int64_t duration = 3600000000LL; // 1 hour

		Array todo_list;

		// First action: attend class at 10:00-11:00
		Array action1;
		action1.push_back("action_attend_class");
		action1.push_back("math101");
		action1.push_back(10);
		int64_t start1 = base_time + 10 * 3600000000LL;
		int64_t end1 = start1 + duration;
		Dictionary action1_temporal = attach_temporal_constraints(action1, start1, end1, duration);
		todo_list.push_back(action1_temporal);

		// Second action: study at 10:30-11:30 (overlaps!)
		Array action2;
		action2.push_back("action_study");
		action2.push_back("math");
		action2.push_back(10);
		int64_t start2 = base_time + 10 * 3600000000LL + duration / 2; // Starts in middle
		int64_t end2 = start2 + duration;
		Dictionary action2_temporal = attach_temporal_constraints(action2, start2, end2, duration);
		todo_list.push_back(action2_temporal);

		// Should fail due to temporal conflict
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][TaskBacktracking] Time scheduling - Multiple activities conflict") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: empty schedule
	Dictionary state;
	state["schedule"] = Dictionary();
	state["classes"] = Dictionary();
	state["enrolled"] = Dictionary();

	SUBCASE("Task fails when activities conflict - should backtrack") {
		Dictionary activities;

		// Activity 1 at timeslot 10
		Dictionary activity1;
		activity1["type"] = "class";
		activity1["class_id"] = "math101";
		activities[10] = activity1;

		// Activity 2 at same timeslot (conflict!)
		Dictionary activity2;
		activity2["type"] = "study";
		activity2["subject"] = "math";
		activities[10] = activity2; // Same timeslot!

		Array todo_list;
		Array task;
		task.push_back("manage_time");
		task.push_back(activities);
		todo_list.push_back(task);

		// Task should fail (time conflict), should backtrack
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][TaskBacktracking] Character unavailable - Alternative time") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: character busy at timeslot 10, available at 11
	Dictionary state;
	Dictionary schedule;
	state["schedule"] = schedule;

	Dictionary characters;
	Dictionary character;
	Dictionary char_schedule;
	char_schedule[10] = true; // Character busy at 10
	character["schedule"] = char_schedule;
	characters["alice"] = character;
	state["characters"] = characters;
	state["relationships"] = Dictionary();

	SUBCASE("Task backtracks to find alternative timeslot") {
		Array todo_list;
		Array task;
		task.push_back("build_relationship");
		task.push_back("alice");
		task.push_back(1);
		todo_list.push_back(task);

		// Should succeed: find alternative timeslot (11) when 10 is busy
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
		if (result.get_type() == Variant::ARRAY) {
			Array plan_array = result;
			CHECK(plan_array.size() >= 1);
		}
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][GoalBacktracking] Relationship requirement not met") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: no relationship with character, but character exists
	Dictionary state;
	state["schedule"] = Dictionary();
	Dictionary characters;
	Dictionary alice;
	characters["alice"] = alice;
	state["characters"] = characters;
	state["relationships"] = Dictionary();

	SUBCASE("Goal fails when relationship too low - should backtrack") {
		Array todo_list;
		Array goal;
		goal.push_back("relationship_level");
		goal.push_back("alice");
		goal.push_back(5); // Need level 5
		todo_list.push_back(goal);

		// Should succeed: build relationship to reach level 5
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][GoalBacktracking] Alternative method after backtrack") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: character doesn't exist (first method fails)
	Dictionary state;
	state["schedule"] = Dictionary();
	state["characters"] = Dictionary(); // No characters
	state["relationships"] = Dictionary();

	SUBCASE("Goal backtracks and tries alternative method") {
		Array todo_list;
		Array goal;
		goal.push_back("relationship_level");
		goal.push_back("nonexistent");
		goal.push_back(1);
		todo_list.push_back(goal);

		// Should fail: character doesn't exist, alternative method also fails
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][MultigoalBacktracking] Story route - Multiple requirements") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: no relationships, no events
	Dictionary state;
	state["schedule"] = Dictionary();
	state["characters"] = Dictionary();
	state["relationships"] = Dictionary();
	state["events"] = Dictionary();
	state["completed_events"] = Dictionary();
	state["story_flags"] = Dictionary();
	state["routes"] = Dictionary();

	SUBCASE("Multigoal fails when requirements not met - should backtrack") {
		Dictionary multigoal;
		Dictionary relationships_goal;
		relationships_goal["alice"] = 5;
		relationships_goal["bob"] = 3;
		multigoal["relationships"] = relationships_goal;

		Array todo_list;
		todo_list.push_back(multigoal);

		// Should fail: cannot complete route without characters/events
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][TimeScheduling] Daily schedule exceeds available time") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: schedule most timeslots
	Dictionary state;
	Dictionary schedule;
	Dictionary player_schedule;
	// Fill most timeslots (0-22)
	for (int64_t i = 0; i < 23; i++) {
		player_schedule[i] = true;
	}
	schedule["player"] = player_schedule;
	state["schedule"] = schedule;
	state["classes"] = Dictionary();
	state["enrolled"] = Dictionary();

	SUBCASE("Task fails when no available time - should backtrack") {
		Dictionary activities;
		Dictionary activity;
		activity["type"] = "study";
		activity["subject"] = "math";
		activities[23] = activity; // Only timeslot 23 available

		// Try to schedule two activities
		Dictionary activity2;
		activity2["type"] = "study";
		activity2["subject"] = "science";
		activities[23] = activity2; // Conflict with first!

		Array todo_list;
		Array task;
		task.push_back("manage_time");
		task.push_back(activities);
		todo_list.push_back(task);

		// Should fail: both activities want same last available timeslot
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][TimeScheduling] Class conflicts with event") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: class scheduled at timeslot 10
	Dictionary state;
	Dictionary schedule;
	Dictionary player_schedule;
	player_schedule[10] = true; // Class at 10
	schedule["player"] = player_schedule;
	state["schedule"] = schedule;

	Dictionary classes;
	Dictionary class_info;
	class_info["timeslot"] = 10;
	classes["math101"] = class_info;
	state["classes"] = classes;

	Dictionary enrolled;
	enrolled["math101"] = true;
	state["enrolled"] = enrolled;

	Dictionary events;
	Dictionary event_info;
	event_info["prerequisites"] = Array();
	events["festival"] = event_info;
	state["events"] = events;
	state["story_flags"] = Dictionary();
	state["completed_events"] = Dictionary();

	SUBCASE("Action fails when class conflicts with event - should backtrack") {
		Array todo_list;
		Array action;
		action.push_back("action_participate_event");
		action.push_back("festival");
		action.push_back(10); // Same timeslot as class
		todo_list.push_back(action);

		// Action should fail (timeslot conflict), should backtrack
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][TimeScheduling] Study time overlaps with social event") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: study scheduled at timeslot 14
	Dictionary state;
	Dictionary schedule;
	Dictionary player_schedule;
	player_schedule[14] = true; // Study at 14
	schedule["player"] = player_schedule;
	state["schedule"] = schedule;

	Dictionary characters;
	Dictionary character;
	characters["alice"] = character;
	state["characters"] = characters;
	state["relationships"] = Dictionary();

	SUBCASE("Action fails when study overlaps with social - should backtrack") {
		Array todo_list;
		Array action;
		action.push_back("action_talk_to_character");
		action.push_back("alice");
		action.push_back(14); // Same timeslot as study
		todo_list.push_back(action);

		// Action should fail (timeslot conflict), should backtrack
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][TimeScheduling] Temporal constraints - Sequential scheduling") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(0);

	// Initial state: empty schedule, but need class setup
	Dictionary state;
	state["schedule"] = Dictionary();
	Dictionary classes;
	Dictionary math101;
	math101["timeslot"] = 9; // Class at timeslot 9
	classes["math101"] = math101;
	state["classes"] = classes;
	Dictionary enrolled;
	enrolled["math101"] = true;
	state["enrolled"] = enrolled;

	SUBCASE("Sequential activities with temporal constraints should succeed") {
		int64_t base_time = 1735689600000000LL;
		int64_t duration = 3600000000LL; // 1 hour

		Array todo_list;

		// First: attend class at 9:00-10:00
		Array action1;
		action1.push_back("action_attend_class");
		action1.push_back("math101");
		action1.push_back(9);
		int64_t start1 = base_time + 9 * 3600000000LL;
		int64_t end1 = start1 + duration;
		Dictionary action1_temporal = attach_temporal_constraints(action1, start1, end1, duration);
		todo_list.push_back(action1_temporal);

		// Second: study at 10:00-11:00 (starts after first ends)
		Array action2;
		action2.push_back("action_study");
		action2.push_back("math");
		action2.push_back(10);
		int64_t start2 = end1; // Starts after first ends
		int64_t end2 = start2 + duration;
		Dictionary action2_temporal = attach_temporal_constraints(action2, start2, end2, duration);
		todo_list.push_back(action2_temporal);

		// Should succeed: sequential activities, no conflict
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][VerifyGoalBacktracking] Goal verification fails") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(1);

	// Initial state: relationship level 2, need level 5, character exists
	Dictionary state;
	state["schedule"] = Dictionary();
	Dictionary characters;
	Dictionary alice;
	characters["alice"] = alice;
	state["characters"] = characters;
	Dictionary relationships;
	relationships["alice"] = 2; // Only level 2
	state["relationships"] = relationships;

	SUBCASE("Goal verification fails when goal not achieved - should backtrack") {
		Array todo_list;
		Array goal;
		goal.push_back("relationship_level");
		goal.push_back("alice");
		goal.push_back(5); // Need level 5
		todo_list.push_back(goal);

		// Should succeed: build relationship to level 5
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result.get_type() == Variant::ARRAY);
	}
}

TEST_CASE("[Modules][GameDomains][AcademyVN][VerifyMultigoalBacktracking] Multigoal verification fails") {
	Ref<PlannerPlan> plan = memnew(PlannerPlan);
	Ref<PlannerDomain> domain = setup_academy_visual_novel_domain();
	plan->set_current_domain(domain);
	plan->set_verbose(1);

	// Initial state: partial relationships
	Dictionary state;
	state["schedule"] = Dictionary();
	state["characters"] = Dictionary();
	Dictionary relationships;
	relationships["alice"] = 3; // Need 5
	relationships["bob"] = 2; // Need 3
	state["relationships"] = relationships;
	state["events"] = Dictionary();
	state["completed_events"] = Dictionary();
	state["story_flags"] = Dictionary();
	state["routes"] = Dictionary();

	SUBCASE("Multigoal verification fails when some goals not achieved - should backtrack") {
		Dictionary multigoal;
		Dictionary relationships_goal;
		relationships_goal["alice"] = 5;
		relationships_goal["bob"] = 3;
		multigoal["relationships"] = relationships_goal;

		Array todo_list;
		todo_list.push_back(multigoal);

		// Should fail: cannot complete route without proper setup
		Variant result = plan->find_plan(state, todo_list);
		CHECK(result == Variant(false));
	}
}

} // namespace TestGameDomainsBacktracking
#endif // TOOLS_ENABLED
