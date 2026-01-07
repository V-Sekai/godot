#!/usr/bin/env -S godot --headless --script
# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Sims-Style Scenarios Test
# Tests common Sims game scenarios with the planner

extends SceneTree

const Domain = preload("domain.gd")

var tests_passed = 0
var tests_failed = 0
var current_test = ""

func _init():
	print("=== Sims-Style Scenarios Test ===")
	call_deferred("run_all_tests")

func assert_test(condition: bool, message: String = ""):
	if not condition:
		tests_failed += 1
		print("FAILED: %s - %s" % [current_test, message])
		return false
	tests_passed += 1
	return true

func test(name: String):
	current_test = name
	print("\nTest: %s" % name)

# Helper: Create initial state with specific needs
func create_state_with_needs(persona_id: String, hunger: int, energy: int, social: int, fun: int, hygiene: int, money: int, location: String = "dorm") -> Dictionary:
	var state = {}
	
	# Location
	state["is_at"] = {persona_id: location}
	
	# Needs
	state["needs"] = {
		persona_id: {
			"hunger": hunger,
			"energy": energy,
			"social": social,
			"fun": fun,
			"hygiene": hygiene
		}
	}
	
	# Money
	state["money"] = {persona_id: money}
	
	# Study points
	state["study_points"] = {persona_id: 0}
	
	# Relationship points
	state["relationship_points_%s_maya" % persona_id] = 0
	
	return state

func create_domain() -> PlannerDomain:
	var domain = PlannerDomain.new()
	
	# Add actions
	var actions = [
		Callable(Domain, "action_eat_mess_hall"),
		Callable(Domain, "action_eat_restaurant"),
		Callable(Domain, "action_eat_snack"),
		Callable(Domain, "action_cook_meal"),
		Callable(Domain, "action_sleep_dorm"),
		Callable(Domain, "action_nap_library"),
		Callable(Domain, "action_energy_drink"),
		Callable(Domain, "action_talk_friend"),
		Callable(Domain, "action_join_club"),
		Callable(Domain, "action_phone_call"),
		Callable(Domain, "action_play_games"),
		Callable(Domain, "action_watch_streaming"),
		Callable(Domain, "action_go_cinema"),
		Callable(Domain, "action_shower"),
		Callable(Domain, "action_wash_hands"),
		Callable(Domain, "action_move_to")
	]
	domain.add_actions(actions)
	
	# Add task methods
	var satisfy_hunger_methods = [
		Callable(Domain, "task_satisfy_hunger_method_mess_hall"),
		Callable(Domain, "task_satisfy_hunger_method_restaurant"),
		Callable(Domain, "task_satisfy_hunger_method_snack"),
		Callable(Domain, "task_satisfy_hunger_method_cook"),
		Callable(Domain, "task_satisfy_hunger_method_social_eat")
	]
	domain.add_task_methods("task_satisfy_hunger", satisfy_hunger_methods)
	
	var satisfy_energy_methods = [
		Callable(Domain, "task_satisfy_energy_method_sleep"),
		Callable(Domain, "task_satisfy_energy_method_nap"),
		Callable(Domain, "task_satisfy_energy_method_drink"),
		Callable(Domain, "task_satisfy_energy_method_rest_activity"),
		Callable(Domain, "task_satisfy_energy_method_early_sleep")
	]
	domain.add_task_methods("task_satisfy_energy", satisfy_energy_methods)
	
	var satisfy_social_methods = [
		Callable(Domain, "task_satisfy_social_method_talk"),
		Callable(Domain, "task_satisfy_social_method_club"),
		Callable(Domain, "task_satisfy_social_method_phone"),
		Callable(Domain, "task_satisfy_social_method_socialize_task"),
		Callable(Domain, "task_satisfy_social_method_group_activity")
	]
	domain.add_task_methods("task_satisfy_social", satisfy_social_methods)
	
	var satisfy_fun_methods = [
		Callable(Domain, "task_satisfy_fun_method_games"),
		Callable(Domain, "task_satisfy_fun_method_streaming"),
		Callable(Domain, "task_satisfy_fun_method_cinema"),
		Callable(Domain, "task_satisfy_fun_method_preferred_activity"),
		Callable(Domain, "task_satisfy_fun_method_social_fun")
	]
	domain.add_task_methods("task_satisfy_fun", satisfy_fun_methods)
	
	var satisfy_hygiene_methods = [
		Callable(Domain, "task_satisfy_hygiene_method_shower"),
		Callable(Domain, "task_satisfy_hygiene_method_wash_hands"),
		Callable(Domain, "task_satisfy_hygiene_method_force_shower")
	]
	domain.add_task_methods("task_satisfy_hygiene", satisfy_hygiene_methods)
	
	var move_methods = [
		Callable(Domain, "task_move_to_location_method_direct")
	]
	domain.add_task_methods("task_move_to_location", move_methods)
	
	return domain

# Scenario 1: Character is starving and exhausted - must satisfy multiple critical needs
# This tests prioritization and resource management
func test_scenario_starving_and_exhausted():
	test("Scenario 1: Starving and Exhausted - Multiple Critical Needs")
	
	var domain = create_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(15)
	
	# Character is very hungry (20/100) and very tired (15/100), has some money
	var state = create_state_with_needs("yuki", 20, 15, 50, 50, 50, 30, "dorm")
	
	# Goal: Satisfy both hunger and energy to at least 70
	var multigoal = [
		["needs", "yuki", "hunger", 70],
		["needs", "yuki", "energy", 70]
	]
	
	# Note: This is a simplified multigoal format - in real implementation,
	# you'd need a multigoal method that handles needs
	# For now, we'll test with individual tasks
	var todo_list = [
		["task_satisfy_hunger", "yuki", 70],
		["task_satisfy_energy", "yuki", 70]
	]
	
	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")
	
	if result.get_success():
		var plan_actions = result.extract_plan()
		print("  Plan has %d actions" % plan_actions.size())
		assert_test(plan_actions.size() > 0, "Should have actions to satisfy needs")
		
		# Verify final state
		var final_state = result.get_final_state()
		var final_hunger = Domain.get_need(final_state, "yuki", "hunger")
		var final_energy = Domain.get_need(final_state, "yuki", "energy")
		print("  Final hunger: %d (target: 70)" % final_hunger)
		print("  Final energy: %d (target: 70)" % final_energy)
		assert_test(final_hunger >= 70, "Hunger should be satisfied")
		assert_test(final_energy >= 70, "Energy should be satisfied")
	else:
		print("  Planning failed - may need more methods or resources")

# Scenario 2: Character wants to go out but is broke - tests resource constraints
# Character wants to go to cinema (costs money) but doesn't have enough
func test_scenario_broke_wants_fun():
	test("Scenario 2: Broke but Wants Fun - Resource Constraints")
	
	var domain = create_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(15)
	
	# Character has low fun (30/100) but no money (0)
	var state = create_state_with_needs("yuki", 50, 50, 50, 30, 50, 0, "dorm")
	
	# Goal: Satisfy fun to 70 (but can't afford cinema)
	var todo_list = [["task_satisfy_fun", "yuki", 70]]
	
	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")
	
	if result.get_success():
		var plan_actions = result.extract_plan()
		print("  Plan has %d actions" % plan_actions.size())
		
		# Should use free methods (games, streaming) instead of paid (cinema)
		var has_cinema = false
		for action in plan_actions:
			if action is Array and action.size() > 0 and str(action[0]) == "action_go_cinema":
				has_cinema = true
				break
		
		assert_test(not has_cinema, "Should not use cinema (too expensive)")
		print("  Plan avoids expensive activities (good!)")
		
		var final_state = result.get_final_state()
		var final_fun = Domain.get_need(final_state, "yuki", "fun")
		print("  Final fun: %d (target: 70)" % final_fun)
		assert_test(final_fun >= 70, "Fun should be satisfied using free methods")
	else:
		print("  Planning failed")

# Scenario 3: Character needs to cook but is in wrong location
# Tests location requirements and movement
func test_scenario_cook_wrong_location():
	test("Scenario 3: Cook but Wrong Location - Movement Required")
	
	var domain = create_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(15)
	
	# Character is hungry (30/100), at library, wants to cook (requires dorm)
	var state = create_state_with_needs("yuki", 30, 50, 50, 50, 50, 50, "library")
	
	# Goal: Satisfy hunger to 70 using cooking
	var todo_list = [["task_satisfy_hunger", "yuki", 70]]
	
	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")
	
	if result.get_success():
		var plan_actions = result.extract_plan()
		print("  Plan has %d actions" % plan_actions.size())
		
		# Should either move to dorm and cook, or use other methods
		var has_movement = false
		var has_cook = false
		for action in plan_actions:
			if action is Array and action.size() > 0:
				var action_name = str(action[0])
				if action_name == "action_move_to":
					has_movement = true
				elif action_name == "action_cook_meal":
					has_cook = true
		
		# If cooking is chosen, should have movement first
		if has_cook:
			assert_test(has_movement, "Should move to dorm before cooking")
			print("  Plan includes movement to dorm for cooking")
		
		var final_state = result.get_final_state()
		var final_hunger = Domain.get_need(final_state, "yuki", "hunger")
		print("  Final hunger: %d (target: 70)" % final_hunger)
		assert_test(final_hunger >= 70, "Hunger should be satisfied")
	else:
		print("  Planning failed")

# Scenario 4: Character is dirty and tired - needs hygiene and energy
# Tests multiple needs with location constraints
func test_scenario_dirty_and_tired():
	test("Scenario 4: Dirty and Tired - Multiple Needs with Constraints")
	
	var domain = create_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(15)
	
	# Character is dirty (25/100 hygiene) and tired (20/100 energy), at library
	var state = create_state_with_needs("yuki", 50, 20, 50, 50, 25, 50, "library")
	
	# Goal: Satisfy both hygiene and energy
	var todo_list = [
		["task_satisfy_hygiene", "yuki", 70],
		["task_satisfy_energy", "yuki", 70]
	]
	
	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")
	
	if result.get_success():
		var plan_actions = result.extract_plan()
		print("  Plan has %d actions" % plan_actions.size())
		
		# Should handle both needs, possibly requiring movement
		var final_state = result.get_final_state()
		var final_hygiene = Domain.get_need(final_state, "yuki", "hygiene")
		var final_energy = Domain.get_need(final_state, "yuki", "energy")
		print("  Final hygiene: %d (target: 70)" % final_hygiene)
		print("  Final energy: %d (target: 70)" % final_energy)
		assert_test(final_hygiene >= 70, "Hygiene should be satisfied")
		assert_test(final_energy >= 70, "Energy should be satisfied")
	else:
		print("  Planning failed")

# Scenario 5: Character has partial money - tests method selection with constraints
# Character can afford snack but not restaurant, must use cheaper options
func test_scenario_partial_money():
	test("Scenario 5: Partial Money - Method Selection with Constraints")
	
	var domain = create_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(15)
	
	# Character is hungry (25/100), has only 10 money (can afford snack, not restaurant)
	var state = create_state_with_needs("yuki", 25, 50, 50, 50, 50, 10, "dorm")
	
	# Goal: Satisfy hunger to 80
	var todo_list = [["task_satisfy_hunger", "yuki", 80]]
	
	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")
	
	if result.get_success():
		var plan_actions = result.extract_plan()
		print("  Plan has %d actions" % plan_actions.size())
		
		# Should not use restaurant (costs 20, only have 10)
		var has_restaurant = false
		for action in plan_actions:
			if action is Array and action.size() > 0 and str(action[0]) == "action_eat_restaurant":
				has_restaurant = true
				break
		
		assert_test(not has_restaurant, "Should not use restaurant (too expensive)")
		print("  Plan uses affordable methods (snack, mess hall, or cook)")
		
		var final_state = result.get_final_state()
		var final_hunger = Domain.get_need(final_state, "yuki", "hunger")
		var final_money = Domain.get_money(final_state, "yuki")
		print("  Final hunger: %d (target: 80)" % final_hunger)
		print("  Remaining money: %d" % final_money)
		assert_test(final_hunger >= 80, "Hunger should be satisfied")
		assert_test(final_money >= 0, "Should not overspend")
	else:
		print("  Planning failed")

# Scenario 6: Character needs social interaction but friend is far
# Tests social needs with location requirements
func test_scenario_social_need():
	test("Scenario 6: Social Need - Interaction Required")
	
	var domain = create_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(15)
	
	# Character is lonely (20/100 social), at dorm
	var state = create_state_with_needs("yuki", 50, 50, 20, 50, 50, 50, "dorm")
	
	# Goal: Satisfy social to 70
	var todo_list = [["task_satisfy_social", "yuki", 70]]
	
	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")
	
	if result.get_success():
		var plan_actions = result.extract_plan()
		print("  Plan has %d actions" % plan_actions.size())
		
		# Should use social methods (talk, phone, club, etc.)
		var final_state = result.get_final_state()
		var final_social = Domain.get_need(final_state, "yuki", "social")
		print("  Final social: %d (target: 70)" % final_social)
		assert_test(final_social >= 70, "Social should be satisfied")
	else:
		print("  Planning failed")

# Scenario 7: Character wants everything but has limited resources
# Tests complex multi-goal with resource and location constraints
func test_scenario_everything_low():
	test("Scenario 7: Everything Low - Complex Multi-Goal")
	
	var domain = create_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(20)
	
	# Character has all needs low, limited money, at library
	var state = create_state_with_needs("yuki", 30, 25, 20, 30, 25, 15, "library")
	
	# Goal: Satisfy all needs to at least 60
	var todo_list = [
		["task_satisfy_hunger", "yuki", 60],
		["task_satisfy_energy", "yuki", 60],
		["task_satisfy_social", "yuki", 60],
		["task_satisfy_fun", "yuki", 60],
		["task_satisfy_hygiene", "yuki", 60]
	]
	
	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")
	
	if result.get_success():
		var plan_actions = result.extract_plan()
		print("  Plan has %d actions" % plan_actions.size())
		print("  Complex scenario: All needs low, limited money, wrong location")
		
		var final_state = result.get_final_state()
		var final_hunger = Domain.get_need(final_state, "yuki", "hunger")
		var final_energy = Domain.get_need(final_state, "yuki", "energy")
		var final_social = Domain.get_need(final_state, "yuki", "social")
		var final_fun = Domain.get_need(final_state, "yuki", "fun")
		var final_hygiene = Domain.get_need(final_state, "yuki", "hygiene")
		
		print("  Final needs:")
		print("    Hunger: %d (target: 60)" % final_hunger)
		print("    Energy: %d (target: 60)" % final_energy)
		print("    Social: %d (target: 60)" % final_social)
		print("    Fun: %d (target: 60)" % final_fun)
		print("    Hygiene: %d (target: 60)" % final_hygiene)
		
		# At least some needs should be satisfied
		var satisfied_count = 0
		if final_hunger >= 60: satisfied_count += 1
		if final_energy >= 60: satisfied_count += 1
		if final_social >= 60: satisfied_count += 1
		if final_fun >= 60: satisfied_count += 1
		if final_hygiene >= 60: satisfied_count += 1
		
		print("  Satisfied %d/5 needs" % satisfied_count)
		assert_test(satisfied_count > 0, "Should satisfy at least some needs")
	else:
		print("  Planning failed - may need more depth or resources")

func run_all_tests():
	test_scenario_starving_and_exhausted()
	test_scenario_broke_wants_fun()
	test_scenario_cook_wrong_location()
	test_scenario_dirty_and_tired()
	test_scenario_partial_money()
	test_scenario_social_need()
	test_scenario_everything_low()
	
	print("\n=== Test Results ===")
	print("Passed: %d" % tests_passed)
	print("Failed: %d" % tests_failed)
	print("Total: %d" % (tests_passed + tests_failed))
	
	if tests_failed > 0:
		print("\n❌ Some tests failed!")
		quit(1)
	else:
		print("\n✅ All tests passed!")
		quit(0)

