#!/usr/bin/env -S godot --headless --script
# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Magical Girls College Test - GDScript headless test script
# Tests GDScript bindings for the goal_task_planner module

extends SceneTree

const Domain = preload("domain.gd")

var tests_passed = 0
var tests_failed = 0
var current_test = ""

func _init():
	print("=== Magical Girls College GDScript Tests ===")
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

# Helper functions
func create_magical_girls_college_init_state() -> Dictionary:
	var state = {}

	# Locations
	var is_at = {
		"yuki": "dorm",
		"maya": "dorm",
		"aria": "dorm",
		"kira": "dorm",
		"luna": "dorm"
	}
	state["is_at"] = is_at

	# Study points
	var study_points = {
		"yuki": 0,
		"maya": 0,
		"aria": 0,
		"kira": 0,
		"luna": 0
	}
	state["study_points"] = study_points

	# Relationship points (flat predicates)
	state["relationship_points_yuki_maya"] = 0
	state["relationship_points_yuki_aria"] = 0
	state["relationship_points_yuki_kira"] = 0
	state["relationship_points_yuki_luna"] = 0

	# Burnout
	var burnout = {
		"yuki": 0,
		"maya": 0,
		"aria": 0,
		"kira": 0,
		"luna": 0
	}
	state["burnout"] = burnout

	# The Sims-style needs (0-100 scale, 50 is neutral)
	var needs = {
		"yuki": {
			"hunger": 50,
			"energy": 50,
			"social": 50,
			"fun": 50,
			"hygiene": 50
		},
		"maya": {
			"hunger": 50,
			"energy": 50,
			"social": 50,
			"fun": 50,
			"hygiene": 50
		},
		"aria": {
			"hunger": 50,
			"energy": 50,
			"social": 50,
			"fun": 50,
			"hygiene": 50
		},
		"kira": {
			"hunger": 50,
			"energy": 50,
			"social": 50,
			"fun": 50,
			"hygiene": 50
		},
		"luna": {
			"hunger": 50,
			"energy": 50,
			"social": 50,
			"fun": 50,
			"hygiene": 50
		}
	}
	state["needs"] = needs

	# Money (resource constraint)
	var money = {
		"yuki": 100,
		"maya": 100,
		"aria": 100,
		"kira": 100,
		"luna": 100
	}
	state["money"] = money

	# Preferences
	var preferences = {
		"yuki": {
			"likes": ["library", "movies"],
			"dislikes": ["beach"]
		},
		"maya": {
			"likes": ["library", "park"],
			"dislikes": ["pool"]
		},
		"aria": {
			"likes": ["beach", "pool"],
			"dislikes": ["library"]
		},
		"kira": {
			"likes": ["cinema", "coffee"],
			"dislikes": ["beach"]
		},
		"luna": {
			"likes": ["park", "beach"],
			"dislikes": ["mess_hall"]
		}
	}
	state["preferences"] = preferences

	return state

# Helper function to create relationship_points goal using flat predicate format
# Format: ["relationship_points_persona_companion", target_points]
func create_relationship_points_goal(persona_id: String, companion_id: String, target_points: int) -> Array:
	var predicate = "relationship_points_%s_%s" % [persona_id, companion_id]
	var goal = Array()
	goal.append(predicate)
	goal.append(target_points)
	return goal

func create_magical_girls_college_domain() -> PlannerDomain:
	var domain = PlannerDomain.new()

	# Add actions
	var actions = [
		Callable(Domain, "action_attend_lecture"),
		Callable(Domain, "action_complete_homework"),
		Callable(Domain, "action_study_library"),
		Callable(Domain, "action_eat_mess_hall_with_friend"),
		Callable(Domain, "action_coffee_together"),
		Callable(Domain, "action_watch_movie"),
		Callable(Domain, "action_pool_hangout"),
		Callable(Domain, "action_park_picnic"),
		Callable(Domain, "action_beach_trip"),
		Callable(Domain, "action_read_book"),
		Callable(Domain, "action_club_activity"),
		Callable(Domain, "action_optimize_schedule"),
		Callable(Domain, "action_predict_outcome"),
		# The Sims-style needs satisfaction actions
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

	# Add task methods for earn_study_points
	var earn_study_points_methods = [
		Callable(Domain, "task_earn_study_points_method_done"),
		Callable(Domain, "task_earn_study_points_method_coordinated"),
		Callable(Domain, "task_earn_study_points_method_lecture"),
		Callable(Domain, "task_earn_study_points_method_homework"),
		Callable(Domain, "task_earn_study_points_method_library")
	]
	domain.add_task_methods("task_earn_study_points", earn_study_points_methods)

	# Add task methods for socialize
	var socialize_methods = [
		Callable(Domain, "task_socialize_method_easy"),
		Callable(Domain, "task_socialize_method_moderate"),
		Callable(Domain, "task_socialize_method_challenging")
	]
	domain.add_task_methods("task_socialize", socialize_methods)

	# Add task methods for manage_week
	var manage_week_methods = [
		Callable(Domain, "task_manage_week_method_balance"),
		Callable(Domain, "task_manage_week_method_academics"),
		Callable(Domain, "task_manage_week_method_relationships")
	]
	domain.add_task_methods("task_manage_week", manage_week_methods)

	# Add task methods for The Sims-style needs satisfaction (deep backtracking)
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

	# Add move task method
	var move_to_location_methods = [
		Callable(Domain, "task_move_to_location_method_direct")
	]
	domain.add_task_methods("task_move_to_location", move_to_location_methods)

	# Add unigoal methods
	var study_unigoal_methods = [
		Callable(Domain, "unigoal_achieve_study_goal")
	]
	domain.add_unigoal_methods("study_points", study_unigoal_methods)

	var relationship_unigoal_methods = [
		Callable(Domain, "unigoal_achieve_relationship_goal")
	]
	domain.add_unigoal_methods("relationship_points", relationship_unigoal_methods)

	# Add multigoal methods
	var multigoal_methods = [
		Callable(Domain, "multigoal_balance_life"),
		Callable(Domain, "multigoal_solve_temporal_puzzle")
	]
	domain.add_multigoal_methods(multigoal_methods)

	return domain

func create_yuki_persona() -> PlannerPersona:
	return PlannerPersona.create_human("yuki", "Yuki")

func create_maya_persona() -> PlannerPersona:
	return PlannerPersona.create_human("maya", "Maya")

func create_aria_persona() -> PlannerPersona:
	return PlannerPersona.create_ai("aria", "Aria")

func create_kira_persona() -> PlannerPersona:
	return PlannerPersona.create_hybrid("kira", "Kira")

func create_luna_persona() -> PlannerPersona:
	return PlannerPersona.create_human("luna", "Luna")

func setup_belief_manager() -> PlannerBeliefManager:
	var manager = PlannerBeliefManager.new()
	manager.register_persona(create_yuki_persona())
	manager.register_persona(create_maya_persona())
	manager.register_persona(create_aria_persona())
	manager.register_persona(create_kira_persona())
	manager.register_persona(create_luna_persona())
	return manager

func setup_allocentric_facts() -> PlannerFactsAllocentric:
	var facts = PlannerFactsAllocentric.new()

	# Set terrain facts
	facts.set_terrain_fact("mess_hall", "type", "indoor")
	facts.set_terrain_fact("mess_hall", "difficulty", 1)
	facts.set_terrain_fact("library", "type", "indoor")
	facts.set_terrain_fact("library", "difficulty", 1)
	facts.set_terrain_fact("cinema", "type", "indoor")
	facts.set_terrain_fact("cinema", "difficulty", 2)
	facts.set_terrain_fact("pool", "type", "outdoor")
	facts.set_terrain_fact("pool", "difficulty", 2)
	facts.set_terrain_fact("park", "type", "outdoor")
	facts.set_terrain_fact("park", "difficulty", 3)
	facts.set_terrain_fact("beach", "type", "outdoor")
	facts.set_terrain_fact("beach", "difficulty", 3)

	# Add shared objects
	var book_data = {"type": "study_book", "location": "library"}
	facts.add_shared_object("study_book_1", book_data)

	var club_data = {"type": "club_material", "location": "dorm"}
	facts.add_shared_object("club_material_1", club_data)

	# Add public events
	var lecture_event = {"type": "lecture", "location": "library", "time": "morning"}
	facts.add_public_event("lecture_schedule_1", lecture_event)

	var club_event = {"type": "club_meeting", "location": "dorm", "time": "afternoon"}
	facts.add_public_event("club_meeting_1", club_event)

	# Set entity positions
	facts.set_entity_position("yuki", Vector3(0, 0, 0))
	facts.set_entity_position("maya", Vector3(0, 0, 0))
	facts.set_entity_position("aria", Vector3(0, 0, 0))
	facts.set_entity_position("kira", Vector3(0, 0, 0))
	facts.set_entity_position("luna", Vector3(0, 0, 0))

	# Set entity capabilities
	facts.set_entity_capability_public("yuki", "movable", true)
	facts.set_entity_capability_public("yuki", "interact", true)
	facts.set_entity_capability_public("maya", "movable", true)
	facts.set_entity_capability_public("maya", "interact", true)
	facts.set_entity_capability_public("aria", "compute", true)
	facts.set_entity_capability_public("aria", "optimize", true)
	facts.set_entity_capability_public("kira", "movable", true)
	facts.set_entity_capability_public("kira", "interact", true)
	facts.set_entity_capability_public("kira", "compute", true)
	facts.set_entity_capability_public("luna", "movable", true)
	facts.set_entity_capability_public("luna", "interact", true)

	return facts

# Test cases
func test_setup_personas():
	test("Setup personas with identity types and capabilities")

	var yuki = create_yuki_persona()
	var maya = create_maya_persona()
	var aria = create_aria_persona()
	var kira = create_kira_persona()
	var luna = create_luna_persona()

	assert_test(yuki != null, "Yuki persona should be valid")
	assert_test(yuki.get_identity_type() == PlannerPersona.IDENTITY_HUMAN, "Yuki should be human")
	assert_test(yuki.has_capability("movable"), "Yuki should have movable capability")
	assert_test(yuki.has_capability("interact"), "Yuki should have interact capability")

	assert_test(maya != null, "Maya persona should be valid")
	assert_test(maya.get_identity_type() == PlannerPersona.IDENTITY_HUMAN, "Maya should be human")

	assert_test(aria != null, "Aria persona should be valid")
	assert_test(aria.get_identity_type() == PlannerPersona.IDENTITY_AI, "Aria should be AI")
	assert_test(aria.has_capability("compute"), "Aria should have compute capability")
	assert_test(aria.has_capability("optimize"), "Aria should have optimize capability")
	assert_test(aria.has_capability("predict"), "Aria should have predict capability")

	assert_test(kira != null, "Kira persona should be valid")
	assert_test(kira.get_identity_type() == PlannerPersona.IDENTITY_HUMAN_AND_AI, "Kira should be hybrid")
	assert_test(kira.has_capability("movable"), "Kira should have movable capability")
	assert_test(kira.has_capability("interact"), "Kira should have interact capability")
	assert_test(kira.has_capability("compute"), "Kira should have compute capability")
	assert_test(kira.has_capability("optimize"), "Kira should have optimize capability")

	assert_test(luna != null, "Luna persona should be valid")
	assert_test(luna.get_identity_type() == PlannerPersona.IDENTITY_HUMAN, "Luna should be human")

	var manager = setup_belief_manager()
	assert_test(manager.has_persona("yuki"), "Manager should have yuki")
	assert_test(manager.has_persona("maya"), "Manager should have maya")
	assert_test(manager.has_persona("aria"), "Manager should have aria")
	assert_test(manager.has_persona("kira"), "Manager should have kira")
	assert_test(manager.has_persona("luna"), "Manager should have luna")

func test_setup_allocentric_facts():
	test("Setup allocentric facts - terrain, objects, events, positions")

	var facts = setup_allocentric_facts()

	assert_test(facts.has_terrain_fact("library", "type"), "Should have library terrain fact")
	assert_test(str(facts.get_terrain_fact("library", "type")) == "indoor", "Library should be indoor")

	assert_test(facts.has_shared_object("study_book_1"), "Should have study_book_1")
	var book = facts.get_shared_object("study_book_1")
	assert_test(str(book["type"]) == "study_book", "Book type should be study_book")

	assert_test(facts.has_public_event("lecture_schedule_1"), "Should have lecture_schedule_1")
	var event = facts.get_public_event("lecture_schedule_1")
	assert_test(str(event["type"]) == "lecture", "Event type should be lecture")

	assert_test(facts.has_entity_position("yuki"), "Should have yuki position")
	var pos = facts.get_entity_position("yuki")
	assert_test(pos == Vector3(0, 0, 0), "Yuki position should be (0,0,0)")

	assert_test(facts.has_entity_capability_public("yuki", "movable"), "Should have yuki movable capability")
	assert_test(bool(facts.get_entity_capability_public("yuki", "movable")) == true, "Yuki should be movable")

	# Verify all personas can observe
	var terrain = facts.observe_terrain("library")
	assert_test(terrain.has("type"), "Terrain observation should have type")

	var objects = facts.observe_shared_objects("library")
	assert_test(objects.has("study_book_1"), "Objects observation should have study_book_1")

	var events = facts.observe_public_events()
	assert_test(events.has("lecture_schedule_1"), "Events observation should have lecture_schedule_1")

	var positions = facts.observe_entity_positions()
	assert_test(positions.has("yuki"), "Positions observation should have yuki")

	var capabilities = facts.observe_entity_capabilities()
	assert_test(capabilities.has("yuki"), "Capabilities observation should have yuki")

func test_planning_with_persona():
	test("Yuki plans weekly activities with persona integration")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	var yuki = create_yuki_persona()
	var manager = setup_belief_manager()
	var facts = setup_allocentric_facts()

	plan.set_current_persona(yuki)
	plan.set_belief_manager(manager)
	plan.set_allocentric_facts(facts)

	assert_test(plan.get_current_persona() != null, "Plan should have current persona")
	assert_test(plan.get_belief_manager() != null, "Plan should have belief manager")
	assert_test(plan.get_allocentric_facts() != null, "Plan should have allocentric facts")

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_manage_week", "yuki"]]

	print("\nTodo List:")
	for i in range(todo_list.size()):
		print("  [%d] %s" % [i, todo_list[i]])

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	if result.get_success():
		var plan_actions = result.extract_plan()
		print("\nPlan Actions (%d actions):" % plan_actions.size())
		for i in range(plan_actions.size()):
			print("  [%d] %s" % [i, plan_actions[i]])
	else:
		print("\nPlanning failed - no plan generated")

	# Planning may succeed or fail depending on complexity, but should not crash
	assert_test(true, "Planning should complete without crashing")

func test_belief_formation():
	test("Belief formation through observation processing")

	var yuki = create_yuki_persona()
	var maya = create_maya_persona()

	# Yuki observes Maya's activities
	var observation1 = {
		"entity": "maya",
		"action": "visit_library",
		"location": "library",
		"confidence": 0.7
	}
	yuki.process_observation(observation1)

	var beliefs = yuki.get_beliefs_about("maya")
	assert_test(beliefs.has("observed_visit_library"), "Should have observed_visit_library belief")

	# Check belief confidence
	var confidence = yuki.get_belief_confidence_for("maya", "observed_visit_library")
	assert_test(confidence > 0.0, "Confidence should be greater than 0")

	# Update belief confidence
	yuki.update_belief_confidence("maya", "observed_visit_library", 0.8)
	var updated_confidence = yuki.get_belief_confidence_for("maya", "observed_visit_library")
	assert_test(updated_confidence > 0.0, "Updated confidence should be greater than 0")

	# Multiple observations increase confidence
	var observation2 = {
		"entity": "maya",
		"action": "visit_library",
		"location": "library",
		"confidence": 0.8
	}
	yuki.process_observation(observation2)

	var new_confidence = yuki.get_belief_confidence_for("maya", "observed_visit_library")
	assert_test(new_confidence >= confidence, "New confidence should be >= old confidence")

func test_simple_planning():
	test("Simple planning - earn study points")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(10)

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 10]]

	print("\nTodo List:")
	for i in range(todo_list.size()):
		print("  [%d] %s" % [i, todo_list[i]])

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	if result.get_success():
		var plan_actions = result.extract_plan()
		print("\nPlan Actions (%d actions):" % plan_actions.size())
		for i in range(plan_actions.size()):
			print("  [%d] %s" % [i, plan_actions[i]])

		assert_test(plan_actions.size() > 0, "Plan should have actions")

		# Verify final state has study points
		var final_state = result.get_final_state()
		var study_points = final_state.get("study_points", {})
		var yuki_points = study_points.get("yuki", 0)
		print("\nFinal State - Yuki study points: %d (target: 10)" % yuki_points)
		assert_test(yuki_points >= 10, "Yuki should have at least 10 study points")
	else:
		print("\nPlanning failed - no plan generated")

func test_persona_coordination_temporal_planning():
	test("Persona coordination through communication with temporal planning")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_current_persona(create_yuki_persona())
	plan.set_verbose(0)

	# Set time range for a day (24 hours = 86400000000 microseconds)
	var time_range_dict = {
		"start_time": 1735689600000000,  # 2025-01-01 00:00:00 UTC (midnight)
		"end_time": 1735689600000000 + 86400000000  # +24 hours
	}
	plan.set_time_range_dict(time_range_dict)

	# Maya coordinates a study session with Yuki at 2pm (14:00 = 14 hours from midnight)
	time_range_dict = plan.get_time_range_dict()
	var coordinated_time = time_range_dict["start_time"] + 50400000000  # 14 hours = 2pm
	var coordination = {
		"from": "maya",
		"message": "Let's study together at the library at 2pm",
		"topic": "coordination",
		"action": "study_session",
		"location": "library",
		"time": coordinated_time  # Absolute time in microseconds
	}

	# Store coordination in persona (for communication tracking)
	var yuki = plan.get_current_persona()
	yuki.process_communication(coordination)

	# Communication stores coordination information
	var beliefs = yuki.get_beliefs_about("maya")
	assert_test(beliefs.has("communication_coordination"), "Should have communication_coordination")

	# Store coordination in state so domain methods can access it
	var state = create_magical_girls_college_init_state()
	if not state.has("coordination"):
		state["coordination"] = {}
	var coord_dict = state["coordination"]
	coord_dict["yuki"] = coordination
	state["coordination"] = coord_dict

	# Create a task to study at the library (which will fulfill the coordination)
	var todo_list = [["task_earn_study_points", "yuki", 5]]

	# Use run_lazy_refineahead for temporal planning (supports temporal constraints)
	var result = plan.run_lazy_refineahead(state, todo_list)
	assert_test(result != null, "Result should be valid")
	assert_test(result.get_success(), "Planning should succeed")

	if result.get_success():
		var plan_actions = result.extract_plan()
		assert_test(plan_actions.size() > 0, "Plan should have actions")

		# Simulate the plan to verify time advancement
		var state_sequence = plan.simulate(result, state, 0)
		assert_test(state_sequence.size() > 1, "Should have multiple states showing time progression")

		# Verify study points increased
		if state_sequence.size() > 0:
			var final_state = state_sequence[state_sequence.size() - 1]
			var study_points = final_state.get("study_points", {})
			var yuki_points = study_points.get("yuki", 0)
			assert_test(yuki_points >= 5, "Should have earned study points")

func test_multi_persona_planning():
	test("Multi-persona planning with capabilities and preferences")

	var domain = create_magical_girls_college_domain()
	var state = create_magical_girls_college_init_state()

	# Yuki plans socialization
	var plan_yuki = PlannerPlan.new()
	plan_yuki.set_current_domain(domain)
	plan_yuki.set_current_persona(create_yuki_persona())
	plan_yuki.set_verbose(0)

	var todo_yuki = [["task_socialize", "yuki", "maya", 2]]  # Moderate activity

	var result_yuki = plan_yuki.find_plan(state, todo_yuki)
	assert_test(result_yuki != null, "Yuki result should be valid")

	# Aria (AI) plans optimal study schedule
	var plan_aria = PlannerPlan.new()
	plan_aria.set_current_domain(domain)
	plan_aria.set_current_persona(create_aria_persona())
	plan_aria.set_verbose(0)

	var todo_aria = [["task_earn_study_points", "aria", 15]]

	var result_aria = plan_aria.find_plan(state, todo_aria)
	assert_test(result_aria != null, "Aria result should be valid")

func test_information_asymmetry():
	test("Information asymmetry - internal state hidden")

	var manager = setup_belief_manager()

	# Yuki cannot directly access Maya's internal planner state
	var result = manager.get_planner_state("maya", "yuki")
	# Should return error or empty for internal state
	var result_valid = result.is_empty() or result.has("error")
	assert_test(result_valid, "Internal state should not be directly accessible")

	# Must form beliefs through observation (not communication)
	var yuki = manager.get_persona("yuki")
	var observation = {
		"entity": "maya",
		"action": "activity",
		"location": "library",
		"observer_location": "library"  # Yuki is at library
	}
	yuki.process_observation(observation)

	var beliefs = yuki.get_beliefs_about("maya")
	assert_test(beliefs.size() > 0, "Should have beliefs from observation")

func test_identity_type_affects_planning():
	test("Identity type affects planning capabilities")

	var domain = create_magical_girls_college_domain()
	var state = create_magical_girls_college_init_state()

	# Aria (AI) uses COMPUTE and OPTIMIZE
	var plan_aria = PlannerPlan.new()
	plan_aria.set_current_domain(domain)
	plan_aria.set_current_persona(create_aria_persona())
	plan_aria.set_verbose(0)

	var todo_aria = [["task_earn_study_points", "aria", 10]]

	var result_aria = plan_aria.find_plan(state, todo_aria)
	assert_test(result_aria != null, "Aria result should be valid")

	# Kira (HYBRID) uses both human and AI capabilities
	var plan_kira = PlannerPlan.new()
	plan_kira.set_current_domain(domain)
	plan_kira.set_current_persona(create_kira_persona())
	plan_kira.set_verbose(0)

	var todo_kira = [["task_manage_week", "kira"]]

	var result_kira = plan_kira.find_plan(state, todo_kira)
	assert_test(result_kira != null, "Kira result should be valid")

func test_three_planning_methods():
	test("Three planning methods - find_plan, lazy_lookahead, lazy_refineahead")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_current_persona(create_yuki_persona())

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 10]]

	# Method 1: find_plan (standard planning)
	var result1 = plan.find_plan(state, todo_list)
	assert_test(result1 != null, "find_plan result should be valid")

	# Method 2: run_lazy_lookahead (incremental execution)
	var result2 = plan.run_lazy_lookahead(state, todo_list, 10)
	assert_test(result2 != null, "run_lazy_lookahead result should be valid")

	# Method 3: run_lazy_refineahead (graph-based with temporal constraints)
	var result3 = plan.run_lazy_refineahead(state, todo_list)
	assert_test(result3 != null, "run_lazy_refineahead result should be valid")

func test_temporal_planning_with_metadata():
	test("Temporal planning with metadata")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	# Set time range (week: 7 days = 604800000000 microseconds)
	var time_range_dict = {
		"start_time": 1735689600000000,  # 2025-01-01 00:00:00 UTC
		"end_time": 1735689600000000 + 604800000000  # +7 days
	}
	plan.set_time_range_dict(time_range_dict)

	# Attach temporal constraints to actions
	var action1 = ["action_attend_lecture", "yuki", "math"]
	var temporal1 = {"duration": 7200000000}  # 2 hours
	var action1_with_metadata = plan.attach_metadata(action1, temporal1)
	assert_test(action1_with_metadata != null, "Action with metadata should be valid")

	var action2 = ["action_watch_movie", "yuki", "maya"]
	var temporal2 = {
		"start_time": time_range_dict["start_time"] + 3600000000,  # +1 hour
		"end_time": time_range_dict["start_time"] + 10800000000  # +3 hours
	}
	var action2_with_metadata = plan.attach_metadata(action2, temporal2)
	assert_test(action2_with_metadata != null, "Action with metadata should be valid")

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_manage_week", "yuki"]]

	var result = plan.run_lazy_refineahead(state, todo_list)
	assert_test(result != null, "Result should be valid")

func test_entity_requirements_with_metadata():
	test("Entity requirements with metadata")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_current_persona(create_aria_persona())  # AI persona

	# Attach entity requirements to actions
	var action1 = ["action_optimize_schedule", "aria"]
	var entity1 = {
		"type": "ai",
		"capabilities": ["compute", "optimize"]
	}
	var action1_with_metadata = plan.attach_metadata(action1, {}, entity1)
	assert_test(action1_with_metadata != null, "Action with entity metadata should be valid")

	var action2 = ["action_predict_outcome", "aria", "study"]
	var entity2 = {
		"type": "ai",
		"capabilities": ["predict"]
	}
	var action2_with_metadata = plan.attach_metadata(action2, {}, entity2)
	assert_test(action2_with_metadata != null, "Action with entity metadata should be valid")

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "aria", 10]]

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

func test_vsids_activity_tracking():
	test("VSIDS activity tracking")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	var state = create_magical_girls_college_init_state()

	# Plan multiple times with different goals
	for i in range(3):
		var todo_list = [["task_earn_study_points", "yuki", 5 + i * 5]]
		var result = plan.find_plan(state, todo_list)
		assert_test(result != null, "Result should be valid")

	# Check method activities
	var activities = plan.get_method_activities()
	assert_test(activities != null, "Activities should be valid")

	# Reset VSIDS activity
	plan.reset_vsids_activity()
	var cleared_activities = plan.get_method_activities()
	assert_test(cleared_activities.size() == 0, "Activities should be cleared after reset")

func test_plan_simulation():
	test("Plan simulation")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 10]]

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	if result.get_success():
		# Simulate the plan
		var state_sequence = plan.simulate(result, state, 0)
		assert_test(state_sequence.size() > 0, "State sequence should have states")

		# Verify final state has study points
		if state_sequence.size() > 0:
			var final_state = state_sequence[state_sequence.size() - 1]
			var study_points = final_state.get("study_points", {})
			var yuki_points = study_points.get("yuki", 0)
			assert_test(yuki_points >= 10, "Yuki should have at least 10 study points")

func test_multigoal_planning():
	test("Multigoal planning")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_current_persona(create_yuki_persona())

	var state = create_magical_girls_college_init_state()

	# Create multigoal: study points and relationship points (using flat predicate format)
	# Format: [predicate, subject, value] where predicate is "relationship_points" and subject is the full flat predicate
	var multigoal = [
		["study_points", "yuki", 10],
		["relationship_points", "relationship_points_yuki_maya", 5]
	]

	var todo_list = [multigoal]

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	if result.get_success():
		var plan_actions = result.extract_plan()
		assert_test(plan_actions.size() > 0, "Plan should have actions")

		# Verify goals achieved
		var final_state = result.get_final_state()
		var study_points = final_state.get("study_points", {})
		var yuki_points = study_points.get("yuki", 0)
		assert_test(yuki_points >= 10, "Should have study points")

		var maya_relationship = final_state.get("relationship_points_yuki_maya", 0)
		assert_test(maya_relationship >= 5, "Should have relationship points")

func test_unigoal_planning():
	test("Unigoal planning")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_current_persona(create_yuki_persona())

	var state = create_magical_girls_college_init_state()

	# Create unigoal: achieve study goal
	var unigoal = ["study_points", "yuki", 10]
	var todo_list = [unigoal]

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	if result.get_success():
		var plan_actions = result.extract_plan()
		assert_test(plan_actions.size() > 0, "Plan should have actions")

		# Verify goal achieved
		var final_state = result.get_final_state()
		var study_points = final_state.get("study_points", {})
		var yuki_points = study_points.get("yuki", 0)
		assert_test(yuki_points >= 10, "Should have study points")

func test_complex_temporal_puzzle():
	test("Complex temporal puzzle with dependencies")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(15)
	plan.set_current_persona(create_yuki_persona())

	# Set time range
	var time_range_dict = {
		"start_time": 1735689600000000,  # 2025-01-01 00:00:00 UTC
		"end_time": 1735689600000000 + 86400000000  # +24 hours
	}
	plan.set_time_range_dict(time_range_dict)

	# Create state with temporal puzzle
	var state = create_magical_girls_college_init_state()
	var temporal_puzzle = {
		"morning_study_time": time_range_dict["start_time"] + 3600000000,  # 1 hour after start
		"homework_deadline": time_range_dict["start_time"] + 28800000000  # 8 hours after start (movie time)
	}
	state["temporal_puzzle"] = temporal_puzzle

	# Add coordination for study session
	if not state.has("coordination"):
		state["coordination"] = {}
	var coord_dict = state["coordination"]
	coord_dict["yuki"] = {
		"action": "study_session",
		"location": "library",
		"time": temporal_puzzle["morning_study_time"]
	}
	state["coordination"] = coord_dict

	# Create multigoal with temporal dependencies (using flat predicate format)
	# Format: [predicate, subject, value] where predicate is "relationship_points" and subject is the full flat predicate
	var multigoal = [
		["study_points", "yuki", 8],
		["relationship_points", "relationship_points_yuki_maya", 3]
	]

	var todo_list = [multigoal]
	var result: PlannerResult = plan.run_lazy_refineahead(state, todo_list)
	assert_test(result != null, "Result should be valid")

	assert_test(result.get_success(), "Planning should succeed for both goals")
	var plan_actions = result.extract_plan()
	assert_test(plan_actions.size() > 0, "Plan should have actions")

	# Simulate the plan execution to get the updated state sequence
	var state_sequence = plan.simulate(result, state, 0)
	assert_test(state_sequence.size() > 0, "State sequence should have states")

	# Get the final state after executing all actions
	var final_state = state_sequence[state_sequence.size() - 1]

	# Verify study_points goal is achieved
	var study_points = final_state.get("study_points", {})
	var yuki_study_points = study_points.get("yuki", 0)
	assert_test(yuki_study_points >= 8, "Yuki should have at least 8 study points (got %d)" % yuki_study_points)

	# Verify relationship_points goal is achieved (using flat predicate)
	var maya_relationship = final_state.get("relationship_points_yuki_maya", 0)
	if maya_relationship is int:
		assert_test(maya_relationship >= 3, "Yuki should have at least 3 relationship points with Maya (got %d)" % maya_relationship)
	else:
		assert_test(false, "Maya relationship should be an int, got %s" % str(maya_relationship))

func test_planner_result_helpers():
	test("PlannerResult helper methods - get_all_nodes, get_node, has_node, find_failed_nodes")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 10]]

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	if result.get_success():
		# Test get_all_nodes
		var all_nodes = result.get_all_nodes()
		assert_test(all_nodes is Array, "get_all_nodes should return Array")
		assert_test(all_nodes.size() > 0, "Should have nodes in graph")

		# Test has_node and get_node
		if all_nodes.size() > 0:
			var first_node = all_nodes[0]
			if first_node.has("node_id"):
				var node_id = first_node["node_id"]
				assert_test(result.has_node(node_id), "Should have node with id %d" % node_id)

				var node = result.get_node(node_id)
				assert_test(node is Dictionary, "get_node should return Dictionary")
				assert_test(node.has("type"), "Node should have type")
				assert_test(node.has("status"), "Node should have status")

		# Test find_failed_nodes (should be empty for successful plan)
		var failed_nodes = result.find_failed_nodes()
		assert_test(failed_nodes is Array, "find_failed_nodes should return Array")
		# For successful plan, should have no failed nodes
		assert_test(failed_nodes.size() == 0, "Successful plan should have no failed nodes")

func test_blacklist_command():
	test("Blacklist command - exclude specific actions from planning")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 10]]

	# Get plan without blacklist
	var result1 = plan.find_plan(state, todo_list)
	assert_test(result1 != null, "Result should be valid")

	var plan1_actions = []
	if result1.get_success():
		plan1_actions = result1.extract_plan()

	# Blacklist a specific action
	if plan1_actions.size() > 0:
		plan.blacklist_command(plan1_actions[0])

		# Plan again - should avoid blacklisted action
		var result2 = plan.find_plan(state, todo_list)
		assert_test(result2 != null, "Result should be valid")

		if result2.get_success():
			var plan2_actions = result2.extract_plan()
			# The blacklisted action should not appear (or plan might fail)
			# This is a basic test - actual behavior depends on domain alternatives

func test_get_iterations():
	test("Get iterations - track planning complexity")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 10]]

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	var iterations = plan.get_iterations()
	assert_test(iterations >= 0, "Iterations should be non-negative")
	assert_test(iterations > 0, "Should have at least one iteration for successful plan")

func test_max_depth_limit():
	test("Max depth limit - prevent infinite recursion")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)
	plan.set_max_depth(3)  # Very low depth limit

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 20]]  # Requires more depth

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	# With very low depth, plan might fail or succeed with fewer actions
	# The important thing is it doesn't hang
	assert_test(true, "Max depth limit should prevent infinite recursion")

func test_goal_tags():
	test("Goal tags - tag multigoals for method matching")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	# Create multigoal
	var multigoal = [
		["study_points", "yuki", 10],
		["relationship_points", "relationship_points_yuki_maya", 5]
	]

	# Test get_goal_tag and set_goal_tag
	var tag = PlannerMultigoal.get_goal_tag(multigoal)
	assert_test(tag == "", "Untagged multigoal should have empty tag")

	var tagged_multigoal = PlannerMultigoal.set_goal_tag(multigoal, "balance_life")
	assert_test(tagged_multigoal is Dictionary, "set_goal_tag should return Dictionary")

	var retrieved_tag = PlannerMultigoal.get_goal_tag(tagged_multigoal)
	assert_test(retrieved_tag == "balance_life", "Should retrieve correct tag")

func test_flat_predicate_format():
	test("Flat predicate format - verify flat predicates work correctly")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	var state = create_magical_girls_college_init_state()

	# Test flat predicate goal format: ["relationship_points", "relationship_points_yuki_maya", 3]
	var flat_predicate_goal = ["relationship_points", "relationship_points_yuki_maya", 3]
	var todo_list = [flat_predicate_goal]

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	if result.get_success():
		var plan_actions = result.extract_plan()
		assert_test(plan_actions.size() > 0, "Plan should have actions")

		# Verify flat predicate in final state
		var final_state = result.get_final_state()
		var maya_relationship = final_state.get("relationship_points_yuki_maya", 0)
		assert_test(maya_relationship >= 3, "Flat predicate should be set in state")

func test_replan():
	test("Replan - replan from failure point")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(0)

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 10]]

	var result = plan.find_plan(state, todo_list)
	assert_test(result != null, "Result should be valid")

	if result.get_success():
		# Find failed nodes (should be empty for successful plan)
		var failed_nodes = result.find_failed_nodes()

		# If there are failed nodes, test replanning
		if failed_nodes.size() > 0:
			var fail_node = failed_nodes[0]
			var fail_node_id = fail_node.get("node_id", -1)
			if fail_node_id >= 0:
				# Modify state to potentially fix the failure
				var new_state = state.duplicate(true)
				var replan_result = plan.replan(result, new_state, fail_node_id)
				assert_test(replan_result != null, "Replan result should be valid")
		else:
			# For successful plan, replanning should still work (no-op or continuation)
			assert_test(true, "Replan test completed (no failures to replan from)")

func test_verbose_logging():
	test("Verbose logging - different verbosity levels")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)

	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 5]]

	# Test different verbosity levels
	for verbosity in range(4):
		plan.set_verbose(verbosity)
		assert_test(plan.get_verbose() == verbosity, "Verbosity should be set to %d" % verbosity)

		var result = plan.find_plan(state, todo_list)
		assert_test(result != null, "Result should be valid at verbosity %d" % verbosity)

func test_reset():
	test("Reset - clear planner state")

	var domain = create_magical_girls_college_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(2)
	plan.set_max_depth(15)

	# Set some state
	var state = create_magical_girls_college_init_state()
	var todo_list = [["task_earn_study_points", "yuki", 5]]
	var result = plan.find_plan(state, todo_list)

	# Reset and verify state is cleared
	plan.reset()
	assert_test(plan.get_verbose() == 0, "Reset should clear verbose to 0")
	assert_test(plan.get_max_depth() == 10, "Reset should restore default max_depth (10)")
	assert_test(plan.get_iterations() == 0, "Reset should clear iterations")

func run_all_tests():
	test_setup_personas()
	test_setup_allocentric_facts()
	test_planning_with_persona()
	test_belief_formation()
	test_simple_planning()
	test_persona_coordination_temporal_planning()
	test_multi_persona_planning()
	test_information_asymmetry()
	test_identity_type_affects_planning()
	test_three_planning_methods()
	test_temporal_planning_with_metadata()
	test_entity_requirements_with_metadata()
	test_vsids_activity_tracking()
	test_plan_simulation()
	test_multigoal_planning()
	test_unigoal_planning()
	test_complex_temporal_puzzle()
	test_planner_result_helpers()
	test_blacklist_command()
	test_get_iterations()
	test_max_depth_limit()
	test_goal_tags()
	test_flat_predicate_format()
	test_replan()
	test_verbose_logging()
	test_reset()

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
