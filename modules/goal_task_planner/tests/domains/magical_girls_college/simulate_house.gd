#!/usr/bin/env -S godot --headless --script
# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Sims House Simulation - 20 Minute Continuous Simulation
# Simulates a character living in a Sims-style house for 20 minutes
# Needs decay over time, and the planner continuously generates plans to satisfy them

extends SceneTree

const Domain = preload("domain.gd")

var simulation_time_seconds = 0.0
var total_simulation_time = 20 * 60  # 20 minutes in seconds
var time_step = 30.0  # Update every 30 seconds
var needs_decay_rate = 1.0  # Needs decay by 1.0 per time step (per 30 seconds)
var planning_check_interval = 60.0  # Check for planning every 60 seconds

var domain: PlannerDomain
var plan: PlannerPlan
var current_state: Dictionary
var persona_id = "yuki"

var total_actions_executed = 0
var total_plans_generated = 0

func _init():
	print("=== Sims House Simulation - 20 Minutes ===")
	print("Starting simulation for character: %s" % persona_id)
	call_deferred("start_simulation")

func create_initial_state() -> Dictionary:
	var state = {}
	
	# Location - start at dorm
	state["is_at"] = {persona_id: "dorm"}
	
	# Needs - start with moderate levels
	state["needs"] = {
		persona_id: {
			"hunger": 70,
			"energy": 60,
			"social": 50,
			"fun": 55,
			"hygiene": 65
		}
	}
	
	# Money - start with some money
	state["money"] = {persona_id: 50}
	
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

func decay_needs(state: Dictionary) -> Dictionary:
	# Decay all needs slightly over time
	var needs = state["needs"][persona_id]
	
	needs["hunger"] = max(0, needs["hunger"] - needs_decay_rate)
	needs["energy"] = max(0, needs["energy"] - needs_decay_rate)
	needs["social"] = max(0, needs["social"] - needs_decay_rate * 0.3)  # Social decays slower
	needs["fun"] = max(0, needs["fun"] - needs_decay_rate * 0.4)  # Fun decays slower
	needs["hygiene"] = max(0, needs["hygiene"] - needs_decay_rate * 0.2)  # Hygiene decays slowest
	
	state["needs"][persona_id] = needs
	return state

func get_critical_needs(state: Dictionary) -> Array:
	# Return list of needs that are below threshold (55) or very low (below 40)
	var critical = []
	var needs = state["needs"][persona_id]
	var threshold = 55
	var urgent_threshold = 40
	
	# Check each need - use guard style to skip if not critical
	if needs["hunger"] < urgent_threshold:
		critical.append(["task_satisfy_hunger", persona_id, 70])
	elif needs["hunger"] < threshold:
		critical.append(["task_satisfy_hunger", persona_id, 65])
	
	if needs["energy"] < urgent_threshold:
		critical.append(["task_satisfy_energy", persona_id, 70])
	elif needs["energy"] < threshold:
		critical.append(["task_satisfy_energy", persona_id, 65])
	
	if needs["social"] < urgent_threshold:
		critical.append(["task_satisfy_social", persona_id, 70])
	elif needs["social"] < threshold:
		critical.append(["task_satisfy_social", persona_id, 65])
	
	if needs["fun"] < urgent_threshold:
		critical.append(["task_satisfy_fun", persona_id, 70])
	elif needs["fun"] < threshold:
		critical.append(["task_satisfy_fun", persona_id, 65])
	
	if needs["hygiene"] < urgent_threshold:
		critical.append(["task_satisfy_hygiene", persona_id, 70])
	elif needs["hygiene"] < threshold:
		critical.append(["task_satisfy_hygiene", persona_id, 65])
	
	return critical

func execute_plan(state: Dictionary, plan_actions: Array) -> Dictionary:
	# Execute all actions in the plan sequentially
	var new_state = state.duplicate(true)
	
	for action in plan_actions:
		if not (action is Array and action.size() > 0):
			continue
		
		var action_name = str(action[0])
		var needs = new_state["needs"][persona_id]
		
		# Simulate action effects based on action name
		if action_name.begins_with("action_eat"):
			needs["hunger"] = min(100, needs["hunger"] + 30)
			new_state["needs"][persona_id] = needs
			
			if action_name == "action_eat_restaurant":
				new_state["money"][persona_id] = max(0, new_state["money"][persona_id] - 20)
				continue
			if action_name == "action_eat_snack":
				new_state["money"][persona_id] = max(0, new_state["money"][persona_id] - 5)
				continue
			if action_name == "action_eat_mess_hall":
				new_state["money"][persona_id] = max(0, new_state["money"][persona_id] - 10)
				continue
			continue
		
		if action_name.begins_with("action_sleep") or action_name == "action_nap_library":
			needs["energy"] = min(100, needs["energy"] + 40)
			new_state["needs"][persona_id] = needs
			continue
		
		if action_name == "action_energy_drink":
			needs["energy"] = min(100, needs["energy"] + 20)
			new_state["needs"][persona_id] = needs
			new_state["money"][persona_id] = max(0, new_state["money"][persona_id] - 3)
			continue
		
		if action_name.begins_with("action_talk") or action_name == "action_phone_call" or action_name == "action_join_club":
			needs["social"] = min(100, needs["social"] + 25)
			new_state["needs"][persona_id] = needs
			continue
		
		if action_name == "action_play_games" or action_name == "action_watch_streaming":
			needs["fun"] = min(100, needs["fun"] + 30)
			new_state["needs"][persona_id] = needs
			continue
		
		if action_name == "action_go_cinema":
			needs["fun"] = min(100, needs["fun"] + 40)
			new_state["needs"][persona_id] = needs
			new_state["money"][persona_id] = max(0, new_state["money"][persona_id] - 15)
			continue
		
		if action_name == "action_shower":
			needs["hygiene"] = min(100, needs["hygiene"] + 50)
			new_state["needs"][persona_id] = needs
			continue
		
		if action_name == "action_wash_hands":
			needs["hygiene"] = min(100, needs["hygiene"] + 15)
			new_state["needs"][persona_id] = needs
			continue
		
		if action_name == "action_move_to" and action.size() > 2:
			new_state["is_at"][persona_id] = str(action[2])
			continue
		
		if action_name == "action_cook_meal":
			needs["hunger"] = min(100, needs["hunger"] + 35)
			new_state["needs"][persona_id] = needs
			continue
	
	return new_state

func print_state(time_minutes: float, state: Dictionary):
	var needs = state["needs"][persona_id]
	var location = Domain.get_location(state, persona_id)
	var money = Domain.get_money(state, persona_id)
	
	print("\n[%.1f min] Location: %s | Money: $%d" % [time_minutes, location, money])
	print("  Needs: Hunger=%d Energy=%d Social=%d Fun=%d Hygiene=%d" % [
		needs["hunger"], needs["energy"], needs["social"], needs["fun"], needs["hygiene"]
	])

func start_simulation():
	# Initialize
	domain = create_domain()
	plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(1)  # Enable verbose output
	plan.set_max_depth(15)
	
	current_state = create_initial_state()
	
	print("\nInitial State:")
	print_state(0.0, current_state)
	
	# Run simulation loop
	call_deferred("simulation_step")

var last_planning_check = 0.0

func simulation_step():
	# Early return: simulation complete
	if simulation_time_seconds >= total_simulation_time:
		print("\n=== Simulation Complete ===")
		print("Total simulation time: %.1f minutes" % (total_simulation_time / 60.0))
		print("Total plans generated: %d" % total_plans_generated)
		print("Total actions executed: %d" % total_actions_executed)
		print("\nFinal State:")
		print_state(total_simulation_time / 60.0, current_state)
		quit(0)
		return
	
	# Decay needs over time
	current_state = decay_needs(current_state)
	
	# Check for critical needs periodically
	var time_since_last_plan = simulation_time_seconds - last_planning_check
	if time_since_last_plan >= planning_check_interval:
		last_planning_check = simulation_time_seconds
		handle_planning()
	
	# Print state every 2 minutes
	if int(simulation_time_seconds) % 120 < time_step:
		print_state(simulation_time_seconds / 60.0, current_state)
	
	# Advance time
	simulation_time_seconds += time_step
	
	# Schedule next step
	call_deferred("simulation_step")

func handle_planning():
	var critical_needs = get_critical_needs(current_state)
	
	# Early return: no critical needs
	if critical_needs.size() == 0:
		return
	
	var time_minutes = simulation_time_seconds / 60.0
	print("\n[%.1f min] Planning to satisfy needs..." % time_minutes)
	print("  Critical needs detected:")
	for need_task in critical_needs:
		if not (need_task is Array and need_task.size() >= 2):
			continue
		var need_type = need_task[0] if need_task.size() > 0 else "unknown"
		var target = need_task[2] if need_task.size() > 2 else "?"
		print("    - %s (target: %s)" % [need_type, target])
	
	var result = plan.find_plan(current_state, critical_needs)
	total_plans_generated += 1
	
	# Early return: planning failed
	if result == null or not result.get_success():
		print("\n[%.1f min] ✗ Planning failed - needs may be too complex or resources insufficient" % time_minutes)
		if result != null:
			var failed_nodes = result.find_failed_nodes()
			if failed_nodes.size() > 0:
				print("  Failed nodes: %d" % failed_nodes.size())
		return
	
	var plan_actions = result.extract_plan(1)  # Get verbose plan output
	
	# Early return: no actions in plan
	if plan_actions.size() == 0:
		return
	
	# Execute the plan
	current_state = execute_plan(current_state, plan_actions)
	total_actions_executed += plan_actions.size()
	
	# Print what we're doing
	print("\n[%.1f min] ✓ Plan found! Executing %d actions..." % [time_minutes, plan_actions.size()])
	for i in range(plan_actions.size()):
		var action = plan_actions[i]
		if not (action is Array and action.size() > 0):
			continue
		
		var action_str = str(action[0])
		if action.size() > 1:
			action_str += "("
			for j in range(1, action.size()):
				if j > 1:
					action_str += ", "
				action_str += str(action[j])
			action_str += ")"
		print("  [%d] %s" % [i + 1, action_str])
	
	# Show state after action
	print_state(time_minutes, current_state)

