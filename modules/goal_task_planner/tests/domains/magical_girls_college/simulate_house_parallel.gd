#!/usr/bin/env -S godot --headless --script
# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Sims House Simulation - Multi-Agent Parallel Version
# Uses WorkerThreadPool to run multiple agents in parallel (one per core)
# Each agent runs on a time-shared thread from the pool

extends SceneTree

const Domain = preload("domain.gd")

# Simulation parameters
var simulation_time_seconds = 0.0
var total_simulation_time = 20 * 60  # 20 minutes in seconds
var time_step = 30.0  # Update every 30 seconds
var needs_decay_rate = 1.0  # Needs decay by 1.0 per time step
var planning_check_interval = 60.0  # Check for planning every 60 seconds

# Agent data - each agent has its own state and planner
var agents: Array[Dictionary] = []
var agent_states: Array[Dictionary] = []
var agent_plans: Array[PlannerPlan] = []
var agent_domains: Array[PlannerDomain] = []

# Thread synchronization
var state_mutex = Mutex.new()
var simulation_time_mutex = Mutex.new()

# Statistics
var total_actions_executed = 0
var total_plans_generated = 0
var active_tasks: Array[int] = []

# Profiling metrics
var profile_data = {
	"total_planning_time": 0.0,
	"total_mutex_wait_time": 0.0,
	"total_state_ops_time": 0.0,
	"total_decay_time": 0.0,
	"total_execution_time": 0.0,
	"planning_calls": 0,
	"mutex_locks": 0,
	"step_times": []
}

func _init():
	print("=== Sims House Simulation - Multi-Agent Parallel ===")
	print("Using WorkerThreadPool for parallel agent simulation")
	call_deferred("start_simulation")

func create_agent_state(persona_id: String) -> Dictionary:
	var state = {}

	# Location - start at dorm
	state["is_at"] = {persona_id: "dorm"}

	# Needs - start with moderate levels (slightly randomized)
	var rng = RandomNumberGenerator.new()
	rng.randomize()
	state["needs"] = {
		persona_id: {
			"hunger": 70 + rng.randi_range(-10, 10),
			"energy": 60 + rng.randi_range(-10, 10),
			"social": 50 + rng.randi_range(-10, 10),
			"fun": 55 + rng.randi_range(-10, 10),
			"hygiene": 65 + rng.randi_range(-10, 10)
		}
	}

	# Money - start with some money
	state["money"] = {persona_id: 50 + rng.randi_range(-10, 10)}

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

func decay_needs(state: Dictionary, persona_id: String) -> Dictionary:
	# Decay all needs slightly over time
	var needs = state["needs"][persona_id]

	needs["hunger"] = max(0, needs["hunger"] - needs_decay_rate)
	needs["energy"] = max(0, needs["energy"] - needs_decay_rate)
	needs["social"] = max(0, needs["social"] - needs_decay_rate * 0.3)
	needs["fun"] = max(0, needs["fun"] - needs_decay_rate * 0.4)
	needs["hygiene"] = max(0, needs["hygiene"] - needs_decay_rate * 0.2)

	state["needs"][persona_id] = needs
	return state

func get_critical_needs(state: Dictionary, persona_id: String) -> Array:
	var critical = []
	var needs = state["needs"][persona_id]
	var threshold = 55
	var urgent_threshold = 40

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

func execute_plan(state: Dictionary, plan_actions: Array, persona_id: String) -> Dictionary:
	var new_state = state.duplicate(true)

	for action in plan_actions:
		if not (action is Array and action.size() > 0):
			continue

		var action_name = str(action[0])
		var needs = new_state["needs"][persona_id]

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

# Process a single agent - this runs in a worker thread
func process_agent(agent_index: int):
	var agent = agents[agent_index]
	var persona_id = agent["persona_id"]
	var plan = agent_plans[agent_index]
	var domain = agent_domains[agent_index]

	# Lock to read state - minimize lock time
	var mutex_start = Time.get_ticks_usec()
	state_mutex.lock()
	var mutex_wait = Time.get_ticks_usec() - mutex_start

	var state = agent_states[agent_index].duplicate(true)
	state_mutex.unlock()

	# Track mutex wait time (batch update at end)
	var mutex_wait_accumulated = mutex_wait / 1000000.0

	# Decay needs
	var decay_start = Time.get_ticks_usec()
	state = decay_needs(state, persona_id)
	var decay_time = Time.get_ticks_usec() - decay_start
	var decay_time_accumulated = decay_time / 1000000.0

	# Check for critical needs periodically
	simulation_time_mutex.lock()
	var current_time = simulation_time_seconds
	simulation_time_mutex.unlock()

	# Initialize statistics variables (early return style)
	var planning_time_accumulated = 0.0
	var plans_generated_this_step = 0
	var exec_time_accumulated = 0.0
	var actions_executed_this_step = 0

	var time_since_last_plan = current_time - agent.get("last_planning_check", 0.0)
	if time_since_last_plan >= planning_check_interval:
		agent["last_planning_check"] = current_time

		var critical_needs = get_critical_needs(state, persona_id)

		if critical_needs.size() > 0:
			# Planning - this is likely the bottleneck
			var planning_start = Time.get_ticks_usec()
			var result = plan.find_plan(state, critical_needs)
			var planning_time = Time.get_ticks_usec() - planning_start
			planning_time_accumulated = planning_time / 1000000.0
			plans_generated_this_step = 1

			if result != null and result.get_success():
				var plan_actions = result.extract_plan(0)

				if plan_actions.size() > 0:
					# Execute the plan
					var exec_start = Time.get_ticks_usec()
					state = execute_plan(state, plan_actions, persona_id)
					var exec_time = Time.get_ticks_usec() - exec_start
					exec_time_accumulated = exec_time / 1000000.0
					actions_executed_this_step = plan_actions.size()

	# Batch update: lock once to update state and all statistics
	state_mutex.lock()
	agent_states[agent_index] = state
	total_plans_generated += plans_generated_this_step
	total_actions_executed += actions_executed_this_step
	profile_data["total_planning_time"] += planning_time_accumulated
	profile_data["planning_calls"] += plans_generated_this_step
	profile_data["total_execution_time"] += exec_time_accumulated
	profile_data["total_decay_time"] += decay_time_accumulated
	profile_data["total_mutex_wait_time"] += mutex_wait_accumulated
	profile_data["mutex_locks"] += 1
	state_mutex.unlock()

func start_simulation():
	# Determine number of agents (one per CPU core, or use a reasonable default)
	var num_cores = OS.get_processor_count()
	var num_agents = min(num_cores, 4)  # Limit to 4 agents for demo

	print("Initializing %d agents (CPU cores: %d)" % [num_agents, num_cores])

	# Create agents
	for i in range(num_agents):
		var persona_id = "agent_%d" % i
		var agent = {
			"persona_id": persona_id,
			"last_planning_check": 0.0
		}
		agents.append(agent)

		# Create domain and plan for each agent
		var domain = create_domain()
		agent_domains.append(domain)

		var plan = PlannerPlan.new()
		plan.set_current_domain(domain)
		plan.set_verbose(0)
		plan.set_max_depth(15)
		agent_plans.append(plan)

		# Create initial state
		var state = create_agent_state(persona_id)
		agent_states.append(state)

		print("  Agent %d (%s) initialized" % [i, persona_id])

	print("\nInitial States:")
	for i in range(agents.size()):
		print_agent_state(i)

	# Run simulation loop
	call_deferred("simulation_step")

func print_agent_state(agent_index: int):
	var agent = agents[agent_index]
	var persona_id = agent["persona_id"]
	var state = agent_states[agent_index]
	var needs = state["needs"][persona_id]
	var location = Domain.get_location(state, persona_id)
	var money = Domain.get_money(state, persona_id)

	print("  [%s] Location: %s | Money: $%d | Needs: H=%d E=%d S=%d F=%d Hy=%d" % [
		persona_id, location, money,
		needs["hunger"], needs["energy"], needs["social"], needs["fun"], needs["hygiene"]
	])

func simulation_step():
	simulation_time_mutex.lock()
	var current_time = simulation_time_seconds
	simulation_time_mutex.unlock()

	# Early return: simulation complete
	if current_time >= total_simulation_time:
		print_profiling_summary()
		print("\n=== Simulation Complete ===")
		print("Total simulation time: %.1f minutes" % (total_simulation_time / 60.0))
		print("Total plans generated: %d" % total_plans_generated)
		print("Total actions executed: %d" % total_actions_executed)
		print("\nFinal States:")
		for i in range(agents.size()):
			print_agent_state(i)
		quit(0)
		return

	# Process all agents in parallel using WorkerThreadPool
	# Each agent gets processed on a separate thread from the pool
	var step_start = Time.get_ticks_usec()
	var task_id = WorkerThreadPool.add_group_task(process_agent, agents.size(), -1, true, "Process agents")
	active_tasks.append(task_id)

	# Wait for all agents to complete processing
	WorkerThreadPool.wait_for_group_task_completion(task_id)
	var step_duration = Time.get_ticks_usec() - step_start
	active_tasks.erase(task_id)

	# Track step timing
	state_mutex.lock()
	profile_data["step_times"].append(step_duration / 1000000.0)
	state_mutex.unlock()

	# Print state every 2 minutes
	simulation_time_mutex.lock()
	var current_sim_time = simulation_time_seconds
	simulation_time_mutex.unlock()

	if int(current_sim_time) % 120 < time_step:
		var time_minutes = current_sim_time / 60.0
		print("\n[%.1f min] Agent States:" % time_minutes)
		for i in range(agents.size()):
			print_agent_state(i)
		print_profiling_stats()

	# Advance time
	simulation_time_mutex.lock()
	simulation_time_seconds += time_step
	simulation_time_mutex.unlock()

	# Schedule next step
	call_deferred("simulation_step")

func print_profiling_stats():
	state_mutex.lock()
	var data = profile_data.duplicate(true)
	state_mutex.unlock()

	if data["planning_calls"] == 0:
		return

	var total_step_time = 0.0
	for t in data["step_times"]:
		total_step_time += t

	print("\n  [Profiling] Step timing:")
	print("    Planning: %.3fms avg (%.1f%% of time)" % [
		(data["total_planning_time"] / data["planning_calls"]) * 1000.0 if data["planning_calls"] > 0 else 0.0,
		(data["total_planning_time"] / total_step_time) * 100.0 if total_step_time > 0 else 0.0
	])
	print("    Mutex wait: %.3fms total" % (data["total_mutex_wait_time"] * 1000.0))
	print("    State ops: %.3fms total" % (data["total_state_ops_time"] * 1000.0))

func print_profiling_summary():
	state_mutex.lock()
	var data = profile_data.duplicate(true)
	state_mutex.unlock()

	var total_time = 0.0
	var max_step_time = 0.0
	var min_step_time = 999999.0
	for t in data["step_times"]:
		total_time += t
		if t > max_step_time:
			max_step_time = t
		if t < min_step_time:
			min_step_time = t
	if data["step_times"].size() == 0:
		min_step_time = 0.0

	var avg_step_time = total_time / data["step_times"].size() if data["step_times"].size() > 0 else 0.0

	print("\n=== Profiling Summary ===")
	print("Total steps: %d" % data["step_times"].size())
	print("Step timing:")
	print("  Average: %.3fms" % (avg_step_time * 1000.0))
	print("  Min: %.3fms" % (min_step_time * 1000.0))
	print("  Max: %.3fms" % (max_step_time * 1000.0))

	if data["planning_calls"] > 0:
		var avg_planning = data["total_planning_time"] / data["planning_calls"]
		var planning_pct = (data["total_planning_time"] / total_time) * 100.0 if total_time > 0 else 0.0
		print("\nPlanning (bottleneck analysis):")
		print("  Total calls: %d" % data["planning_calls"])
		print("  Total time: %.3fms" % (data["total_planning_time"] * 1000.0))
		print("  Average per call: %.3fms" % (avg_planning * 1000.0))
		print("  Percentage of total: %.1f%%" % planning_pct)

	print("\nOther operations:")
	print("  Mutex wait time: %.3fms (%.1f%% of total)" % [
		data["total_mutex_wait_time"] * 1000.0,
		(data["total_mutex_wait_time"] / total_time) * 100.0 if total_time > 0 else 0.0
	])
	print("  State operations: %.3fms (%.1f%% of total)" % [
		data["total_state_ops_time"] * 1000.0,
		(data["total_state_ops_time"] / total_time) * 100.0 if total_time > 0 else 0.0
	])
	print("  Decay operations: %.3fms (%.1f%% of total)" % [
		data["total_decay_time"] * 1000.0,
		(data["total_decay_time"] / total_time) * 100.0 if total_time > 0 else 0.0
	])
	print("  Execution operations: %.3fms (%.1f%% of total)" % [
		data["total_execution_time"] * 1000.0,
		(data["total_execution_time"] / total_time) * 100.0 if total_time > 0 else 0.0
	])

	# Thread utilization estimate
	var num_threads = min(agents.size(), OS.get_processor_count())
	var theoretical_min_time = max_step_time / num_threads if num_threads > 0 else max_step_time
	var efficiency = (theoretical_min_time / avg_step_time) * 100.0 if avg_step_time > 0 else 0.0
	print("\nThread utilization:")
	print("  CPU cores: %d" % OS.get_processor_count())
	print("  Agents: %d" % agents.size())
	print("  Estimated efficiency: %.1f%%" % efficiency)
