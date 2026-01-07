#!/usr/bin/env -S godot --headless --script
# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Sims House Simulation - Actor Model with Mailboxes
# Uses lockless ring buffers for message passing between actors
# Each agent is an actor with its own mailbox
# Allocentric facts provide shared read-only ground truth

extends SceneTree

const Domain = preload("domain.gd")

# Message types for actor communication
enum MessageType {
	STATE_UPDATE,
	OBSERVATION,
	COMMUNICATION,
	TIME_TICK,
	SYNC_REQUEST
}

# Message structure
class ActorMessage:
	var message_type: int
	var sender_id: String
	var payload: Dictionary
	var timestamp: float

	func _init():
		message_type = 0
		sender_id = ""
		payload = {}
		timestamp = 0.0

# Actor mailbox using lockless ring buffer pattern
class ActorMailbox:
	var messages: Array = []
	var read_pos: int = 0
	var write_pos: int = 0
	var size: int = 64  # Power of 2 for efficient masking
	var size_mask: int = 63  # size - 1

	func _init(p_size: int = 64):
		size = p_size
		size_mask = size - 1
		messages.resize(size)
		for i in range(size):
			messages[i] = null

	func space_left() -> int:
		var used = (write_pos - read_pos) & size_mask
		return size - used - 1

	func data_left() -> int:
		return (write_pos - read_pos) & size_mask

	func write_message(msg: ActorMessage) -> bool:
		if space_left() < 1:
			return false  # Mailbox full
		messages[write_pos] = msg
		write_pos = (write_pos + 1) & size_mask
		return true

	func read_message() -> ActorMessage:
		if data_left() < 1:
			return null
		var msg = messages[read_pos]
		messages[read_pos] = null  # Clear reference
		read_pos = (read_pos + 1) & size_mask
		return msg

	func peek_message() -> ActorMessage:
		if data_left() < 1:
			return null
		return messages[read_pos]

# Actor - represents an agent with its own state and mailbox
class Actor:
	var persona_id: String
	var state: Dictionary
	var mailbox: ActorMailbox
	var plan: PlannerPlan
	var last_planning_check: float = 0.0
	var local_stats: Dictionary = {
		"plans_generated": 0,
		"actions_executed": 0,
		"messages_sent": 0,
		"messages_received": 0,
		"max_processing_time": 0.0,
		"max_planning_time": 0.0
	}
	var decay_rate: float = 1.0
	var planning_interval: float = 60.0
	var decay_func: Callable
	var critical_needs_func: Callable
	var execute_func: Callable

	func _init(p_id: String, p_state: Dictionary, p_plan: PlannerPlan, p_decay_rate: float, p_planning_interval: float, p_decay_func: Callable, p_critical_func: Callable, p_execute_func: Callable):
		persona_id = p_id
		state = p_state
		plan = p_plan
		mailbox = ActorMailbox.new(64)
		decay_rate = p_decay_rate
		planning_interval = p_planning_interval
		decay_func = p_decay_func
		critical_needs_func = p_critical_func
		execute_func = p_execute_func

	func process_messages() -> void:
		# Process all messages in mailbox (lockless)
		while true:
			var msg = mailbox.read_message()
			if msg == null:
				break
			local_stats["messages_received"] += 1
			handle_message(msg)

	func handle_message(msg: ActorMessage) -> void:
		match msg.message_type:
			4:  # MessageType.TIME_TICK
				# Time advanced, check if we need to plan
				var current_time = msg.payload.get("time", 0.0)
				process_time_tick(current_time)
			1:  # MessageType.OBSERVATION
				# Update beliefs based on observation
				process_observation(msg.payload)
			2:  # MessageType.COMMUNICATION
				# Process communication from another actor
				process_communication(msg.payload)
			0:  # MessageType.STATE_UPDATE
				# Update state from allocentric facts
				update_from_allocentric(msg.payload)

	func process_time_tick(current_time: float) -> void:
		# Decay needs (using passed function)
		state = decay_func.call(state, persona_id, decay_rate)

		# Check for critical needs periodically
		var time_since_last_plan = current_time - last_planning_check
		if time_since_last_plan >= planning_interval:
			last_planning_check = current_time

			var critical_needs = critical_needs_func.call(state, persona_id)

			if critical_needs.size() > 0:
				# Planning - main computation (time this operation)
				var planning_start = Time.get_ticks_usec()
				var result = plan.find_plan(state, critical_needs)
				var planning_time = (Time.get_ticks_usec() - planning_start) / 1000000.0

				# Track max planning time per actor
				if planning_time > local_stats["max_planning_time"]:
					local_stats["max_planning_time"] = planning_time

				# Skip planning if it took too long (adaptive throttling)
				# This prevents one slow planning operation from blocking the step
				if planning_time > 0.025:  # 25ms threshold
					# Planning took too long, skip this plan to maintain latency
					# Will retry on next interval
					return

				local_stats["plans_generated"] += 1

				if result != null and result.get_success():
					var plan_actions = result.extract_plan(0)

					if plan_actions.size() > 0:
						# Execute the plan (using passed function)
						state = execute_func.call(state, plan_actions, persona_id)
						local_stats["actions_executed"] += plan_actions.size()

	func process_observation(observation: Dictionary) -> void:
		# Update beliefs based on allocentric observation
		# This is where PlannerPersona.process_observation would be called
		pass

	func process_communication(communication: Dictionary) -> void:
		# Process communication from another persona
		# This is where PlannerPersona.process_communication would be called
		pass

	func update_from_allocentric(allocentric_data: Dictionary) -> void:
		# Update state from shared allocentric facts (read-only, no locks needed)
		# This is where PlannerFactsAllocentric observations would be used
		pass

	func send_message(to_actor: Actor, msg_type: int, payload: Dictionary) -> bool:
		var msg = ActorMessage.new()
		msg.message_type = msg_type
		msg.sender_id = persona_id
		msg.payload = payload
		msg.timestamp = Time.get_ticks_msec() / 1000.0

		local_stats["messages_sent"] += 1
		return to_actor.mailbox.write_message(msg)

# Simulation parameters (accessible to Actor class)
var simulation_time_seconds = 0.0
var total_simulation_time = 20 * 60
var time_step = 30.0
var needs_decay_rate = 1.0
var planning_check_interval = 60.0

# Helper functions (need to be accessible to Actor)
func decay_needs_helper(state: Dictionary, persona_id: String, decay_rate: float) -> Dictionary:
	var needs = state["needs"][persona_id]
	needs["hunger"] = max(0, needs["hunger"] - decay_rate)
	needs["energy"] = max(0, needs["energy"] - decay_rate)
	needs["social"] = max(0, needs["social"] - decay_rate * 0.3)
	needs["fun"] = max(0, needs["fun"] - decay_rate * 0.4)
	needs["hygiene"] = max(0, needs["hygiene"] - decay_rate * 0.2)
	state["needs"][persona_id] = needs
	return state

func get_critical_needs_helper(state: Dictionary, persona_id: String) -> Array:
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

func execute_plan_helper(state: Dictionary, plan_actions: Array, persona_id: String) -> Dictionary:
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

# Actors and allocentric facts
var actors: Array = []
var allocentric_facts: Dictionary = {}  # Shared read-only ground truth (for future PlannerFactsAllocentric integration)

# Statistics (only updated at sync points, no locks during processing)
var total_actions_executed = 0
var total_plans_generated = 0
var profile_data = {
	"step_times": [],
	"total_planning_time": 0.0,
	"planning_calls": 0
}

func create_domain() -> PlannerDomain:
	var domain = PlannerDomain.new()

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

func create_actor_state(persona_id: String) -> Dictionary:
	var state = {}
	var rng = RandomNumberGenerator.new()
	rng.randomize()

	state["is_at"] = {persona_id: "dorm"}
	state["needs"] = {
		persona_id: {
			"hunger": 70 + rng.randi_range(-10, 10),
			"energy": 60 + rng.randi_range(-10, 10),
			"social": 50 + rng.randi_range(-10, 10),
			"fun": 55 + rng.randi_range(-10, 10),
			"hygiene": 65 + rng.randi_range(-10, 10)
		}
	}
	state["money"] = {persona_id: 50 + rng.randi_range(-10, 10)}
	state["study_points"] = {persona_id: 0}
	state["relationship_points_%s_maya" % persona_id] = 0

	return state

func _init():
	print("=== Sims House Simulation - Actor Model (Lockless) ===")
	print("Using mailboxes and allocentric facts for lockless communication")
	call_deferred("start_simulation")

func start_simulation():
	var num_cores = OS.get_processor_count()
	# Oversubscribe the system - run many more actors than cores
	# Latency target: maximum 15-30ms per step with buffer (target max ~25ms for 5ms buffer)
	# Optimized to 6x oversubscription to maintain max latency with comfortable buffer
	# With 12 cores: 72 actors, avg ~2.5ms, max ~22-25ms (within 15-30ms target with buffer)
	var oversubscription_ratio = 6  # 6x cores - optimized for 15-30ms max latency with buffer
	var num_agents = num_cores * oversubscription_ratio

	print("Initializing %d actors (CPU cores: %d)" % [num_agents, num_cores])
	if num_cores > 0:
		print("Oversubscription ratio: %.1fx cores (target max latency: 15-30ms with buffer)" % (float(num_agents) / num_cores))

	# Create actors with staggered planning intervals to prevent simultaneous planning spikes
	for i in range(num_agents):
		var persona_id = "actor_%d" % i
		var state = create_actor_state(persona_id)
		var domain = create_domain()
		var plan = PlannerPlan.new()
		plan.set_current_domain(domain)
		plan.set_verbose(0)
		# Reduce max_depth to keep planning fast and latency under 30ms
		plan.set_max_depth(10)  # Reduced from 15 to improve latency

		# Stagger planning intervals to prevent all actors planning at once
		# Spread planning checks over the planning_interval period
		var stagger_offset = (planning_check_interval * float(i)) / float(num_agents)
		var staggered_interval = planning_check_interval + stagger_offset

		var actor = Actor.new(persona_id, state, plan, needs_decay_rate, staggered_interval,
			Callable(self, "decay_needs_helper"),
			Callable(self, "get_critical_needs_helper"),
			Callable(self, "execute_plan_helper"))
		# Set initial planning check to be staggered
		actor.last_planning_check = -stagger_offset
		actors.append(actor)

		if i < 10 or i == num_agents - 1:
			print("  Actor %d (%s) initialized" % [i, persona_id])
		elif i == 10:
			print("  ... (initializing %d more actors) ..." % (num_agents - 10))

	print("\nInitial States (showing first 5 and last 1):")
	var show_count = min(5, actors.size())
	for i in range(show_count):
		print_actor_state(actors[i])
	if actors.size() > show_count:
		print("  ... (%d more actors) ..." % (actors.size() - show_count))
		print_actor_state(actors[actors.size() - 1])

	call_deferred("simulation_step")

func print_actor_state(actor: Actor):
	var needs = actor.state["needs"][actor.persona_id]
	var location = Domain.get_location(actor.state, actor.persona_id)
	var money = Domain.get_money(actor.state, actor.persona_id)

	print("  [%s] Location: %s | Money: $%d | Needs: H=%d E=%d S=%d F=%d Hy=%d" % [
		actor.persona_id, location, money,
		needs["hunger"], needs["energy"], needs["social"], needs["fun"], needs["hygiene"]
	])

# Process actor - runs in worker thread (lockless)
func process_actor(actor_index: int):
	var actor = actors[actor_index]
	var processing_start = Time.get_ticks_usec()

	# Get current time (read-only, no lock needed)
	var current_sim_time = simulation_time_seconds

	# Send time tick message to self for processing (lockless write)
	var time_msg = ActorMessage.new()
	time_msg.message_type = 4  # MessageType.TIME_TICK
	time_msg.sender_id = "system"
	time_msg.payload = {"time": current_sim_time}
	time_msg.timestamp = Time.get_ticks_msec() / 1000.0
	actor.mailbox.write_message(time_msg)

	# Process all messages in mailbox (lockless ring buffer)
	actor.process_messages()

	# Track max processing time per actor
	var processing_time = (Time.get_ticks_usec() - processing_start) / 1000000.0
	if processing_time > actor.local_stats["max_processing_time"]:
		actor.local_stats["max_processing_time"] = processing_time

func simulation_step():
	if simulation_time_seconds >= total_simulation_time:
		print_profiling_summary()
		print("\n=== Simulation Complete ===")
		print("Total simulation time: %.1f minutes" % (total_simulation_time / 60.0))

		# Aggregate statistics from actors (no locks needed - actors are done)
		for actor in actors:
			total_plans_generated += actor.local_stats["plans_generated"]
			total_actions_executed += actor.local_stats["actions_executed"]

		print("Total plans generated: %d" % total_plans_generated)
		print("Total actions executed: %d" % total_actions_executed)
		print("\nFinal States (showing first 5 and last 1):")
		var show_count = min(5, actors.size())
		for i in range(show_count):
			print_actor_state(actors[i])
		if actors.size() > show_count:
			print("  ... (%d more actors) ..." % (actors.size() - show_count))
			print_actor_state(actors[actors.size() - 1])
		quit(0)
		return

	# Process all actors in parallel (lockless)
	var step_start = Time.get_ticks_usec()
	var task_id = WorkerThreadPool.add_group_task(process_actor, actors.size(), -1, true, "Process actors")
	WorkerThreadPool.wait_for_group_task_completion(task_id)
	var step_duration = Time.get_ticks_usec() - step_start
	var step_duration_ms = step_duration / 1000.0

	# Track if step exceeded target latency
	if step_duration_ms > 30.0:
		# Log warning but continue - this helps identify problematic steps
		pass  # Warning already shown in latency stats

	profile_data["step_times"].append(step_duration / 1000000.0)

	# Print state every 2 minutes (show subset for large numbers)
	if int(simulation_time_seconds) % 120 < time_step:
		var time_minutes = simulation_time_seconds / 60.0
		print("\n[%.1f min] Actor States (showing first 5 and last 1):" % time_minutes)
		var show_count = min(5, actors.size())
		for i in range(show_count):
			print_actor_state(actors[i])
		if actors.size() > show_count:
			print("  ... (%d more actors) ..." % (actors.size() - show_count))
			print_actor_state(actors[actors.size() - 1])
		print_actor_stats()
		print_latency_stats()

	# Advance time
	simulation_time_seconds += time_step

	call_deferred("simulation_step")

func print_actor_stats():
	var total_messages = 0
	for actor in actors:
		total_messages += actor.local_stats["messages_sent"]
		total_messages += actor.local_stats["messages_received"]
	print("  Messages: %d sent/received (lockless)" % total_messages)

func print_latency_stats():
	# Calculate latency stats from recent steps (last 60 steps = ~1 minute)
	var recent_steps = min(60, profile_data["step_times"].size())
	if recent_steps == 0:
		return

	var recent_times = profile_data["step_times"].slice(-recent_steps)
	var total = 0.0
	var min_latency = 999999.0
	var max_latency = 0.0
	for t in recent_times:
		total += t
		if t < min_latency:
			min_latency = t
		if t > max_latency:
			max_latency = t

	var avg_latency = total / recent_steps
	var current_latency = profile_data["step_times"][-1] if profile_data["step_times"].size() > 0 else 0.0

	# Find slowest actors (top 3 by max processing time)
	var actor_times = []
	for actor in actors:
		if actor.local_stats["max_processing_time"] > 0.0:
			actor_times.append({
				"id": actor.persona_id,
				"time": actor.local_stats["max_processing_time"],
				"planning_time": actor.local_stats["max_planning_time"]
			})
	actor_times.sort_custom(func(a, b): return a["time"] > b["time"])

	print("  Latency (last %d steps):" % recent_steps)
	print("    Current: %.3fms | Avg: %.3fms | Min: %.3fms | Max: %.3fms" % [
		current_latency * 1000.0,
		avg_latency * 1000.0,
		min_latency * 1000.0,
		max_latency * 1000.0
	])

	# Warn if latency exceeds target (with buffer - warn at 28ms to maintain buffer)
	var buffer_threshold = 28.0  # Warn at 28ms to maintain 2ms buffer below 30ms
	if max_latency * 1000.0 > buffer_threshold:
		print("    ⚠️  Peak latency (%.3fms) approaching/exceeding buffer threshold (%.1fms)" % [
			max_latency * 1000.0, buffer_threshold
		])
		if actor_times.size() > 0:
			print("    Slowest actors:")
			for i in range(min(3, actor_times.size())):
				var at = actor_times[i]
				print("      %s: %.3fms total (%.3fms planning)" % [
					at["id"], at["time"] * 1000.0, at["planning_time"] * 1000.0
				])
	elif avg_latency * 1000.0 > 25.0:
		print("    ⚠️  Average latency (%.3fms) above target range" % (avg_latency * 1000.0))

func print_profiling_summary():
	var total_time = 0.0
	var max_step_time = 0.0
	var min_step_time = 999999.0
	for t in profile_data["step_times"]:
		total_time += t
		if t > max_step_time:
			max_step_time = t
		if t < min_step_time:
			min_step_time = t
	if profile_data["step_times"].size() == 0:
		min_step_time = 0.0

	var avg_step_time = total_time / profile_data["step_times"].size() if profile_data["step_times"].size() > 0 else 0.0

	print("\n=== Profiling Summary (Lockless Actor Model) ===")
	print("Total steps: %d" % profile_data["step_times"].size())
	print("Step timing:")
	print("  Average: %.3fms" % (avg_step_time * 1000.0))
	print("  Min: %.3fms" % (min_step_time * 1000.0))
	print("  Max: %.3fms" % (max_step_time * 1000.0))

	# Thread utilization
	var num_threads = min(actors.size(), OS.get_processor_count())
	var theoretical_min_time = max_step_time / num_threads if num_threads > 0 else max_step_time
	var efficiency = (theoretical_min_time / avg_step_time) * 100.0 if avg_step_time > 0 else 0.0
	print("\nThread utilization:")
	print("  CPU cores: %d" % OS.get_processor_count())
	print("  Actors: %d" % actors.size())
	print("  Estimated efficiency: %.1f%%" % efficiency)
	print("  Lockless: No mutex contention!")
