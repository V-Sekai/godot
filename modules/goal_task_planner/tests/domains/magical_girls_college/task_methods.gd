# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Magical Girls College Domain - Task Methods
# All task method functions that decompose tasks into subtasks

const Helpers = preload("helpers.gd")

# Task methods - Earn study points
static func task_earn_study_points_method_done(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	var current_points = Helpers.get_study_points(state, persona_id)
	if current_points >= target_points:
		return []  # Done, no subtasks needed
	return null  # Not done, try other methods

static func task_earn_study_points_method_lecture(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	var current_points = Helpers.get_study_points(state, persona_id)
	if current_points >= target_points:
		return null  # Already have enough

	# Check if homework is required
	if state.has("temporal_puzzle"):
		var puzzle = state["temporal_puzzle"]
		if puzzle.has("homework_deadline"):
			return null  # Homework required, cannot use lecture

	# Check coordination for movie that requires homework
	if state.has("coordination"):
		var coord_dict = state["coordination"]
		if coord_dict.has(persona_id):
			var coordination = coord_dict[persona_id]
			if coordination.has("requires_homework") and bool(coordination["requires_homework"]):
				return null  # Homework required, cannot use lecture

	# Execute action and recursively refine task
	var subtasks = []
	var action = ["action_attend_lecture", persona_id, "math"]

	# Attach temporal metadata: lecture duration = 2 hours = 7200000000 microseconds
	var temporal_constraints = {"duration": 7200000000}
	var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
	subtasks.append(action_with_metadata)

	# After executing the action, we'll have current_points + 5
	# If that's still not enough, recursively refine the task
	var points_after_action = current_points + 5
	if points_after_action < target_points:
		var recursive_task = ["task_earn_study_points", persona_id, target_points]
		subtasks.append(recursive_task)

	return subtasks

static func task_earn_study_points_method_homework(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	var current_points = Helpers.get_study_points(state, persona_id)
	if current_points >= target_points:
		return null  # Already have enough

	var homework_required = false
	if state.has("temporal_puzzle"):
		var puzzle = state["temporal_puzzle"]
		if puzzle.has("homework_deadline"):
			homework_required = true

	if not homework_required and state.has("coordination"):
		var coord_dict = state["coordination"]
		if coord_dict.has(persona_id):
			var coord = coord_dict[persona_id]
			if coord.has("requires_homework") and bool(coord["requires_homework"]):
				homework_required = true

	# Check if there's an unused study session
	var has_unused_study_session = false
	if state.has("temporal_puzzle"):
		var puzzle = state["temporal_puzzle"]
		if puzzle.has("morning_study_time"):
			var coordination = Helpers.get_coordination(state, persona_id)
			if coordination.is_empty() or not coordination.has("used") or not bool(coordination["used"]):
				has_unused_study_session = true

	if not has_unused_study_session:
		var coordination = Helpers.get_coordination(state, persona_id)
		if not coordination.is_empty() and coordination.has("action") and str(coordination["action"]) == "study_session":
			if coordination.has("location") and str(coordination["location"]) == "library":
				if not coordination.has("used") or not bool(coordination["used"]):
					has_unused_study_session = true

	if has_unused_study_session and not homework_required:
		return null  # Let coordinated method handle it first

	# Execute action and recursively refine task
	var subtasks = []
	var action = ["action_complete_homework", persona_id, "math"]

	# Attach temporal metadata: homework duration = 1.5 hours = 5400000000 microseconds
	var temporal_constraints = {"duration": 5400000000}

	# If homework_deadline exists, set end_time to deadline
	if homework_required and state.has("temporal_puzzle"):
		var puzzle = state["temporal_puzzle"]
		if puzzle.has("homework_deadline"):
			var deadline = puzzle["homework_deadline"]
			if deadline is int:
				temporal_constraints["end_time"] = deadline
				var homework_duration = 5400000000
				temporal_constraints["start_time"] = deadline - homework_duration

	var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
	subtasks.append(action_with_metadata)

	var points_after_action = current_points + 3
	if points_after_action < target_points:
		var recursive_task = ["task_earn_study_points", persona_id, target_points]
		subtasks.append(recursive_task)

	return subtasks

static func task_earn_study_points_method_library(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	var current_points = Helpers.get_study_points(state, persona_id)
	if current_points >= target_points:
		return null  # Already have enough

	# Check if homework is required
	if state.has("temporal_puzzle"):
		var puzzle = state["temporal_puzzle"]
		if puzzle.has("homework_deadline"):
			return null  # Homework required, cannot use library

	if state.has("coordination"):
		var coord_dict = state["coordination"]
		if coord_dict.has(persona_id):
			var coord = coord_dict[persona_id]
			if coord.has("requires_homework") and bool(coord["requires_homework"]):
				return null  # Homework required, cannot use library

	# Execute action and recursively refine task
	var subtasks = []
	var action = ["action_study_library", persona_id]

	# Attach temporal metadata: library study duration = 2 hours = 7200000000 microseconds
	var temporal_constraints = {"duration": 7200000000}
	var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
	subtasks.append(action_with_metadata)

	var points_after_action = current_points + 4
	if points_after_action < target_points:
		var recursive_task = ["task_earn_study_points", persona_id, target_points]
		subtasks.append(recursive_task)

	return subtasks

static func task_earn_study_points_method_coordinated(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	var current_points = Helpers.get_study_points(state, persona_id)
	if current_points >= target_points:
		return null  # Already have enough

	# Check for study session coordination
	var coord_time = 0
	var is_study_session = false
	var coordination = Helpers.get_coordination(state, persona_id)

	# First, check temporal_puzzle for morning_study_time
	if state.has("temporal_puzzle"):
		var puzzle = state["temporal_puzzle"]
		if puzzle.has("morning_study_time"):
			var morning_time_var = puzzle["morning_study_time"]
			if morning_time_var is int:
				coord_time = morning_time_var
				is_study_session = true

	# Fallback: check coordination dictionary
	if not is_study_session and not coordination.is_empty() and coordination.has("time"):
		var coord_time_var = coordination.get("time", null)
		if coord_time_var is int:
			var temp_time = coord_time_var
			var coord_location = coordination.get("location", "")
			var coord_action = coordination.get("action", "")

			if coord_action == "study_session" and coord_location == "library":
				coord_time = temp_time
				is_study_session = true
			elif coord_location == "library" and state.has("temporal_puzzle"):
				var puzzle = state["temporal_puzzle"]
				if puzzle.has("homework_deadline"):
					var deadline_var = puzzle["homework_deadline"]
					if deadline_var is int:
						var movie_time = deadline_var
						if temp_time < movie_time:
							coord_time = temp_time
							is_study_session = true

	if not is_study_session or coord_time == 0:
		return null  # No study session coordination found

	if not coordination.is_empty() and coordination.has("used") and bool(coordination["used"]):
		return null  # Coordination already used

	var action = ["action_study_library", persona_id]
	var study_duration = 3600000000
	var coord_end_time = coord_time + study_duration

	var temporal_constraints = {}
	temporal_constraints["start_time"] = coord_time
	temporal_constraints["end_time"] = coord_end_time
	temporal_constraints["duration"] = study_duration

	var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
	var subtasks = []
	subtasks.append(action_with_metadata)

	var points_after_action = current_points + 4
	if points_after_action < target_points:
		var recursive_task = ["task_earn_study_points", persona_id, target_points]
		subtasks.append(recursive_task)

	return subtasks

# Task methods - Socialize
static func task_socialize_method_easy(state: Dictionary, persona_id: Variant, companion_id: Variant, activity_level: Variant) -> Variant:
	if activity_level > 1:
		return null  # Not easy activity

	var is_kira = (str(companion_id) == "kira")
	var movie_scheduled = false
	var movie_end_time = 0

	if state.has("coordination"):
		var coord_dict = state["coordination"]
		if coord_dict.has(persona_id):
			var coordination = coord_dict[persona_id]
			if coordination.has("action") and str(coordination["action"]) == "movie":
				movie_scheduled = true
				if coordination.has("time"):
					var movie_time_var = coordination["time"]
					if movie_time_var is int:
						var movie_time = movie_time_var
						movie_end_time = movie_time + 7200000000

	if not movie_scheduled and state.has("temporal_puzzle"):
		var puzzle = state["temporal_puzzle"]
		if puzzle.has("homework_deadline") and is_kira:
			var deadline_var = puzzle["homework_deadline"]
			if deadline_var is int:
				var movie_time = deadline_var
				movie_scheduled = true
				movie_end_time = movie_time + 7200000000

	var subtasks = []
	var action = []

	if is_kira and movie_scheduled and movie_end_time > 0:
		action = ["action_coffee_together", persona_id, companion_id]
		var temporal_constraints = {}
		temporal_constraints["start_time"] = movie_end_time
		temporal_constraints["duration"] = 3600000000
		temporal_constraints["end_time"] = movie_end_time + 3600000000
		var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
		subtasks.append(action_with_metadata)
	else:
		var coordination = Helpers.get_coordination(state, persona_id)
		if not coordination.is_empty() and coordination.has("action") and str(coordination["action"]) == "lunch" and str(coordination.get("companion", "")) == str(companion_id):
			action = ["action_eat_mess_hall_with_friend", persona_id, companion_id]
			var temporal_constraints = {}
			if coordination.has("time"):
				var lunch_time = coordination["time"]
				if lunch_time is int:
					temporal_constraints["start_time"] = lunch_time
					temporal_constraints["end_time"] = lunch_time + 1800000000
			var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
			subtasks.append(action_with_metadata)
		else:
			action = ["action_eat_mess_hall_with_friend", persona_id, companion_id]
			var temporal_constraints = {"duration": 1800000000}
			var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
			subtasks.append(action_with_metadata)

	return subtasks

static func task_socialize_method_moderate(state: Dictionary, persona_id: Variant, companion_id: Variant, activity_level: Variant) -> Variant:
	if activity_level != 2:
		return null  # Not moderate activity

	var subtasks = []
	var action = ["action_watch_movie", persona_id, companion_id]
	var temporal_constraints = {"duration": 7200000000}
	var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
	subtasks.append(action_with_metadata)
	return subtasks

static func task_socialize_method_challenging(state: Dictionary, persona_id: Variant, companion_id: Variant, activity_level: Variant) -> Variant:
	if activity_level < 3:
		return null  # Not challenging activity

	var subtasks = []
	var action = ["action_park_picnic", persona_id, companion_id]
	var temporal_constraints = {"duration": 10800000000}
	var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
	subtasks.append(action_with_metadata)
	return subtasks

# Task methods - Manage week
static func task_manage_week_method_balance(state: Dictionary, persona_id: Variant) -> Variant:
	var subtasks = []
	subtasks.append(["task_earn_study_points", persona_id, 10])
	subtasks.append(["task_socialize", persona_id, "maya", 2])
	return subtasks

static func task_manage_week_method_academics(state: Dictionary, persona_id: Variant) -> Variant:
	var subtasks = []
	subtasks.append(["task_earn_study_points", persona_id, 20])
	return subtasks

static func task_manage_week_method_relationships(state: Dictionary, persona_id: Variant) -> Variant:
	var subtasks = []
	subtasks.append(["task_socialize", persona_id, "maya", 3])
	subtasks.append(["task_socialize", persona_id, "aria", 2])
	return subtasks

# Task methods - The Sims-style needs satisfaction (5 methods per need for deep backtracking)
# Hunger satisfaction methods (5 methods)
static func task_satisfy_hunger_method_mess_hall(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	var current_hunger = Helpers.get_need(state, persona_id, "hunger")
	if current_hunger >= target_hunger:
		return []
	var subtasks = []
	subtasks.append(["action_eat_mess_hall", persona_id])
	return subtasks

static func task_satisfy_hunger_method_restaurant(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	var current_hunger = Helpers.get_need(state, persona_id, "hunger")
	if current_hunger >= target_hunger:
		return []
	var current_money = Helpers.get_money(state, persona_id)
	if current_money < 20:
		return null
	var subtasks = []
	subtasks.append(["action_eat_restaurant", persona_id])
	return subtasks

static func task_satisfy_hunger_method_snack(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	var current_hunger = Helpers.get_need(state, persona_id, "hunger")
	if current_hunger >= target_hunger:
		return []
	var current_money = Helpers.get_money(state, persona_id)
	if current_money < 5:
		return null
	var subtasks = []
	subtasks.append(["action_eat_snack", persona_id])
	var hunger_after = current_hunger + 20
	if hunger_after < target_hunger:
		subtasks.append(["task_satisfy_hunger", persona_id, target_hunger])
	return subtasks

static func task_satisfy_hunger_method_cook(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	var current_hunger = Helpers.get_need(state, persona_id, "hunger")
	if current_hunger >= target_hunger:
		return []
	var current_location = Helpers.get_location(state, persona_id)
	if current_location != "dorm":
		return null
	var subtasks = []
	subtasks.append(["action_cook_meal", persona_id])
	return subtasks

static func task_satisfy_hunger_method_social_eat(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	var current_hunger = Helpers.get_need(state, persona_id, "hunger")
	if current_hunger >= target_hunger:
		return []
	var subtasks = []
	subtasks.append(["action_eat_mess_hall_with_friend", persona_id, "maya"])
	return subtasks

# Energy/Sleep satisfaction methods (5 methods)
static func task_satisfy_energy_method_sleep(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	var current_energy = Helpers.get_need(state, persona_id, "energy")
	if current_energy >= target_energy:
		return []
	var current_location = Helpers.get_location(state, persona_id)
	if current_location != "dorm":
		var subtasks = []
		subtasks.append(["task_move_to_location", persona_id, "dorm"])
		subtasks.append(["action_sleep_dorm", persona_id])
		return subtasks
	var subtasks = []
	subtasks.append(["action_sleep_dorm", persona_id])
	return subtasks

static func task_satisfy_energy_method_nap(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	var current_energy = Helpers.get_need(state, persona_id, "energy")
	if current_energy >= target_energy:
		return []
	var current_location = Helpers.get_location(state, persona_id)
	if current_location != "library":
		return null
	var subtasks = []
	subtasks.append(["action_nap_library", persona_id])
	var energy_after = min(100, current_energy + 30)
	if energy_after < target_energy:
		subtasks.append(["task_satisfy_energy", persona_id, target_energy])
	return subtasks

static func task_satisfy_energy_method_drink(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	var current_energy = Helpers.get_need(state, persona_id, "energy")
	if current_energy >= target_energy:
		return []
	var current_money = Helpers.get_money(state, persona_id)
	if current_money < 3:
		return null
	var subtasks = []
	subtasks.append(["action_energy_drink", persona_id])
	var energy_after = min(100, current_energy + 20)
	if energy_after < target_energy:
		subtasks.append(["task_satisfy_energy", persona_id, target_energy])
	return subtasks

static func task_satisfy_energy_method_rest_activity(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	var current_energy = Helpers.get_need(state, persona_id, "energy")
	if current_energy >= target_energy:
		return []
	var subtasks = []
	subtasks.append(["action_read_book", persona_id])
	var energy_after = min(100, current_energy + 10)
	if energy_after < target_energy:
		subtasks.append(["task_satisfy_energy", persona_id, target_energy])
	return subtasks

static func task_satisfy_energy_method_early_sleep(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	var current_energy = Helpers.get_need(state, persona_id, "energy")
	if current_energy >= target_energy:
		return []
	var subtasks = []
	subtasks.append(["task_move_to_location", persona_id, "dorm"])
	subtasks.append(["action_sleep_dorm", persona_id])
	return subtasks

# Social satisfaction methods (5 methods)
static func task_satisfy_social_method_talk(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	var current_social = Helpers.get_need(state, persona_id, "social")
	if current_social >= target_social:
		return []
	var subtasks = []
	subtasks.append(["action_talk_friend", persona_id, "maya"])
	var social_after = min(100, current_social + 15)
	if social_after < target_social:
		subtasks.append(["task_satisfy_social", persona_id, target_social])
	return subtasks

static func task_satisfy_social_method_club(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	var current_social = Helpers.get_need(state, persona_id, "social")
	if current_social >= target_social:
		return []
	var subtasks = []
	subtasks.append(["action_join_club", persona_id, "magic_club"])
	return subtasks

static func task_satisfy_social_method_phone(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	var current_social = Helpers.get_need(state, persona_id, "social")
	if current_social >= target_social:
		return []
	var subtasks = []
	subtasks.append(["action_phone_call", persona_id, "maya"])
	var social_after = min(100, current_social + 10)
	if social_after < target_social:
		subtasks.append(["task_satisfy_social", persona_id, target_social])
	return subtasks

static func task_satisfy_social_method_socialize_task(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	var current_social = Helpers.get_need(state, persona_id, "social")
	if current_social >= target_social:
		return []
	var subtasks = []
	subtasks.append(["task_socialize", persona_id, "maya", 2])
	return subtasks

static func task_satisfy_social_method_group_activity(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	var current_social = Helpers.get_need(state, persona_id, "social")
	if current_social >= target_social:
		return []
	var subtasks = []
	subtasks.append(["action_watch_movie", persona_id, "maya"])
	return subtasks

# Fun satisfaction methods (5 methods)
static func task_satisfy_fun_method_games(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	var current_fun = Helpers.get_need(state, persona_id, "fun")
	if current_fun >= target_fun:
		return []
	var current_location = Helpers.get_location(state, persona_id)
	if current_location != "dorm":
		return null
	var subtasks = []
	subtasks.append(["action_play_games", persona_id])
	var fun_after = min(100, current_fun + 30)
	if fun_after < target_fun:
		subtasks.append(["task_satisfy_fun", persona_id, target_fun])
	return subtasks

static func task_satisfy_fun_method_streaming(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	var current_fun = Helpers.get_need(state, persona_id, "fun")
	if current_fun >= target_fun:
		return []
	var current_location = Helpers.get_location(state, persona_id)
	if current_location != "dorm":
		return null
	var subtasks = []
	subtasks.append(["action_watch_streaming", persona_id])
	var fun_after = min(100, current_fun + 25)
	if fun_after < target_fun:
		subtasks.append(["task_satisfy_fun", persona_id, target_fun])
	return subtasks

static func task_satisfy_fun_method_cinema(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	var current_fun = Helpers.get_need(state, persona_id, "fun")
	if current_fun >= target_fun:
		return []
	var current_money = Helpers.get_money(state, persona_id)
	if current_money < 15:
		return null
	var subtasks = []
	subtasks.append(["action_go_cinema", persona_id])
	return subtasks

static func task_satisfy_fun_method_preferred_activity(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	var current_fun = Helpers.get_need(state, persona_id, "fun")
	if current_fun >= target_fun:
		return []
	var subtasks = []
	if Helpers.likes_activity(state, persona_id, "movies"):
		subtasks.append(["action_watch_movie", persona_id, "maya"])
	elif Helpers.likes_activity(state, persona_id, "beach"):
		subtasks.append(["action_beach_trip", persona_id, "maya"])
	else:
		return null
	return subtasks

static func task_satisfy_fun_method_social_fun(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	var current_fun = Helpers.get_need(state, persona_id, "fun")
	if current_fun >= target_fun:
		return []
	var subtasks = []
	subtasks.append(["task_socialize", persona_id, "maya", 2])
	return subtasks

# Hygiene satisfaction methods (3 methods)
static func task_satisfy_hygiene_method_shower(state: Dictionary, persona_id: Variant, target_hygiene: Variant) -> Variant:
	var current_hygiene = Helpers.get_need(state, persona_id, "hygiene")
	if current_hygiene >= target_hygiene:
		return []
	var current_location = Helpers.get_location(state, persona_id)
	if current_location != "dorm":
		var subtasks = []
		subtasks.append(["task_move_to_location", persona_id, "dorm"])
		subtasks.append(["action_shower", persona_id])
		return subtasks
	var subtasks = []
	subtasks.append(["action_shower", persona_id])
	return subtasks

static func task_satisfy_hygiene_method_wash_hands(state: Dictionary, persona_id: Variant, target_hygiene: Variant) -> Variant:
	var current_hygiene = Helpers.get_need(state, persona_id, "hygiene")
	if current_hygiene >= target_hygiene:
		return []
	var subtasks = []
	subtasks.append(["action_wash_hands", persona_id])
	var hygiene_after = min(100, current_hygiene + 20)
	if hygiene_after < target_hygiene:
		subtasks.append(["task_satisfy_hygiene", persona_id, target_hygiene])
	return subtasks

static func task_satisfy_hygiene_method_force_shower(state: Dictionary, persona_id: Variant, target_hygiene: Variant) -> Variant:
	var current_hygiene = Helpers.get_need(state, persona_id, "hygiene")
	if current_hygiene >= target_hygiene:
		return []
	var subtasks = []
	subtasks.append(["task_move_to_location", persona_id, "dorm"])
	subtasks.append(["action_shower", persona_id])
	return subtasks

# Simple move task method
static func task_move_to_location_method_direct(state: Dictionary, persona_id: Variant, location: Variant) -> Variant:
	var current_location = Helpers.get_location(state, persona_id)
	var target_location = str(location)
	if current_location == target_location:
		return []
	var subtasks = []
	subtasks.append(["action_move_to", persona_id, target_location])
	return subtasks
