# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Magical Girls College Domain - GDScript version
# Domain definition with actions and methods for testing GDScript bindings

class_name MagicalGirlsCollegeDomain

# Helper functions
static func get_int(state: Dictionary, key: String, default_value: int = 0) -> int:
	if not state.has(key):
		return default_value
	var val = state[key]
	if val is int:
		return val
	return default_value

static func get_string(state: Dictionary, key: String, default_value: String = "") -> String:
	if not state.has(key):
		return default_value
	var val = state[key]
	if val is String:
		return val
	return default_value

static func get_study_points(state: Dictionary, persona_id: String) -> int:
	if not state.has("study_points"):
		return 0
	var study_points = state["study_points"]
	if study_points.has(persona_id):
		var val = study_points[persona_id]
		if val is int:
			return val
	return 0

static func set_study_points(state: Dictionary, persona_id: String, points: int) -> void:
	if not state.has("study_points"):
		state["study_points"] = {}
	var study_points = state["study_points"]
	study_points[persona_id] = points
	state["study_points"] = study_points

static func get_relationship_points(state: Dictionary, persona_id: String, companion_id: String) -> int:
	var predicate = "relationship_points_%s_%s" % [persona_id, companion_id]
	if not state.has(predicate):
		return 0
	var val = state[predicate]
	if val is int:
		return val
	return 0

static func set_relationship_points(state: Dictionary, persona_id: String, companion_id: String, points: int) -> void:
	var predicate = "relationship_points_%s_%s" % [persona_id, companion_id]
	state[predicate] = points

static func get_location(state: Dictionary, persona_id: String) -> String:
	if not state.has("is_at"):
		return "dorm"
	var is_at = state["is_at"]
	if is_at.has(persona_id):
		var val = is_at[persona_id]
		if val is String:
			return val
	return "dorm"

static func set_location(state: Dictionary, persona_id: String, location: String) -> void:
	if not state.has("is_at"):
		state["is_at"] = {}
	var is_at = state["is_at"]
	is_at[persona_id] = location
	state["is_at"] = is_at

static func get_burnout(state: Dictionary, persona_id: String) -> int:
	if not state.has("burnout"):
		return 0
	var burnout = state["burnout"]
	if burnout.has(persona_id):
		var val = burnout[persona_id]
		if val is int:
			return val
	return 0

static func set_burnout(state: Dictionary, persona_id: String, burnout_level: int) -> void:
	if not state.has("burnout"):
		state["burnout"] = {}
	var burnout = state["burnout"]
	burnout[persona_id] = burnout_level
	state["burnout"] = burnout

static func likes_activity(state: Dictionary, persona_id: String, activity: String) -> bool:
	if not state.has("preferences"):
		return false
	var preferences = state["preferences"]
	if not preferences.has(persona_id):
		return false
	var persona_prefs = preferences[persona_id]
	if not persona_prefs.has("likes"):
		return false
	var likes = persona_prefs["likes"]
	return activity in likes

static func dislikes_activity(state: Dictionary, persona_id: String, activity: String) -> bool:
	if not state.has("preferences"):
		return false
	var preferences = state["preferences"]
	if not preferences.has(persona_id):
		return false
	var persona_prefs = preferences[persona_id]
	if not persona_prefs.has("dislikes"):
		return false
	var dislikes = persona_prefs["dislikes"]
	return activity in dislikes

static func get_coordination(state: Dictionary, persona_id: String) -> Dictionary:
	if not state.has("coordination"):
		return {}
	var coordination = state["coordination"]
	if coordination.has(persona_id):
		var coord_val = coordination[persona_id]
		if coord_val is Dictionary:
			return coord_val
	return {}

static func set_coordination(state: Dictionary, persona_id: String, coordination: Dictionary) -> void:
	if not state.has("coordination"):
		state["coordination"] = {}
	var coord_dict = state["coordination"]
	coord_dict[persona_id] = coordination
	state["coordination"] = coord_dict

# Actions - Study
static func action_attend_lecture(state: Dictionary, persona_id: Variant, subject: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_points = get_study_points(new_state, persona_id)
	set_study_points(new_state, persona_id, current_points + 5)
	return new_state

static func action_complete_homework(state: Dictionary, persona_id: Variant, subject: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_points = get_study_points(new_state, persona_id)
	set_study_points(new_state, persona_id, current_points + 3)
	return new_state

static func action_study_library(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	set_location(new_state, persona_id, "library")
	var current_points = get_study_points(new_state, persona_id)
	set_study_points(new_state, persona_id, current_points + 4)

	# Consume coordination if it matches a study session at library
	var coordination = get_coordination(new_state, persona_id)
	if not coordination.is_empty() and coordination.has("action") and str(coordination["action"]) == "study_session" and coordination.has("location") and str(coordination["location"]) == "library":
		coordination["used"] = true
		set_coordination(new_state, persona_id, coordination)

	return new_state

# Actions - Socialization
static func action_eat_mess_hall(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	set_location(new_state, persona_id, "mess_hall")
	set_location(new_state, companion_id, "mess_hall")
	var current_points = get_relationship_points(new_state, persona_id, companion_id)
	set_relationship_points(new_state, persona_id, companion_id, current_points + 2)
	return new_state

static func action_coffee_together(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_points = get_relationship_points(new_state, persona_id, companion_id)
	set_relationship_points(new_state, persona_id, companion_id, current_points + 3)
	return new_state

static func action_watch_movie(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	set_location(new_state, persona_id, "cinema")
	set_location(new_state, companion_id, "cinema")
	var current_points = get_relationship_points(new_state, persona_id, companion_id)
	set_relationship_points(new_state, persona_id, companion_id, current_points + 5)
	return new_state

static func action_pool_hangout(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	set_location(new_state, persona_id, "pool")
	set_location(new_state, companion_id, "pool")
	var current_points = get_relationship_points(new_state, persona_id, companion_id)
	set_relationship_points(new_state, persona_id, companion_id, current_points + 6)
	return new_state

static func action_park_picnic(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	set_location(new_state, persona_id, "park")
	set_location(new_state, companion_id, "park")
	var current_points = get_relationship_points(new_state, persona_id, companion_id)
	set_relationship_points(new_state, persona_id, companion_id, current_points + 7)
	return new_state

static func action_beach_trip(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	set_location(new_state, persona_id, "beach")
	set_location(new_state, companion_id, "beach")
	var current_points = get_relationship_points(new_state, persona_id, companion_id)
	set_relationship_points(new_state, persona_id, companion_id, current_points + 10)
	return new_state

# Actions - Rest/Recreation
static func action_read_book(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_burnout = get_burnout(new_state, persona_id)
	set_burnout(new_state, persona_id, max(0, current_burnout - 2))
	return new_state

static func action_club_activity(state: Dictionary, persona_id: Variant, club: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_burnout = get_burnout(new_state, persona_id)
	set_burnout(new_state, persona_id, max(0, current_burnout - 3))
	return new_state

# Actions - AI-specific
static func action_optimize_schedule(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_burnout = get_burnout(new_state, persona_id)
	set_burnout(new_state, persona_id, max(0, current_burnout - 10))
	return new_state

static func action_predict_outcome(state: Dictionary, persona_id: Variant, activity: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_points = get_study_points(new_state, persona_id)
	set_study_points(new_state, persona_id, current_points + 2)
	return new_state

# Task methods - Earn study points
static func task_earn_study_points_method_done(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	var current_points = get_study_points(state, persona_id)
	if current_points >= target_points:
		return []  # Done, no subtasks needed
	return null  # Not done, try other methods

static func task_earn_study_points_method_lecture(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	var current_points = get_study_points(state, persona_id)
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
	var current_points = get_study_points(state, persona_id)
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
			var coordination = get_coordination(state, persona_id)
			if coordination.is_empty() or not coordination.has("used") or not bool(coordination["used"]):
				has_unused_study_session = true

	if not has_unused_study_session:
		var coordination = get_coordination(state, persona_id)
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
	var current_points = get_study_points(state, persona_id)
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
	var current_points = get_study_points(state, persona_id)
	if current_points >= target_points:
		return null  # Already have enough

	# Check for study session coordination
	# Since coordinations can be overwritten when merged, we check temporal_puzzle for the morning study time
	var coord_time = 0
	var is_study_session = false
	var coordination = get_coordination(state, persona_id)

	# First, check temporal_puzzle for morning_study_time (most reliable since it's not overwritten)
	if state.has("temporal_puzzle"):
		var puzzle = state["temporal_puzzle"]
		if puzzle.has("morning_study_time"):
			var morning_time_var = puzzle["morning_study_time"]
			if morning_time_var is int:
				coord_time = morning_time_var
				is_study_session = true  # If morning_study_time exists, there's a study session

	# Fallback: check coordination dictionary (may have been overwritten, but worth checking)
	if not is_study_session and not coordination.is_empty() and coordination.has("time"):
		var coord_time_var = coordination.get("time", null)
		if coord_time_var is int:
			var temp_time = coord_time_var
			var coord_location = coordination.get("location", "")
			var coord_action = coordination.get("action", "")

			# Check if it's explicitly a study session
			if coord_action == "study_session" and coord_location == "library":
				coord_time = temp_time
				is_study_session = true
			elif coord_location == "library" and state.has("temporal_puzzle"):
				# Check if time is before movie (homework_deadline)
				var puzzle = state["temporal_puzzle"]
				if puzzle.has("homework_deadline"):
					var deadline_var = puzzle["homework_deadline"]
					if deadline_var is int:
						var movie_time = deadline_var
						if temp_time < movie_time:
							coord_time = temp_time
							is_study_session = true

	if not is_study_session or coord_time == 0:
		return null  # No study session coordination found, try other methods

	# Check if coordination has already been used (by checking if we're already at the library
	# and have gained points, or if the coordination has a "used" flag)
	if not coordination.is_empty() and coordination.has("used") and bool(coordination["used"]):
		return null  # Coordination already used, try other methods

	# Create action with temporal metadata attached
	var action = ["action_study_library", persona_id]

	# Wrap action with temporal constraints based on coordination time
	# Study session duration: 1 hour = 3600000000 microseconds
	var study_duration = 3600000000
	var coord_end_time = coord_time + study_duration

	var temporal_constraints = {}
	temporal_constraints["start_time"] = coord_time
	temporal_constraints["end_time"] = coord_end_time
	temporal_constraints["duration"] = study_duration

	# Wrap action with temporal metadata (same format as attach_metadata returns)
	var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}

	var subtasks = []
	subtasks.append(action_with_metadata)

	# After executing the action, we'll have current_points + 4
	# If that's still not enough, recursively refine the task
	var points_after_action = current_points + 4
	if points_after_action < target_points:
		# Still need more points, so recursively refine the task
		var recursive_task = ["task_earn_study_points", persona_id, target_points]
		subtasks.append(recursive_task)

	return subtasks

# Task methods - Socialize
static func task_socialize_method_easy(state: Dictionary, persona_id: Variant, companion_id: Variant, activity_level: Variant) -> Variant:
	if activity_level > 1:
		return null  # Not easy activity

	# Check if this is for Kira and we need an evening activity (after movie)
	# If companion is Kira and we have a movie scheduled, prefer coffee/reading over mess hall
	var is_kira = (str(companion_id) == "kira")
	var movie_scheduled = false
	var movie_end_time = 0

	# Check coordination for movie
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
						# Movie duration = 2 hours = 7200000000 microseconds
						movie_end_time = movie_time + 7200000000

	# Also check temporal_puzzle for movie information (if coordination not found)
	if not movie_scheduled and state.has("temporal_puzzle"):
		var puzzle = state["temporal_puzzle"]
		# If homework_deadline exists, it's likely the movie time
		# For Kira, we want an evening activity after the movie
		if puzzle.has("homework_deadline") and is_kira:
			var deadline_var = puzzle["homework_deadline"]
			if deadline_var is int:
				var movie_time = deadline_var
				movie_scheduled = true
				# Movie duration = 2 hours = 7200000000 microseconds
				movie_end_time = movie_time + 7200000000

	var subtasks = []
	var action = []

	# For Kira after movie, prefer coffee or reading over mess hall
	if is_kira and movie_scheduled and movie_end_time > 0:
		# Use coffee_together for evening activity
		action = ["action_coffee_together", persona_id, companion_id]

		# Attach temporal metadata: coffee duration = 1 hour = 3600000000 microseconds
		# Set start_time to after movie ends
		var temporal_constraints = {}
		temporal_constraints["start_time"] = movie_end_time
		temporal_constraints["duration"] = 3600000000  # 1 hour
		temporal_constraints["end_time"] = movie_end_time + 3600000000
		var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
		subtasks.append(action_with_metadata)
	else:
		# Check coordination for lunch
		var coordination = get_coordination(state, persona_id)
		if not coordination.is_empty() and coordination.has("action") and str(coordination["action"]) == "lunch" and str(coordination.get("companion", "")) == str(companion_id):
			# If coordinated lunch with companion, use it
			action = ["action_eat_mess_hall", persona_id, companion_id]
			var temporal_constraints = {}
			if coordination.has("time"):
				var lunch_time = coordination["time"]
				if lunch_time is int:
					temporal_constraints["start_time"] = lunch_time
					temporal_constraints["end_time"] = lunch_time + 1800000000  # 30 minutes
			var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
			subtasks.append(action_with_metadata)
		else:
			# Default: use mess hall
			action = ["action_eat_mess_hall", persona_id, companion_id]

			# Attach temporal metadata: mess hall meal duration = 30 minutes = 1800000000 microseconds
			var temporal_constraints = {"duration": 1800000000}  # 30 minutes
			var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
			subtasks.append(action_with_metadata)

	return subtasks

static func task_socialize_method_moderate(state: Dictionary, persona_id: Variant, companion_id: Variant, activity_level: Variant) -> Variant:
	if activity_level != 2:
		return null  # Not moderate activity

	var subtasks = []
	var action = ["action_watch_movie", persona_id, companion_id]

	# Attach temporal metadata: movie duration = 2 hours = 7200000000 microseconds
	var temporal_constraints = {"duration": 7200000000}  # 2 hours
	var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
	subtasks.append(action_with_metadata)
	return subtasks

static func task_socialize_method_challenging(state: Dictionary, persona_id: Variant, companion_id: Variant, activity_level: Variant) -> Variant:
	if activity_level < 3:
		return null  # Not challenging activity

	var subtasks = []
	var action = ["action_park_picnic", persona_id, companion_id]

	# Attach temporal metadata: park picnic duration = 3 hours = 10800000000 microseconds
	var temporal_constraints = {"duration": 10800000000}  # 3 hours
	var action_with_metadata = {"item": action, "temporal_constraints": temporal_constraints}
	subtasks.append(action_with_metadata)
	return subtasks

# Task methods - Manage week
static func task_manage_week_method_balance(state: Dictionary, persona_id: Variant) -> Variant:
	# Balance academics and relationships
	var subtasks = []
	var task1 = ["task_earn_study_points", persona_id, 10]  # Target 10 study points
	subtasks.append(task1)
	var task2 = ["task_socialize", persona_id, "maya", 2]  # Moderate activity
	subtasks.append(task2)
	return subtasks

static func task_manage_week_method_academics(state: Dictionary, persona_id: Variant) -> Variant:
	# Focus on academics
	var subtasks = []
	subtasks.append(["task_earn_study_points", persona_id, 20])
	return subtasks

static func task_manage_week_method_relationships(state: Dictionary, persona_id: Variant) -> Variant:
	# Focus on relationships
	var subtasks = []
	var task1 = ["task_socialize", persona_id, "maya", 3]  # Challenging activity
	subtasks.append(task1)
	var task2 = ["task_socialize", persona_id, "aria", 2]  # Moderate activity
	subtasks.append(task2)
	return subtasks

# Unigoal methods
static func unigoal_achieve_study_goal(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	var current_points = get_study_points(state, persona_id)
	if current_points >= target_points:
		return []  # Goal achieved
	return [["task_earn_study_points", persona_id, target_points]]

static func unigoal_achieve_relationship_goal(state: Dictionary, subject: Variant, target: Variant) -> Variant:
	# Handle flat predicate format: ["relationship_points", "relationship_points_persona_companion", target]
	# For flat predicates, we use the subject field to pass the full flat predicate
	# The planner calls us with (state, subject, value) where:
	# - subject is "relationship_points_yuki_maya" (the full flat predicate)
	# - value is the target (e.g., 3)

	if not (subject is String):
		return null  # Fail early - return null so planner can try other methods

	var predicate_str: String = subject
	if not predicate_str.begins_with("relationship_points_"):
		return null  # Invalid format - fail early

	# Parse persona_id and companion_id from predicate
	# Format: "relationship_points_persona_companion"
	var parts = predicate_str.split("_")
	if parts.size() < 4:
		return null  # Invalid format - fail early

	var persona_id = parts[2]  # e.g., "yuki" from "relationship_points_yuki_maya"
	var companion_id = parts[3]  # e.g., "maya" from "relationship_points_yuki_maya"
	# Handle multi-part companion names (e.g., "relationship_points_yuki_maya_smith" -> "maya_smith")
	if parts.size() > 4:
		companion_id = "_".join(parts.slice(3))

	if not (target is int):
		return null  # Fail early - target must be an int

	var current = get_relationship_points(state, persona_id, companion_id)
	if current >= target:
		return []  # Goal achieved - return empty array for success with no work

	# Return task to achieve relationship points
	var task = ["task_socialize", persona_id, companion_id]
	var points_needed = target - current
	var activity_level = 1 if points_needed <= 2 else (2 if points_needed <= 4 else 3)
	task.append(activity_level)
	return [task]

# Multigoal methods
static func multigoal_balance_life(state: Dictionary, multigoal: Array) -> Array:
	# Handles "at least" goals for numeric predicates (study_points, relationship_points)
	var goals = []

	# Check study points goal
	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 2 and str(goal[0]) == "study_points":
			var persona_id = str(goal[1])
			var target = goal[2] if goal.size() >= 3 else 10
			var current = get_study_points(state, persona_id)
			# "At least" goal: check if current < target (not ==)
			if current < target:
				# Return task method instead of unigoal (task methods handle "at least" logic)
				var task = ["task_earn_study_points", persona_id, target]
				goals.append(task)

	# Check relationship points goal (flat predicate format)
	# Format: ["relationship_points", "relationship_points_persona_companion", target]
	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 3:
			var predicate = str(goal[0])
			if predicate == "relationship_points":
				# 3-element format: ["relationship_points", "relationship_points_persona_companion", target]
				var flat_predicate = str(goal[1])  # e.g., "relationship_points_yuki_maya"
				var target = goal[2]
				if not (target is int):
					continue  # Invalid format

				if not flat_predicate.begins_with("relationship_points_"):
					continue  # Invalid format

				# Parse persona_id and companion_id from flat predicate
				var parts = flat_predicate.split("_")
				if parts.size() < 4:
					continue  # Invalid format

				var persona_id = parts[2]  # e.g., "yuki" from "relationship_points_yuki_maya"
				var companion_id = parts[3]  # e.g., "maya" from "relationship_points_yuki_maya"
				# Handle multi-part companion names
				if parts.size() > 4:
					companion_id = "_".join(parts.slice(3))

				var current = get_relationship_points(state, persona_id, companion_id)
				# IPyHOP approach: return goal with exact current value if achieved
				# Otherwise, return as unigoal for unigoal method to handle
				if current >= target:
					# Goal already achieved - return goal with exact achieved value
					var achieved_goal = ["relationship_points", flat_predicate, current]
					goals.append(achieved_goal)
				elif current < target:
					# Return as unigoal - let unigoal method handle task creation
					var unigoal = ["relationship_points", flat_predicate, target]
					goals.append(unigoal)

	# Check burnout goal
	# Note: Burnout is an "at most" goal (burnout <= target), but we use unigoal here
	# because there's no task method for managing burnout. This has the same limitation
	# as "at least" goals: if burnout is reduced below target, exact equality fails.
	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 2 and str(goal[0]) == "burnout":
			var persona_id = str(goal[1])
			var target = goal[2] if goal.size() >= 3 else 50  # Max burnout
			var current = get_burnout(state, persona_id)
			if current > target:
				var unigoal = ["burnout", persona_id, target]
				goals.append(unigoal)

	return goals

static func multigoal_solve_temporal_puzzle(state: Dictionary, multigoal: Array) -> Array:
	var goals = []
	var persona_id = "yuki"  # Assuming Yuki is the main persona for this puzzle

	# Check study points goal
	var study_target = 0
	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 2 and str(goal[0]) == "study_points":
			study_target = goal[2]
			break

	var current_study = get_study_points(state, persona_id)
	# For "at least" goals: IPyHOP approach - return goal with exact current value if achieved
	# This allows the planner to verify the goal individually and mark it as achieved
	# If not achieved, return tasks to achieve it
	if current_study >= study_target:
		# Goal already achieved - return goal with exact achieved value
		# Planner will add this as a unigoal node and verify it (which will pass)
		var achieved_goal = ["study_points", persona_id, current_study]
		goals.append(achieved_goal)
	elif current_study < study_target:
		# Prioritize coordinated study if available and not yet done
		var coordination = get_coordination(state, persona_id)
		if coordination.has("action") and str(coordination["action"]) == "study_session" and str(coordination["location"]) == "library":
			# If coordinated study is available, try to use it
			var task = ["task_earn_study_points", persona_id, study_target]  # Pass the overall target
			goals.append(task)
		else:
			# Otherwise, try other study methods
			var task = ["task_earn_study_points", persona_id, study_target]
			goals.append(task)

	# Check relationship goals (flat predicate format)
	# Format: ["relationship_points", "relationship_points_persona_companion", target]
	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 3:
			var predicate = str(goal[0])
			if predicate == "relationship_points":
				# 3-element format: ["relationship_points", "relationship_points_persona_companion", target]
				var flat_predicate = str(goal[1])  # e.g., "relationship_points_yuki_maya"
				var target = goal[2]
				if not (target is int):
					continue  # Invalid format

				if not flat_predicate.begins_with("relationship_points_"):
					continue  # Invalid format

				# Parse persona_id and companion_id from flat predicate
				var parts = flat_predicate.split("_")
				if parts.size() < 4:
					continue  # Invalid format

				var goal_persona_id = parts[2]  # e.g., "yuki" from "relationship_points_yuki_maya"
				var companion_id = parts[3]  # e.g., "maya" from "relationship_points_yuki_maya"
				# Handle multi-part companion names
				if parts.size() > 4:
					companion_id = "_".join(parts.slice(3))

				# Use the persona_id from the puzzle context (usually "yuki")
				var current = get_relationship_points(state, persona_id, companion_id)
				# IPyHOP approach: return goal with exact current value if achieved
				# Otherwise, return as unigoal for unigoal method to handle
				if current >= target:
					# Goal already achieved - return goal with exact achieved value
					var achieved_goal = ["relationship_points", flat_predicate, current]
					goals.append(achieved_goal)
				elif current < target:
					# Check for coordination
					var coordination = get_coordination(state, persona_id)
					if coordination.has("action") and str(coordination["action"]) == "lunch" and companion_id == "aria":
						var task = ["task_socialize", persona_id, companion_id, 1]
						goals.append(task)
					elif coordination.has("action") and str(coordination["action"]) == "movie" and companion_id == "maya":
						var task = ["task_socialize", persona_id, companion_id, 2]
						goals.append(task)
					else:
						# Return as unigoal - let unigoal method handle task creation
						var unigoal = ["relationship_points", flat_predicate, target]
						goals.append(unigoal)

	return goals
