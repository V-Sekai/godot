# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Magical Girls College Domain - Multigoal Methods
# Methods that handle multiple goals simultaneously

const Helpers = preload("helpers.gd")

static func multigoal_balance_life(state: Dictionary, multigoal: Array) -> Array:
	var goals = []

	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 2 and str(goal[0]) == "study_points":
			var persona_id = str(goal[1])
			var target = goal[2] if goal.size() >= 3 else 10
			var current = Helpers.get_study_points(state, persona_id)
			if current < target:
				var task = ["task_earn_study_points", persona_id, target]
				goals.append(task)

	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 3:
			var predicate = str(goal[0])
			if predicate == "relationship_points":
				var flat_predicate = str(goal[1])
				var target = goal[2]
				if not (target is int):
					continue
				if not flat_predicate.begins_with("relationship_points_"):
					continue
				var parts = flat_predicate.split("_")
				if parts.size() < 4:
					continue
				var persona_id = parts[2]
				var companion_id = parts[3]
				if parts.size() > 4:
					companion_id = "_".join(parts.slice(3))
				var current = Helpers.get_relationship_points(state, persona_id, companion_id)
				if current >= target:
					var achieved_goal = ["relationship_points", flat_predicate, current]
					goals.append(achieved_goal)
				elif current < target:
					var unigoal = ["relationship_points", flat_predicate, target]
					goals.append(unigoal)

	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 2 and str(goal[0]) == "burnout":
			var persona_id = str(goal[1])
			var target = goal[2] if goal.size() >= 3 else 50
			var current = Helpers.get_burnout(state, persona_id)
			if current > target:
				var unigoal = ["burnout", persona_id, target]
				goals.append(unigoal)

	return goals

static func multigoal_solve_temporal_puzzle(state: Dictionary, multigoal: Array) -> Array:
	var goals = []
	var persona_id = "yuki"

	var study_target = 0
	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 2 and str(goal[0]) == "study_points":
			study_target = goal[2]
			break

	var current_study = Helpers.get_study_points(state, persona_id)
	if current_study >= study_target:
		var achieved_goal = ["study_points", persona_id, current_study]
		goals.append(achieved_goal)
	elif current_study < study_target:
		var coordination = Helpers.get_coordination(state, persona_id)
		if coordination.has("action") and str(coordination["action"]) == "study_session" and str(coordination["location"]) == "library":
			var task = ["task_earn_study_points", persona_id, study_target]
			goals.append(task)
		else:
			var task = ["task_earn_study_points", persona_id, study_target]
			goals.append(task)

	for i in range(multigoal.size()):
		var goal = multigoal[i]
		if goal is Array and goal.size() >= 3:
			var predicate = str(goal[0])
			if predicate == "relationship_points":
				var flat_predicate = str(goal[1])
				var target = goal[2]
				if not (target is int):
					continue
				if not flat_predicate.begins_with("relationship_points_"):
					continue
				var parts = flat_predicate.split("_")
				if parts.size() < 4:
					continue
				var goal_persona_id = parts[2]
				var companion_id = parts[3]
				if parts.size() > 4:
					companion_id = "_".join(parts.slice(3))
				var current = Helpers.get_relationship_points(state, persona_id, companion_id)
				if current >= target:
					var achieved_goal = ["relationship_points", flat_predicate, current]
					goals.append(achieved_goal)
				elif current < target:
					var coordination = Helpers.get_coordination(state, persona_id)
					if coordination.has("action") and str(coordination["action"]) == "lunch" and companion_id == "aria":
						var task = ["task_socialize", persona_id, companion_id, 1]
						goals.append(task)
					elif coordination.has("action") and str(coordination["action"]) == "movie" and companion_id == "maya":
						var task = ["task_socialize", persona_id, companion_id, 2]
						goals.append(task)
					else:
						var unigoal = ["relationship_points", flat_predicate, target]
						goals.append(unigoal)

	return goals
