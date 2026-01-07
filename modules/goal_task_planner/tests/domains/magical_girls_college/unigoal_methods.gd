# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Magical Girls College Domain - Unigoal Methods
# Methods that handle single goal achievement

const Helpers = preload("helpers.gd")

static func unigoal_achieve_study_goal(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	var current_points = Helpers.get_study_points(state, persona_id)
	if current_points >= target_points:
		return []  # Goal achieved
	return [["task_earn_study_points", persona_id, target_points]]

static func unigoal_achieve_relationship_goal(state: Dictionary, subject: Variant, target: Variant) -> Variant:
	if not (subject is String):
		return null

	var predicate_str: String = subject
	if not predicate_str.begins_with("relationship_points_"):
		return null

	var parts = predicate_str.split("_")
	if parts.size() < 4:
		return null

	var persona_id = parts[2]
	var companion_id = parts[3]
	if parts.size() > 4:
		companion_id = "_".join(parts.slice(3))

	if not (target is int):
		return null

	var current = Helpers.get_relationship_points(state, persona_id, companion_id)
	if current >= target:
		return []

	var task = ["task_socialize", persona_id, companion_id]
	var points_needed = target - current
	var activity_level = 1 if points_needed <= 2 else (2 if points_needed <= 4 else 3)
	task.append(activity_level)
	return [task]
