# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Magical Girls College Domain - Helper Functions
# Shared helper functions for state manipulation

# Generic helpers
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

# Study points helpers
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

# Relationship points helpers
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

# Location helpers
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

# Burnout helpers
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

# Preference helpers
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

# Coordination helpers
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

# The Sims-style needs helpers
static func get_need(state: Dictionary, persona_id: String, need_name: String, default_value: int = 50) -> int:
	var needs_key = "needs"
	if not state.has(needs_key):
		return default_value
	var needs = state[needs_key]
	if not needs.has(persona_id):
		return default_value
	var persona_needs = needs[persona_id]
	if not persona_needs.has(need_name):
		return default_value
	var val = persona_needs[need_name]
	if val is int:
		return val
	return default_value

static func set_need(state: Dictionary, persona_id: String, need_name: String, value: int) -> void:
	if not state.has("needs"):
		state["needs"] = {}
	var needs = state["needs"]
	if not needs.has(persona_id):
		needs[persona_id] = {}
	var persona_needs = needs[persona_id]
	persona_needs[need_name] = clamp(value, 0, 100)  # Needs are 0-100
	needs[persona_id] = persona_needs
	state["needs"] = needs

# Money helpers
static func get_money(state: Dictionary, persona_id: String) -> int:
	if not state.has("money"):
		return 0
	var money = state["money"]
	if money.has(persona_id):
		var val = money[persona_id]
		if val is int:
			return val
	return 0

static func set_money(state: Dictionary, persona_id: String, amount: int) -> void:
	if not state.has("money"):
		state["money"] = {}
	var money = state["money"]
	money[persona_id] = amount
	state["money"] = money
