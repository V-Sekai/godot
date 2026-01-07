# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Magical Girls College Domain - Actions
# All action functions that modify state

const Helpers = preload("helpers.gd")

# Actions - Study
static func action_attend_lecture(state: Dictionary, persona_id: Variant, subject: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_points = Helpers.get_study_points(new_state, persona_id)
	Helpers.set_study_points(new_state, persona_id, current_points + 5)
	return new_state

static func action_complete_homework(state: Dictionary, persona_id: Variant, subject: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_points = Helpers.get_study_points(new_state, persona_id)
	Helpers.set_study_points(new_state, persona_id, current_points + 3)
	return new_state

static func action_study_library(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	Helpers.set_location(new_state, persona_id, "library")
	var current_points = Helpers.get_study_points(new_state, persona_id)
	Helpers.set_study_points(new_state, persona_id, current_points + 4)

	# Consume coordination if it matches a study session at library
	var coordination = Helpers.get_coordination(new_state, persona_id)
	if not coordination.is_empty() and coordination.has("action") and str(coordination["action"]) == "study_session" and coordination.has("location") and str(coordination["location"]) == "library":
		coordination["used"] = true
		Helpers.set_coordination(new_state, persona_id, coordination)

	return new_state

# Actions - Socialization (with companions)
static func action_eat_mess_hall_with_friend(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	Helpers.set_location(new_state, persona_id, "mess_hall")
	Helpers.set_location(new_state, companion_id, "mess_hall")
	var current_points = Helpers.get_relationship_points(new_state, persona_id, companion_id)
	Helpers.set_relationship_points(new_state, persona_id, companion_id, current_points + 2)
	# Also satisfies hunger
	var current_hunger = Helpers.get_need(new_state, persona_id, "hunger")
	Helpers.set_need(new_state, persona_id, "hunger", current_hunger + 40)
	return new_state

static func action_coffee_together(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_points = Helpers.get_relationship_points(new_state, persona_id, companion_id)
	Helpers.set_relationship_points(new_state, persona_id, companion_id, current_points + 3)
	return new_state

static func action_watch_movie(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	Helpers.set_location(new_state, persona_id, "cinema")
	Helpers.set_location(new_state, companion_id, "cinema")
	var current_points = Helpers.get_relationship_points(new_state, persona_id, companion_id)
	Helpers.set_relationship_points(new_state, persona_id, companion_id, current_points + 5)
	return new_state

static func action_pool_hangout(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	Helpers.set_location(new_state, persona_id, "pool")
	Helpers.set_location(new_state, companion_id, "pool")
	var current_points = Helpers.get_relationship_points(new_state, persona_id, companion_id)
	Helpers.set_relationship_points(new_state, persona_id, companion_id, current_points + 6)
	return new_state

static func action_park_picnic(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	Helpers.set_location(new_state, persona_id, "park")
	Helpers.set_location(new_state, companion_id, "park")
	var current_points = Helpers.get_relationship_points(new_state, persona_id, companion_id)
	Helpers.set_relationship_points(new_state, persona_id, companion_id, current_points + 7)
	return new_state

static func action_beach_trip(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	Helpers.set_location(new_state, persona_id, "beach")
	Helpers.set_location(new_state, companion_id, "beach")
	var current_points = Helpers.get_relationship_points(new_state, persona_id, companion_id)
	Helpers.set_relationship_points(new_state, persona_id, companion_id, current_points + 10)
	return new_state

# Actions - Rest/Recreation
static func action_read_book(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_burnout = Helpers.get_burnout(new_state, persona_id)
	Helpers.set_burnout(new_state, persona_id, max(0, current_burnout - 2))
	return new_state

static func action_club_activity(state: Dictionary, persona_id: Variant, club: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_burnout = Helpers.get_burnout(new_state, persona_id)
	Helpers.set_burnout(new_state, persona_id, max(0, current_burnout - 3))
	return new_state

# Actions - AI-specific
static func action_optimize_schedule(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_burnout = Helpers.get_burnout(new_state, persona_id)
	Helpers.set_burnout(new_state, persona_id, max(0, current_burnout - 10))
	return new_state

static func action_predict_outcome(state: Dictionary, persona_id: Variant, activity: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_points = Helpers.get_study_points(new_state, persona_id)
	Helpers.set_study_points(new_state, persona_id, current_points + 2)
	return new_state

# Actions - The Sims-style needs satisfaction
# Hunger satisfaction actions
static func action_eat_mess_hall(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	Helpers.set_location(new_state, persona_id, "mess_hall")
	var current_hunger = Helpers.get_need(new_state, persona_id, "hunger")
	Helpers.set_need(new_state, persona_id, "hunger", current_hunger + 40)  # Mess hall is filling
	return new_state

static func action_eat_restaurant(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_money = Helpers.get_money(new_state, persona_id)
	if current_money < 20:
		return null  # Not enough money
	Helpers.set_money(new_state, persona_id, current_money - 20)
	var current_hunger = Helpers.get_need(new_state, persona_id, "hunger")
	Helpers.set_need(new_state, persona_id, "hunger", current_hunger + 60)  # Restaurant is very filling
	return new_state

static func action_eat_snack(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_money = Helpers.get_money(new_state, persona_id)
	if current_money < 5:
		return null  # Not enough money
	Helpers.set_money(new_state, persona_id, current_money - 5)
	var current_hunger = Helpers.get_need(new_state, persona_id, "hunger")
	Helpers.set_need(new_state, persona_id, "hunger", current_hunger + 20)  # Snack is light
	return new_state

static func action_cook_meal(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_location = Helpers.get_location(new_state, persona_id)
	if current_location != "dorm":
		return null  # Can only cook in dorm
	var current_hunger = Helpers.get_need(new_state, persona_id, "hunger")
	Helpers.set_need(new_state, persona_id, "hunger", current_hunger + 50)  # Home-cooked is good
	return new_state

# Sleep/Energy satisfaction actions
static func action_sleep_dorm(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_location = Helpers.get_location(new_state, persona_id)
	if current_location != "dorm":
		return null  # Must be in dorm to sleep
	var current_energy = Helpers.get_need(new_state, persona_id, "energy")
	Helpers.set_need(new_state, persona_id, "energy", 100)  # Full rest
	return new_state

static func action_nap_library(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_location = Helpers.get_location(new_state, persona_id)
	if current_location != "library":
		return null  # Must be in library
	var current_energy = Helpers.get_need(new_state, persona_id, "energy")
	Helpers.set_need(new_state, persona_id, "energy", min(100, current_energy + 30))  # Light nap
	return new_state

static func action_energy_drink(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_money = Helpers.get_money(new_state, persona_id)
	if current_money < 3:
		return null  # Not enough money
	Helpers.set_money(new_state, persona_id, current_money - 3)
	var current_energy = Helpers.get_need(new_state, persona_id, "energy")
	Helpers.set_need(new_state, persona_id, "energy", min(100, current_energy + 20))  # Temporary boost
	return new_state

# Social satisfaction actions
static func action_talk_friend(state: Dictionary, persona_id: Variant, friend_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_social = Helpers.get_need(new_state, persona_id, "social")
	Helpers.set_need(new_state, persona_id, "social", min(100, current_social + 15))
	# Also improves relationship
	var current_rel = Helpers.get_relationship_points(new_state, persona_id, friend_id)
	Helpers.set_relationship_points(new_state, persona_id, friend_id, current_rel + 1)
	return new_state

static func action_join_club(state: Dictionary, persona_id: Variant, club: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_social = Helpers.get_need(new_state, persona_id, "social")
	Helpers.set_need(new_state, persona_id, "social", min(100, current_social + 25))  # Club is very social
	return new_state

static func action_phone_call(state: Dictionary, persona_id: Variant, friend_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_social = Helpers.get_need(new_state, persona_id, "social")
	Helpers.set_need(new_state, persona_id, "social", min(100, current_social + 10))  # Phone call is less social
	return new_state

# Fun satisfaction actions
static func action_play_games(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_location = Helpers.get_location(new_state, persona_id)
	if current_location != "dorm":
		return null  # Games are in dorm
	var current_fun = Helpers.get_need(new_state, persona_id, "fun")
	Helpers.set_need(new_state, persona_id, "fun", min(100, current_fun + 30))
	return new_state

static func action_watch_streaming(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_location = Helpers.get_location(new_state, persona_id)
	if current_location != "dorm":
		return null  # Streaming is in dorm
	var current_fun = Helpers.get_need(new_state, persona_id, "fun")
	Helpers.set_need(new_state, persona_id, "fun", min(100, current_fun + 25))
	return new_state

static func action_go_cinema(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_money = Helpers.get_money(new_state, persona_id)
	if current_money < 15:
		return null  # Not enough money
	Helpers.set_money(new_state, persona_id, current_money - 15)
	Helpers.set_location(new_state, persona_id, "cinema")
	var current_fun = Helpers.get_need(new_state, persona_id, "fun")
	Helpers.set_need(new_state, persona_id, "fun", min(100, current_fun + 40))  # Cinema is very fun
	return new_state

# Hygiene satisfaction actions
static func action_shower(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_location = Helpers.get_location(new_state, persona_id)
	if current_location != "dorm":
		return null  # Must be in dorm to shower
	var current_hygiene = Helpers.get_need(new_state, persona_id, "hygiene")
	Helpers.set_need(new_state, persona_id, "hygiene", 100)  # Shower is full hygiene
	return new_state

static func action_wash_hands(state: Dictionary, persona_id: Variant) -> Variant:
	var new_state = state.duplicate(true)
	var current_hygiene = Helpers.get_need(new_state, persona_id, "hygiene")
	Helpers.set_need(new_state, persona_id, "hygiene", min(100, current_hygiene + 20))  # Partial hygiene
	return new_state

# Simple move action (used by move task)
static func action_move_to(state: Dictionary, persona_id: Variant, location: Variant) -> Variant:
	var new_state = state.duplicate(true)
	Helpers.set_location(new_state, persona_id, str(location))
	return new_state

