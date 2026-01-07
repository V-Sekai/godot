# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Magical Girls College Domain - Main Domain Class
# This file provides a unified interface to all domain components
# The actual implementations are split across multiple files for better organization

class_name MagicalGirlsCollegeDomain

# Import all domain components
const Helpers = preload("helpers.gd")
const Actions = preload("actions.gd")
const TaskMethods = preload("task_methods.gd")
const UnigoalMethods = preload("unigoal_methods.gd")
const MultigoalMethods = preload("multigoal_methods.gd")

# Re-export helper functions for convenience
static func get_study_points(state: Dictionary, persona_id: String) -> int:
	return Helpers.get_study_points(state, persona_id)

static func set_study_points(state: Dictionary, persona_id: String, points: int) -> void:
	Helpers.set_study_points(state, persona_id, points)

static func get_relationship_points(state: Dictionary, persona_id: String, companion_id: String) -> int:
	return Helpers.get_relationship_points(state, persona_id, companion_id)

static func set_relationship_points(state: Dictionary, persona_id: String, companion_id: String, points: int) -> void:
	Helpers.set_relationship_points(state, persona_id, companion_id, points)

static func get_location(state: Dictionary, persona_id: String) -> String:
	return Helpers.get_location(state, persona_id)

static func set_location(state: Dictionary, persona_id: String, location: String) -> void:
	Helpers.set_location(state, persona_id, location)

static func get_burnout(state: Dictionary, persona_id: String) -> int:
	return Helpers.get_burnout(state, persona_id)

static func set_burnout(state: Dictionary, persona_id: String, burnout_level: int) -> void:
	Helpers.set_burnout(state, persona_id, burnout_level)

static func likes_activity(state: Dictionary, persona_id: String, activity: String) -> bool:
	return Helpers.likes_activity(state, persona_id, activity)

static func dislikes_activity(state: Dictionary, persona_id: String, activity: String) -> bool:
	return Helpers.dislikes_activity(state, persona_id, activity)

# Unified domain creation function
# Parameters:
#   include_study: Include study-related actions and task methods (default: true)
#   include_social: Include social/relationship actions and methods (default: true)
#   include_unigoal: Include unigoal methods (default: true)
#   include_multigoal: Include multigoal methods (default: true)
static func create_planner_domain(
	include_study: bool = true,
	include_social: bool = true,
	include_unigoal: bool = true,
	include_multigoal: bool = true
) -> PlannerDomain:
	var domain = PlannerDomain.new()

	# Add actions - always include Sims-style needs satisfaction actions
	var actions = [
		Callable(MagicalGirlsCollegeDomain, "action_eat_mess_hall"),
		Callable(MagicalGirlsCollegeDomain, "action_eat_restaurant"),
		Callable(MagicalGirlsCollegeDomain, "action_eat_snack"),
		Callable(MagicalGirlsCollegeDomain, "action_cook_meal"),
		Callable(MagicalGirlsCollegeDomain, "action_sleep_dorm"),
		Callable(MagicalGirlsCollegeDomain, "action_nap_library"),
		Callable(MagicalGirlsCollegeDomain, "action_energy_drink"),
		Callable(MagicalGirlsCollegeDomain, "action_talk_friend"),
		Callable(MagicalGirlsCollegeDomain, "action_join_club"),
		Callable(MagicalGirlsCollegeDomain, "action_phone_call"),
		Callable(MagicalGirlsCollegeDomain, "action_play_games"),
		Callable(MagicalGirlsCollegeDomain, "action_watch_streaming"),
		Callable(MagicalGirlsCollegeDomain, "action_go_cinema"),
		Callable(MagicalGirlsCollegeDomain, "action_shower"),
		Callable(MagicalGirlsCollegeDomain, "action_wash_hands"),
		Callable(MagicalGirlsCollegeDomain, "action_move_to")
	]

	# Add study actions if requested
	if include_study:
		actions.append_array([
			Callable(MagicalGirlsCollegeDomain, "action_attend_lecture"),
			Callable(MagicalGirlsCollegeDomain, "action_complete_homework"),
			Callable(MagicalGirlsCollegeDomain, "action_study_library")
		])

	# Add social actions if requested
	if include_social:
		actions.append_array([
			Callable(MagicalGirlsCollegeDomain, "action_eat_mess_hall_with_friend"),
			Callable(MagicalGirlsCollegeDomain, "action_coffee_together"),
			Callable(MagicalGirlsCollegeDomain, "action_watch_movie"),
			Callable(MagicalGirlsCollegeDomain, "action_pool_hangout"),
			Callable(MagicalGirlsCollegeDomain, "action_park_picnic"),
			Callable(MagicalGirlsCollegeDomain, "action_beach_trip"),
			Callable(MagicalGirlsCollegeDomain, "action_read_book"),
			Callable(MagicalGirlsCollegeDomain, "action_club_activity"),
			Callable(MagicalGirlsCollegeDomain, "action_optimize_schedule"),
			Callable(MagicalGirlsCollegeDomain, "action_predict_outcome")
		])

	domain.add_actions(actions)

	# Add task methods for The Sims-style needs satisfaction (always included)
	var satisfy_hunger_methods = [
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_hunger_method_mess_hall"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_hunger_method_restaurant"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_hunger_method_snack"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_hunger_method_cook"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_hunger_method_social_eat")
	]
	domain.add_task_methods("task_satisfy_hunger", satisfy_hunger_methods)

	var satisfy_energy_methods = [
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_energy_method_sleep"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_energy_method_nap"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_energy_method_drink"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_energy_method_rest_activity"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_energy_method_early_sleep")
	]
	domain.add_task_methods("task_satisfy_energy", satisfy_energy_methods)

	var satisfy_social_methods = [
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_social_method_talk"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_social_method_club"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_social_method_phone"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_social_method_socialize_task"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_social_method_group_activity")
	]
	domain.add_task_methods("task_satisfy_social", satisfy_social_methods)

	var satisfy_fun_methods = [
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_fun_method_games"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_fun_method_streaming"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_fun_method_cinema"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_fun_method_preferred_activity"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_fun_method_social_fun")
	]
	domain.add_task_methods("task_satisfy_fun", satisfy_fun_methods)

	var satisfy_hygiene_methods = [
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_hygiene_method_shower"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_hygiene_method_wash_hands"),
		Callable(MagicalGirlsCollegeDomain, "task_satisfy_hygiene_method_force_shower")
	]
	domain.add_task_methods("task_satisfy_hygiene", satisfy_hygiene_methods)

	# Add move task method (always included)
	var move_to_location_methods = [
		Callable(MagicalGirlsCollegeDomain, "task_move_to_location_method_direct")
	]
	domain.add_task_methods("task_move_to_location", move_to_location_methods)

	# Add study task methods if requested
	if include_study:
		var earn_study_points_methods = [
			Callable(MagicalGirlsCollegeDomain, "task_earn_study_points_method_done"),
			Callable(MagicalGirlsCollegeDomain, "task_earn_study_points_method_lecture"),
			Callable(MagicalGirlsCollegeDomain, "task_earn_study_points_method_homework"),
			Callable(MagicalGirlsCollegeDomain, "task_earn_study_points_method_library"),
			Callable(MagicalGirlsCollegeDomain, "task_earn_study_points_method_coordinated")
		]
		domain.add_task_methods("task_earn_study_points", earn_study_points_methods)

	# Add social task methods if requested
	if include_social:
		var socialize_methods = [
			Callable(MagicalGirlsCollegeDomain, "task_socialize_method_easy"),
			Callable(MagicalGirlsCollegeDomain, "task_socialize_method_moderate"),
			Callable(MagicalGirlsCollegeDomain, "task_socialize_method_challenging")
		]
		domain.add_task_methods("task_socialize", socialize_methods)

		var manage_week_methods = [
			Callable(MagicalGirlsCollegeDomain, "task_manage_week_method_balance"),
			Callable(MagicalGirlsCollegeDomain, "task_manage_week_method_academics"),
			Callable(MagicalGirlsCollegeDomain, "task_manage_week_method_relationships")
		]
		domain.add_task_methods("task_manage_week", manage_week_methods)

	# Add unigoal methods if requested
	if include_unigoal:
		var study_unigoal_methods = [
			Callable(MagicalGirlsCollegeDomain, "unigoal_achieve_study_goal")
		]
		domain.add_unigoal_methods("study_points", study_unigoal_methods)

		var relationship_unigoal_methods = [
			Callable(MagicalGirlsCollegeDomain, "unigoal_achieve_relationship_goal")
		]
		domain.add_unigoal_methods("relationship_points", relationship_unigoal_methods)

	# Add multigoal methods if requested
	if include_multigoal:
		var multigoal_methods = [
			Callable(MagicalGirlsCollegeDomain, "multigoal_balance_life"),
			Callable(MagicalGirlsCollegeDomain, "multigoal_solve_temporal_puzzle")
		]
		domain.add_multigoal_methods(multigoal_methods)

	return domain

static func get_coordination(state: Dictionary, persona_id: String) -> Dictionary:
	return Helpers.get_coordination(state, persona_id)

static func set_coordination(state: Dictionary, persona_id: String, coordination: Dictionary) -> void:
	Helpers.set_coordination(state, persona_id, coordination)

static func get_need(state: Dictionary, persona_id: String, need_name: String, default_value: int = 50) -> int:
	return Helpers.get_need(state, persona_id, need_name, default_value)

static func set_need(state: Dictionary, persona_id: String, need_name: String, value: int) -> void:
	Helpers.set_need(state, persona_id, need_name, value)

static func get_money(state: Dictionary, persona_id: String) -> int:
	return Helpers.get_money(state, persona_id)

static func set_money(state: Dictionary, persona_id: String, amount: int) -> void:
	Helpers.set_money(state, persona_id, amount)

# Re-export actions for convenience
static func action_attend_lecture(state: Dictionary, persona_id: Variant, subject: Variant) -> Variant:
	return Actions.action_attend_lecture(state, persona_id, subject)

static func action_complete_homework(state: Dictionary, persona_id: Variant, subject: Variant) -> Variant:
	return Actions.action_complete_homework(state, persona_id, subject)

static func action_study_library(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_study_library(state, persona_id)

static func action_eat_mess_hall_with_friend(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	return Actions.action_eat_mess_hall_with_friend(state, persona_id, companion_id)

static func action_coffee_together(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	return Actions.action_coffee_together(state, persona_id, companion_id)

static func action_watch_movie(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	return Actions.action_watch_movie(state, persona_id, companion_id)

static func action_pool_hangout(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	return Actions.action_pool_hangout(state, persona_id, companion_id)

static func action_park_picnic(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	return Actions.action_park_picnic(state, persona_id, companion_id)

static func action_beach_trip(state: Dictionary, persona_id: Variant, companion_id: Variant) -> Variant:
	return Actions.action_beach_trip(state, persona_id, companion_id)

static func action_read_book(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_read_book(state, persona_id)

static func action_club_activity(state: Dictionary, persona_id: Variant, club: Variant) -> Variant:
	return Actions.action_club_activity(state, persona_id, club)

static func action_optimize_schedule(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_optimize_schedule(state, persona_id)

static func action_predict_outcome(state: Dictionary, persona_id: Variant, activity: Variant) -> Variant:
	return Actions.action_predict_outcome(state, persona_id, activity)

static func action_eat_mess_hall(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_eat_mess_hall(state, persona_id)

static func action_eat_restaurant(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_eat_restaurant(state, persona_id)

static func action_eat_snack(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_eat_snack(state, persona_id)

static func action_cook_meal(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_cook_meal(state, persona_id)

static func action_sleep_dorm(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_sleep_dorm(state, persona_id)

static func action_nap_library(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_nap_library(state, persona_id)

static func action_energy_drink(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_energy_drink(state, persona_id)

static func action_talk_friend(state: Dictionary, persona_id: Variant, friend_id: Variant) -> Variant:
	return Actions.action_talk_friend(state, persona_id, friend_id)

static func action_join_club(state: Dictionary, persona_id: Variant, club: Variant) -> Variant:
	return Actions.action_join_club(state, persona_id, club)

static func action_phone_call(state: Dictionary, persona_id: Variant, friend_id: Variant) -> Variant:
	return Actions.action_phone_call(state, persona_id, friend_id)

static func action_play_games(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_play_games(state, persona_id)

static func action_watch_streaming(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_watch_streaming(state, persona_id)

static func action_go_cinema(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_go_cinema(state, persona_id)

static func action_shower(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_shower(state, persona_id)

static func action_wash_hands(state: Dictionary, persona_id: Variant) -> Variant:
	return Actions.action_wash_hands(state, persona_id)

static func action_move_to(state: Dictionary, persona_id: Variant, location: Variant) -> Variant:
	return Actions.action_move_to(state, persona_id, location)

# Re-export task methods for convenience
static func task_earn_study_points_method_done(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	return TaskMethods.task_earn_study_points_method_done(state, persona_id, target_points)

static func task_earn_study_points_method_lecture(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	return TaskMethods.task_earn_study_points_method_lecture(state, persona_id, target_points)

static func task_earn_study_points_method_homework(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	return TaskMethods.task_earn_study_points_method_homework(state, persona_id, target_points)

static func task_earn_study_points_method_library(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	return TaskMethods.task_earn_study_points_method_library(state, persona_id, target_points)

static func task_earn_study_points_method_coordinated(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	return TaskMethods.task_earn_study_points_method_coordinated(state, persona_id, target_points)

static func task_socialize_method_easy(state: Dictionary, persona_id: Variant, companion_id: Variant, activity_level: Variant) -> Variant:
	return TaskMethods.task_socialize_method_easy(state, persona_id, companion_id, activity_level)

static func task_socialize_method_moderate(state: Dictionary, persona_id: Variant, companion_id: Variant, activity_level: Variant) -> Variant:
	return TaskMethods.task_socialize_method_moderate(state, persona_id, companion_id, activity_level)

static func task_socialize_method_challenging(state: Dictionary, persona_id: Variant, companion_id: Variant, activity_level: Variant) -> Variant:
	return TaskMethods.task_socialize_method_challenging(state, persona_id, companion_id, activity_level)

static func task_manage_week_method_balance(state: Dictionary, persona_id: Variant) -> Variant:
	return TaskMethods.task_manage_week_method_balance(state, persona_id)

static func task_manage_week_method_academics(state: Dictionary, persona_id: Variant) -> Variant:
	return TaskMethods.task_manage_week_method_academics(state, persona_id)

static func task_manage_week_method_relationships(state: Dictionary, persona_id: Variant) -> Variant:
	return TaskMethods.task_manage_week_method_relationships(state, persona_id)

static func task_satisfy_hunger_method_mess_hall(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	return TaskMethods.task_satisfy_hunger_method_mess_hall(state, persona_id, target_hunger)

static func task_satisfy_hunger_method_restaurant(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	return TaskMethods.task_satisfy_hunger_method_restaurant(state, persona_id, target_hunger)

static func task_satisfy_hunger_method_snack(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	return TaskMethods.task_satisfy_hunger_method_snack(state, persona_id, target_hunger)

static func task_satisfy_hunger_method_cook(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	return TaskMethods.task_satisfy_hunger_method_cook(state, persona_id, target_hunger)

static func task_satisfy_hunger_method_social_eat(state: Dictionary, persona_id: Variant, target_hunger: Variant) -> Variant:
	return TaskMethods.task_satisfy_hunger_method_social_eat(state, persona_id, target_hunger)

static func task_satisfy_energy_method_sleep(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	return TaskMethods.task_satisfy_energy_method_sleep(state, persona_id, target_energy)

static func task_satisfy_energy_method_nap(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	return TaskMethods.task_satisfy_energy_method_nap(state, persona_id, target_energy)

static func task_satisfy_energy_method_drink(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	return TaskMethods.task_satisfy_energy_method_drink(state, persona_id, target_energy)

static func task_satisfy_energy_method_rest_activity(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	return TaskMethods.task_satisfy_energy_method_rest_activity(state, persona_id, target_energy)

static func task_satisfy_energy_method_early_sleep(state: Dictionary, persona_id: Variant, target_energy: Variant) -> Variant:
	return TaskMethods.task_satisfy_energy_method_early_sleep(state, persona_id, target_energy)

static func task_satisfy_social_method_talk(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	return TaskMethods.task_satisfy_social_method_talk(state, persona_id, target_social)

static func task_satisfy_social_method_club(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	return TaskMethods.task_satisfy_social_method_club(state, persona_id, target_social)

static func task_satisfy_social_method_phone(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	return TaskMethods.task_satisfy_social_method_phone(state, persona_id, target_social)

static func task_satisfy_social_method_socialize_task(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	return TaskMethods.task_satisfy_social_method_socialize_task(state, persona_id, target_social)

static func task_satisfy_social_method_group_activity(state: Dictionary, persona_id: Variant, target_social: Variant) -> Variant:
	return TaskMethods.task_satisfy_social_method_group_activity(state, persona_id, target_social)

static func task_satisfy_fun_method_games(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	return TaskMethods.task_satisfy_fun_method_games(state, persona_id, target_fun)

static func task_satisfy_fun_method_streaming(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	return TaskMethods.task_satisfy_fun_method_streaming(state, persona_id, target_fun)

static func task_satisfy_fun_method_cinema(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	return TaskMethods.task_satisfy_fun_method_cinema(state, persona_id, target_fun)

static func task_satisfy_fun_method_preferred_activity(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	return TaskMethods.task_satisfy_fun_method_preferred_activity(state, persona_id, target_fun)

static func task_satisfy_fun_method_social_fun(state: Dictionary, persona_id: Variant, target_fun: Variant) -> Variant:
	return TaskMethods.task_satisfy_fun_method_social_fun(state, persona_id, target_fun)

static func task_satisfy_hygiene_method_shower(state: Dictionary, persona_id: Variant, target_hygiene: Variant) -> Variant:
	return TaskMethods.task_satisfy_hygiene_method_shower(state, persona_id, target_hygiene)

static func task_satisfy_hygiene_method_wash_hands(state: Dictionary, persona_id: Variant, target_hygiene: Variant) -> Variant:
	return TaskMethods.task_satisfy_hygiene_method_wash_hands(state, persona_id, target_hygiene)

static func task_satisfy_hygiene_method_force_shower(state: Dictionary, persona_id: Variant, target_hygiene: Variant) -> Variant:
	return TaskMethods.task_satisfy_hygiene_method_force_shower(state, persona_id, target_hygiene)

static func task_move_to_location_method_direct(state: Dictionary, persona_id: Variant, location: Variant) -> Variant:
	return TaskMethods.task_move_to_location_method_direct(state, persona_id, location)

# Re-export unigoal methods for convenience
static func unigoal_achieve_study_goal(state: Dictionary, persona_id: Variant, target_points: Variant) -> Variant:
	return UnigoalMethods.unigoal_achieve_study_goal(state, persona_id, target_points)

static func unigoal_achieve_relationship_goal(state: Dictionary, subject: Variant, target: Variant) -> Variant:
	return UnigoalMethods.unigoal_achieve_relationship_goal(state, subject, target)

# Re-export multigoal methods for convenience
static func multigoal_balance_life(state: Dictionary, multigoal: Array) -> Array:
	return MultigoalMethods.multigoal_balance_life(state, multigoal)

static func multigoal_solve_temporal_puzzle(state: Dictionary, multigoal: Array) -> Array:
	return MultigoalMethods.multigoal_solve_temporal_puzzle(state, multigoal)
