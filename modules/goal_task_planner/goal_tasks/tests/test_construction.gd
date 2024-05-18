# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# test_construction_project.gd
# SPDX-License-Identifier: MIT

extends GutTest
#"""
# A simplified construction project planning example using the task_goal
# plugin framework for Godot. The goal is to create a plan for the
# construction activities based on provided dependencies and durations.
#"""

var domain_name = "construction_project"
var the_domain = preload("res://addons/task_goal/core/domain.gd").new(domain_name)

var planner = preload("res://addons/task_goal/core/plan.gd").new()

@export var types = {
	"activity": [
		"finish",
		"pour_walks_and_complete_landscaping",
		"finish_grading",
		"finish_electrical_work",
		"paint",
		"sand_and_varnish_flooring",
		"lay_storm_drains_for_rain_water",
		"fasten_gutters_and_downspouts",
		"finish_roofing_and_flashing",
		"finish_carpentry",
		"install_finish_plumbing",
		"install_kitchen_fixtures",
		"lay_finish_flooring",
		"fasten_plaster_board_and_plaster",
		"install_heating_and_ventilating",
		"install_rough_wiring",
		"install_rough_plumbing",
		"pour_basement_floor",
		"install_basement_drains_and_plumbing",
		"lay_brickwork",
		"erect_wooden_frame_including_rough_roof",
		"pour_concrete_foundation",
		"excavate_and_pour_footers",
		"start"
	]
}

@export var dependencies: Dictionary = {
	"finish": ["sand_and_varnish_flooring", "finish_electrical_work", "pour_walks_and_complete_landscaping"],
	"pour_walks_and_complete_landscaping": ["finish_grading"],
	"finish_grading": ["fasten_gutters_and_downspouts", "lay_storm_drains_for_rain_water"],
	"finish_electrical_work": ["start"],
	"paint": ["install_kitchen_fixtures", "install_finish_plumbing"],
	"sand_and_varnish_flooring": ["finish_carpentry", "install_basement_drains_and_plumbing"],
	"lay_storm_drains_for_rain_water": ["pour_concrete_foundation"],
	"fasten_gutters_and_downspouts": ["finish_roofing_and_flashing"],
	"finish_roofing_and_flashing": ["lay_brickwork"],
	"finish_carpentry": ["lay_finish_flooring"],
	"install_finish_plumbing": ["lay_finish_flooring"],
	"install_kitchen_fixtures": ["lay_finish_flooring"],
	"lay_finish_flooring": ["fasten_plaster_board_and_plaster"],
	"fasten_plaster_board_and_plaster": ["install_rough_wiring", "install_heating_and_ventilating", "install_rough_plumbing"],
	"install_heating_and_ventilating": ["erect_wooden_frame_including_rough_roof", "pour_basement_floor"],
	"install_rough_wiring": ["erect_wooden_frame_including_rough_roof"],
	"install_rough_plumbing": ["install_basement_drains_and_plumbing"],
	"pour_basement_floor": ["install_basement_drains_and_plumbing"],
	"install_basement_drains_and_plumbing": ["pour_concrete_foundation"],
	"lay_brickwork": ["erect_wooden_frame_including_rough_roof"],
	"erect_wooden_frame_including_rough_roof": ["pour_concrete_foundation"],
	"pour_concrete_foundation": ["excavate_and_pour_footers"],
	"excavate_and_pour_footers": ["start"],
	"start": []
}


# Prototypical initial state, can include current progress, resources etc.
var state0: Dictionary = {
	# Example state properties
	"completed_activities": [],
	"pending_activities": types["activity"]  # Initial list from 'types'
}

# Helper functions:

func set_as_completed(state, activity_name):
	if activity_name in state.pending_activities:
		state.pending_activities.erase(activity_name)
		state.completed_activities.append(activity_name)
	return state


## Actions:

func complete_activity(state, activity) -> Variant:
	if activity in state.pending_activities:
		return set_as_completed(state, activity)
	return false


## Methods:


func schedule_activity(state: Dictionary, activity: String) -> Array:
	var actions = []
	
	if not activity in state["pending_activities"]:
		return []

	for predecessor in dependencies[activity]:
		if not predecessor in state["completed_activities"]:
			actions.push_back(["schedule_activity", predecessor])

	actions.append(["complete_activity", activity])
	
	return actions


func _ready() -> void:
	planner.domains.push_back(the_domain)
	planner.current_domain = the_domain
	planner.declare_actions([Callable(self, "complete_activity")])
	planner.declare_task_methods(
		"schedule_activity",
		[
			Callable(self, "schedule_activity"),
		]
	)

func test_construction_planning() -> void:
	planner.current_domain = the_domain

	# Example: Executing a single activity
	var state1 = state0.duplicate(true)
	var goals = [["schedule_activity", "finish"]]
	var plan: Variant = planner.find_plan(state1, goals)

	# Assert that plan is not false and has a positive size
	assert_true(not plan is bool and plan.size() > 0)
	
	gut.p("Plan to finish:")
	if typeof(plan) == TYPE_ARRAY:
		for action in plan:
			gut.p("- %s" % str(action))
