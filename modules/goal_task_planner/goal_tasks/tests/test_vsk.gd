# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# test_vsk.gd
# SPDX-License-Identifier: MIT

extends GutTest

## A simplified VR project planning example using the task_goal
## plugin framework for Godot. The goal is to create a plan for the
## VR development activities based on provided dependencies and durations.
var domain_name = "vr_project"
var the_domain = preload("res://addons/task_goal/core/domain.gd").new(domain_name)

var planner = preload("res://addons/task_goal/core/plan.gd").new()


@export var types = {
	"activity": [
		"Tip Method (Controller, Pen)",
		"Hand Method (Fingers)",
		"Pointer Ray Method (Shun)",
		"Handler receives input when method is in proximity",
		"Create a snap-to-object feature",
		"Develop an object debug system (display position and parameters)",
		"For each input method, handle input by distance (Distance link)",
		"Input action for pinch",
		"Input action for gaze",
		"Compound input action that starts when pinch and button press occur",
		"Pull cord handler",
		"Push button handler",
		"Spinner volume knob handler",
		"System control as diagetic for loading"
	]
}


@export var dependencies: Dictionary = {
	"Develop an object debug system (display position and parameters)": ["Create a snap-to-object feature"],
	"Create a snap-to-object feature": ["Tip Method (Controller, Pen)", "Hand Method (Fingers)", "Pointer Ray Method (Shun)"],
	"For each input method, handle input by distance (Distance link)": ["Handler receives input when method is in proximity"],
	"Input action for pinch": ["For each input method, handle input by distance (Distance link)"],
	"Input action for gaze": ["For each input method, handle input by distance (Distance link)"],
	"Pull cord handler": ["For each input method, handle input by distance (Distance link)"],
	"Push button handler": ["For each input method, handle input by distance (Distance link)"],
	"Spinner volume knob handler": ["For each input method, handle input by distance (Distance link)"],
	"Compound input action that starts when pinch and button press occur": ["Input action for pinch", "Push button handler"],
	"System control as diagetic for loading": [
		"Develop an object debug system (display position and parameters)",
		"Create a snap-to-object feature",
		"For each input method, handle input by distance (Distance link)",
		"Input action for pinch",
		"Input action for gaze",
		"Compound input action that starts when pinch and button press occur",
		"Pull cord handler",
		"Push button handler",
		"Spinner volume knob handler"
	]
}


## Prototypical initial state, can include current progress, resources etc.
var state0: Dictionary = {
	"completed_activities": [],
	"pending_activities": types["activity"]
}


## This function sets an activity as completed in the given state. 
## It removes the activity from pending activities and adds it to completed activities.
func set_as_completed(state, activity_name):
	if activity_name in state.pending_activities:
		state.pending_activities.erase(activity_name)
		state.completed_activities.append(activity_name)
	return state


## This function completes an activity in the given state if it is a pending activity.
func complete_activity(state, activity) -> Variant:
	if activity in state.pending_activities:
		return set_as_completed(state, activity)
	return false


## This function schedules an activity in the given state.
## It returns an array of actions needed to complete the activity and its dependencies.
func schedule_activity(state: Dictionary, activity: String) -> Variant:
	var actions = []
	
	if not activity in state["pending_activities"]:
		return []

	if activity in dependencies:
		for predecessor in dependencies[activity]:
			if not predecessor in state["completed_activities"]:
				actions.push_back(["schedule_activity", predecessor])

	actions.append(["complete_activity", activity])
	
	return actions


## This function initializes the planner with the domain and declares the actions and task methods.
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


## This function tests the VR project planning by executing a single activity and asserting the plan.
func test_vsk_planning() -> void:
	planner.current_domain = the_domain

	# Example: Executing all activities
	var state1 = state0.duplicate(true)
	var goals = []
	for activity in types["activity"]:
		goals.append(["schedule_activity", activity])
	
	RandomNumberGenerator.new().seed = 12345
	goals.shuffle()
	
	var plan: Variant = planner.find_plan(state1, goals)

	# Assert that plan is not false and has a positive size
	assert_true(not plan is bool and plan.size() > 0)

	gut.p("Plan to finish:")
	if typeof(plan) == TYPE_ARRAY:
		for action in plan:
			gut.p("- %s" % str(action))


## This function tests if an activity can be completed when it is not in pending activities.
func test_complete_activity_error_case() -> void:
	var state = state0.duplicate(true)
	state["pending_activities"] = []
	var result = complete_activity(state, "Understand the project requirements")
	assert_eq(result, false, "Activity should not be completed if it is not in pending activities.")


## This function tests if an activity can be scheduled when it is not in pending activities.
func test_schedule_activity_error_case() -> void:
	var state = state0.duplicate(true)
	state["pending_activities"] = []
	var actions = schedule_activity(state, "Understand the project requirements")
	assert_eq(actions.size(), 0, "No actions should be returned if the activity is not in pending activities.")


## This function tests if an activity can be set as completed when it is not in pending activities.
func test_set_as_completed_error_case() -> void:
	var state = state0.duplicate(true)
	state["pending_activities"] = []
	var result = set_as_completed(state, "Understand the project requirements")
	assert_eq(result, state, "State should remain unchanged if the activity is not in pending activities.")
