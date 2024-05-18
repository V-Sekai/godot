# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# test_anime.gd
# SPDX-License-Identifier: MIT

extends GutTest

var the_domain = preload("res://goal_tasks/tests/domains/anime_domain.gd").new()

var planner = null


func before_each():
	planner = preload("res://addons/task_goal/core/plan.gd").new()
	planner.verbose = 0
	var new_domain = the_domain.duplicate(true)
	planner.domains.push_back(new_domain)
	planner.current_domain = new_domain

	planner.declare_actions(
		[
			Callable(planner.current_domain, "use_move"),
			Callable(planner.current_domain, "apply_status_move"),
			Callable(planner.current_domain, "apply_damage_move"),
			Callable(planner.current_domain, "walk"),
			Callable(planner.current_domain, "close_door"),
			Callable(planner.current_domain, "call_car"),
			Callable(planner.current_domain, "do_nothing"),
			Callable(planner.current_domain, "ride_car"),
			Callable(planner.current_domain, "pay_driver"),
			Callable(planner.current_domain, "do_nothing"),
			Callable(planner.current_domain, "idle"),
			Callable(planner.current_domain, "wait_for_everyone")
		]
	)
	planner.declare_unigoal_methods(
		"loc",
		[
			Callable(planner.current_domain, "travel_by_foot"),
			Callable(planner.current_domain, "travel_by_car"),
			Callable(planner.current_domain, "find_path")
		]
	)
	planner.declare_task_methods(
		"travel",
		[
			Callable(planner.current_domain, "travel_by_foot"),
			Callable(planner.current_domain, "travel_by_car"),
			Callable(planner.current_domain, "find_path")
		]
	)
	planner.declare_task_methods(
		"travel_by_car", [Callable(planner.current_domain, "travel_by_car")]
	)
	planner.declare_unigoal_methods("time", [Callable(planner.current_domain, "do_idle")])
	planner.declare_multigoal_methods([planner.m_split_multigoal])
	planner.declare_unigoal_methods("door", [Callable(planner.current_domain, "do_mia_close_door")])
	planner.declare_task_methods("find_path", [Callable(planner.current_domain, "find_path")])
	planner.declare_task_methods(
		"do_close_door", [Callable(planner.current_domain, "do_close_door")]
	)
	planner.declare_task_methods("do_walk", [Callable(planner.current_domain, "do_walk")])
	planner.declare_multigoal_methods([planner.m_split_multigoal])


func test_walk():
	planner.verbose = 0
	var state0: Dictionary = {
		"loc":
		{
			"Mia": "home_Mia",
			"Chair": "home_Mia",
			"Frank": "home_Frank",
			"car1": "cinema",
			"car2": "station"
		},
		"cash": {"Mia": 30, "Frank": 35},
		"owe": {"Mia": 0, "Frank": 0},
		"time": {"Mia": 0, "Frank": 0},
		"stn":
		{
			"Mia": SimpleTemporalNetwork.new(),
			"Frank": SimpleTemporalNetwork.new(),
			"Chair": SimpleTemporalNetwork.new()
		},
		"door": {}
	}
	var state1 = state0.duplicate(true)
	var expected = [["walk", "Mia", "home_Mia", "mall", 8]]
	var result = planner.find_plan(state1, [["walk", "Mia", "home_Mia", "mall", 8]])
	assert_eq_deep(result, expected)
	assert_eq(state1["time"]["Mia"], 8)
	assert_eq(state1["loc"]["Mia"], "mall")


func test_find_path():
	planner.verbose = 0
	var state0: Dictionary = {
		"loc":
		{
			"Mia": "home_Mia",
			"Chair": "home_Mia",
			"Frank": "home_Frank",
			"car1": "cinema",
			"car2": "station"
		},
		"cash": {"Mia": 30, "Frank": 35},
		"owe": {"Mia": 0, "Frank": 0},
		"time": {"Mia": 0, "Frank": 0},
		"stn":
		{
			"Mia": SimpleTemporalNetwork.new(),
			"Frank": SimpleTemporalNetwork.new(),
			"Chair": SimpleTemporalNetwork.new()
		},
		"door": {}
	}
	var expected = [["walk", "Mia", "home_Mia", "mall", 8]]
	var result = planner.find_plan(state0.duplicate(true), [["find_path", "Mia", "mall"]])
	assert_eq_deep(result, expected)


func test_find_path_next_node():
	planner.verbose = 0
	var state0: Dictionary = {
		"loc":
		{
			"Mia": "home_Mia",
			"Chair": "home_Mia",
			"Frank": "home_Frank",
			"car1": "cinema",
			"car2": "station"
		},
		"cash": {"Mia": 30, "Frank": 35},
		"owe": {"Mia": 0, "Frank": 0},
		"time": {"Mia": 0, "Frank": 0},
		"stn":
		{
			"Mia": SimpleTemporalNetwork.new(),
			"Frank": SimpleTemporalNetwork.new(),
			"Chair": SimpleTemporalNetwork.new()
		},
		"door": {}
	}
	var expected = [
		["walk", "Mia", "home_Mia", "park", 5],
		["walk", "Mia", "park", "museum", 13],
		["walk", "Mia", "museum", "zoo", 14],
		["walk", "Mia", "zoo", "airport", 15]
	]
	var result = planner.find_plan(state0, [["find_path", "Mia", "airport"]])
	assert_eq_deep(result, expected)
	assert_eq(state0["time"]["Mia"], 15)
	assert_eq(state0["loc"]["Mia"], "airport")


func test_isekai_anime_01():
	planner.verbose = 0
	var state0: Dictionary = {
		"loc":
		{
			"Mia": "home_Mia",
			"Chair": "home_Mia",
			"Frank": "home_Frank",
			"car1": "cinema",
			"car2": "station"
		},
		"cash": {"Mia": 30, "Frank": 35},
		"owe": {"Mia": 0, "Frank": 0},
		"time": {"Mia": 0, "Frank": 0},
		"stn":
		{
			"Mia": SimpleTemporalNetwork.new(),
			"Frank": SimpleTemporalNetwork.new(),
			"Chair": SimpleTemporalNetwork.new()
		},
		"door": {}
	}
	var expected = [["walk", "Mia", "home_Mia", "mall", 8]]
	var result = planner.find_plan(state0.duplicate(true), [["travel", "Mia", "mall"]])
	assert_eq_deep(result, expected)


func test_isekai_anime_02():
	planner.verbose = 0
	var state0: Dictionary = {
		"loc":
		{
			"Mia": "home_Mia",
			"Chair": "home_Mia",
			"Frank": "home_Frank",
			"car1": "cinema",
			"car2": "station"
		},
		"cash": {"Mia": 30, "Frank": 35},
		"owe": {"Mia": 0, "Frank": 0},
		"time": {"Mia": 0, "Frank": 0},
		"stn":
		{
			"Mia": SimpleTemporalNetwork.new(),
			"Frank": SimpleTemporalNetwork.new(),
			"Chair": SimpleTemporalNetwork.new()
		},
		"door": {}
	}
	var goal2 = Multigoal.new("goal2", {"loc": {"Mia": "mall", "Frank": "mall"}})
	var plan = planner.find_plan(state0.duplicate(true), [goal2])
	assert_eq_deep(
		plan, [["walk", "Mia", "home_Mia", "mall", 8], ["walk", "Frank", "home_Frank", "mall", 10]]
	)


func test_visit_all_the_doors() -> void:
	planner.verbose = 0
	var state0: Dictionary = {
		"loc":
		{
			"Mia": "home_Mia",
			"Chair": "home_Mia",
			"Frank": "home_Frank",
			"car1": "cinema",
			"car2": "station"
		},
		"cash": {"Mia": 30, "Frank": 35},
		"owe": {"Mia": 0, "Frank": 0},
		"time": {"Mia": 0, "Frank": 0},
		"stn":
		{
			"Mia": SimpleTemporalNetwork.new(),
			"Frank": SimpleTemporalNetwork.new(),
			"Chair": SimpleTemporalNetwork.new()
		},
		"door": {}
	}
	var door_goals = []
	for location in the_domain.types["location"]:
		var task = ["travel", "Mia", location]
		gut.p(task)
		door_goals.append(task)
	var result = planner.find_plan(state0, door_goals)
	assert_eq_deep(
		result,
		[
			["call_car", "Mia", "cinema", 13],
			["ride_car", "Mia", "home_Frank", 15],
			["pay_driver", "Mia", "home_Frank", 16],
			["walk", "Mia", "home_Frank", "cinema", 21],
			["call_car", "Mia", "home_Mia", 34],
			["ride_car", "Mia", "station", 37],
			["pay_driver", "Mia", "station", 38],
			["walk", "Mia", "station", "home_Mia", 40],
			["walk", "Mia", "home_Mia", "mall", 47],
			["call_car", "Mia", "home_Mia", 60],
			["ride_car", "Mia", "park", 62],
			["pay_driver", "Mia", "park", 63],
			["walk", "Mia", "park", "restaurant", 69],
			["walk", "Mia", "restaurant", "school", 75],
			["walk", "Mia", "school", "office", 82],
			["walk", "Mia", "office", "gym", 90],
			["walk", "Mia", "gym", "library", 99],
			["walk", "Mia", "library", "hospital", 109],
			["walk", "Mia", "hospital", "beach", 120],
			["walk", "Mia", "beach", "supermarket", 132],
			["call_car", "Mia", "park", 146],
			["ride_car", "Mia", "museum", 149],
			["pay_driver", "Mia", "museum", 150],
			["walk", "Mia", "museum", "zoo", 165],
			["walk", "Mia", "zoo", "airport", 180]
		]
	)
	assert_ne_deep(result, false)


func test_close_all_the_door_goal() -> void:
	planner.verbose = 0
	var state0: Dictionary = {
		"loc": {"Mia": "home_Mia", "Chair": "home_Mia"},
		"cash": {"Mia": 30, "Frank": 35},
		"owe": {"Mia": 0, "Frank": 0},
		"time": {"Mia": 0, "Frank": 0},
		"stn":
		{
			"Mia": SimpleTemporalNetwork.new(),
			"Frank": SimpleTemporalNetwork.new(),
			"Chair": SimpleTemporalNetwork.new()
		},
		"door": {}
	}
	var state1 = state0.duplicate(true)
	for location in planner.current_domain.types["location"]:
		state1["door"][location] = "opened"
	var goals = []
	for location in planner.current_domain.types["location"]:
		if not planner.current_domain.is_a(location, "location"):
			continue
		goals.append(
			Multigoal.new(
				"goal_%s" % location, {"loc": {"Mia": location}, "door": {location: "closed"}}
			)
		)
	state1 = planner.run_lazy_lookahead(state1, goals)
	var is_doors_closed = true
	for location in planner.current_domain.types["location"]:
		gut.p("Location and door state: %s %s" % [location, state1["door"][location]])
		if state1["door"][location] != "closed":
			is_doors_closed = false
			gut.p("Door is still open: %s" % location)
	gut.p("Mia is at %s" % state1["loc"]["Mia"])
	gut.p("What is Mia's time?: %s" % state1["time"]["Mia"])
	assert_true(is_doors_closed)


func test_travel_by_car():
	planner.verbose = 0
	var state: Dictionary = {
		"loc":
		{
			"Mia": "home_Mia",
			"Chair": "home_Mia",
			"Frank": "home_Frank",
			"car1": "cinema",
			"car2": "station"
		},
		"cash": {"Mia": 30, "Frank": 35},
		"owe": {"Mia": 0, "Frank": 0},
		"time": {"Mia": 0, "Frank": 0},
		"stn":
		{
			"Mia": SimpleTemporalNetwork.new(),
			"Frank": SimpleTemporalNetwork.new(),
			"Chair": SimpleTemporalNetwork.new()
		},
		"door": {}
	}

	var expected = [
		["call_car", "Mia", "home_Mia", 1],
		["ride_car", "Mia", "mall", 3],
		["pay_driver", "Mia", "mall", 4]
	]

	var result = planner.find_plan(state, [["travel_by_car", "Mia", "mall"]])
	assert_eq_deep(result, expected)


func test_do_walk():
	planner.verbose = 0
	var state: Dictionary = {
		"loc":
		{
			"Mia": "home_Mia",
			"Chair": "home_Mia",
			"Frank": "home_Frank",
			"car1": "cinema",
			"car2": "station"
		},
		"cash": {"Mia": 30, "Frank": 35},
		"owe": {"Mia": 0, "Frank": 0},
		"time": {"Mia": 0, "Frank": 0},
		"stn":
		{
			"Mia": SimpleTemporalNetwork.new(),
			"Frank": SimpleTemporalNetwork.new(),
			"Chair": SimpleTemporalNetwork.new()
		},
		"door": {}
	}

	var expected = [["walk", "Mia", "home_Mia", "mall", 8]]

	var result = planner.find_plan(state, [["do_walk", "Mia", "home_Mia", "mall", 8]])
	assert_eq_deep(result, expected)


func test_use_move():
	planner.verbose = 0
	var state = {
		"time": {"user1": 0},
		"stats": {"target1": {"Attack": 5, "Defense": 5}},
		"health": {"target1": 100},
		"level": {"user1": 10},
		"stn": {"user1": SimpleTemporalNetwork.new(), "target1": SimpleTemporalNetwork.new()},
	}
	var user = "user1"
	var target = "target1"
	var move = "Growl"
	var time = 1

	planner.find_plan(state, [["use_move", user, target, move, time]])
	assert_eq(state["stats"][target]["Attack"], 4)

