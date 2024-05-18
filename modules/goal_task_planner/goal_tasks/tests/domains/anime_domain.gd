# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# anime_domain.gd
# SPDX-License-Identifier: MIT

extends "res://addons/task_goal/core/domain.gd"

@export var types = {
	"character": ["Mia", "Frank", "Chair", "Hero", "Villain", "user1", "target1"],
	"location":
	[
		"home_Mia",
		"home_Frank",
		"cinema",
		"station",
		"mall",
		"park",
		"restaurant",
		"school",
		"office",
		"gym",
		"library",
		"hospital",
		"beach",
		"supermarket",
		"museum",
		"zoo",
		"airport"
	],
	"door":
	[
		"home_Mia",
		"home_Frank",
		"cinema",
		"station",
		"mall",
		"park",
		"restaurant",
		"school",
		"office",
		"gym",
		"library",
		"hospital",
		"beach",
		"supermarket",
		"museum",
		"zoo",
		"airport"
	],
	"vehicle": ["car1", "car2"],
	"owe": [],
	"stn": [],
}

@export var distance: Dictionary = {
	["home_Mia", "cinema"]: 12,
	["cinema", "home_Mia"]: 12,
	["home_Frank", "cinema"]: 4,
	["cinema", "home_Frank"]: 4,
	["station", "home_Mia"]: 1,
	["station", "home_Frank"]: 10,
	["station", "cinema"]: 12,
	["home_Mia", "station"]: 12,
	["home_Mia", "mall"]: 8,
	["home_Frank", "mall"]: 10,
	["home_Frank", "hospital"]: 20,
	["hospital", "home_Frank"]: 20,
	["mall", "cinema"]: 7,
	["home_Mia", "park"]: 5,
	["park", "restaurant"]: 5,
	["restaurant", "school"]: 6,
	["school", "office"]: 7,
	["office", "gym"]: 8,
	["gym", "library"]: 9,
	["library", "hospital"]: 10,
	["hospital", "beach"]: 11,
	["beach", "supermarket"]: 12,
	["supermarket", "park"]: 13,
	["park", "museum"]: 13,
	["museum", "zoo"]: 14,
	["zoo", "airport"]: 15,
	["airport", "home_Mia"]: 16,
	["airport", "home_Frank"]: 17,
}


func _init() -> void:
	set_name("romantic")


func is_a(variable, type) -> bool:
	return variable in types[type]


func close_door(state, person, location, status) -> Variant:
	if check_location_and_action(state, person, location, "door"):
		state["door"][location] = status
		if verbose > 0:
			print("The door at location %s has been closed." % [location])
		return state
	return false


func handle_temporal_constraint(state, person, current_time, goal_time, constraint_name) -> Variant:
	var constraint = TemporalConstraint.new(
		current_time,
		goal_time,
		goal_time - current_time,
		TemporalConstraint.TemporalQualifier.AT_END,
		constraint_name
	)
	if state["stn"][person].check_overlap(constraint):
		if verbose > 0:
			print("Error: Temporal constraint overlaped %s" % [str(constraint)])
		return false
	if state["stn"][person].add_temporal_constraint(constraint):
		state["time"][person] = goal_time
		return state
	else:
		if verbose > 0:
			print("Error: Failed to add temporal constraint %s" % str(constraint))
	return false


@export var moves = {
	"Tackle": {"power": 40, "type": "Normal", "category": "Physical"},
	"Growl": {"power": 0, "type": "Normal", "category": "Status"},
	"Ember": {"power": 40, "type": "Fire", "category": "Special"},
	"Tail Whip": {"power": 0, "type": "Normal", "category": "Status"}
}


func apply_status_move(state, _user, target, move) -> Variant:
	if move == "Growl":
		state["stats"][target]["Attack"] -= 1
	elif move == "Tail Whip":
		state["stats"][target]["Defense"] -= 1
	return state


func apply_damage_move(state, user, target, move) -> Variant:
	var move_data = moves[move]
	var damage = calculate_damage(state, user, target, move_data)
	state["health"][target] -= damage
	return state


func calculate_damage(state, user, target, move_data) -> int:
	var level = state["level"][user]
	var power = move_data["power"]
	var A = state["stats"][user]["Attack"]
	var D = state["stats"][target]["Defense"]
	var damage = (((2 * level / 5 + 2) * power * A / D) / 50) + 2
	return damage


func use_move(state, user, target, move, time) -> Variant:
	if is_a(user, "character") and is_a(target, "character") and move in moves.keys():
		var move_data = moves[move]
		var current_time = state["time"][user]
		var goal_time = time + 1
		var constraint_name = "%s_uses_%s" % [user, move]
		state = handle_temporal_constraint(state, user, current_time, goal_time, constraint_name)
		if state:
			if move_data["category"] == "Status":
				return apply_status_move(state, user, target, move)
			else:
				return apply_damage_move(state, user, target, move)
	return false


func idle(state, person, time) -> Variant:
	if is_a(person, "character"):
		var current_time = state["time"][person]
		if current_time >= time:
			return false
		var constraint_name = "%s_idle_until_%s" % [person, time]
		return handle_temporal_constraint(state, person, current_time, time, constraint_name)
	return false


func wait_for_everyone(state, persons):
	var max_time = 0
	for person in persons:
		if not is_a(person, "character"):
			return false
		var time = state.time[person]  # Get the time for each person
		if time > max_time:
			max_time = time  # Update the maximum time

		# Have everyone do nothing until the slowest person arrives
	for person in persons:
		var time = max_time - state.time[person]
		if time > 0:
			state.time[person] += time
	return state


func ride_car(state, p, y, time) -> Variant:
	if is_a(p, "character") and is_a(state.loc[p], "vehicle") and is_a(y, "location"):
		var car = state.loc[p]
		var x = state.loc[car]
		if is_a(x, "location") and x != y:
			var current_time = state.time[p]
			if current_time >= time:
				print(
					(
						"ride_car error: Current time %s is bigger than ride car time %s"
						% [current_time, time]
					)
				)
				return false
			var constraint_name = "%s_ride_car_from_%s_to_%s" % [p, x, y]
			state = handle_temporal_constraint(state, p, current_time, time, constraint_name)
			if state:
				state.loc[car] = y
				state.owe[p] = taxi_rate(distance_to(x, y))
				return state
	return false


func call_car(state, p, x, goal_time) -> Variant:
	if is_a(p, "character") and is_a(x, "location"):
		var current_time = state.time[p]
		if current_time < goal_time:
			current_time = goal_time
		var _travel_time = 1
		var arrival_time = goal_time + _travel_time
		var constraint_name = "%s_call_car_at_%s" % [p, x]
		state = handle_temporal_constraint(state, p, current_time, arrival_time, constraint_name)
		if state:
			for car in types["vehicle"]:
				state.loc[car] = x
				state.loc[p] = car
				return state
			print("call_car error: No available cars at location.")
			return state
	return false


func pay_driver(state, p, y, goal_time) -> Variant:
	if is_a(p, "character"):
		if state.cash[p] >= state.owe[p]:
			var payment_time = 1  # Assuming payment takes no time
			var current_time = state.time[p]
			# Ensure the payment starts after the ride
			if current_time < goal_time:
				current_time = goal_time
			var post_payment_time = current_time + payment_time
			var constraint_name = "%s_pay_driver_at_%s" % [p, y]
			state = handle_temporal_constraint(
				state, p, current_time, post_payment_time, constraint_name
			)
			if state:
				state.cash[p] = state.cash[p] - state.owe[p]
				state.owe[p] = 0
				state.loc[p] = y
				return state
			else:
				print("Current STN: ", state["stn"][p])
				print("Temporal constraint could not be added")
	return false


func walk(state, p, x, y, time) -> Variant:
	if is_a(p, "character") and is_a(x, "location") and is_a(y, "location"):
		if x == y:
			if verbose > 0:
				print("Origin and destination are the same.")
			return false
		if state.loc[p] == x:
			var constraint_name = "%s_walk_from_%s_to_%s" % [p, x, y]
			var current_time = state["time"][p]
			if current_time >= time:
				return false
			state = handle_temporal_constraint(state, p, current_time, time, constraint_name)
			if state and state["stn"][p].is_consistent():
				state.loc[p] = y
				return state
	if verbose > 0:
		print("Invalid parameters.")
	return false


func do_mia_close_door(state, location, status) -> Variant:
	var person = "Mia"
	if is_a(person, "character") and is_a(location, "location") and is_a(location, "door"):
		if state["door"][location] == "closed":
			if verbose > 0:
				print("Door at location: %s is already closed" % location)
			return []
		else:
			if verbose > 0:
				print("Attempting to change door status at location: %s to %s" % [location, status])
			var actions = []
			actions.append(["travel", person, location])
			actions.append(["close_door", person, location, "closed"])
			return actions
	else:
		if verbose > 0:
			print("Cant close door %s %s:" % [person, location])
	return false


func do_close_door(state, person, location, status, time) -> Variant:
	if check_location_and_action(state, person, location, "door"):
		if state["time"][person] >= time:
			return false
		if verbose > 0:
			print("Attempting to change door status at location: %s to %s" % [location, status])
		var actions = []
		actions.append(["close_door", person, location, "closed"])
		return actions
	else:
		if verbose > 0:
			print("Cant close door %s %s:" % [person, location])
	return false


func do_idle(state, person, time) -> Variant:
	if time == 0:
		return false
	if is_a(person, "character"):
		if verbose > 0:
			print("Current time for %s: %s" % [person, state["time"][person]])
			print("Goal time: %s" % time)
		if state["time"][person] >= time:
			return false
		return [["idle", person, time]]
	return false


func do_walk(state, person, x, y, time) -> Variant:
	if time == 0:
		return false
	if is_a(person, "character") and is_a(x, "location") and is_a(y, "location") and y != x:
		if verbose > 0:
			print("Current time for %s: %s" % [person, state["time"][person]])
			print("Goal time: %s" % time)
		if state["time"][person] >= time:
			return false
		return [["walk", person, x, y, time]]
	return false


func travel_time(x, y, mode) -> int:
	var _distance = distance_to(x, y)
	if mode == "foot":
		return _distance / 1
	elif mode == "car":
		return _distance / 5
	else:
		print("Error: Invalid mode of transportation")
		return -1


func do_nothing(state, p, y) -> Variant:
	if is_a(p, "character"):
		state["time"][p] += y
		return []
	return false


func travel_by_foot(state, p, destination) -> Variant:
	if is_a(p, "character") and is_a(destination, "location"):
		var current_location = state.loc[p]
		if current_location == destination:
			return false
		elif current_location != null:
			return [["find_path", p, destination]]
	return false


var memo = {}


func find_path(state, p, destination) -> Variant:
	var current_location = state["loc"][p]
	var pq = [[0, current_location]]  # Initialize priority queue with current location and distance as 0
	var dist = {}
	var prev = {}
	for loc in types["location"]:
		dist[loc] = INF
		prev[loc] = null
	dist[current_location] = 0

	while pq.size() > 0:
		pq.sort()
		var top = pq.pop_front()
		var d = top[0]
		var u = top[1]

		if d != dist[u]:
			continue

		for loc in types["location"]:
			if loc == u or distance_to(u, loc) == INF:
				continue
			var travel_time_to_loc = travel_time(u, loc, "foot")
			if dist[u] + travel_time_to_loc < dist[loc]:
				dist[loc] = dist[u] + travel_time_to_loc
				prev[loc] = u
				pq.push_back([dist[loc], loc])

	if not dist.has(destination):
		return false

	var path = []
	var uu = destination
	while uu != null:
		if prev[uu] != null and prev[uu] != uu:
			var _travel_time = travel_time(prev[uu], uu, "foot")
			path.push_front(["walk", p, prev[uu], uu, state["time"][p] + _travel_time])
		uu = prev[uu]

	return path


func travel_by_car(state, p, y) -> Variant:
	if is_a(p, "character") and is_a(y, "location"):
		var x = state.loc[p]
		if x == y:
			return false
		if x != y and state.cash[p] >= taxi_rate(distance_to(x, y)):
			var call_car_goal_time = state.time[p] + 1  # Assuming calling a car takes 1 unit of time
			var ride_car_goal_time = max(
				call_car_goal_time + travel_time(x, y, "car") + 1, call_car_goal_time + 2
			)
			var actions = [
				["call_car", p, x, call_car_goal_time], ["ride_car", p, y, ride_car_goal_time]
			]
			if actions[0][0] == "call_car" and actions[1][0] == "ride_car":
				var pay_driver_goal_time = ride_car_goal_time + 1  # Assuming payment takes 1 unit of time
				actions.append(["pay_driver", p, y, pay_driver_goal_time])
			return actions
	return false


func compare_goal_times(a, b) -> bool:
	return a[1] > b[1]


func check_location_and_action(state, person, location, action) -> bool:
	return (
		is_a(location, "location")
		and is_a(person, "character")
		and state["loc"][person] == location
		and state[action][location] != "closed"
	)


func path_has_location(path, location) -> bool:
	for step in path:
		if step[3] == location:  # Check the destination of each step
			if verbose > 0:
				print("Location %s found in path" % location)
			return true
	if verbose > 0:
		print("Location %s not found in path" % location)
	return false


func taxi_rate(taxi_dist) -> float:
	return 1.5 + 0.5 * taxi_dist


func distance_to(x: String, y: String) -> float:
	var result = distance.get([x, y])
	if result == null:
		return INF
	if result > 0:
		return result
	result = distance.get([y, x])
	if result == null:
		return INF
	if result > 0:
		return result
	return INF
