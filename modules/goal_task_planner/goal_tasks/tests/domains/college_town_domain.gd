# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# college_town_domain.gd
# SPDX-License-Identifier: MIT

extends "res://addons/task_goal/core/domain.gd"

@export var distance: Dictionary = {
# ["home_Mia", "cinema"]: 12,
}


func _init() -> void:
	set_name("college_town")


@export var types = {}

@export var simple_temporal_networks = {}


func is_a(variable, type) -> bool:
	return variable in types[type]


func travel_time(x, y, mode) -> int:
	var _distance = distance_to(x, y)
	if mode == "foot":
		return _distance / 1
	elif mode == "car":
		return _distance / 5
	else:
		print("Error: Invalid mode of transportation")
		return -1


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
