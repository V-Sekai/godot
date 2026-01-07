extends Node

class_name RescueDomain

static func get_location(location: Variant) -> Array:
	if location is Array:
		return location
	if location is Vector2:
		return [int(location.x), int(location.y)]
	if location is String:
		var parts = location.strip_edges().split(",")
		if parts.size() >= 2:
			return [int(parts[0]), int(parts[1])]
	return []

static func get_string_from_dict(dict: Dictionary, key: String) -> String:
	if dict.has(key):
		return String(dict[key])
	return ""

static func get_int_from_dict(dict: Dictionary, key: String) -> int:
	if dict.has(key):
		return int(dict[key])
	return 0

static func euclidean_distance(loc1: Array, loc2: Array) -> float:
	if loc1.size() < 2 || loc2.size() < 2:
		return INF
	return sqrt(pow(loc1[0] - loc2[0], 2) + pow(loc1[1] - loc2[1], 2))

static func manhattan_distance(loc1: Array, loc2: Array) -> float:
	if loc1.size() < 2 || loc2.size() < 2:
		return INF
	return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

static func obstacle_in_path(loc1: Array, loc2: Array) -> bool:
	# Simplified obstacle check
	return false

# Actions

static func action_free_robot(state: Dictionary, robot: String) -> Dictionary:
	var new_state = state.duplicate(true)
	var status_dict = new_state.get("status", {})
	status_dict[robot] = "free"
	new_state["status"] = status_dict
	return new_state

static func action_die_update(state: Dictionary, person: String) -> Dictionary:
	var new_state = state.duplicate(true)
	var real_status_dict = new_state.get("real_status", {})
	real_status_dict[person] = "dead"
	new_state["real_status"] = real_status_dict
	return new_state

static func action_move_euclidean(state: Dictionary, robot: String, from_loc: Array, to_loc: Array, dist: Variant) -> Variant:
	if obstacle_in_path(from_loc, to_loc):
		return null
	var new_state = state.duplicate(true)
	var loc_dict = new_state.get("loc", {})
	loc_dict[robot] = to_loc
	new_state["loc"] = loc_dict
	return new_state

static func action_move_manhattan(state: Dictionary, robot: String, from_loc: Array, to_loc: Array, dist: Variant) -> Variant:
	if obstacle_in_path(from_loc, to_loc):
		return null
	var new_state = state.duplicate(true)
	var loc_dict = new_state.get("loc", {})
	loc_dict[robot] = to_loc
	new_state["loc"] = loc_dict
	return new_state

static func action_move_curved(state: Dictionary, robot: String, from_loc: Array, to_loc: Array, dist: Variant) -> Variant:
	if obstacle_in_path(from_loc, to_loc):
		return null
	var new_state = state.duplicate(true)
	var loc_dict = new_state.get("loc", {})
	loc_dict[robot] = to_loc
	new_state["loc"] = loc_dict
	return new_state

static func action_move_fly(state: Dictionary, robot: String, from_loc: Array, to_loc: Array) -> Dictionary:
	var new_state = state.duplicate(true)
	var loc_dict = new_state.get("loc", {})
	loc_dict[robot] = to_loc
	new_state["loc"] = loc_dict
	return new_state

static func action_move_alt_fly(state: Dictionary, robot: String, from_loc: Array, to_loc: Array) -> Dictionary:
	var new_state = state.duplicate(true)
	var loc_dict = new_state.get("loc", {})
	loc_dict[robot] = to_loc
	new_state["loc"] = loc_dict
	return new_state

static func action_inspect_person(state: Dictionary, robot: String, person: String) -> Dictionary:
	var new_state = state.duplicate(true)
	var status_dict = new_state.get("status", {})
	var real_status_dict = new_state.get("real_status", {})
	if real_status_dict.has(person):
		status_dict[person] = real_status_dict[person]
	new_state["status"] = status_dict
	return new_state

static func action_support_person(state: Dictionary, robot: String, person: String) -> Variant:
	var status_dict = state.get("status", {})
	var person_status = get_string_from_dict(status_dict, person)
	if person_status == "dead":
		return null
	var new_state = state.duplicate(true)
	var new_status_dict = new_state.get("status", {})
	var real_status_dict = new_state.get("real_status", {})
	new_status_dict[person] = "OK"
	real_status_dict[person] = "OK"
	new_state["status"] = new_status_dict
	new_state["real_status"] = real_status_dict
	return new_state

static func action_support_person_2(state: Dictionary, robot: String, person: String) -> Variant:
	return action_support_person(state, robot, person)

static func action_inspect_location(state: Dictionary, robot: String, location: Variant) -> Dictionary:
	var new_state = state.duplicate(true)
	var status_dict = new_state.get("status", {})
	var real_status_dict = new_state.get("real_status", {})
	var loc_key = str(location)
	if real_status_dict.has(loc_key):
		status_dict[loc_key] = real_status_dict[loc_key]
	new_state["status"] = status_dict
	return new_state

static func action_clear_location(state: Dictionary, robot: String, location: Variant) -> Dictionary:
	var new_state = state.duplicate(true)
	var status_dict = new_state.get("status", {})
	var real_status_dict = new_state.get("real_status", {})
	var loc_key = str(location)
	status_dict[loc_key] = "clear"
	real_status_dict[loc_key] = "clear"
	new_state["status"] = status_dict
	new_state["real_status"] = real_status_dict
	return new_state

static func action_clear_location_2(state: Dictionary, robot: String, location: Variant) -> Dictionary:
	return action_clear_location(state, robot, location)

static func action_replenish_supplies(state: Dictionary, robot: String) -> Variant:
	var loc_dict = state.get("loc", {})
	var robot_loc = get_location(loc_dict.get(robot))
	if robot_loc.size() < 2 or robot_loc[0] != 1 or robot_loc[1] != 1:
		return null
	var new_state = state.duplicate(true)
	var has_medicine_dict = new_state.get("has_medicine", {})
	has_medicine_dict[robot] = 2
	new_state["has_medicine"] = has_medicine_dict
	return new_state

static func action_transfer(state: Dictionary, robot_from: String, robot_to: String) -> Variant:
	var loc_dict = state.get("loc", {})
	var loc_from = get_location(loc_dict.get(robot_from))
	var loc_to = get_location(loc_dict.get(robot_to))
	if loc_from.size() < 2 or loc_to.size() < 2 or loc_from[0] != loc_to[0] or loc_from[1] != loc_to[1]:
		return null
	var has_medicine_dict = state.get("has_medicine", {})
	var medicine_from = get_int_from_dict(has_medicine_dict, robot_from)
	if medicine_from <= 0:
		return null
	var new_state = state.duplicate(true)
	var new_medicine_dict = new_state.get("has_medicine", {})
	var medicine_to = get_int_from_dict(has_medicine_dict, robot_to)
	new_medicine_dict[robot_from] = medicine_from - 1
	new_medicine_dict[robot_to] = medicine_to + 1
	new_state["has_medicine"] = new_medicine_dict
	return new_state

static func sense_image(state: Dictionary, robot: String, camera: String, location: Variant) -> Dictionary:
	var img = {"loc": null, "person": null}
	var weather_dict = state.get("weather", {})
	var loc_key = str(location)
	var weather = get_string_from_dict(weather_dict, loc_key)
	var altitude_dict = state.get("altitude", {})
	var robot_altitude = get_string_from_dict(altitude_dict, robot)
	var visibility = false
	if weather == "rainy":
		if camera == "bottom_camera" and robot_altitude == "low":
			visibility = true
	elif weather == "foggy":
		if camera == "front_camera" and robot_altitude == "low":
			visibility = true
	elif weather == "dusts_storm":
		if robot_altitude == "low":
			visibility = true
	elif weather == "clear":
		visibility = true
	if visibility:
		img["loc"] = location
		var real_person_dict = state.get("real_person", {})
		if real_person_dict.has(loc_key):
			img["person"] = real_person_dict[loc_key]
	return img

static func action_capture_image(state: Dictionary, robot: String, camera: String, location: Variant) -> Dictionary:
	var new_state = state.duplicate(true)
	var current_image_dict = new_state.get("current_image", {})
	var img = sense_image(state, robot, camera, location)
	current_image_dict[robot] = img
	new_state["current_image"] = current_image_dict
	return new_state

static func action_change_altitude(state: Dictionary, robot: String, new_altitude: String) -> Dictionary:
	var new_state = state.duplicate(true)
	var altitude_dict = new_state.get("altitude", {})
	altitude_dict[robot] = new_altitude
	new_state["altitude"] = altitude_dict
	return new_state

static func action_check_real(state: Dictionary, location: Variant) -> Dictionary:
	var loc_key = str(location)
	var real_person_dict = state.get("real_person", {})
	if not real_person_dict.has(loc_key):
		return state
	var person = real_person_dict[loc_key]
	if person == "":
		return state
	var real_status_dict = state.get("real_status", {})
	var person_status = get_string_from_dict(real_status_dict, person)
	var loc_status = get_string_from_dict(real_status_dict, loc_key)
	if person_status == "injured" or person_status == "dead" or loc_status == "has_debri":
		return action_die_update(state, person)
	return state

static func action_engage_robot(state: Dictionary, robot: String) -> Dictionary:
	var new_state = state.duplicate(true)
	var status_dict = new_state.get("status", {})
	status_dict[robot] = "busy"
	new_state["status"] = status_dict
	var new_robot_dict = new_state.get("new_robot", {})
	new_robot_dict[1] = robot
	new_state["new_robot"] = new_robot_dict
	return new_state

static func action_force_engage_robot(state: Dictionary) -> Variant:
	var rigid = state.get("rigid", {})
	var wheeled_robots = rigid.get("wheeled_robots", [])
	if wheeled_robots.size() == 0:
		return null
	return action_engage_robot(state, wheeled_robots[0])

# Task Methods

static func task_move_method_euclidean(state: Dictionary, robot: Variant, location: Variant) -> Variant:
	var robot_str = str(robot)
	var target_loc = get_location(location)
	var loc_dict = state.get("loc", {})
	var current_loc = get_location(loc_dict.get(robot_str))
	if current_loc == target_loc:
		return []
	var robot_type_dict = state.get("robot_type", {})
	if get_string_from_dict(robot_type_dict, robot_str) != "wheeled":
		return null
	return [["action_move_euclidean", robot_str, current_loc, target_loc, null]]

static func task_move_method_manhattan(state: Dictionary, robot: Variant, location: Variant) -> Variant:
	var robot_str = str(robot)
	var target_loc = get_location(location)
	var loc_dict = state.get("loc", {})
	var current_loc = get_location(loc_dict.get(robot_str))
	if current_loc == target_loc:
		return []
	var robot_type_dict = state.get("robot_type", {})
	if get_string_from_dict(robot_type_dict, robot_str) != "wheeled":
		return null
	return [["action_move_manhattan", robot_str, current_loc, target_loc, null]]

static func task_move_method_curved(state: Dictionary, robot: Variant, location: Variant) -> Variant:
	var robot_str = str(robot)
	var target_loc = get_location(location)
	var loc_dict = state.get("loc", {})
	var current_loc = get_location(loc_dict.get(robot_str))
	if current_loc == target_loc:
		return []
	var robot_type_dict = state.get("robot_type", {})
	if get_string_from_dict(robot_type_dict, robot_str) != "wheeled":
		return null
	return [["action_move_curved", robot_str, current_loc, target_loc, null]]

static func task_move_method_fly(state: Dictionary, robot: Variant, location: Variant) -> Variant:
	var robot_str = str(robot)
	var target_loc = get_location(location)
	var loc_dict = state.get("loc", {})
	var current_loc = get_location(loc_dict.get(robot_str))
	if current_loc == target_loc:
		return []
	var robot_type_dict = state.get("robot_type", {})
	if get_string_from_dict(robot_type_dict, robot_str) != "uav":
		return null
	return [["action_move_fly", robot_str, current_loc, target_loc]]

static func task_move_method_alt_fly(state: Dictionary, robot: Variant, location: Variant) -> Variant:
	var robot_str = str(robot)
	var target_loc = get_location(location)
	var loc_dict = state.get("loc", {})
	var current_loc = get_location(loc_dict.get(robot_str))
	if current_loc == target_loc:
		return []
	var robot_type_dict = state.get("robot_type", {})
	if get_string_from_dict(robot_type_dict, robot_str) != "uav":
		return null
	return [["action_move_alt_fly", robot_str, current_loc, target_loc]]

static func task_new_robot_encap(state: Dictionary, person: Variant) -> Variant:
	var new_robot_dict = state.get("new_robot", {})
	var r2 = new_robot_dict.get(1)
	if r2 == null:
		return null
	var robot_str = str(r2)
	var subtasks = []
	var has_medicine_dict = state.get("has_medicine", {})
	if get_int_from_dict(has_medicine_dict, robot_str) == 0:
		subtasks.append(["get_supplies_task", robot_str])
	subtasks.append(["help_person_task", robot_str, str(person)])
	subtasks.append(["action_free_robot", robot_str])
	return subtasks

static func task_rescue_method_ground(state: Dictionary, robot: Variant, person: Variant) -> Variant:
	var robot_str = str(robot)
	var robot_type_dict = state.get("robot_type", {})
	if get_string_from_dict(robot_type_dict, robot_str) == "uav":
		return null
	var subtasks = []
	var has_medicine_dict = state.get("has_medicine", {})
	if get_int_from_dict(has_medicine_dict, robot_str) == 0:
		subtasks.append(["get_supplies_task", robot_str])
	subtasks.append(["help_person_task", robot_str, str(person)])
	return subtasks

static func task_rescue_method_uav(state: Dictionary, robot: Variant, person: Variant) -> Variant:
	var robot_str = str(robot)
	var robot_type_dict = state.get("robot_type", {})
	if get_string_from_dict(robot_type_dict, robot_str) != "uav":
		return null
	return [["get_robot_task"], ["new_robot_encap_task", str(person)]]

static func task_support_person_method_1(state: Dictionary, robot: Variant, person: Variant) -> Variant:
	var person_str = str(person)
	var status_dict = state.get("status", {})
	if get_string_from_dict(status_dict, person_str) != "injured":
		return null
	return [["action_support_person", str(robot), person_str]]

static func task_support_person_method_2(state: Dictionary, robot: Variant, person: Variant) -> Variant:
	var person_str = str(person)
	var status_dict = state.get("status", {})
	if get_string_from_dict(status_dict, person_str) != "injured":
		return null
	return [["action_support_person_2", str(robot), person_str]]

static func task_support_person_method_3(state: Dictionary, robot: Variant, person: Variant) -> Variant:
	var person_str = str(person)
	var loc_dict = state.get("loc", {})
	var person_loc = loc_dict.get(person_str)
	var loc_key = str(person_loc)
	var status_dict = state.get("status", {})
	if get_string_from_dict(status_dict, loc_key) != "has_debri":
		return null
	return [["action_clear_location", str(robot), person_loc]]

static func task_support_person_method_4(state: Dictionary, robot: Variant, person: Variant) -> Variant:
	var person_str = str(person)
	var loc_dict = state.get("loc", {})
	var person_loc = loc_dict.get(person_str)
	var loc_key = str(person_loc)
	var status_dict = state.get("status", {})
	if get_string_from_dict(status_dict, loc_key) != "has_debri":
		return null
	return [["action_clear_location_2", str(robot), person_loc]]

static func task_support_person_method_5(state: Dictionary, robot: Variant, person: Variant) -> Variant:
	var person_str = str(person)
	var loc_dict = state.get("loc", {})
	var person_loc = loc_dict.get(person_str)
	return [["action_check_real", person_loc]]

static func task_help_person(state: Dictionary, robot: Variant, person: Variant) -> Variant:
	var robot_str = str(robot)
	var person_str = str(person)
	var loc_dict = state.get("loc", {})
	var person_loc = loc_dict.get(person_str)
	return [
		["move_task", robot_str, person_loc],
		["action_inspect_location", robot_str, person_loc],
		["action_inspect_person", robot_str, person_str],
		["support_task", robot_str, person_str]
	]

static func task_get_supplies_method_robot(state: Dictionary, robot: Variant) -> Variant:
	var robot_str = str(robot)
	var loc_dict = state.get("loc", {})
	var robot_loc = get_location(loc_dict.get(robot_str))
	var rigid = state.get("rigid", {})
	var wheeled_robots = rigid.get("wheeled_robots", [])
	var has_medicine_dict = state.get("has_medicine", {})
	var nearest_robot = ""
	var nearest_dist = INF
	for r1 in wheeled_robots:
		if get_int_from_dict(has_medicine_dict, r1) > 0:
			var r1_loc = get_location(loc_dict.get(r1))
			var dist = euclidean_distance(robot_loc, r1_loc)
			if dist < nearest_dist:
				nearest_dist = dist
				nearest_robot = r1
	if nearest_robot == "":
		return null
	return [
		["move_task", robot_str, loc_dict[nearest_robot]],
		["action_transfer", nearest_robot, robot_str]
	]

static func task_get_supplies_method_base(state: Dictionary, robot: Variant) -> Variant:
	var robot_str = str(robot)
	var base_loc = [1, 1]
	return [
		["move_task", robot_str, base_loc],
		["action_replenish_supplies", robot_str]
	]

static func task_rescue_encap(state: Dictionary, robot: Variant) -> Variant:
	var robot_str = str(robot)
	var current_image_dict = state.get("current_image", {})
	var img = current_image_dict.get(robot_str, {})
	var person = img.get("person")
	if person == null:
		return []
	return [["rescue_task", robot_str, str(person)]]

static func task_survey_method_front(state: Dictionary, robot: Variant, location: Variant) -> Variant:
	var robot_str = str(robot)
	var robot_type_dict = state.get("robot_type", {})
	if get_string_from_dict(robot_type_dict, robot_str) != "uav":
		return null
	return [
		["adjust_altitude_task", robot_str],
		["action_capture_image", robot_str, "front_camera", location],
		["rescue_encap_task", robot_str],
		["action_check_real", location]
	]

static func task_survey_method_bottom(state: Dictionary, robot: Variant, location: Variant) -> Variant:
	var robot_str = str(robot)
	var robot_type_dict = state.get("robot_type", {})
	if get_string_from_dict(robot_type_dict, robot_str) != "uav":
		return null
	return [
		["adjust_altitude_task", robot_str],
		["action_capture_image", robot_str, "bottom_camera", location],
		["rescue_encap_task", robot_str],
		["action_check_real", location]
	]

static func task_get_robot_method_free(state: Dictionary) -> Variant:
	var rigid = state.get("rigid", {})
	var wheeled_robots = rigid.get("wheeled_robots", [])
	var status_dict = state.get("status", {})
	var loc_dict = state.get("loc", {})
	var base_loc = [1, 1]
	var nearest_robot = ""
	var nearest_dist = INF
	for r in wheeled_robots:
		if get_string_from_dict(status_dict, r) == "free":
			var r_loc = get_location(loc_dict.get(r))
			var dist = euclidean_distance(r_loc, base_loc)
			if dist < nearest_dist:
				nearest_dist = dist
				nearest_robot = r
	if nearest_robot == "":
		return null
	return [["action_engage_robot", nearest_robot]]

static func task_get_robot_method_force(state: Dictionary) -> Variant:
	return [["action_force_engage_robot"]]

static func task_adjust_altitude_method_low(state: Dictionary, robot: Variant) -> Variant:
	var robot_str = str(robot)
	var altitude_dict = state.get("altitude", {})
	if get_string_from_dict(altitude_dict, robot_str) != "high":
		return null
	return [["action_change_altitude", robot_str, "low"]]

static func task_adjust_altitude_method_high(state: Dictionary, robot: Variant) -> Variant:
	var robot_str = str(robot)
	var altitude_dict = state.get("altitude", {})
	if get_string_from_dict(altitude_dict, robot_str) != "low":
		return null
	return [["action_change_altitude", robot_str, "high"]]
