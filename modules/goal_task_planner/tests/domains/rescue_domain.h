/**************************************************************************/
/*  rescue_domain.h                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/io/file_access.h"
#include "core/variant/callable.h"
#include <cmath>
#include <limits>

// Wrapper class used to construct Callables for free functions in RescueDomain.
class RescueDomainCallable {
public:
	// Actions - movement
	static Variant action_free_robot(Dictionary p_state, Variant p_robot);
	static Variant action_die_update(Dictionary p_state, Variant p_person);
	static Variant action_move_euclidean(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc, Variant p_dist);
	static Variant action_move_manhattan(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc, Variant p_dist);
	static Variant action_move_curved(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc, Variant p_dist);
	static Variant action_move_fly(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc);
	static Variant action_move_alt_fly(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc);

	// Actions - inspection and support
	static Variant action_inspect_person(Dictionary p_state, Variant p_robot, Variant p_person);
	static Variant action_support_person(Dictionary p_state, Variant p_robot, Variant p_person);
	static Variant action_support_person_2(Dictionary p_state, Variant p_robot, Variant p_person);
	static Variant action_inspect_location(Dictionary p_state, Variant p_robot, Variant p_location);
	static Variant action_clear_location(Dictionary p_state, Variant p_robot, Variant p_location);
	static Variant action_clear_location_2(Dictionary p_state, Variant p_robot, Variant p_location);

	// Actions - supplies and transfer
	static Variant action_replenish_supplies(Dictionary p_state, Variant p_robot);
	static Variant action_transfer(Dictionary p_state, Variant p_robot_from, Variant p_robot_to);

	// Actions - image capture and altitude
	static Variant action_capture_image(Dictionary p_state, Variant p_robot, Variant p_camera, Variant p_location);
	static Variant action_change_altitude(Dictionary p_state, Variant p_robot, Variant p_new_altitude);
	static Variant action_check_real(Dictionary p_state, Variant p_location);

	// Actions - robot engagement
	static Variant action_engage_robot(Dictionary p_state, Variant p_robot);
	static Variant action_force_engage_robot(Dictionary p_state);

	// Task methods
	static Variant task_move_method_euclidean(Dictionary p_state, Variant p_robot, Variant p_location);
	static Variant task_move_method_manhattan(Dictionary p_state, Variant p_robot, Variant p_location);
	static Variant task_move_method_curved(Dictionary p_state, Variant p_robot, Variant p_location);
	static Variant task_move_method_fly(Dictionary p_state, Variant p_robot, Variant p_location);
	static Variant task_move_method_alt_fly(Dictionary p_state, Variant p_robot, Variant p_location);

	static Variant task_new_robot_encap(Dictionary p_state, Variant p_person);

	static Variant task_rescue_method_ground(Dictionary p_state, Variant p_robot, Variant p_person);
	static Variant task_rescue_method_uav(Dictionary p_state, Variant p_robot, Variant p_person);

	static Variant task_support_person_method_1(Dictionary p_state, Variant p_robot, Variant p_person);
	static Variant task_support_person_method_2(Dictionary p_state, Variant p_robot, Variant p_person);
	static Variant task_support_person_method_3(Dictionary p_state, Variant p_robot, Variant p_person);
	static Variant task_support_person_method_4(Dictionary p_state, Variant p_robot, Variant p_person);
	static Variant task_support_person_method_5(Dictionary p_state, Variant p_robot, Variant p_person);

	static Variant task_help_person(Dictionary p_state, Variant p_robot, Variant p_person);

	static Variant task_get_supplies_method_robot(Dictionary p_state, Variant p_robot);
	static Variant task_get_supplies_method_base(Dictionary p_state, Variant p_robot);

	static Variant task_rescue_encap(Dictionary p_state, Variant p_robot);

	static Variant task_survey_method_front(Dictionary p_state, Variant p_robot, Variant p_location);
	static Variant task_survey_method_bottom(Dictionary p_state, Variant p_robot, Variant p_location);

	static Variant task_get_robot_method_free(Dictionary p_state);
	static Variant task_get_robot_method_force(Dictionary p_state);

	static Variant task_adjust_altitude_method_low(Dictionary p_state, Variant p_robot);
	static Variant task_adjust_altitude_method_high(Dictionary p_state, Variant p_robot);
};

// Helper functions for rescue domain
namespace RescueDomain {

// Helper: Get location tuple from Variant (Array of 2 ints)
Array get_location(Variant p_loc) {
	if (p_loc.get_type() == Variant::ARRAY) {
		Array arr = p_loc;
		if (arr.size() >= 2) {
			return arr;
		}
	}
	return Array();
}

// Helper: Check if location is tuple (x, y)
bool is_location_tuple(Variant p_loc) {
	if (p_loc.get_type() == Variant::ARRAY) {
		Array arr = p_loc;
		return arr.size() == 2;
	}
	return false;
}

// Helper: Get string value from nested dictionary (e.g., state["status"][key])
String get_string_from_dict(Dictionary p_dict, Variant p_key, const String &p_default = "") {
	if (!p_dict.has(p_key)) {
		return p_default;
	}
	Variant val = p_dict[p_key];
	if (val.get_type() == Variant::STRING) {
		return val;
	}
	return p_default;
}

// Helper: Get int value from nested dictionary (e.g., state["has_medicine"][key])
int get_int_from_dict(Dictionary p_dict, Variant p_key, int p_default = 0) {
	if (!p_dict.has(p_key)) {
		return p_default;
	}
	Variant val = p_dict[p_key];
	if (val.get_type() == Variant::INT) {
		return val;
	}
	return p_default;
}

// Helper: Calculate euclidean distance between two locations
double euclidean_distance(Array loc1, Array loc2) {
	if (loc1.size() < 2 || loc2.size() < 2) {
		return 0.0;
	}
	double x1 = loc1[0];
	double y1 = loc1[1];
	double x2 = loc2[0];
	double y2 = loc2[1];
	return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Helper: Check if obstacle is in path (simplified - matches Python logic)
bool obstacle_in_path(Array from_loc, Array to_loc, Array obstacles) {
	if (from_loc.size() < 2 || to_loc.size() < 2) {
		return false;
	}

	double x1 = from_loc[0];
	double y1 = from_loc[1];
	double x2 = to_loc[0];
	double y2 = to_loc[1];

	double x_low = (x1 < x2) ? x1 : x2;
	double x_high = (x1 > x2) ? x1 : x2;
	double y_low = (y1 < y2) ? y1 : y2;
	double y_high = (y1 > y2) ? y1 : y2;

	for (int i = 0; i < obstacles.size(); i++) {
		Array obs = obstacles[i];
		if (obs.size() < 2) {
			continue;
		}
		double ox = obs[0];
		double oy = obs[1];

		if (x_low <= ox && ox <= x_high && y_low <= oy && oy <= y_high) {
			return true;
		}
	}
	return false;
}

// Action: Free robot (set status to 'free')
Dictionary action_free_robot(Dictionary state, String robot) {
	Dictionary new_state = state.duplicate();
	Dictionary status_dict;
	if (new_state.has("status")) {
		status_dict = new_state["status"];
	} else {
		status_dict = Dictionary();
	}
	status_dict[robot] = "free";
	new_state["status"] = status_dict;
	return new_state;
}

// Action: Die update (mark person as dead)
Variant action_die_update(Dictionary state, String person) {
	Dictionary new_state = state.duplicate();
	Dictionary status_dict;
	if (new_state.has("status")) {
		status_dict = new_state["status"];
	} else {
		status_dict = Dictionary();
	}
	Dictionary real_status_dict;
	if (new_state.has("real_status")) {
		real_status_dict = new_state["real_status"];
	} else {
		real_status_dict = Dictionary();
	}
	status_dict[person] = "dead";
	real_status_dict[person] = "dead";
	new_state["status"] = status_dict;
	new_state["real_status"] = real_status_dict;
	return new_state;
}

// Action: Move euclidean (straight path with obstacle checking)
Variant action_move_euclidean(Dictionary state, String robot, Array from_loc, Array to_loc, Variant dist) {
	if (from_loc.size() < 2 || to_loc.size() < 2) {
		return Variant(); // Invalid locations
	}

	// Check if already at destination
	if (from_loc == to_loc) {
		return state;
	}

	// Get robot location from state
	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Variant();
	}

	Array current_loc = get_location(loc_dict[robot]);
	if (current_loc.size() < 2) {
		return Variant();
	}

	// Check if robot is at from_loc
	if (current_loc[0] != from_loc[0] || current_loc[1] != from_loc[1]) {
		return Variant();
	}

	// Check obstacles
	Dictionary rigid;
	if (state.has("rigid")) {
		rigid = state["rigid"];
	} else {
		return Variant();
	}

	Array obstacles;
	if (rigid.has("obstacles")) {
		obstacles = rigid["obstacles"];
	} else {
		obstacles = Array();
	}

	// Check if obstacle blocks path (simplified euclidean check)
	double x1 = from_loc[0];
	double y1 = from_loc[1];
	double x2 = to_loc[0];
	double y2 = to_loc[1];

	double x_low = (x1 < x2) ? x1 : x2;
	double x_high = (x1 > x2) ? x1 : x2;
	double y_low = (y1 < y2) ? y1 : y2;
	double y_high = (y1 > y2) ? y1 : y2;

	for (int i = 0; i < obstacles.size(); i++) {
		Array obs = obstacles[i];
		if (obs.size() < 2) {
			continue;
		}
		double ox = obs[0];
		double oy = obs[1];

		if (x_low <= ox && ox <= x_high && y_low <= oy && oy <= y_high) {
			// Check if obstacle is on the line (same slope)
			if (std::abs(x2 - x1) > 0.0001) {
				double slope = (y2 - y1) / (x2 - x1);
				double expected_y = y1 + slope * (ox - x1);
				if (std::abs(oy - expected_y) <= 0.0001) {
					return Variant(); // Obstacle blocks path
				}
			} else if (std::abs(ox - x1) <= 0.0001) {
				return Variant(); // Vertical line with obstacle
			}
		}
	}

	// Move robot
	Dictionary new_state = state.duplicate();
	Dictionary new_loc_dict = loc_dict.duplicate();
	new_loc_dict[robot] = to_loc;
	new_state["loc"] = new_loc_dict;
	return new_state;
}

// Action: Move manhattan (L-shaped path with obstacle checking)
Variant action_move_manhattan(Dictionary state, String robot, Array from_loc, Array to_loc, Variant dist) {
	if (from_loc.size() < 2 || to_loc.size() < 2) {
		return Variant();
	}

	if (from_loc == to_loc) {
		return state;
	}

	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Variant();
	}

	Array current_loc = get_location(loc_dict[robot]);
	if (current_loc.size() < 2) {
		return Variant();
	}

	if (current_loc[0] != from_loc[0] || current_loc[1] != from_loc[1]) {
		return Variant();
	}

	// Check obstacles for manhattan path
	Dictionary rigid;
	if (state.has("rigid")) {
		rigid = state["rigid"];
	} else {
		return Variant();
	}

	Array obstacles;
	if (rigid.has("obstacles")) {
		obstacles = rigid["obstacles"];
	} else {
		obstacles = Array();
	}

	double x1 = from_loc[0];
	double y1 = from_loc[1];
	double x2 = to_loc[0];
	double y2 = to_loc[1];

	double x_low = (x1 < x2) ? x1 : x2;
	double x_high = (x1 > x2) ? x1 : x2;
	double y_low = (y1 < y2) ? y1 : y2;
	double y_high = (y1 > y2) ? y1 : y2;

	for (int i = 0; i < obstacles.size(); i++) {
		Array obs = obstacles[i];
		if (obs.size() < 2) {
			continue;
		}
		double ox = obs[0];
		double oy = obs[1];

		// Manhattan path checks: horizontal then vertical, or vertical then horizontal
		if (std::abs(oy - y1) <= 0.0001 && x_low <= ox && ox <= x_high) {
			return Variant(); // Obstacle on horizontal segment
		}
		if (std::abs(ox - x2) <= 0.0001 && y_low <= oy && oy <= y_high) {
			return Variant(); // Obstacle on vertical segment
		}
	}

	Dictionary new_state = state.duplicate();
	Dictionary new_loc_dict = loc_dict.duplicate();
	new_loc_dict[robot] = to_loc;
	new_state["loc"] = new_loc_dict;
	return new_state;
}

// Action: Move curved (arc path with obstacle checking)
Variant action_move_curved(Dictionary state, String robot, Array from_loc, Array to_loc, Variant dist) {
	if (from_loc.size() < 2 || to_loc.size() < 2) {
		return Variant();
	}

	if (from_loc == to_loc) {
		return state;
	}

	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Variant();
	}

	Array current_loc = get_location(loc_dict[robot]);
	if (current_loc.size() < 2) {
		return Variant();
	}

	if (current_loc[0] != from_loc[0] || current_loc[1] != from_loc[1]) {
		return Variant();
	}

	// Check obstacles for curved path (arc with center at midpoint)
	Dictionary rigid;
	if (state.has("rigid")) {
		rigid = state["rigid"];
	} else {
		return Variant();
	}

	Array obstacles;
	if (rigid.has("obstacles")) {
		obstacles = rigid["obstacles"];
	} else {
		obstacles = Array();
	}

	double x1 = from_loc[0];
	double y1 = from_loc[1];
	double x2 = to_loc[0];
	double y2 = to_loc[1];
	double center_x = (x1 + x2) / 2.0;
	double center_y = (y1 + y2) / 2.0;

	double r2 = (x2 - center_x) * (x2 - center_x) + (y2 - center_y) * (y2 - center_y);

	for (int i = 0; i < obstacles.size(); i++) {
		Array obs = obstacles[i];
		if (obs.size() < 2) {
			continue;
		}
		double ox = obs[0];
		double oy = obs[1];
		double ro = (ox - center_x) * (ox - center_x) + (oy - center_y) * (oy - center_y);

		if (std::abs(r2 - ro) <= 0.0001) {
			return Variant(); // Obstacle on arc
		}
	}

	Dictionary new_state = state.duplicate();
	Dictionary new_loc_dict = loc_dict.duplicate();
	new_loc_dict[robot] = to_loc;
	new_state["loc"] = new_loc_dict;
	return new_state;
}

// Action: Move fly (UAV movement, no obstacle checking)
Variant action_move_fly(Dictionary state, String robot, Array from_loc, Array to_loc) {
	if (from_loc.size() < 2 || to_loc.size() < 2) {
		return Variant();
	}

	if (from_loc == to_loc) {
		return state;
	}

	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Variant();
	}

	Array current_loc = get_location(loc_dict[robot]);
	if (current_loc.size() < 2) {
		return Variant();
	}

	if (current_loc[0] != from_loc[0] || current_loc[1] != from_loc[1]) {
		return Variant();
	}

	Dictionary new_state = state.duplicate();
	Dictionary new_loc_dict = loc_dict.duplicate();
	new_loc_dict[robot] = to_loc;
	new_state["loc"] = new_loc_dict;
	return new_state;
}

// Action: Move alt fly (alternative UAV movement)
Variant action_move_alt_fly(Dictionary state, String robot, Array from_loc, Array to_loc) {
	return action_move_fly(state, robot, from_loc, to_loc); // Same as move_fly
}

// Action: Inspect person (update status from real_status)
Dictionary action_inspect_person(Dictionary state, String robot, String person) {
	Dictionary new_state = state.duplicate();
	Dictionary status_dict;
	if (new_state.has("status")) {
		status_dict = new_state["status"];
	} else {
		status_dict = Dictionary();
	}
	Dictionary real_status_dict;
	if (new_state.has("real_status")) {
		real_status_dict = new_state["real_status"];
	} else {
		real_status_dict = Dictionary();
	}

	if (real_status_dict.has(person)) {
		status_dict[person] = real_status_dict[person];
	}
	new_state["status"] = status_dict;
	return new_state;
}

// Action: Support person (heal injured person)
Variant action_support_person(Dictionary state, String robot, String person) {
	Dictionary status_dict;
	if (state.has("status")) {
		status_dict = state["status"];
	} else {
		return Variant();
	}

	String person_status = get_string_from_dict(status_dict, person);
	if (person_status == "dead") {
		return Variant(); // Can't support dead person
	}

	Dictionary new_state = state.duplicate();
	Dictionary new_status_dict = status_dict.duplicate();
	Dictionary real_status_dict;
	if (new_state.has("real_status")) {
		real_status_dict = new_state["real_status"];
	} else {
		real_status_dict = Dictionary();
	}

	new_status_dict[person] = "OK";
	real_status_dict[person] = "OK";
	new_state["status"] = new_status_dict;
	new_state["real_status"] = real_status_dict;
	return new_state;
}

// Action: Support person 2 (alternative support method)
Variant action_support_person_2(Dictionary state, String robot, String person) {
	return action_support_person(state, robot, person); // Same as support_person
}

// Action: Inspect location (update location status from real_status)
Dictionary action_inspect_location(Dictionary state, String robot, Variant location) {
	Dictionary new_state = state.duplicate();
	Dictionary status_dict;
	if (new_state.has("status")) {
		status_dict = new_state["status"];
	} else {
		status_dict = Dictionary();
	}
	Dictionary real_status_dict;
	if (new_state.has("real_status")) {
		real_status_dict = new_state["real_status"];
	} else {
		real_status_dict = Dictionary();
	}

	String loc_key = String(location);
	if (real_status_dict.has(loc_key)) {
		status_dict[loc_key] = real_status_dict[loc_key];
	}
	new_state["status"] = status_dict;
	return new_state;
}

// Action: Clear location (remove debris)
Dictionary action_clear_location(Dictionary state, String robot, Variant location) {
	Dictionary new_state = state.duplicate();
	Dictionary status_dict;
	if (new_state.has("status")) {
		status_dict = new_state["status"];
	} else {
		status_dict = Dictionary();
	}
	Dictionary real_status_dict;
	if (new_state.has("real_status")) {
		real_status_dict = new_state["real_status"];
	} else {
		real_status_dict = Dictionary();
	}

	String loc_key = String(location);
	status_dict[loc_key] = "clear";
	real_status_dict[loc_key] = "clear";
	new_state["status"] = status_dict;
	new_state["real_status"] = real_status_dict;
	return new_state;
}

// Action: Clear location 2 (alternative clear method)
Dictionary action_clear_location_2(Dictionary state, String robot, Variant location) {
	return action_clear_location(state, robot, location); // Same as clear_location
}

// Action: Replenish supplies (at base location (1, 1))
Variant action_replenish_supplies(Dictionary state, String robot) {
	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Variant();
	}

	Array robot_loc = get_location(loc_dict[robot]);
	if (robot_loc.size() < 2) {
		return Variant();
	}

	// Check if at base (1, 1)
	int x = robot_loc[0];
	int y = robot_loc[1];
	if (x != 1 || y != 1) {
		return Variant();
	}

	Dictionary new_state = state.duplicate();
	Dictionary has_medicine_dict;
	if (new_state.has("has_medicine")) {
		has_medicine_dict = new_state["has_medicine"];
	} else {
		has_medicine_dict = Dictionary();
	}
	has_medicine_dict[robot] = 2;
	new_state["has_medicine"] = has_medicine_dict;
	return new_state;
}

// Action: Transfer medicine between robots
Variant action_transfer(Dictionary state, String robot_from, String robot_to) {
	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Variant();
	}

	Array loc_from = get_location(loc_dict[robot_from]);
	Array loc_to = get_location(loc_dict[robot_to]);

	if (loc_from.size() < 2 || loc_to.size() < 2) {
		return Variant();
	}

	// Check if robots are at same location
	if (loc_from[0] != loc_to[0] || loc_from[1] != loc_to[1]) {
		return Variant();
	}

	Dictionary has_medicine_dict;
	if (state.has("has_medicine")) {
		has_medicine_dict = state["has_medicine"];
	} else {
		return Variant();
	}

	int medicine_from = get_int_from_dict(has_medicine_dict, robot_from);
	if (medicine_from <= 0) {
		return Variant();
	}

	Dictionary new_state = state.duplicate();
	Dictionary new_medicine_dict = has_medicine_dict.duplicate();
	int medicine_to = get_int_from_dict(has_medicine_dict, robot_to);
	new_medicine_dict[robot_from] = medicine_from - 1;
	new_medicine_dict[robot_to] = medicine_to + 1;
	new_state["has_medicine"] = new_medicine_dict;
	return new_state;
}

// Helper: Sense image (determines visibility based on weather and camera)
Dictionary sense_image(Dictionary state, String robot, String camera, Variant location) {
	Dictionary img;
	img["loc"] = Variant();
	img["person"] = Variant();

	Dictionary weather_dict;
	if (state.has("weather")) {
		weather_dict = state["weather"];
	} else {
		return img;
	}

	String loc_key = String(location);
	String weather = get_string_from_dict(weather_dict, loc_key);

	Dictionary altitude_dict;
	if (state.has("altitude")) {
		altitude_dict = state["altitude"];
	} else {
		return img;
	}
	String robot_altitude = get_string_from_dict(altitude_dict, robot);

	bool visibility = false;
	if (weather == "rainy") {
		if (camera == "bottom_camera" && robot_altitude == "low") {
			visibility = true;
		}
	} else if (weather == "foggy") {
		if (camera == "front_camera" && robot_altitude == "low") {
			visibility = true;
		}
	} else if (weather == "dusts_storm") {
		if (robot_altitude == "low") {
			visibility = true;
		}
	} else if (weather == "clear") {
		visibility = true;
	}

	if (visibility) {
		img["loc"] = location;
		Dictionary real_person_dict;
		if (state.has("real_person")) {
			real_person_dict = state["real_person"];
		} else {
			return img;
		}
		if (real_person_dict.has(loc_key)) {
			img["person"] = real_person_dict[loc_key];
		}
	}

	return img;
}

// Action: Capture image
Dictionary action_capture_image(Dictionary state, String robot, String camera, Variant location) {
	Dictionary new_state = state.duplicate();
	Dictionary current_image_dict;
	if (new_state.has("current_image")) {
		current_image_dict = new_state["current_image"];
	} else {
		current_image_dict = Dictionary();
	}

	Dictionary img = sense_image(state, robot, camera, location);
	current_image_dict[robot] = img;
	new_state["current_image"] = current_image_dict;
	return new_state;
}

// Action: Change altitude
Dictionary action_change_altitude(Dictionary state, String robot, String new_altitude) {
	Dictionary new_state = state.duplicate();
	Dictionary altitude_dict;
	if (new_state.has("altitude")) {
		altitude_dict = new_state["altitude"];
	} else {
		altitude_dict = Dictionary();
	}
	altitude_dict[robot] = new_altitude;
	new_state["altitude"] = altitude_dict;
	return new_state;
}

// Action: Check real (check if person dies based on conditions)
Variant action_check_real(Dictionary state, Variant location) {
	String loc_key = String(location);
	Dictionary real_person_dict;
	if (state.has("real_person")) {
		real_person_dict = state["real_person"];
	} else {
		return state;
	}

	if (!real_person_dict.has(loc_key)) {
		return state;
	}

	String person = real_person_dict[loc_key];
	if (person.is_empty()) {
		return state;
	}

	Dictionary real_status_dict;
	if (state.has("real_status")) {
		real_status_dict = state["real_status"];
	} else {
		return state;
	}

	String person_status = get_string_from_dict(real_status_dict, person);
	String loc_status = get_string_from_dict(real_status_dict, loc_key);

	if (person_status == "injured" || person_status == "dead" || loc_status == "has_debri") {
		return action_die_update(state, person);
	}

	return state;
}

// Action: Engage robot (set robot to busy and assign to new_robot[1])
Dictionary action_engage_robot(Dictionary state, String robot) {
	Dictionary new_state = state.duplicate();
	Dictionary status_dict;
	if (new_state.has("status")) {
		status_dict = new_state["status"];
	} else {
		status_dict = Dictionary();
	}
	status_dict[robot] = "busy";
	new_state["status"] = status_dict;

	Dictionary new_robot_dict;
	if (new_state.has("new_robot")) {
		new_robot_dict = new_state["new_robot"];
	} else {
		new_robot_dict = Dictionary();
	}
	new_robot_dict[1] = robot;
	new_state["new_robot"] = new_robot_dict;
	return new_state;
}

// Action: Force engage robot (force engage first wheeled robot)
Variant action_force_engage_robot(Dictionary state) {
	Dictionary rigid;
	if (state.has("rigid")) {
		rigid = state["rigid"];
	} else {
		return Variant();
	}

	Array wheeled_robots;
	if (rigid.has("wheeled_robots")) {
		wheeled_robots = rigid["wheeled_robots"];
	} else {
		return Variant();
	}

	if (wheeled_robots.size() == 0) {
		return Variant();
	}

	String robot = wheeled_robots[0];
	Dictionary new_state = action_engage_robot(state, robot);
	return new_state;
}

} // namespace RescueDomain

// Implementations of RescueDomainCallable static methods
inline Variant RescueDomainCallable::action_free_robot(Dictionary p_state, Variant p_robot) {
	return RescueDomain::action_free_robot(p_state, String(p_robot));
}

inline Variant RescueDomainCallable::action_die_update(Dictionary p_state, Variant p_person) {
	return RescueDomain::action_die_update(p_state, String(p_person));
}

inline Variant RescueDomainCallable::action_move_euclidean(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc, Variant p_dist) {
	return RescueDomain::action_move_euclidean(p_state, String(p_robot), Array(p_from_loc), Array(p_to_loc), p_dist);
}

inline Variant RescueDomainCallable::action_move_manhattan(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc, Variant p_dist) {
	return RescueDomain::action_move_manhattan(p_state, String(p_robot), Array(p_from_loc), Array(p_to_loc), p_dist);
}

inline Variant RescueDomainCallable::action_move_curved(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc, Variant p_dist) {
	return RescueDomain::action_move_curved(p_state, String(p_robot), Array(p_from_loc), Array(p_to_loc), p_dist);
}

inline Variant RescueDomainCallable::action_move_fly(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc) {
	return RescueDomain::action_move_fly(p_state, String(p_robot), Array(p_from_loc), Array(p_to_loc));
}

inline Variant RescueDomainCallable::action_move_alt_fly(Dictionary p_state, Variant p_robot, Variant p_from_loc, Variant p_to_loc) {
	return RescueDomain::action_move_alt_fly(p_state, String(p_robot), Array(p_from_loc), Array(p_to_loc));
}

inline Variant RescueDomainCallable::action_inspect_person(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::action_inspect_person(p_state, String(p_robot), String(p_person));
}

inline Variant RescueDomainCallable::action_support_person(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::action_support_person(p_state, String(p_robot), String(p_person));
}

inline Variant RescueDomainCallable::action_support_person_2(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::action_support_person_2(p_state, String(p_robot), String(p_person));
}

inline Variant RescueDomainCallable::action_inspect_location(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::action_inspect_location(p_state, String(p_robot), p_location);
}

inline Variant RescueDomainCallable::action_clear_location(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::action_clear_location(p_state, String(p_robot), p_location);
}

inline Variant RescueDomainCallable::action_clear_location_2(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::action_clear_location_2(p_state, String(p_robot), p_location);
}

inline Variant RescueDomainCallable::action_replenish_supplies(Dictionary p_state, Variant p_robot) {
	return RescueDomain::action_replenish_supplies(p_state, String(p_robot));
}

inline Variant RescueDomainCallable::action_transfer(Dictionary p_state, Variant p_robot_from, Variant p_robot_to) {
	return RescueDomain::action_transfer(p_state, String(p_robot_from), String(p_robot_to));
}

inline Variant RescueDomainCallable::action_capture_image(Dictionary p_state, Variant p_robot, Variant p_camera, Variant p_location) {
	return RescueDomain::action_capture_image(p_state, String(p_robot), String(p_camera), p_location);
}

inline Variant RescueDomainCallable::action_change_altitude(Dictionary p_state, Variant p_robot, Variant p_new_altitude) {
	return RescueDomain::action_change_altitude(p_state, String(p_robot), String(p_new_altitude));
}

inline Variant RescueDomainCallable::action_check_real(Dictionary p_state, Variant p_location) {
	return RescueDomain::action_check_real(p_state, p_location);
}

inline Variant RescueDomainCallable::action_engage_robot(Dictionary p_state, Variant p_robot) {
	return RescueDomain::action_engage_robot(p_state, String(p_robot));
}

inline Variant RescueDomainCallable::action_force_engage_robot(Dictionary p_state) {
	return RescueDomain::action_force_engage_robot(p_state);
}

// Task method implementations
namespace RescueDomain {

// Task method: Move euclidean
Variant task_move_method_euclidean(Dictionary state, Variant robot, Variant location) {
	String robot_str = String(robot);
	Array target_loc = get_location(location);

	Dictionary loc_dict = state["loc"];
	Array current_loc = get_location(loc_dict[robot_str]);

	if (current_loc == target_loc) {
		return Array(); // Already at location
	}

	Dictionary robot_type_dict = state["robot_type"];
	String robot_type = get_string_from_dict(robot_type_dict, robot_str);

	if (robot_type != "wheeled") {
		return Variant(); // Only for wheeled robots
	}

	Array subtasks;
	Array action;
	action.push_back("action_move_euclidean");
	action.push_back(robot_str);
	action.push_back(current_loc);
	action.push_back(target_loc);
	action.push_back(Variant()); // dist parameter (None)
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Move manhattan
Variant task_move_method_manhattan(Dictionary state, Variant robot, Variant location) {
	String robot_str = String(robot);
	Array target_loc = get_location(location);

	Dictionary loc_dict = state["loc"];
	Array current_loc = get_location(loc_dict[robot_str]);

	if (current_loc == target_loc) {
		return Array();
	}

	Dictionary robot_type_dict = state["robot_type"];
	String robot_type = get_string_from_dict(robot_type_dict, robot_str);

	if (robot_type != "wheeled") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_move_manhattan");
	action.push_back(robot_str);
	action.push_back(current_loc);
	action.push_back(target_loc);
	action.push_back(Variant());
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Move curved
Variant task_move_method_curved(Dictionary state, Variant robot, Variant location) {
	String robot_str = String(robot);
	Array target_loc = get_location(location);

	Dictionary loc_dict = state["loc"];
	Array current_loc = get_location(loc_dict[robot_str]);

	if (current_loc == target_loc) {
		return Array();
	}

	Dictionary robot_type_dict = state["robot_type"];
	String robot_type = get_string_from_dict(robot_type_dict, robot_str);

	if (robot_type != "wheeled") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_move_curved");
	action.push_back(robot_str);
	action.push_back(current_loc);
	action.push_back(target_loc);
	action.push_back(Variant());
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Move fly
Variant task_move_method_fly(Dictionary state, Variant robot, Variant location) {
	String robot_str = String(robot);
	Array target_loc = get_location(location);

	Dictionary loc_dict = state["loc"];
	Array current_loc = get_location(loc_dict[robot_str]);

	if (current_loc == target_loc) {
		return Array();
	}

	Dictionary robot_type_dict = state["robot_type"];
	String robot_type = get_string_from_dict(robot_type_dict, robot_str);

	if (robot_type != "uav") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_move_fly");
	action.push_back(robot_str);
	action.push_back(current_loc);
	action.push_back(target_loc);
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Move alt fly
Variant task_move_method_alt_fly(Dictionary state, Variant robot, Variant location) {
	String robot_str = String(robot);
	Array target_loc = get_location(location);

	Dictionary loc_dict = state["loc"];
	Array current_loc = get_location(loc_dict[robot_str]);

	if (current_loc == target_loc) {
		return Array();
	}

	Dictionary robot_type_dict = state["robot_type"];
	String robot_type = get_string_from_dict(robot_type_dict, robot_str);

	if (robot_type != "uav") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_move_alt_fly");
	action.push_back(robot_str);
	action.push_back(current_loc);
	action.push_back(target_loc);
	subtasks.push_back(action);
	return subtasks;
}

// Task method: New robot encap
Variant task_new_robot_encap(Dictionary state, Variant person) {
	Dictionary new_robot_dict = state["new_robot"];
	Variant r2 = new_robot_dict[1];

	if (r2.get_type() == Variant::NIL) {
		return Variant();
	}

	String robot_str = String(r2);
	Array subtasks;

	Dictionary has_medicine_dict = state["has_medicine"];
	int medicine = get_int_from_dict(has_medicine_dict, robot_str);

	if (medicine == 0) {
		Array get_supplies_task;
		get_supplies_task.push_back("get_supplies_task");
		get_supplies_task.push_back(robot_str);
		subtasks.push_back(get_supplies_task);
	}

	Array help_task;
	help_task.push_back("help_person_task");
	help_task.push_back(robot_str);
	help_task.push_back(String(person));
	subtasks.push_back(help_task);

	Array free_action;
	free_action.push_back("action_free_robot");
	free_action.push_back(robot_str);
	subtasks.push_back(free_action);

	return subtasks;
}

// Task method: Rescue (ground robot)
Variant task_rescue_method_ground(Dictionary state, Variant robot, Variant person) {
	String robot_str = String(robot);
	Dictionary robot_type_dict = state["robot_type"];
	String robot_type = get_string_from_dict(robot_type_dict, robot_str);

	if (robot_type == "uav") {
		return Variant();
	}

	Array subtasks;

	Dictionary has_medicine_dict = state["has_medicine"];
	int medicine = get_int_from_dict(has_medicine_dict, robot_str);

	if (medicine == 0) {
		Array get_supplies_task;
		get_supplies_task.push_back("get_supplies_task");
		get_supplies_task.push_back(robot_str);
		subtasks.push_back(get_supplies_task);
	}

	Array help_task;
	help_task.push_back("help_person_task");
	help_task.push_back(robot_str);
	help_task.push_back(String(person));
	subtasks.push_back(help_task);

	return subtasks;
}

// Task method: Rescue (UAV)
Variant task_rescue_method_uav(Dictionary state, Variant robot, Variant person) {
	String robot_str = String(robot);
	Dictionary robot_type_dict = state["robot_type"];
	String robot_type = get_string_from_dict(robot_type_dict, robot_str);

	if (robot_type != "uav") {
		return Variant();
	}

	Array subtasks;
	Array get_robot_task;
	get_robot_task.push_back("get_robot_task");
	subtasks.push_back(get_robot_task);

	Array new_robot_encap_task;
	new_robot_encap_task.push_back("new_robot_encap_task");
	new_robot_encap_task.push_back(String(person));
	subtasks.push_back(new_robot_encap_task);

	return subtasks;
}

// Task method: Support person method 1
Variant task_support_person_method_1(Dictionary state, Variant robot, Variant person) {
	String person_str = String(person);
	Dictionary status_dict = state["status"];
	String person_status = get_string_from_dict(status_dict, person_str);

	if (person_status != "injured") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_support_person");
	action.push_back(String(robot));
	action.push_back(person_str);
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Support person method 2
Variant task_support_person_method_2(Dictionary state, Variant robot, Variant person) {
	String person_str = String(person);
	Dictionary status_dict = state["status"];
	String person_status = get_string_from_dict(status_dict, person_str);

	if (person_status != "injured") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_support_person_2");
	action.push_back(String(robot));
	action.push_back(person_str);
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Support person method 3
Variant task_support_person_method_3(Dictionary state, Variant robot, Variant person) {
	String person_str = String(person);
	Dictionary loc_dict = state["loc"];
	Variant person_loc = loc_dict[person_str];
	String loc_key = String(person_loc);

	Dictionary status_dict = state["status"];
	String loc_status = get_string_from_dict(status_dict, loc_key);

	if (loc_status != "has_debri") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_clear_location");
	action.push_back(String(robot));
	action.push_back(person_loc);
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Support person method 4
Variant task_support_person_method_4(Dictionary state, Variant robot, Variant person) {
	String person_str = String(person);
	Dictionary loc_dict = state["loc"];
	Variant person_loc = loc_dict[person_str];
	String loc_key = String(person_loc);

	Dictionary status_dict = state["status"];
	String loc_status = get_string_from_dict(status_dict, loc_key);

	if (loc_status != "has_debri") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_clear_location_2");
	action.push_back(String(robot));
	action.push_back(person_loc);
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Support person method 5
Variant task_support_person_method_5(Dictionary state, Variant robot, Variant person) {
	String person_str = String(person);
	Dictionary loc_dict = state["loc"];
	Variant person_loc = loc_dict[person_str];

	Array subtasks;
	Array action;
	action.push_back("action_check_real");
	action.push_back(person_loc);
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Help person
Variant task_help_person(Dictionary state, Variant robot, Variant person) {
	String robot_str = String(robot);
	String person_str = String(person);
	Dictionary loc_dict = state["loc"];
	Variant person_loc = loc_dict[person_str];

	Array subtasks;
	Array move_task;
	move_task.push_back("move_task");
	move_task.push_back(robot_str);
	move_task.push_back(person_loc);
	subtasks.push_back(move_task);

	Array inspect_loc_action;
	inspect_loc_action.push_back("action_inspect_location");
	inspect_loc_action.push_back(robot_str);
	inspect_loc_action.push_back(person_loc);
	subtasks.push_back(inspect_loc_action);

	Array inspect_person_action;
	inspect_person_action.push_back("action_inspect_person");
	inspect_person_action.push_back(robot_str);
	inspect_person_action.push_back(person_str);
	subtasks.push_back(inspect_person_action);

	Array support_task;
	support_task.push_back("support_task");
	support_task.push_back(robot_str);
	support_task.push_back(person_str);
	subtasks.push_back(support_task);

	return subtasks;
}

// Task method: Get supplies from robot
Variant task_get_supplies_method_robot(Dictionary state, Variant robot) {
	String robot_str = String(robot);
	Dictionary loc_dict = state["loc"];
	Array robot_loc = get_location(loc_dict[robot_str]);

	Dictionary rigid = state["rigid"];
	Array wheeled_robots = rigid["wheeled_robots"];
	Dictionary has_medicine_dict = state["has_medicine"];

	String nearest_robot;
	double nearest_dist = std::numeric_limits<double>::infinity();

	for (int i = 0; i < wheeled_robots.size(); i++) {
		String r1 = wheeled_robots[i];
		int medicine = get_int_from_dict(has_medicine_dict, r1);

		if (medicine > 0) {
			Array r1_loc = get_location(loc_dict[r1]);
			double dist = euclidean_distance(robot_loc, r1_loc);

			if (dist < nearest_dist) {
				nearest_dist = dist;
				nearest_robot = r1;
			}
		}
	}

	if (nearest_robot.is_empty()) {
		return Variant();
	}

	Array subtasks;
	Array move_task;
	move_task.push_back("move_task");
	move_task.push_back(robot_str);
	move_task.push_back(loc_dict[nearest_robot]);
	subtasks.push_back(move_task);

	Array transfer_action;
	transfer_action.push_back("action_transfer");
	transfer_action.push_back(nearest_robot);
	transfer_action.push_back(robot_str);
	subtasks.push_back(transfer_action);

	return subtasks;
}

// Task method: Get supplies from base
Variant task_get_supplies_method_base(Dictionary state, Variant robot) {
	String robot_str = String(robot);
	Array base_loc;
	base_loc.push_back(1);
	base_loc.push_back(1);

	Array subtasks;
	Array move_task;
	move_task.push_back("move_task");
	move_task.push_back(robot_str);
	move_task.push_back(base_loc);
	subtasks.push_back(move_task);

	Array replenish_action;
	replenish_action.push_back("action_replenish_supplies");
	replenish_action.push_back(robot_str);
	subtasks.push_back(replenish_action);

	return subtasks;
}

// Task method: Rescue encap
Variant task_rescue_encap(Dictionary state, Variant robot) {
	String robot_str = String(robot);
	Dictionary current_image_dict = state["current_image"];
	Dictionary img = current_image_dict[robot_str];

	Variant position = img["loc"];
	Variant person = img["person"];

	if (person.get_type() == Variant::NIL) {
		return Array(); // No person found
	}

	Array subtasks;
	Array rescue_task;
	rescue_task.push_back("rescue_task");
	rescue_task.push_back(robot_str);
	rescue_task.push_back(String(person));
	subtasks.push_back(rescue_task);

	return subtasks;
}

// Task method: Survey front camera
Variant task_survey_method_front(Dictionary state, Variant robot, Variant location) {
	String robot_str = String(robot);
	Dictionary robot_type_dict = state["robot_type"];
	String robot_type = get_string_from_dict(robot_type_dict, robot_str);

	if (robot_type != "uav") {
		return Variant();
	}

	Array subtasks;
	Array adjust_task;
	adjust_task.push_back("adjust_altitude_task");
	adjust_task.push_back(robot_str);
	subtasks.push_back(adjust_task);

	Array capture_action;
	capture_action.push_back("action_capture_image");
	capture_action.push_back(robot_str);
	capture_action.push_back("front_camera");
	capture_action.push_back(location);
	subtasks.push_back(capture_action);

	Array rescue_encap_task;
	rescue_encap_task.push_back("rescue_encap_task");
	rescue_encap_task.push_back(robot_str);
	subtasks.push_back(rescue_encap_task);

	Array check_action;
	check_action.push_back("action_check_real");
	check_action.push_back(location);
	subtasks.push_back(check_action);

	return subtasks;
}

// Task method: Survey bottom camera
Variant task_survey_method_bottom(Dictionary state, Variant robot, Variant location) {
	String robot_str = String(robot);
	Dictionary robot_type_dict = state["robot_type"];
	String robot_type = get_string_from_dict(robot_type_dict, robot_str);

	if (robot_type != "uav") {
		return Variant();
	}

	Array subtasks;
	Array adjust_task;
	adjust_task.push_back("adjust_altitude_task");
	adjust_task.push_back(robot_str);
	subtasks.push_back(adjust_task);

	Array capture_action;
	capture_action.push_back("action_capture_image");
	capture_action.push_back(robot_str);
	capture_action.push_back("bottom_camera");
	capture_action.push_back(location);
	subtasks.push_back(capture_action);

	Array rescue_encap_task;
	rescue_encap_task.push_back("rescue_encap_task");
	rescue_encap_task.push_back(robot_str);
	subtasks.push_back(rescue_encap_task);

	Array check_action;
	check_action.push_back("action_check_real");
	check_action.push_back(location);
	subtasks.push_back(check_action);

	return subtasks;
}

// Task method: Get robot (free)
Variant task_get_robot_method_free(Dictionary state) {
	Dictionary rigid = state["rigid"];
	Array wheeled_robots = rigid["wheeled_robots"];
	Dictionary status_dict = state["status"];
	Dictionary loc_dict = state["loc"];

	Array base_loc;
	base_loc.push_back(1);
	base_loc.push_back(1);

	String nearest_robot;
	double nearest_dist = std::numeric_limits<double>::infinity();

	for (int i = 0; i < wheeled_robots.size(); i++) {
		String r = wheeled_robots[i];
		String status = get_string_from_dict(status_dict, r);

		if (status == "free") {
			Array r_loc = get_location(loc_dict[r]);
			double dist = euclidean_distance(r_loc, base_loc);

			if (dist < nearest_dist) {
				nearest_dist = dist;
				nearest_robot = r;
			}
		}
	}

	if (nearest_robot.is_empty()) {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_engage_robot");
	action.push_back(nearest_robot);
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Get robot (force)
Variant task_get_robot_method_force(Dictionary state) {
	Array subtasks;
	Array action;
	action.push_back("action_force_engage_robot");
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Adjust altitude to low
Variant task_adjust_altitude_method_low(Dictionary state, Variant robot) {
	String robot_str = String(robot);
	Dictionary altitude_dict = state["altitude"];
	String current_altitude = get_string_from_dict(altitude_dict, robot_str);

	if (current_altitude != "high") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_change_altitude");
	action.push_back(robot_str);
	action.push_back("low");
	subtasks.push_back(action);
	return subtasks;
}

// Task method: Adjust altitude to high
Variant task_adjust_altitude_method_high(Dictionary state, Variant robot) {
	String robot_str = String(robot);
	Dictionary altitude_dict = state["altitude"];
	String current_altitude = get_string_from_dict(altitude_dict, robot_str);

	if (current_altitude != "low") {
		return Variant();
	}

	Array subtasks;
	Array action;
	action.push_back("action_change_altitude");
	action.push_back(robot_str);
	action.push_back("high");
	subtasks.push_back(action);
	return subtasks;
}

} // namespace RescueDomain

// Task method callable wrappers
inline Variant RescueDomainCallable::task_move_method_euclidean(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::task_move_method_euclidean(p_state, p_robot, p_location);
}

inline Variant RescueDomainCallable::task_move_method_manhattan(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::task_move_method_manhattan(p_state, p_robot, p_location);
}

inline Variant RescueDomainCallable::task_move_method_curved(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::task_move_method_curved(p_state, p_robot, p_location);
}

inline Variant RescueDomainCallable::task_move_method_fly(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::task_move_method_fly(p_state, p_robot, p_location);
}

inline Variant RescueDomainCallable::task_move_method_alt_fly(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::task_move_method_alt_fly(p_state, p_robot, p_location);
}

inline Variant RescueDomainCallable::task_new_robot_encap(Dictionary p_state, Variant p_person) {
	return RescueDomain::task_new_robot_encap(p_state, p_person);
}

inline Variant RescueDomainCallable::task_rescue_method_ground(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::task_rescue_method_ground(p_state, p_robot, p_person);
}

inline Variant RescueDomainCallable::task_rescue_method_uav(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::task_rescue_method_uav(p_state, p_robot, p_person);
}

inline Variant RescueDomainCallable::task_support_person_method_1(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::task_support_person_method_1(p_state, p_robot, p_person);
}

inline Variant RescueDomainCallable::task_support_person_method_2(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::task_support_person_method_2(p_state, p_robot, p_person);
}

inline Variant RescueDomainCallable::task_support_person_method_3(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::task_support_person_method_3(p_state, p_robot, p_person);
}

inline Variant RescueDomainCallable::task_support_person_method_4(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::task_support_person_method_4(p_state, p_robot, p_person);
}

inline Variant RescueDomainCallable::task_support_person_method_5(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::task_support_person_method_5(p_state, p_robot, p_person);
}

inline Variant RescueDomainCallable::task_help_person(Dictionary p_state, Variant p_robot, Variant p_person) {
	return RescueDomain::task_help_person(p_state, p_robot, p_person);
}

inline Variant RescueDomainCallable::task_get_supplies_method_robot(Dictionary p_state, Variant p_robot) {
	return RescueDomain::task_get_supplies_method_robot(p_state, p_robot);
}

inline Variant RescueDomainCallable::task_get_supplies_method_base(Dictionary p_state, Variant p_robot) {
	return RescueDomain::task_get_supplies_method_base(p_state, p_robot);
}

inline Variant RescueDomainCallable::task_rescue_encap(Dictionary p_state, Variant p_robot) {
	return RescueDomain::task_rescue_encap(p_state, p_robot);
}

inline Variant RescueDomainCallable::task_survey_method_front(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::task_survey_method_front(p_state, p_robot, p_location);
}

inline Variant RescueDomainCallable::task_survey_method_bottom(Dictionary p_state, Variant p_robot, Variant p_location) {
	return RescueDomain::task_survey_method_bottom(p_state, p_robot, p_location);
}

inline Variant RescueDomainCallable::task_get_robot_method_free(Dictionary p_state) {
	return RescueDomain::task_get_robot_method_free(p_state);
}

inline Variant RescueDomainCallable::task_get_robot_method_force(Dictionary p_state) {
	return RescueDomain::task_get_robot_method_force(p_state);
}

inline Variant RescueDomainCallable::task_adjust_altitude_method_low(Dictionary p_state, Variant p_robot) {
	return RescueDomain::task_adjust_altitude_method_low(p_state, p_robot);
}

inline Variant RescueDomainCallable::task_adjust_altitude_method_high(Dictionary p_state, Variant p_robot) {
	return RescueDomain::task_adjust_altitude_method_high(p_state, p_robot);
}
