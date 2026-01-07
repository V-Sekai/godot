/**************************************************************************/
/*  temporal_travel_domain.h                                              */
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

#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

// Temporal Travel Domain - Demonstrates temporal planning with action durations
// Uses absolute time in microseconds instead of ISO 8601 durations (PT5M, PT10M, etc.)
// Derived from IPyHOP temporal_travel_domain.py
class TemporalTravelDomainCallable {
public:
	// Actions
	static Dictionary action_walk(Dictionary p_state, Variant p_person, Variant p_from_loc, Variant p_to_loc);
	static Dictionary action_call_taxi(Dictionary p_state, Variant p_person, Variant p_location);
	static Dictionary action_ride_taxi(Dictionary p_state, Variant p_person, Variant p_to_loc);
	static Dictionary action_pay_driver(Dictionary p_state, Variant p_person, Variant p_location);

	// Task methods
	static Variant task_travel_do_nothing(Dictionary p_state, Variant p_person, Variant p_location);
	static Variant task_travel_by_foot(Dictionary p_state, Variant p_person, Variant p_location);
	static Variant task_travel_by_taxi(Dictionary p_state, Variant p_person, Variant p_location);
};

namespace TemporalTravelDomain {

// Helper: Calculate taxi rate based on distance
// Rate = 1.5 + 0.5 * distance
double taxi_rate(int p_distance) {
	return 1.5 + 0.5 * p_distance;
}

// Helper: Get distance between two locations from rigid relations
// IPyHOP uses tuples as keys, but we'll use string keys like "(home_a, park)"
int get_distance(Dictionary p_state, Variant p_x, Variant p_y) {
	if (!p_state.has("rigid")) {
		return 0;
	}
	Dictionary rigid = p_state["rigid"];
	if (!rigid.has("dist")) {
		return 0;
	}
	Dictionary dist_dict = rigid["dist"];

	String x_str = String(p_x);
	String y_str = String(p_y);

	// Try (x, y) key format
	String key_xy = vformat("(%s, %s)", x_str, y_str);
	if (dist_dict.has(key_xy)) {
		Variant dist_val = dist_dict[key_xy];
		if (dist_val.get_type() == Variant::INT) {
			return int(dist_val);
		}
	}

	// Try (y, x) key format (symmetric)
	String key_yx = vformat("(%s, %s)", y_str, x_str);
	if (dist_dict.has(key_yx)) {
		Variant dist_val = dist_dict[key_yx];
		if (dist_val.get_type() == Variant::INT) {
			return int(dist_val);
		}
	}

	return 0;
}

// Helper: Check if variable is of a certain type
bool is_a(Dictionary p_state, Variant p_var, String p_type) {
	if (!p_state.has("rigid")) {
		return false;
	}
	Dictionary rigid = p_state["rigid"];
	if (!rigid.has("types")) {
		return false;
	}
	Dictionary types = rigid["types"];
	if (!types.has(p_type)) {
		return false;
	}
	Array type_list = types[p_type];
	String var_str = String(p_var);
	for (int i = 0; i < type_list.size(); i++) {
		if (String(type_list[i]) == var_str) {
			return true;
		}
	}
	return false;
}

// Action: Walk from one location to another
// Duration: 5 minutes per unit distance (300000000 microseconds per unit)
Dictionary action_walk(Dictionary state, String person, String from_loc, String to_loc) {
	if (!is_a(state, person, "person") || !is_a(state, from_loc, "location") || !is_a(state, to_loc, "location")) {
		return Dictionary();
	}
	if (from_loc == to_loc) {
		return Dictionary();
	}

	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Dictionary();
	}

	if (String(loc_dict[person]) != from_loc) {
		return Dictionary();
	}

	Dictionary new_state = state.duplicate();
	Dictionary new_loc = loc_dict.duplicate();
	new_loc[person] = to_loc;
	new_state["loc"] = new_loc;
	return new_state;
}

// Action: Call taxi to a location (instant, 0 duration)
Dictionary action_call_taxi(Dictionary state, String person, String location) {
	if (!is_a(state, person, "person") || !is_a(state, location, "location")) {
		return Dictionary();
	}

	Dictionary new_state = state.duplicate();
	Dictionary new_loc;
	if (new_state.has("loc")) {
		new_loc = new_state["loc"];
	} else {
		new_loc = Dictionary();
	}

	// Set taxi1 location to the specified location
	new_loc["taxi1"] = location;
	// Person enters taxi
	new_loc[person] = "taxi1";
	new_state["loc"] = new_loc;
	return new_state;
}

// Action: Ride taxi to destination
// Duration: 10 minutes per unit distance (600000000 microseconds per unit)
Dictionary action_ride_taxi(Dictionary state, String person, String to_loc) {
	if (!is_a(state, person, "person") || !is_a(state, to_loc, "location")) {
		return Dictionary();
	}

	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Dictionary();
	}

	Variant person_loc = loc_dict[person];
	if (String(person_loc) != "taxi1") {
		return Dictionary();
	}

	if (!is_a(state, person_loc, "taxi")) {
		return Dictionary();
	}

	Variant taxi_loc = loc_dict[person_loc];
	String from_loc = String(taxi_loc);

	if (!is_a(state, from_loc, "location") || from_loc == to_loc) {
		return Dictionary();
	}

	int dist = get_distance(state, from_loc, to_loc);
	if (dist == 0) {
		return Dictionary();
	}

	Dictionary new_state = state.duplicate();
	Dictionary new_loc = loc_dict.duplicate();
	Dictionary new_owe;
	if (new_state.has("owe")) {
		new_owe = new_state["owe"];
	} else {
		new_owe = Dictionary();
	}

	// Move taxi to destination
	new_loc[person_loc] = to_loc;

	// Calculate and set fare
	double fare = taxi_rate(dist);
	new_owe[person] = fare;

	new_state["loc"] = new_loc;
	new_state["owe"] = new_owe;
	return new_state;
}

// Action: Pay driver (instant, 0 duration)
Dictionary action_pay_driver(Dictionary state, String person, String location) {
	if (!is_a(state, person, "person")) {
		return Dictionary();
	}

	Dictionary owe_dict;
	Dictionary cash_dict;
	if (state.has("owe")) {
		owe_dict = state["owe"];
	} else {
		return Dictionary();
	}
	if (state.has("cash")) {
		cash_dict = state["cash"];
	} else {
		return Dictionary();
	}

	double owe_amount = 0.0;
	double cash_amount = 0.0;
	if (owe_dict.has(person)) {
		owe_amount = double(owe_dict[person]);
	}
	if (cash_dict.has(person)) {
		cash_amount = double(cash_dict[person]);
	}

	if (cash_amount < owe_amount) {
		return Dictionary(); // Not enough cash
	}

	Dictionary new_state = state.duplicate();
	Dictionary new_owe = owe_dict.duplicate();
	Dictionary new_cash = cash_dict.duplicate();
	Dictionary new_loc;
	if (new_state.has("loc")) {
		new_loc = new_state["loc"];
	} else {
		new_loc = Dictionary();
	}

	// Pay the fare
	new_cash[person] = cash_amount - owe_amount;
	new_owe[person] = 0.0;
	// Person exits taxi at destination
	new_loc[person] = location;

	new_state["owe"] = new_owe;
	new_state["cash"] = new_cash;
	new_state["loc"] = new_loc;
	return new_state;
}

// Task method: Do nothing if already at destination
Variant task_travel_do_nothing(Dictionary state, String person, String location) {
	if (!is_a(state, person, "person") || !is_a(state, location, "location")) {
		return Variant();
	}

	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Variant();
	}

	Variant person_loc = loc_dict[person];
	if (String(person_loc) == location) {
		return Array(); // Already at destination, no actions needed
	}

	return Variant();
}

// Task method: Travel by foot (if distance <= 2)
Variant task_travel_by_foot(Dictionary state, String person, String location) {
	if (!is_a(state, person, "person") || !is_a(state, location, "location")) {
		return Variant();
	}

	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Variant();
	}

	Variant person_loc = loc_dict[person];
	String from_loc = String(person_loc);

	if (from_loc == location) {
		return Variant();
	}

	int dist = get_distance(state, from_loc, location);
	if (dist > 2) {
		return Variant(); // Too far to walk
	}

	Array subtasks;
	Array walk_action;
	walk_action.push_back("action_walk");
	walk_action.push_back(person);
	walk_action.push_back(from_loc);
	walk_action.push_back(location);
	subtasks.push_back(walk_action);
	return subtasks;
}

// Task method: Travel by taxi (if has enough cash)
Variant task_travel_by_taxi(Dictionary state, String person, String location) {
	if (!is_a(state, person, "person") || !is_a(state, location, "location")) {
		return Variant();
	}

	Dictionary loc_dict;
	if (state.has("loc")) {
		loc_dict = state["loc"];
	} else {
		return Variant();
	}

	Variant person_loc = loc_dict[person];
	String from_loc = String(person_loc);

	if (from_loc == location) {
		return Variant();
	}

	int dist = get_distance(state, from_loc, location);
	if (dist == 0) {
		return Variant();
	}

	double fare = taxi_rate(dist);

	Dictionary cash_dict;
	if (state.has("cash")) {
		cash_dict = state["cash"];
	} else {
		return Variant();
	}

	double cash = 0.0;
	if (cash_dict.has(person)) {
		cash = double(cash_dict[person]);
	}

	if (cash < fare) {
		return Variant(); // Not enough cash
	}

	Array subtasks;
	Array call_action;
	call_action.push_back("action_call_taxi");
	call_action.push_back(person);
	call_action.push_back(from_loc);
	subtasks.push_back(call_action);

	Array ride_action;
	ride_action.push_back("action_ride_taxi");
	ride_action.push_back(person);
	ride_action.push_back(location);
	subtasks.push_back(ride_action);

	Array pay_action;
	pay_action.push_back("action_pay_driver");
	pay_action.push_back(person);
	pay_action.push_back(location);
	subtasks.push_back(pay_action);

	return subtasks;
}

} // namespace TemporalTravelDomain

// Implementations of TemporalTravelDomainCallable static methods
inline Dictionary TemporalTravelDomainCallable::action_walk(Dictionary p_state, Variant p_person, Variant p_from_loc, Variant p_to_loc) {
	return TemporalTravelDomain::action_walk(p_state, String(p_person), String(p_from_loc), String(p_to_loc));
}

inline Dictionary TemporalTravelDomainCallable::action_call_taxi(Dictionary p_state, Variant p_person, Variant p_location) {
	return TemporalTravelDomain::action_call_taxi(p_state, String(p_person), String(p_location));
}

inline Dictionary TemporalTravelDomainCallable::action_ride_taxi(Dictionary p_state, Variant p_person, Variant p_to_loc) {
	return TemporalTravelDomain::action_ride_taxi(p_state, String(p_person), String(p_to_loc));
}

inline Dictionary TemporalTravelDomainCallable::action_pay_driver(Dictionary p_state, Variant p_person, Variant p_location) {
	return TemporalTravelDomain::action_pay_driver(p_state, String(p_person), String(p_location));
}

inline Variant TemporalTravelDomainCallable::task_travel_do_nothing(Dictionary p_state, Variant p_person, Variant p_location) {
	return TemporalTravelDomain::task_travel_do_nothing(p_state, String(p_person), String(p_location));
}

inline Variant TemporalTravelDomainCallable::task_travel_by_foot(Dictionary p_state, Variant p_person, Variant p_location) {
	return TemporalTravelDomain::task_travel_by_foot(p_state, String(p_person), String(p_location));
}

inline Variant TemporalTravelDomainCallable::task_travel_by_taxi(Dictionary p_state, Variant p_person, Variant p_location) {
	return TemporalTravelDomain::task_travel_by_taxi(p_state, String(p_person), String(p_location));
}
