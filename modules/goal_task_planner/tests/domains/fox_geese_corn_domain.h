/**************************************************************************/
/*  fox_geese_corn_domain.h                                               */
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

// Wrapper class used to construct Callables for free functions in FoxGeeseCornDomain.
class FoxGeeseCornDomainCallable {
public:
	// Actions - use Variant parameters to match PlannerPlan expectations
	static Variant action_cross_east(Dictionary p_state, Variant p_fox_count, Variant p_geese_count, Variant p_corn_count);
	static Variant action_cross_west(Dictionary p_state, Variant p_fox_count, Variant p_geese_count, Variant p_corn_count);

	// Task methods for transport_all (HTN decomposition)
	static Variant task_transport_all_method_done(Dictionary p_state);
	static Variant task_transport_all_method_return_boat(Dictionary p_state);
	static Variant task_transport_all_method_transport_one(Dictionary p_state);

	// Task methods for transport_one (HTN decomposition)
	static Variant task_transport_one_method_geese(Dictionary p_state);
	static Variant task_transport_one_method_fox_corn(Dictionary p_state);
	static Variant task_transport_one_method_corn(Dictionary p_state);
	static Variant task_transport_one_method_fox(Dictionary p_state);

	// Multigoal methods
	static Array multigoal_transport_all(Dictionary p_state, Array p_multigoal);
};

// Helper functions for fox-geese-corn domain
namespace FoxGeeseCornDomain {

// Helper: Get value from state with default (nested dictionary format)
int get_int(Dictionary p_state, const String &p_key, int p_default = 0) {
	if (!p_state.has(p_key)) {
		return p_default;
	}
	Variant predicate_val = p_state[p_key];
	if (predicate_val.get_type() == Variant::DICTIONARY) {
		Dictionary predicate_dict = predicate_val;
		if (predicate_dict.has("value")) {
			Variant value = predicate_dict["value"];
			if (value.get_type() == Variant::INT) {
				return value;
			}
		}
	}
	return p_default;
}

String get_string(Dictionary p_state, const String &p_key, const String &p_default = "") {
	if (!p_state.has(p_key)) {
		return p_default;
	}
	Variant predicate_val = p_state[p_key];
	if (predicate_val.get_type() == Variant::DICTIONARY) {
		Dictionary predicate_dict = predicate_val;
		if (predicate_dict.has("value")) {
			Variant value = predicate_dict["value"];
			if (value.get_type() == Variant::STRING) {
				return value;
			}
		}
	}
	return p_default;
}

// Initialize state from parameters (nested dictionary format)
Dictionary initialize_state(int p_f, int p_g, int p_c, int p_k, int p_pf, int p_pg, int p_pc) {
	Dictionary state;

	// Create nested dictionaries for all state variables
	Dictionary west_fox_dict;
	west_fox_dict["value"] = p_f;
	state["west_fox"] = west_fox_dict;

	Dictionary west_geese_dict;
	west_geese_dict["value"] = p_g;
	state["west_geese"] = west_geese_dict;

	Dictionary west_corn_dict;
	west_corn_dict["value"] = p_c;
	state["west_corn"] = west_corn_dict;

	Dictionary east_fox_dict;
	east_fox_dict["value"] = 0;
	state["east_fox"] = east_fox_dict;

	Dictionary east_geese_dict;
	east_geese_dict["value"] = 0;
	state["east_geese"] = east_geese_dict;

	Dictionary east_corn_dict;
	east_corn_dict["value"] = 0;
	state["east_corn"] = east_corn_dict;

	Dictionary boat_location_dict;
	boat_location_dict["value"] = "west";
	state["boat_location"] = boat_location_dict;

	Dictionary boat_capacity_dict;
	boat_capacity_dict["value"] = p_k;
	state["boat_capacity"] = boat_capacity_dict;

	Dictionary pf_dict;
	pf_dict["value"] = p_pf;
	state["pf"] = pf_dict;

	Dictionary pg_dict;
	pg_dict["value"] = p_pg;
	state["pg"] = pg_dict;

	Dictionary pc_dict;
	pc_dict["value"] = p_pc;
	state["pc"] = pc_dict;

	return state;
}

// Check if a side is safe (no fox with geese alone, no geese with corn alone)
bool check_side_safe(int p_fox, int p_geese, int p_corn) {
	// Only one type present - always safe
	if ((p_fox > 0 && p_geese == 0 && p_corn == 0) ||
			(p_fox == 0 && p_geese > 0 && p_corn == 0) ||
			(p_fox == 0 && p_geese == 0 && p_corn > 0)) {
		return true;
	}

	// All together - safe
	if (p_fox > 0 && p_geese > 0 && p_corn > 0) {
		return true;
	}

	// Fox and geese alone - unsafe
	if (p_fox > 0 && p_geese > 0 && p_corn == 0) {
		return false;
	}

	// Geese and corn alone - unsafe
	if (p_fox == 0 && p_geese > 0 && p_corn > 0) {
		return false;
	}

	// Empty - safe
	if (p_fox == 0 && p_geese == 0 && p_corn == 0) {
		return true;
	}

	return true;
}

// Check if a state is safe (both sides must be safe)
bool is_safe(Dictionary p_state) {
	int west_fox = get_int(p_state, "west_fox");
	int west_geese = get_int(p_state, "west_geese");
	int west_corn = get_int(p_state, "west_corn");
	int east_fox = get_int(p_state, "east_fox");
	int east_geese = get_int(p_state, "east_geese");
	int east_corn = get_int(p_state, "east_corn");

	bool west_safe = check_side_safe(west_fox, west_geese, west_corn);
	bool east_safe = check_side_safe(east_fox, east_geese, east_corn);

	return west_safe && east_safe;
}

// Check if transporting items would result in a safe state (IPyHOP pattern: check state conditions, don't call action)
// This simulates the state change without actually calling the action
bool would_action_be_safe(Dictionary p_state, String p_direction, int p_fox, int p_geese, int p_corn) {
	int west_fox = get_int(p_state, "west_fox");
	int west_geese = get_int(p_state, "west_geese");
	int west_corn = get_int(p_state, "west_corn");
	int east_fox = get_int(p_state, "east_fox");
	int east_geese = get_int(p_state, "east_geese");
	int east_corn = get_int(p_state, "east_corn");

	// Simulate the state change
	if (p_direction == "east") {
		// Moving from west to east
		west_fox -= p_fox;
		west_geese -= p_geese;
		west_corn -= p_corn;
		east_fox += p_fox;
		east_geese += p_geese;
		east_corn += p_corn;
	} else if (p_direction == "west") {
		// Moving from east to west
		east_fox -= p_fox;
		east_geese -= p_geese;
		east_corn -= p_corn;
		west_fox += p_fox;
		west_geese += p_geese;
		west_corn += p_corn;
	}

	// Check if resulting state would be safe
	bool west_safe = check_side_safe(west_fox, west_geese, west_corn);
	bool east_safe = check_side_safe(east_fox, east_geese, east_corn);

	return west_safe && east_safe;
}

// Calculate objective value (points for items on east side)
int calculate_objective(Dictionary p_state) {
	int east_fox = get_int(p_state, "east_fox");
	int east_geese = get_int(p_state, "east_geese");
	int east_corn = get_int(p_state, "east_corn");
	int pf = get_int(p_state, "pf", 1);
	int pg = get_int(p_state, "pg", 1);
	int pc = get_int(p_state, "pc", 1);

	return east_fox * pf + east_geese * pg + east_corn * pc;
}

// Parse MiniZinc .dzn file
Dictionary parse_dzn_file(const String &p_path) {
	Dictionary params;

	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
	if (!file.is_valid()) {
		return params;
	}

	String content = file->get_as_text();
	PackedStringArray lines = content.split("\n");

	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i].strip_edges();
		if (line.is_empty() || line.begins_with("%")) {
			continue;
		}

		// Parse lines like "f = 6;"
		if (line.contains("=")) {
			PackedStringArray parts = line.split("=");
			if (parts.size() == 2) {
				String key = parts[0].strip_edges();
				String value_str = parts[1].strip_edges().trim_suffix(";").strip_edges();

				if (key == "f") {
					params["f"] = value_str.to_int();
				} else if (key == "g") {
					params["g"] = value_str.to_int();
				} else if (key == "c") {
					params["c"] = value_str.to_int();
				} else if (key == "k") {
					params["k"] = value_str.to_int();
				} else if (key == "pf") {
					params["pf"] = value_str.to_int();
				} else if (key == "pg") {
					params["pg"] = value_str.to_int();
				} else if (key == "pc") {
					params["pc"] = value_str.to_int();
				}
			}
		}
	}

	return params;
}

// Action: Cross from west to east
Variant action_cross_east(Dictionary p_state, int p_fox_count, int p_geese_count, int p_corn_count) {
	// Check boat location
	String boat_location = get_string(p_state, "boat_location");
	if (boat_location != "west") {
		return Variant(); // Return NIL on failure
	}

	// Check capacity
	int total = p_fox_count + p_geese_count + p_corn_count;
	int capacity = get_int(p_state, "boat_capacity", 2);
	if (total > capacity) {
		return Variant();
	}

	// Must transport at least one item
	if (total <= 0) {
		return Variant();
	}

	// Check sufficient items on west side
	int west_fox = get_int(p_state, "west_fox");
	int west_geese = get_int(p_state, "west_geese");
	int west_corn = get_int(p_state, "west_corn");

	if (west_fox < p_fox_count || west_geese < p_geese_count || west_corn < p_corn_count) {
		return Variant();
	}

	// Calculate new state - use deep copy to preserve all fields including nested structures
	// Use duplicate(true) which works correctly for nested dictionaries in Godot (matches plan.cpp)
	Dictionary new_state = p_state.duplicate(true);

	// Update nested dictionaries (create new dictionaries to ensure deep copy)
	Dictionary west_fox_dict;
	west_fox_dict["value"] = west_fox - p_fox_count;
	new_state["west_fox"] = west_fox_dict;

	Dictionary west_geese_dict;
	west_geese_dict["value"] = west_geese - p_geese_count;
	new_state["west_geese"] = west_geese_dict;

	Dictionary west_corn_dict;
	west_corn_dict["value"] = west_corn - p_corn_count;
	new_state["west_corn"] = west_corn_dict;

	Dictionary east_fox_dict;
	east_fox_dict["value"] = get_int(p_state, "east_fox") + p_fox_count;
	new_state["east_fox"] = east_fox_dict;

	Dictionary east_geese_dict;
	east_geese_dict["value"] = get_int(p_state, "east_geese") + p_geese_count;
	new_state["east_geese"] = east_geese_dict;

	Dictionary east_corn_dict;
	east_corn_dict["value"] = get_int(p_state, "east_corn") + p_corn_count;
	new_state["east_corn"] = east_corn_dict;

	Dictionary boat_location_dict;
	boat_location_dict["value"] = "east";
	new_state["boat_location"] = boat_location_dict;

	// Check safety constraints
	if (!is_safe(new_state)) {
		return Variant();
	}

	return new_state;
}

// Action: Cross from east to west
Variant action_cross_west(Dictionary p_state, int p_fox_count, int p_geese_count, int p_corn_count) {
	// Check boat location
	String boat_location = get_string(p_state, "boat_location");
	if (boat_location != "east") {
		return Variant();
	}

	// Check capacity
	int total = p_fox_count + p_geese_count + p_corn_count;
	int capacity = get_int(p_state, "boat_capacity", 2);
	if (total > capacity) {
		return Variant();
	}

	// Check sufficient items on east side (only if transporting items)
	if (total > 0) {
		int east_fox = get_int(p_state, "east_fox");
		int east_geese = get_int(p_state, "east_geese");
		int east_corn = get_int(p_state, "east_corn");

		if (east_fox < p_fox_count || east_geese < p_geese_count || east_corn < p_corn_count) {
			return Variant();
		}
	}

	// Calculate new state - use deep copy to preserve all fields including nested structures
	// Use duplicate(true) which works correctly for nested dictionaries in Godot (matches plan.cpp)
	Dictionary new_state = p_state.duplicate(true);

	// Update nested dictionaries (create new dictionaries to ensure deep copy)
	Dictionary west_fox_dict;
	west_fox_dict["value"] = get_int(p_state, "west_fox") + p_fox_count;
	new_state["west_fox"] = west_fox_dict;

	Dictionary west_geese_dict;
	west_geese_dict["value"] = get_int(p_state, "west_geese") + p_geese_count;
	new_state["west_geese"] = west_geese_dict;

	Dictionary west_corn_dict;
	west_corn_dict["value"] = get_int(p_state, "west_corn") + p_corn_count;
	new_state["west_corn"] = west_corn_dict;

	Dictionary east_fox_dict;
	east_fox_dict["value"] = get_int(p_state, "east_fox") - p_fox_count;
	new_state["east_fox"] = east_fox_dict;

	Dictionary east_geese_dict;
	east_geese_dict["value"] = get_int(p_state, "east_geese") - p_geese_count;
	new_state["east_geese"] = east_geese_dict;

	Dictionary east_corn_dict;
	east_corn_dict["value"] = get_int(p_state, "east_corn") - p_corn_count;
	new_state["east_corn"] = east_corn_dict;

	Dictionary boat_location_dict;
	boat_location_dict["value"] = "west";
	new_state["boat_location"] = boat_location_dict;

	// Check safety constraints (only if transporting items)
	// Empty returns are always safe since no items are moved
	if (total > 0 && !is_safe(new_state)) {
		return Variant();
	}

	return new_state;
}

// Task: Transport all items from west to east
// IPyHOP/HTN pattern: Decompose into smaller subtasks
// Method 1: Goal already achieved - return empty (success)
Variant task_transport_all_method_done(Dictionary p_state) {
	int west_fox = get_int(p_state, "west_fox");
	int west_geese = get_int(p_state, "west_geese");
	int west_corn = get_int(p_state, "west_corn");

	// Guard: Goal already achieved
	if (west_fox == 0 && west_geese == 0 && west_corn == 0) {
		return Array(); // All items transported
	}

	return Variant(); // Not done yet, try next method
}

// Method 2: Boat is on east side - return empty boat, then recurse
Variant task_transport_all_method_return_boat(Dictionary p_state) {
	String boat_location = get_string(p_state, "boat_location");

	// Guard: Boat must be on east side
	if (boat_location != "east") {
		return Variant();
	}

	// Check if empty return would be safe (IPyHOP pattern: check state conditions)
	if (would_action_be_safe(p_state, "west", 0, 0, 0)) {
		Array result;
		Array action;
		action.push_back("action_cross_west");
		action.push_back(0); // fox
		action.push_back(0); // geese
		action.push_back(0); // corn
		result.push_back(action);
		Array recurse_task;
		recurse_task.push_back("transport_all");
		result.push_back(recurse_task);
		return result;
	}

	return Variant(); // Empty return would be unsafe, try next method
}

// Method 3: Boat is on west side - transport one item, then recurse
Variant task_transport_all_method_transport_one(Dictionary p_state) {
	String boat_location = get_string(p_state, "boat_location");
	int capacity = get_int(p_state, "boat_capacity", 2);

	// Guard: Boat must be on west side
	if (boat_location != "west") {
		return Variant();
	}

	// Guard: Must have capacity
	if (capacity < 1) {
		return Variant();
	}

	// Decompose: transport one item, then recurse
	Array result;
	Array transport_one_task;
	transport_one_task.push_back("transport_one");
	result.push_back(transport_one_task);
	Array recurse_task;
	recurse_task.push_back("transport_all");
	result.push_back(recurse_task);
	return result;
}

// Task: Transport one item from west to east
// IPyHOP/HTN pattern: Decompose into selecting which item(s) to transport
// Method 1: Transport geese (highest priority)
Variant task_transport_one_method_geese(Dictionary p_state) {
	int west_geese = get_int(p_state, "west_geese");
	int capacity = get_int(p_state, "boat_capacity", 2);

	if (west_geese > 0) {
		int geese_to_transport = (west_geese < capacity) ? west_geese : capacity;
		// Check if this action would be safe (IPyHOP pattern: check state conditions)
		if (would_action_be_safe(p_state, "east", 0, geese_to_transport, 0)) {
			Array result;
			Array action;
			action.push_back("action_cross_east");
			action.push_back(0); // fox
			action.push_back(geese_to_transport); // geese
			action.push_back(0); // corn
			result.push_back(action);
			return result;
		}
	}

	return Variant(); // Can't transport geese safely, try next method
}

// Method 2: Transport fox and corn together (safe combination when capacity >= 2)
Variant task_transport_one_method_fox_corn(Dictionary p_state) {
	int west_fox = get_int(p_state, "west_fox");
	int west_corn = get_int(p_state, "west_corn");
	int capacity = get_int(p_state, "boat_capacity", 2);

	if (capacity >= 2 && west_fox > 0 && west_corn > 0) {
		int fox_to_transport = (west_fox < capacity) ? west_fox : 1; // At least 1
		int corn_to_transport = (west_corn < capacity - fox_to_transport) ? west_corn : (capacity - fox_to_transport);
		if (corn_to_transport > 0 && fox_to_transport + corn_to_transport <= capacity) {
			// Check if this action would be safe (IPyHOP pattern: check state conditions)
			if (would_action_be_safe(p_state, "east", fox_to_transport, 0, corn_to_transport)) {
				Array result;
				Array action;
				action.push_back("action_cross_east");
				action.push_back(fox_to_transport); // fox
				action.push_back(0); // geese
				action.push_back(corn_to_transport); // corn
				result.push_back(action);
				return result;
			}
		}
	}

	return Variant(); // Can't transport fox+corn safely, try next method
}

// Method 3: Transport corn alone
Variant task_transport_one_method_corn(Dictionary p_state) {
	int west_corn = get_int(p_state, "west_corn");
	int capacity = get_int(p_state, "boat_capacity", 2);

	if (west_corn > 0) {
		int corn_to_transport = (west_corn < capacity) ? west_corn : capacity;
		// Check if this action would be safe (IPyHOP pattern: check state conditions)
		if (would_action_be_safe(p_state, "east", 0, 0, corn_to_transport)) {
			Array result;
			Array action;
			action.push_back("action_cross_east");
			action.push_back(0); // fox
			action.push_back(0); // geese
			action.push_back(corn_to_transport); // corn
			result.push_back(action);
			return result;
		}
	}

	return Variant(); // Can't transport corn safely, try next method
}

// Method 4: Transport fox alone
Variant task_transport_one_method_fox(Dictionary p_state) {
	int west_fox = get_int(p_state, "west_fox");
	int capacity = get_int(p_state, "boat_capacity", 2);

	if (west_fox > 0) {
		int fox_to_transport = (west_fox < capacity) ? west_fox : capacity;
		// Check if this action would be safe (IPyHOP pattern: check state conditions)
		if (would_action_be_safe(p_state, "east", fox_to_transport, 0, 0)) {
			Array result;
			Array action;
			action.push_back("action_cross_east");
			action.push_back(fox_to_transport); // fox
			action.push_back(0); // geese
			action.push_back(0); // corn
			result.push_back(action);
			return result;
		}
	}

	return Variant(); // Can't transport fox safely, all methods failed
}

// Multigoal: Transport all items from west to east (goal-based)
Array multigoal_transport_all(Dictionary p_state, Array p_multigoal) {
	// Check if goal is achieved (all items on east side)
	int west_fox = get_int(p_state, "west_fox");
	int west_geese = get_int(p_state, "west_geese");
	int west_corn = get_int(p_state, "west_corn");

	if (west_fox == 0 && west_geese == 0 && west_corn == 0) {
		return Array(); // Goal achieved, return empty array
	}

	// Generate unigoal arrays for items still on west side
	// Format: [["east_fox", "value", west_fox_value], ...]
	Array goals;

	if (west_fox > 0) {
		Array unigoal_fox;
		unigoal_fox.push_back("east_fox");
		unigoal_fox.push_back("value");
		unigoal_fox.push_back(west_fox);
		goals.push_back(unigoal_fox);
	}

	if (west_geese > 0) {
		Array unigoal_geese;
		unigoal_geese.push_back("east_geese");
		unigoal_geese.push_back("value");
		unigoal_geese.push_back(west_geese);
		goals.push_back(unigoal_geese);
	}

	if (west_corn > 0) {
		Array unigoal_corn;
		unigoal_corn.push_back("east_corn");
		unigoal_corn.push_back("value");
		unigoal_corn.push_back(west_corn);
		goals.push_back(unigoal_corn);
	}

	return goals;
}

} // namespace FoxGeeseCornDomain

// Callable wrapper implementations - convert Variant to int
Variant FoxGeeseCornDomainCallable::action_cross_east(Dictionary p_state, Variant p_fox_count, Variant p_geese_count, Variant p_corn_count) {
	int fox_count = p_fox_count;
	int geese_count = p_geese_count;
	int corn_count = p_corn_count;
	return FoxGeeseCornDomain::action_cross_east(p_state, fox_count, geese_count, corn_count);
}

Variant FoxGeeseCornDomainCallable::action_cross_west(Dictionary p_state, Variant p_fox_count, Variant p_geese_count, Variant p_corn_count) {
	int fox_count = p_fox_count;
	int geese_count = p_geese_count;
	int corn_count = p_corn_count;
	return FoxGeeseCornDomain::action_cross_west(p_state, fox_count, geese_count, corn_count);
}

// Task method wrappers for transport_all (HTN decomposition)
Variant FoxGeeseCornDomainCallable::task_transport_all_method_done(Dictionary p_state) {
	return FoxGeeseCornDomain::task_transport_all_method_done(p_state);
}

Variant FoxGeeseCornDomainCallable::task_transport_all_method_return_boat(Dictionary p_state) {
	return FoxGeeseCornDomain::task_transport_all_method_return_boat(p_state);
}

Variant FoxGeeseCornDomainCallable::task_transport_all_method_transport_one(Dictionary p_state) {
	return FoxGeeseCornDomain::task_transport_all_method_transport_one(p_state);
}

// Task method wrappers for transport_one (HTN decomposition)
Variant FoxGeeseCornDomainCallable::task_transport_one_method_geese(Dictionary p_state) {
	return FoxGeeseCornDomain::task_transport_one_method_geese(p_state);
}

Variant FoxGeeseCornDomainCallable::task_transport_one_method_fox_corn(Dictionary p_state) {
	return FoxGeeseCornDomain::task_transport_one_method_fox_corn(p_state);
}

Variant FoxGeeseCornDomainCallable::task_transport_one_method_corn(Dictionary p_state) {
	return FoxGeeseCornDomain::task_transport_one_method_corn(p_state);
}

Variant FoxGeeseCornDomainCallable::task_transport_one_method_fox(Dictionary p_state) {
	return FoxGeeseCornDomain::task_transport_one_method_fox(p_state);
}

Array FoxGeeseCornDomainCallable::multigoal_transport_all(Dictionary p_state, Array p_multigoal) {
	return FoxGeeseCornDomain::multigoal_transport_all(p_state, p_multigoal);
}
