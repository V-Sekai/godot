/**************************************************************************/
/*  multigoal.cpp                                                         */
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

#include "multigoal.h"

#include "core/object/class_db.h"

void PlannerMultigoal::_bind_methods() {
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("is_multigoal_dict", "variant"), &PlannerMultigoal::is_multigoal_dict);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("get_goal_variables", "multigoal_dict"), &PlannerMultigoal::get_goal_variables);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("get_goal_conditions_for_variable", "multigoal_dict", "variable"), &PlannerMultigoal::get_goal_conditions_for_variable);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("get_goal_value", "multigoal_dict", "variable", "argument"), &PlannerMultigoal::get_goal_value);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("has_goal_condition", "multigoal_dict", "variable", "argument"), &PlannerMultigoal::has_goal_condition);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_goals_not_achieved", "state", "multigoal_dict"), &PlannerMultigoal::method_goals_not_achieved);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_verify_multigoal", "state", "method", "multigoal_dict", "depth", "verbose"), &PlannerMultigoal::method_verify_multigoal);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_split_multigoal", "state", "multigoal_dict"), &PlannerMultigoal::method_split_multigoal);
}

// Check if a Variant is a Dictionary multigoal (all values are dictionaries)
bool PlannerMultigoal::is_multigoal_dict(const Variant &p_variant) {
	if (p_variant.get_type() != Variant::DICTIONARY) {
		return false;
	}
	Dictionary dict = p_variant;
	// Check if all values are dictionaries (nested structure of multigoal)
	Array keys = dict.keys();
	for (int i = 0; i < keys.size(); i++) {
		Variant key = keys[i];
		Variant value = dict[key];
		if (value.get_type() != Variant::DICTIONARY) {
			return false;
		}
	}
	return true;
}

Array PlannerMultigoal::get_goal_variables(const Dictionary &p_multigoal_dict) {
	return p_multigoal_dict.keys();
}

Dictionary PlannerMultigoal::get_goal_conditions_for_variable(const Dictionary &p_multigoal_dict, const String &p_variable) {
	if (p_multigoal_dict.has(p_variable) && p_multigoal_dict[p_variable].get_type() == Variant::DICTIONARY) {
		return p_multigoal_dict[p_variable];
	}
	return Dictionary();
}

Variant PlannerMultigoal::get_goal_value(const Dictionary &p_multigoal_dict, const String &p_variable, const String &p_argument) {
	if (p_multigoal_dict.has(p_variable) && p_multigoal_dict[p_variable].get_type() == Variant::DICTIONARY) {
		Dictionary conditions = p_multigoal_dict[p_variable];
		if (conditions.has(p_argument)) {
			return conditions[p_argument];
		}
	}
	return Variant();
}

bool PlannerMultigoal::has_goal_condition(const Dictionary &p_multigoal_dict, const String &p_variable, const String &p_argument) {
	if (p_multigoal_dict.has(p_variable) && p_multigoal_dict[p_variable].get_type() == Variant::DICTIONARY) {
		Dictionary conditions = p_multigoal_dict[p_variable];
		return conditions.has(p_argument);
	}
	return false;
}

Dictionary PlannerMultigoal::method_goals_not_achieved(const Dictionary &p_state, const Dictionary &p_multigoal_dict) {
	Dictionary unmatched_states;
	Array goal_variables = get_goal_variables(p_multigoal_dict);

	for (int i = 0; i < goal_variables.size(); ++i) {
		String variable_name = goal_variables[i];
		Dictionary goal_conditions = get_goal_conditions_for_variable(p_multigoal_dict, variable_name);

		for (const Variant *key = goal_conditions.next(nullptr); key; key = goal_conditions.next(key)) {
			String argument = *key;
			Variant desired_value = goal_conditions[*key];

			bool current_state_has_variable = p_state.has(variable_name);
			bool current_state_variable_is_dict = false;
			Dictionary current_variable_conditions;
			bool current_state_has_argument = false;
			Variant current_value;

			if (current_state_has_variable) {
				Variant current_var_val = p_state[variable_name];
				if (current_var_val.get_type() == Variant::DICTIONARY) {
					current_state_variable_is_dict = true;
					current_variable_conditions = current_var_val;
					if (current_variable_conditions.has(argument)) {
						current_state_has_argument = true;
						current_value = current_variable_conditions[argument];
					}
				}
			}

			if (!current_state_has_variable || !current_state_variable_is_dict || !current_state_has_argument || desired_value != current_value) {
				if (!unmatched_states.has(variable_name)) {
					unmatched_states[variable_name] = Dictionary();
				}
				Dictionary temp = unmatched_states[variable_name];
				temp[argument] = desired_value;
				unmatched_states[variable_name] = temp;
			}
		}
	}
	return unmatched_states;
}

Variant PlannerMultigoal::method_verify_multigoal(const Dictionary &p_state, const String &p_method, const Dictionary &p_multigoal_dict, int p_depth, int p_verbose) {
	Dictionary goal_dict = method_goals_not_achieved(p_state, p_multigoal_dict);
	if (!goal_dict.is_empty()) {
		if (p_verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve %s", p_depth, p_method, p_multigoal_dict));
		}
		return false;
	}

	if (p_verbose >= 3) {
		print_line(vformat("Depth %d: method %s achieved %s", p_depth, p_method, p_multigoal_dict));
	}
	return Array();
}

Array PlannerMultigoal::method_split_multigoal(const Dictionary &p_state, const Dictionary &p_multigoal_dict) {
	// Get only the unachieved goals (matching IPyHOP's _goals_not_achieved behavior)
	Dictionary goal_state = method_goals_not_achieved(p_state, p_multigoal_dict);
	Array goal_list;

	// Convert each unachieved goal to a Dictionary multigoal: {variable_name: {argument: value}}
	for (Variant state_variable_name : goal_state.keys()) {
		Variant state_values = goal_state[state_variable_name];
		if (state_values.get_type() != Variant::DICTIONARY) {
			continue;
		}
		Dictionary state_variable = state_values;
		for (Variant state_goal : state_variable.keys()) {
			if (!state_variable.has(state_goal)) {
				continue;
			}
			// Convert single goal to Dictionary multigoal: {variable_name: {argument: value}}
			Dictionary multigoal;
			Dictionary variable_dict;
			variable_dict[state_goal] = state_variable[state_goal];
			multigoal[state_variable_name] = variable_dict;
			goal_list.push_back(multigoal);
		}
	}

	// Match IPyHOP's behavior: if there are unachieved goals, append multigoal to re-check later
	// If all goals are achieved (goal_list is empty), return empty list
	if (!goal_list.is_empty()) {
		goal_list.push_back(p_multigoal_dict);
	}
	return goal_list;
}
