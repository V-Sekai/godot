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

void PlannerMultigoal::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_goal_variables"), &PlannerMultigoal::get_goal_variables);
	ClassDB::bind_method(D_METHOD("get_goal_conditions_for_variable", "variable"), &PlannerMultigoal::get_goal_conditions_for_variable);
	ClassDB::bind_method(D_METHOD("get_goal_value", "variable", "argument"), &PlannerMultigoal::get_goal_value);
	ClassDB::bind_method(D_METHOD("has_goal_condition", "variable", "argument"), &PlannerMultigoal::has_goal_condition);

	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_goals_not_achieved", "state", "multigoal"), &PlannerMultigoal::method_goals_not_achieved);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_verify_multigoal", "state", "method", "multigoal", "depth", "verbose"), &PlannerMultigoal::method_verify_multigoal);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_split_multigoal", "state", "multigoal"), &PlannerMultigoal::method_split_multigoal);
}

PlannerMultigoal::PlannerMultigoal(String p_multigoal_name, Dictionary p_state_variables) {
	set_name(p_multigoal_name);
	state = p_state_variables;
}

Array PlannerMultigoal::get_goal_variables() const {
	return state.keys();
}

Dictionary PlannerMultigoal::get_goal_conditions_for_variable(const String &p_variable) const {
	if (state.has(p_variable) && state[p_variable].get_type() == Variant::DICTIONARY) {
		return state[p_variable];
	}
	return Dictionary();
}

Variant PlannerMultigoal::get_goal_value(const String &p_variable, const String &p_argument) const {
	if (state.has(p_variable) && state[p_variable].get_type() == Variant::DICTIONARY) {
		Dictionary conditions = state[p_variable];
		if (conditions.has(p_argument)) {
			return conditions[p_argument];
		}
	}
	return Variant();
}

bool PlannerMultigoal::has_goal_condition(const String &p_variable, const String &p_argument) const {
	if (state.has(p_variable) && state[p_variable].get_type() == Variant::DICTIONARY) {
		Dictionary conditions = state[p_variable];
		return conditions.has(p_argument);
	}
	return false;
}

Array PlannerMultigoal::method_split_multigoal(const Dictionary &p_state, const Ref<PlannerMultigoal> &p_multigoal) {
	Dictionary goal_state = method_goals_not_achieved(p_state, p_multigoal);
	Array goal_list;
	for (Variant state_variable_name : goal_state.keys()) {
		Variant state_values = goal_state[state_variable_name];
		if (state_values.get_type() != Variant::DICTIONARY) {
			continue;
		}
		Dictionary goal_value_dictionary = state_values;
		Dictionary state_variable = state_values;
		for (Variant state_goal : state_variable.keys()) {
			if (!state_variable.has(state_goal)) {
				continue;
			}
			Array goal;
			goal.resize(3);
			goal[0] = state_variable_name;
			goal[1] = state_goal;
			goal[2] = state_variable[state_goal];
			goal_list.push_back(goal);
		}
	}
	for (Variant state_variable_name : p_state.keys()) {
		Variant state_values = p_state[state_variable_name];
		if (state_values.get_type() != Variant::DICTIONARY) {
			continue;
		}
		Dictionary goal_value_dictionary = state_values;
		Dictionary state_variable = state_values;
		for (Variant state_goal : state_variable.keys()) {
			if (!state_variable.has(state_goal)) {
				continue;
			}
			Array goal;
			goal.resize(3);
			goal[0] = state_variable_name;
			goal[1] = state_goal;
			goal[2] = state_variable[state_goal];
			bool exists = false;
			for (int i = 0; i < goal_list.size(); i++) {
				Array existing_goal = goal_list[i];
				if (existing_goal[0] == goal[0] && existing_goal[1] == goal[1]) {
					exists = true;
					break;
				}
			}

			if (!exists) {
				goal_list.push_back(goal);
			}
		}
	}
	if (!goal_list.is_empty()) {
		goal_list.push_back(p_multigoal);
	}
	return goal_list;
}

Variant PlannerMultigoal::method_verify_multigoal(const Dictionary &p_state, const String &p_method, const Ref<PlannerMultigoal> &p_multigoal, int p_depth, int p_verbose) {
	if (p_multigoal.is_null()) {
		return false;
	}
	Dictionary goal_dict = method_goals_not_achieved(p_state, p_multigoal);
	if (!goal_dict.is_empty()) {
		if (p_verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve %s", p_depth, p_method, p_multigoal));
		}
		return false;
	}

	if (p_verbose >= 3) {
		print_line(vformat("Depth %d: method %s achieved %s", p_depth, p_method, p_multigoal));
	}
	return Array();
}

Dictionary PlannerMultigoal::method_goals_not_achieved(const Dictionary &p_state, const Ref<PlannerMultigoal> &p_multigoal) {
	if (p_multigoal.is_null()) {
		return p_state;
	}

	Dictionary unmatched_states;
	Array goal_variables = p_multigoal->get_goal_variables();

	for (int i = 0; i < goal_variables.size(); ++i) {
		String variable_name = goal_variables[i];
		Dictionary goal_conditions = p_multigoal->get_goal_conditions_for_variable(variable_name);

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
