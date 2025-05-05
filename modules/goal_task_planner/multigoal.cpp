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
	ClassDB::bind_method(D_METHOD("get_state"), &PlannerMultigoal::get_state);
	ClassDB::bind_method(D_METHOD("set_state", "value"), &PlannerMultigoal::set_state);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "state"), "set_state", "get_state");
	ClassDB::bind_method(D_METHOD("state_variables"), &PlannerMultigoal::state_variables);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_goals_not_achieved", "state", "multigoal"), &PlannerMultigoal::method_goals_not_achieved);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_verify_multigoal", "state", "method", "multigoal", "depth", "verbose"), &PlannerMultigoal::method_verify_multigoal);
	ClassDB::bind_static_method("PlannerMultigoal", D_METHOD("method_split_multigoal", "state", "multigoal"), &PlannerMultigoal::method_split_multigoal);
}

Dictionary PlannerMultigoal::get_state() const {
	return state;
}

void PlannerMultigoal::set_state(Dictionary p_value) {
	state = p_value;
}

PlannerMultigoal::PlannerMultigoal(String p_multigoal_name, Dictionary p_state_variables) {
	set_name(p_multigoal_name);
	state = p_state_variables;
}

Array PlannerMultigoal::state_variables() {
	return state.keys();
}

Array PlannerMultigoal::method_split_multigoal(Dictionary p_state, Ref<PlannerMultigoal> p_multigoal) {
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
		// Achieve goals, then check whether they're all simultaneously true.
		goal_list.push_back(p_multigoal);
	}
	return goal_list;
}
Variant PlannerMultigoal::method_verify_multigoal(Dictionary p_state, String p_method, Ref<PlannerMultigoal> p_multigoal, int p_depth, int p_verbose) {
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
Dictionary PlannerMultigoal::method_goals_not_achieved(Dictionary p_state, Ref<PlannerMultigoal> p_multigoal) {
	bool is_multigoal_null = p_multigoal.is_null();
	if (is_multigoal_null) {
		return p_state;
	}
	Dictionary unmatched_states;
	for (Variant element : p_multigoal->get_state().keys()) {
		Dictionary sub_dictionary = p_multigoal->get_state()[element];
		for (Variant argument : sub_dictionary.keys()) {
			Variant value = sub_dictionary[argument];
			bool is_state_element_dictionary = p_state[element].get_type() == Variant::DICTIONARY;
			bool does_state_element_have_arguments = Dictionary(p_state[element]).has(argument);
			bool are_values_different = value != Dictionary(p_state[element])[argument];
			if (is_state_element_dictionary && does_state_element_have_arguments && are_values_different) {
				bool is_missing_element = !unmatched_states.has(element);
				if (is_missing_element) {
					unmatched_states[element] = Dictionary();
				}
				Dictionary temp = unmatched_states[element];
				temp[argument] = value;
				unmatched_states[element] = temp;
			}
		}
	}
	return unmatched_states;
}
