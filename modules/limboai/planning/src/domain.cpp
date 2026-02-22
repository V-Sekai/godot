/**************************************************************************/
/*  domain.cpp                                                            */
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

#include "domain.h"

#include "multigoal.h"
#include "plan.h"

void PlannerDomain::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_multigoal_methods", "methods"), &PlannerDomain::add_multigoal_methods);
	ClassDB::bind_method(D_METHOD("add_unigoal_methods", "task_name", "methods"), &PlannerDomain::add_unigoal_methods);
	ClassDB::bind_method(D_METHOD("add_task_methods", "task_name", "methods"), &PlannerDomain::add_task_methods);
	ClassDB::bind_method(D_METHOD("add_command", "name", "command"), &PlannerDomain::add_command);

	ClassDB::bind_static_method("PlannerDomain", D_METHOD("method_verify_goal", "state", "method", "state_var", "arguments", "desired_values", "depth", "verbose"), &PlannerDomain::method_verify_goal);
}

Variant PlannerDomain::method_verify_goal(Dictionary p_state, String p_method, String p_state_variable, String p_arguments, Variant p_desired_value, int p_depth, int p_verbose) {
	// Check if state variable exists - if not, goal wasn't achieved
	if (!p_state.has(p_state_variable)) {
		if (p_verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve - state variable %s doesn't exist\nGoal %s[%s] = %s", p_depth, p_method, p_state_variable, p_state_variable, p_arguments, p_desired_value));
		}
		return false;
	}

	Variant state_var_val = p_state[p_state_variable];
	if (state_var_val.get_type() != Variant::DICTIONARY) {
		if (p_verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve - state variable %s is not a dictionary\nGoal %s[%s] = %s", p_depth, p_method, p_state_variable, p_state_variable, p_arguments, p_desired_value));
		}
		return false;
	}

	Dictionary state_dict = state_var_val;
	if (!state_dict.has(p_arguments)) {
		if (p_verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve - argument %s doesn't exist in state variable\nGoal %s[%s] = %s", p_depth, p_method, p_arguments, p_state_variable, p_arguments, p_desired_value));
		}
		return false;
	}

	if (state_dict[p_arguments] != p_desired_value) {
		if (p_verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve\nGoal %s[%s] = %s", p_depth, p_method, p_state_variable, p_arguments, p_desired_value));
		}
		return false;
	}

	if (p_verbose >= 3) {
		print_line(vformat("Depth %d: method %s achieved\nGoal %s[%s] = %s", p_depth, p_method, p_state_variable, p_arguments, p_desired_value));
	}
	return Array();
}

PlannerDomain::PlannerDomain() {
	task_method_dictionary["_verify_g"] = varray(callable_mp_static(&PlannerDomain::method_verify_goal));
	task_method_dictionary["_verify_mg"] = varray(callable_mp_static(&PlannerMultigoal::method_verify_multigoal));
}

void PlannerDomain::add_multigoal_methods(TypedArray<Callable> p_methods) {
	// Add domain-specific methods for multigoals
	// Methods return false or an Array of planner elements (goals, PlannerMultigoal, tasks, actions)
	for (int i = 0; i < p_methods.size(); ++i) {
		Callable m = p_methods[i];
		if (m.is_null()) {
			continue;
		}
		if (!multigoal_method_list.has(m)) {
			multigoal_method_list.push_back(m);
		}
	}
}

void PlannerDomain::add_unigoal_methods(String p_task_name, TypedArray<Callable> p_methods) {
	if (!unigoal_method_dictionary.has(p_task_name)) {
		unigoal_method_dictionary[p_task_name] = p_methods;
	} else {
		TypedArray<Callable> existing_methods = unigoal_method_dictionary[p_task_name];
		for (int i = 0; i < p_methods.size(); ++i) {
			Callable m = p_methods[i];
			if (m.is_null()) {
				continue;
			}
			if (!existing_methods.has(m)) {
				existing_methods.push_back(m);
			}
		}
		unigoal_method_dictionary[p_task_name] = existing_methods;
	}
}

void PlannerDomain::add_task_methods(String p_task_name, TypedArray<Callable> p_methods) {
	if (task_method_dictionary.has(p_task_name)) {
		TypedArray<Callable> existing_methods = task_method_dictionary[p_task_name];
		for (int i = 0; i < p_methods.size(); ++i) {
			Callable m = p_methods[i];
			if (m.is_null()) {
				continue;
			}
			if (existing_methods.find(m) == -1) {
				existing_methods.push_back(m);
			}
		}
		task_method_dictionary[p_task_name] = existing_methods;
	} else {
		task_method_dictionary[p_task_name] = p_methods;
	}
}

void PlannerDomain::add_command(String p_name, Callable p_command) {
	if (p_command.is_null() || p_name.is_empty()) {
		return;
	}
	command_dictionary[p_name] = p_command;
}
