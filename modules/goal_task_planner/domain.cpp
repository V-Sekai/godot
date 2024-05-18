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

void Domain::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_m_verify_g", "state", "method", "state_var", "arg", "desired_val", "depth"), &Domain::_m_verify_g);
	ClassDB::bind_static_method("Domain", D_METHOD("_goals_not_achieved", "state", "multigoal"), &Domain::_goals_not_achieved);
	ClassDB::bind_method(D_METHOD("_m_verify_mg", "state", "method", "multigoal", "depth"), &Domain::_m_verify_mg);

	ClassDB::bind_method(D_METHOD("set_verbose", "value"), &Domain::set_verbose);
	ClassDB::bind_method(D_METHOD("get_verbose"), &Domain::get_verbose);

	ClassDB::bind_method(D_METHOD("set_action_dict", "value"), &Domain::set_action_dict);
	ClassDB::bind_method(D_METHOD("get_action_dict"), &Domain::get_action_dict);

	ClassDB::bind_method(D_METHOD("set_task_method_dict", "value"), &Domain::set_task_method_dict);
	ClassDB::bind_method(D_METHOD("get_task_method_dict"), &Domain::get_task_method_dict);

	ClassDB::bind_method(D_METHOD("set_unigoal_method_dict", "value"), &Domain::set_unigoal_method_dict);
	ClassDB::bind_method(D_METHOD("get_unigoal_method_dict"), &Domain::get_unigoal_method_dict);

	ClassDB::bind_method(D_METHOD("set_multigoal_method_list", "value"), &Domain::set_multigoal_method_list);
	ClassDB::bind_method(D_METHOD("get_multigoal_method_list"), &Domain::get_multigoal_method_list);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "verbose"), "set_verbose", "get_verbose");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "action_dict"), "set_action_dict", "get_action_dict");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "task_method_dict"), "set_task_method_dict", "get_task_method_dict");
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "unigoal_method_dict"), "set_unigoal_method_dict", "get_unigoal_method_dict");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "multigoal_method_list"), "set_multigoal_method_list", "get_multigoal_method_list");
}

Variant Domain::_m_verify_g(Dictionary p_state, String p_method, String p_state_variable, String p_arguments, Variant p_desired_value, int p_depth) {
	Dictionary state_dict = p_state[p_state_variable];

	if (state_dict[p_arguments] != p_desired_value) {
		if (verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve\nGoal %s[%s] = %s", p_depth, p_method, p_state_variable, p_arguments, p_desired_value));
		}
		return false;
	}

	// if (!stn->is_consistent()) {
	//     if (verbose >= 3) {
	//         print_line(vformat("Depth %d: method %s resulted in inconsistent STN for %s", depth, method));
	//     }
	//     return false;
	// }

	if (verbose >= 3) {
		print_line(vformat("Depth %d: method %s achieved\nGoal %s[%s] = %s", p_depth, p_method, p_state_variable, p_arguments, p_desired_value));
	}
	return Array();
}

Dictionary Domain::_goals_not_achieved(Dictionary p_state, Ref<Multigoal> p_multigoal) {
	Dictionary incomplete;
	Array keys = p_multigoal->get_state().keys();
	for (int i = 0; i < keys.size(); ++i) {
		String n = keys[i];
		Dictionary sub_dict = p_multigoal->get_state()[n];
		Array sub_keys = sub_dict.keys();
		for (int j = 0; j < sub_keys.size(); ++j) {
			String arg = sub_keys[j];
			Variant val = sub_dict[arg];
			if (p_state[n].get_type() == Variant::DICTIONARY && Dictionary(p_state[n]).has(arg) && val != Dictionary(p_state[n])[arg]) {
				if (!incomplete.has(n)) {
					incomplete[n] = Dictionary();
				}
				Dictionary temp = incomplete[n];
				temp[arg] = val;
				incomplete[n] = temp;
			}
		}
	}
	return incomplete;
}

Variant Domain::_m_verify_mg(Dictionary p_state, String p_method, Ref<Multigoal> p_multigoal, int p_depth) {
	Dictionary goal_dict = _goals_not_achieved(p_state, p_multigoal);
	if (!goal_dict.is_empty()) {
		if (verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve %s", p_depth, p_method, p_multigoal));
		}
		return false;
	}

	// if (!stn->is_consistent()) {
	//     if (verbose >= 3) {
	//         print_line(vformat("Depth %d: method %s resulted in inconsistent STN for %s", depth, method));
	//     }
	//     return false;
	// }

	if (verbose >= 3) {
		print_line(vformat("Depth %d: method %s achieved %s", p_depth, p_method, p_multigoal));
	}
	return Array();
}

