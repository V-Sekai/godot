#include "domain.h"

void Domain::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_m_verify_g", "state", "method", "state_var", "arg", "desired_val", "depth"), &Domain::_m_verify_g);
	ClassDB::bind_static_method("Domain", D_METHOD("_goals_not_achieved", "state", "multigoal"), &Domain::_goals_not_achieved);
	ClassDB::bind_method(D_METHOD("_m_verify_mg", "state", "method", "multigoal", "depth"), &Domain::_m_verify_mg);
	ClassDB::bind_method(D_METHOD("display"), &Domain::display);

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

Variant Domain::_m_verify_g(Dictionary state, String method, String state_var, String arg, Variant desired_val, int depth) {
	Dictionary state_dict = state[state_var];

	if (state_dict[arg] != desired_val) {
		if (verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve\nGoal %s[%s] = %s", depth, method, state_var, arg, desired_val));
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
		print_line(vformat("Depth %d: method %s achieved\nGoal %s[%s] = %s", depth, method, state_var, arg, desired_val));
	}
	return Array();
}

Dictionary Domain::_goals_not_achieved(Dictionary state, Ref<Multigoal> multigoal) {
	Dictionary incomplete;
	Array keys = multigoal->get_state().keys();
	for (int i = 0; i < keys.size(); ++i) {
		String n = keys[i];
		Dictionary sub_dict = multigoal->get_state()[n];
		Array sub_keys = sub_dict.keys();
		for (int j = 0; j < sub_keys.size(); ++j) {
			String arg = sub_keys[j];
			Variant val = sub_dict[arg];
			if (state[n].get_type() == Variant::DICTIONARY && Dictionary(state[n]).has(arg) && val != Dictionary(state[n])[arg]) {
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

Variant Domain::_m_verify_mg(Dictionary state, String method, Ref<Multigoal> multigoal, int depth) {
	Dictionary goal_dict = _goals_not_achieved(state, multigoal);
	if (!goal_dict.is_empty()) {
		if (verbose >= 3) {
			print_line(vformat("Depth %d: method %s didn't achieve %s", depth, method, multigoal));
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
		print_line(vformat("Depth %d: method %s achieved %s", depth, method, multigoal));
	}
	return Array();
}

void Domain::display() {
	print_line(to_string());
}
