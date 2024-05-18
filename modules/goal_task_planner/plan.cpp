// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// plan.gd
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021
// Author: K. S. Ernest (iFire) Lee <ernest.lee@chibifire.com>, August 28, 2022

#include "plan.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "modules/goal_task_planner/domain.h"

void Plan::print_domain(Ref<Domain> domain) {
	if (domain == nullptr) {
		domain = current_domain;
	}

	print_line(vformat("Domain name: %s", get_name()));

	print_actions(domain);
	print_methods(domain);
	// print_simple_temporal_network(domain);
}

// void Plan::print_simple_temporal_network(Ref<Domain> domain) {
// 	if (domain == nullptr) {
// 		domain = current_domain;
// 	}

// 	if (domain->stn) {
// 		std::cout << "-- Simple Temporal Network: " << domain->stn.to_dictionary() << std::endl;
// 	} else {
// 		std::cout << "-- There is no Simple Temporal Network --" << std::endl;
// 	}
// }
void Plan::print_actions(Ref<Domain> domain) {
	if (domain.is_null()) {
		domain = current_domain;
	}

	if (!domain->get_action_dict().is_empty()) {
		String actions = "-- Actions: ";

		// Create an iterator for the action_dict
		Array keys = domain->get_action_dict().keys();
		for (int i = 0; i < keys.size(); ++i) {
			if (i != 0) {
				actions += ", ";
			}
			actions += String(keys[i]);
		}

		print_line(actions);
	} else {
		print_line("-- There are no actions --");
	}
}
void Plan::_print_task_methods(Ref<Domain> domain) {
	if (!domain->get_task_method_dict().is_empty()) {
		print_line("\nTask name:         Relevant task methods:");
		print_line("---------------    ----------------------");

		String string_array;
		Array keys = domain->get_task_method_dict().keys();
		for (int i = 0; i < keys.size(); ++i) {
			string_array += String(keys[i]) + ", ";
		}
		print_line(string_array.substr(0, string_array.length() - 2)); // Remove last comma and space

		print_line("");
	} else {
		print_line("-- There are no task methods --");
	}
}

void Plan::_print_unigoal_methods(Ref<Domain> domain) {
	if (!domain->get_unigoal_method_dict().is_empty()) {
		print_line("Blackboard var name:    Relevant unigoal methods:");
		print_line("---------------    -------------------------");

		Array keys = domain->get_unigoal_method_dict().keys();
		for (int j = 0; j < keys.size(); ++j) {
			String string_array;
			Array methods = domain->get_unigoal_method_dict()[keys[j]];
			for (int i = 0; i < methods.size(); ++i) {
				String m = methods[i];
				string_array += m + ", ";
			}
			print_line(string_array.substr(0, string_array.length() - 2)); // Remove last comma and space
		}

		print_line("");
	} else {
		print_line("-- There are no unigoal methods --");
	}
}
void Plan::_print_multigoal_methods(Ref<Domain> domain) {
	if (!domain->get_multigoal_method_list().is_empty()) {
		String string_array;
		Array methods = domain->get_multigoal_method_list();
		for (int i = 0; i < methods.size(); ++i) {
			Variant m = methods[i];
			string_array += String(m) + ", ";
		}
		print_line("-- Multigoal methods: " + string_array.substr(0, string_array.length() - 2)); // Remove last comma and space
	} else {
		print_line("-- There are no multigoal methods --");
	}
}

Dictionary Plan::declare_actions(Array actions) {
	// TODO: FIX
	// if (current_domain == nullptr) {
	// 	print_line("Cannot declare actions until a domain has been created.");
	// 	return Dictionary();
	// }

	// for (int i = 0; i < actions.size(); ++i) {
	// 	Variant action = actions[i];
	// 	current_domain->get_action_dict()[action.get_method()] = action;
	// }

	return current_domain->get_action_dict();
}

void Plan::print_methods(Ref<Domain> domain) {
	if (domain == nullptr) {
		domain = current_domain;
	}

	_print_task_methods(domain);
	_print_unigoal_methods(domain);
	_print_multigoal_methods(domain);
}
Dictionary Plan::declare_task_methods(String task_name, Array methods) {
	if (current_domain == nullptr) {
		print_line("Cannot declare methods until a domain has been created.");
		return Dictionary();
	}

	if (current_domain->get_task_method_dict().has(task_name)) {
		// task_name is already in the dictionary
		Array existing_methods = current_domain->get_task_method_dict()[task_name];
		for (int i = 0; i < methods.size(); ++i) {
			Variant m = methods[i];
			// check if method is not already in the list
			if (existing_methods.find(m) == -1) {
				existing_methods.push_back(m);
			}
		}
		current_domain->get_task_method_dict()[task_name] = existing_methods;
	} else {
		// task_name is not in the dictionary, so add it
		current_domain->get_task_method_dict()[task_name] = methods;
	}

	return current_domain->get_task_method_dict();
}

Dictionary Plan::declare_unigoal_methods(StringName state_var_name, Array methods) {
	if (current_domain == nullptr) {
		print_line("Cannot declare methods until a domain has been created.");
		return Dictionary();
	}

	if (!current_domain->_unigoal_method_dict.has(state_var_name)) {
		current_domain->_unigoal_method_dict[state_var_name] = methods;
	} else {
		Array old_methods = current_domain->_unigoal_method_dict[state_var_name];
		Array method_array;
		for (int i = 0; i < methods.size(); ++i) {
			Variant m = methods[i];
			if (!old_methods.has(m)) {
				method_array.push_back(m);
			}
		}
		current_domain->_unigoal_method_dict[state_var_name].append_array(method_array);
	}

	return current_domain->_unigoal_method_dict;
}

Array Plan::declare_multigoal_methods(Array methods) {
	if (current_domain == nullptr) {
		print_line("Cannot declare methods until a domain has been created.");
		return Array();
	}

	Array method_array;
	for (int i = 0; i < methods.size(); ++i) {
		Variant m = methods[i];
		if (!current_domain->_multigoal_method_list.has(m)) {
			method_array.push_back(m);
		}
	}

	current_domain->_multigoal_method_list.append_array(method_array);

	return current_domain->_multigoal_method_list;
}

Array Plan::m_split_multigoal(Dictionary state, Ref<Multigoal> multigoal) {
	Dictionary goal_dict = get_current_domain()->_goals_not_achieved(state, multigoal);
	Array goal_list;

	for (int i = 0; i < goal_dict.size(); ++i) {
		String state_var_name = goal_dict.get_key_at_index(i).operator String();
		Array args = goal_dict[state_var_name];

		for (int j = 0; j < args.size(); ++j) {
			Variant arg = args[j];
			if (goal_dict[state_var_name].has(arg) && goal_dict[state_var_name].size() > 0) {
				Variant val = goal_dict[state_var_name][arg];
				Array goal = [ state_var_name, arg, val ];
				goal_list.push_back(goal);
			}
		}
	}

	if (!goal_list.is_empty()) {
		// achieve goals, then check whether they're all simultaneously true
		goal_list.push_back(multigoal);
	}

	return goal_list;
}

Variant Plan::_apply_action_and_continue(Dictionary state, Array task1, Array todo_list, Array plan, int depth) {
	Callable action = current_domain->get_action_dict()[task1[0]];
	if (verbose >= 2) {
		Array action_info = task1.slice(1);
		action_info.insert(0, action.get_method());

		String action_info_str;
		for (int i = 0; i < action_info.size(); ++i) {
			action_info_str += String(action_info[i]);
			if (i != action_info.size() - 1) { // Not the last element
				action_info_str += ", "; // Add comma for separation
			}
		}
		print_line("Depth " + itos(depth) + ", Action " + action_info_str + ": ");
	}

	Array args = task1.slice(1);
	args.insert(0, state);
	Variant new_state = action.get_object()->callv(action.get_method(), args);

	if (new_state) {
		if (verbose >= 3) {
			print_line("Intermediate computation: Action applied successfully.");
			print_line("New state: " + String(new_state));
		}

		Array new_plan = plan;
		new_plan.push_back(task1);
		return seek_plan(new_state, todo_list, new_plan, depth + 1);
	}

	if (verbose >= 3) {
		print_line("Intermediate computation: Failed to apply action. The new state is not valid.");
		print_line("New state: " + String(new_state));
		// TODO: Restore
		// print_line("Task: " + String(task1));
		// print_line("State: " + String(state));
	}

	if (verbose >= 2) {
		Array action_info = task1.slice(1);
		action_info.insert(0, action.get_method());
		print_line("Recursive call: Not applicable action: " + String(action_info));
	}

	return false;
}

Variant Plan::_refine_task_and_continue(Dictionary state, Array task1, Array todo_list, Array plan, int depth) {
	Array relevant = current_domain->get_task_method_dict()[task1[0]];
	// if (verbose >= 3) {
	// 	Array string_array;
	// 	for (int i = 0; i < relevant.size(); i++) {
	// 		string_array.push_back(relevant[i].get_method());
	// 	}
	// 	print_line("Depth " + itos(depth) + ", Task " + String(task1) + ", Methods " + String(string_array));
	// }

	// for (int i = 0; i < relevant.size(); i++) {
	// 	if (verbose >= 2) {
	// 		print_line("Depth " + itos(depth) + ", Trying method " + String(relevant[i].get_method()) + ": ");
	// 	}

	// 	Array args = task1.slice(1);
	// 	args.insert(0, state);
	// 	Variant subtasks = relevant[i].get_object().callv(relevant[i].get_method(), args);

	// 	if (subtasks.get_type() == Variant::ARRAY) {
	// 		if (verbose >= 3) {
	// 			print_line("Intermediate computation: Method applicable.");
	// 			print_line("Depth " + itos(depth) + ", Subtasks: " + String(subtasks));
	// 		}

	// 		Array new_todo_list = subtasks + todo_list;
	// 		Variant result = seek_plan(state, new_todo_list, plan, depth + 1);

	// 		if (result.get_type() == Variant::ARRAY) {
	// 			return result;
	// 		}
	// 	}
	// }

	// if (verbose >= 2) {
	// 	print_line("Recursive call: Failed to accomplish task: " + String(task1));
	// }

	return false;
}

Variant Plan::_refine_unigoal_and_continue(Dictionary state, Array goal1, Array todo_list, Array plan, int depth) {
	if (verbose >= 3) {
		print_line("Depth " + itos(depth) + ", Goal " + String(goal1) + ": ");
	}

	// TODO: Restore

	// String state_var_name = goal1[0];
	// String arg = goal1[1];
	// Variant val = goal1[2];

	// if (state[state_var_name][arg] == val) {
	// 	if (verbose >= 3) {
	// 		print_line("Intermediate computation: Goal already achieved.");
	// 	}
	// 	return seek_plan(state, todo_list, plan, depth + 1);
	// }

	// Array relevant = current_domain._unigoal_method_dict[state_var_name];

	// if (verbose >= 3) {
	// 	Array string_array;
	// 	for (int i = 0; i < relevant.size(); i++) {
	// 		string_array.push_back(relevant[i].get_method());
	// 	}
	// 	print_line("Methods " + String(string_array));
	// }

	// for (int i = 0; i < relevant.size(); i++) {
	// 	if (verbose >= 2) {
	// 		print_line("Depth " + itos(depth) + ", Trying method " + String(relevant[i].get_method()) + ": ");
	// 	}

	// 	Variant subgoals = relevant[i].get_object().callv(relevant[i].get_method(), Array::make(state, arg, val));

	// 	if (subgoals.get_type() == Variant::ARRAY) {
	// 		if (verbose >= 3) {
	// 			print_line("Depth " + itos(depth) + ", Subgoals: " + String(subgoals));
	// 		}

	// 		Array verification;

	// 		if (verify_goals) {
	// 			verification.push_back(Array::make("_verify_g", String(relevant[i].get_method()), state_var_name, arg, val, depth));
	// 		} else {
	// 			verification.clear();
	// 		}

	// 		todo_list = subgoals + verification + todo_list;
	// 		Variant result = seek_plan(state, todo_list, plan, depth + 1);

	// 		if (result.get_type() == Variant::ARRAY) {
	// 			return result;
	// 		}
	// 	}
	// }

	// if (verbose >= 2) {
	// 	print_line("Recursive call: Failed to achieve goal: " + String(goal1));
	// }

	return false;
}

Variant Plan::_refine_multigoal_and_continue(Dictionary state, Ref<Multigoal> goal1, Array todo_list, Array plan, int depth) {
	// if (verbose >= 3) {
	// 	print_line("Depth " + itos(depth) + ", Multigoal " + String(goal1) + ": ");
	// }

	// Array relevant = current_domain->get_multigoal_method_list();

	// if (verbose >= 3) {
	// 	Array string_array;
	// 	for (int i = 0; i < relevant.size(); i++) {
	// 		string_array.push_back(relevant[i].get_method());
	// 	}
	// 	print_line("Methods " + String(string_array));
	// }

	// for (int i = 0; i < relevant.size(); i++) {
	// 	if (verbose >= 2) {
	// 		print_line("Depth " + itos(depth) + ", Trying method " + String(relevant[i].get_method()) + ": ");
	// 	}

	// 	Variant subgoals = relevant[i].get_object().callv(relevant[i].get_method(), Array::make(state, goal1));

	// 	if (subgoals.get_type() == Variant::ARRAY) {
	// 		if (verbose >= 3) {
	// 			print_line("Intermediate computation: Method applicable.");
	// 			print_line("Depth " + itos(depth) + ", Subgoals: " + String(subgoals));
	// 		}

	// 		Array verification;

	// 		if (verify_goals) {
	// 			verification.push_back(Array::make("_verify_mg", String(relevant[i].get_method()), goal1, depth));
	// 		} else {
	// 			verification.clear();
	// 		}

	// 		todo_list = subgoals + verification + todo_list;
	// 		Variant result = seek_plan(state, todo_list, plan, depth + 1);

	// 		if (result.get_type() == Variant::ARRAY) {
	// 			return result;
	// 		}
	// 	} else {
	// 		if (verbose >= 3) {
	// 			print_line("Intermediate computation: Method not applicable: " + String(relevant[i]));
	// 		}
	// 	}
	// }

	// if (verbose >= 2) {
	// 	print_line("Recursive call: Failed to achieve multigoal: " + String(goal1));
	// }

	return false;
}

Variant Plan::find_plan(Dictionary state, Array todo_list) {
	return false; // REMOVE WHEN READY

	// TODO: Fix
	// if (verbose >= 1) {
	// 	String todo_string;
	// 	for (int i = 0; i < todo_list.size(); ++i) {
	// 		todo_string += String(todo_list[i]);
	// 		if (i != todo_list.size() - 1) { // Not the last element
	// 			todo_string += ", "; // Add comma for separation
	// 		}
	// 	}
	// 	print_line("FindPlan> find_plan, verbose=" + itos(verbose) + ":");
	// 	print_line("    state = " + String(state) + "\n    todo_list = " + todo_string);
	// }

	// Variant result = seek_plan(state, todo_list, Array(), 0);

	// if (verbose >= 1) {
	// 	print_line("FindPlan> result = " + String(result) + "\n");
	// }

	// return result;
}

Variant Plan::seek_plan(Dictionary state, Array todo_list, Array plan, int depth) {
	if (verbose >= 2) {
		Array todo_array;
		for (int i = 0; i < todo_list.size(); i++) {
			todo_array.push_back(_item_to_string(todo_list[i]));
		}
		String todo_string = "[" + String(", ").join(todo_array) + "]";
		print_line("Depth " + itos(depth) + " todo_list " + todo_string);
	}

	if (todo_list.is_empty()) {
		if (verbose >= 3) {
			print_line("depth " + itos(depth) + " no more tasks or goals, return plan");
		}
		return plan;
	}

	Variant item1 = todo_list.front();
	todo_list.pop_front();

	if (Object::cast_to<Multigoal>(item1)) {
		return _refine_multigoal_and_continue(state, item1, todo_list, plan, depth);
	} else if (item1.get_type() == Variant::ARRAY) {
		Array item1_array = item1;
		if (current_domain->get_action_dict().has(item1_array[0])) {
			return _apply_action_and_continue(state, item1, todo_list, plan, depth);
		} else if (current_domain->get_task_method_dict().has(item1_array[0])) {
			return _refine_task_and_continue(state, item1, todo_list, plan, depth);
		} else if (current_domain->get_unigoal_method_dict().has(item1_array[0])) {
			return _refine_unigoal_and_continue(state, item1, todo_list, plan, depth);
		}
	}

	print_line("Depth " + itos(depth) + ": " + String(item1) + " isn't an action, task, unigoal, or multigoal\n");

	return false;
}

String Plan::_item_to_string(Variant item) {
	return String(item);
}

Dictionary Plan::run_lazy_lookahead(Dictionary state, Array todo_list, int max_tries) {
	if (verbose >= 1) {
		print_line(vformat("RunLazyLookahead> run_lazy_lookahead, verbose = %s, max_tries = %s", verbose, max_tries));
		print_line(vformat("RunLazyLookahead> initial state: %s", state.keys()));
		print_line(vformat("RunLazyLookahead> To do: %s", todo_list));
	}

	Dictionary ordinals;
	ordinals[1] = "st";
	ordinals[2] = "nd";
	ordinals[3] = "rd";

	for (int tries = 1; tries <= max_tries; tries++) {
		if (verbose >= 1) {
			print_line(vformat("RunLazyLookahead> %sth call to find_plan:", tries, ordinals.get(tries, "")));
		}

		Variant plan = find_plan(state, todo_list);
		if (
				plan.is_null() || (plan.get_type() == Variant::ARRAY && ((Array)plan).is_empty()) || (plan.get_type() == Variant::DICTIONARY && ((Dictionary)plan).is_empty())) {
			if (verbose >= 1) {
				print_line("run_lazy_lookahead: find_plan has failed");
			}
			return state;
		}

		if (
				plan.is_null() || (plan.get_type() == Variant::ARRAY && ((Array)plan).is_empty()) || (plan.get_type() == Variant::DICTIONARY && ((Dictionary)plan).is_empty())) {
			if (verbose >= 1) {
				print_line(vformat("RunLazyLookahead> Empty plan => success\nafter %s calls to find_plan.", tries));
			}
			if (verbose >= 2) {
				print_line(vformat("> final state %s", state));
			}
			return state;
		}

		if (plan.get_type() != Variant::BOOL) {
			Array action_list = plan;
			for (int i = 0; i < action_list.size(); i++) {
				Array action = action_list[i];
				Callable action_name = current_domain->get_action_dict()[action[0]];
				if (verbose >= 1) {
					print_line(vformat("RunLazyLookahead> Task: %s", action_name + action.slice(1, action.size())));
				}

				Dictionary new_state = _apply_task_and_continue(state, action_name, action.slice(1, action.size()));
				if (!new_state.is_empty()) {
					if (verbose >= 2) {
						print_line(new_state);
					}
					state = new_state;
				} else {
					if (verbose >= 1) {
						print_line(vformat("RunLazyLookahead> WARNING: action %s failed; will call find_plan.", action_name));
					}
					break;
				}
			}
		}

		if (verbose >= 1 && !state.is_empty()) {
			print_line("RunLazyLookahead> Plan ended; will call find_plan again.");
		}
	}

	if (verbose >= 1) {
		print_line("RunLazyLookahead> Too many tries, giving up.");
	}
	if (verbose >= 2) {
		print_line(vformat("RunLazyLookahead> final state %s", state));
	}

	return state;
}

Variant Plan::_apply_task_and_continue(Dictionary state, Callable command, Array args) {
	if (verbose >= 3) {
		print_line(vformat("_apply_command_and_continue %s, args = %s", command.get_method().to_string(), args.to_string()));
	}

	Variant next_state;
	Array call_args = { state };
	call_args.append_array(args);
	command.callv(command.get_method(), call_args, &next_state);

	if (!next_state) {
		if (verbose >= 3) {
			print_line(vformat("Not applicable command %s", command.get_method().to_string(), args.to_string()));
		}
		return false;
	}

	if (verbose >= 3) {
		print_line("Applied");
		print_line(next_state);
	}

	return next_state;
}

void Plan::_bind_methods() {}
