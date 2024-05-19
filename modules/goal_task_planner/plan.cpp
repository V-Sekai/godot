/**************************************************************************/
/*  plan.cpp                                                              */
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

#include "plan.h"
#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"
#include "domain.h"

int Plan::get_verbose() const { return verbose; }

TypedArray<Domain> Plan::get_domains() const { return domains; }

Ref<Domain> Plan::get_current_domain() const { return current_domain; }

void Plan::set_verbose(int v) { verbose = v; }

void Plan::set_domains(TypedArray<Domain> d) { domains = d; }

Dictionary Plan::declare_actions(TypedArray<Callable> p_actions) {
	if (current_domain.is_null()) {
		print_line("Cannot declare actions until a domain has been created.");
		return Dictionary();
	}

	Dictionary action_dictionary = current_domain->get_actions();

	for (int64_t i = 0; i < p_actions.size(); ++i) {
		Callable action = p_actions[i];
		if (action.is_null()) {
			continue;
		}
		String method_name = action.get_method();
		action_dictionary[method_name] = action;
	}

	current_domain->set_actions(action_dictionary);

	return action_dictionary;
}

Dictionary Plan::declare_task_methods(String p_task_name, TypedArray<Callable> p_methods) {
	if (current_domain == nullptr) {
		print_line("Cannot declare methods until a domain has been created.");
		return Dictionary();
	}

	Dictionary task_method_dictionary = current_domain->get_task_methods();

	if (task_method_dictionary.has(p_task_name)) {
		// task_name is already in the dictionary
		Array existing_methods = task_method_dictionary[p_task_name];
		for (int i = 0; i < p_methods.size(); ++i) {
			Variant m = p_methods[i];
			// check if method is not already in the list
			if (existing_methods.find(m) == -1) {
				existing_methods.push_back(m);
			}
		}
		task_method_dictionary[p_task_name] = existing_methods;
	} else {
		// The task_name is not in the dictionary, so add it.
		task_method_dictionary[p_task_name] = p_methods;
	}

	current_domain->set_task_methods(task_method_dictionary);

	return task_method_dictionary;
}

Dictionary Plan::declare_unigoal_methods(StringName p_state_variable_name, TypedArray<Callable> p_methods) {
	if (current_domain.is_null()) {
		print_line("Cannot declare methods until a domain has been created.");
		return Dictionary();
	}

	Dictionary unigoal_method_dict = current_domain->get_unigoal_methods();

	if (!unigoal_method_dict.has(p_state_variable_name)) {
		unigoal_method_dict[p_state_variable_name] = p_methods;
	} else {
		Array existing_methods = unigoal_method_dict[p_state_variable_name];
		for (int i = 0; i < p_methods.size(); ++i) {
			Variant m = p_methods[i];
			if (!existing_methods.has(m)) {
				existing_methods.push_back(m);
			}
		}
		unigoal_method_dict[p_state_variable_name] = existing_methods;
	}

	current_domain->set_unigoal_methods(unigoal_method_dict); // Assuming this setter method exists

	return unigoal_method_dict;
}

Array Plan::declare_multigoal_methods(TypedArray<Callable> p_methods) {
	if (current_domain == nullptr) {
		print_line("Cannot declare methods until a domain has been created.");
		return Array();
	}

	Array method_array = current_domain->get_multigoal_methods();

	for (int i = 0; i < p_methods.size(); ++i) {
		Variant m = p_methods[i];
		if (!method_array.has(m)) {
			method_array.push_back(m);
		}
	}

	current_domain->set_multigoal_methods(method_array);

	return method_array;
}

Array Plan::method_split_multigoal(Dictionary p_state, Ref<Multigoal> p_multigoal) {
	Dictionary goal_dict = get_current_domain()->method_goals_not_achieved(p_state, p_multigoal);
	Array goal_list;

	for (int i = 0; i < goal_dict.size(); ++i) {
		String state_var_name = goal_dict.get_key_at_index(i).operator String();
		Array args = goal_dict[state_var_name];
		for (int j = 0; j < args.size(); ++j) {
			Variant arg = args[j];
			if (goal_dict[state_var_name].call("has", arg) && int64_t(goal_dict[state_var_name].call("size")) > 0) {
				Variant val = goal_dict[state_var_name].get(arg);
				Array goal;
				goal.resize(3);
				goal[0] = state_var_name;
				goal[1] = arg;
				goal[2] = val;
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

Variant Plan::_apply_action_and_continue(Dictionary state, Array task1, Array todo_list, Array p_plan, int depth) {
	Callable action = current_domain->get_actions()[task1[0]];

	if (verbose >= 2) {
		Array action_info = task1.slice(1);
		action_info.insert(0, action.get_method());
		print_line("Depth " + itos(depth) + ", Action: " + _item_to_string(action_info));
	}

	Array arguments = task1.slice(1);
	arguments.insert(0, state);
	Variant new_state = action.callv(arguments);

	if (new_state) {
		if (verbose >= 3) {
			print_line("Intermediate computation: Action applied successfully.");
			print_line("New state: " + String(new_state));
		}
		TypedArray<Array> new_plan = p_plan;
		new_plan.push_back(task1);
		return seek_plan(new_state, todo_list, new_plan, depth + 1);
	}

	if (verbose >= 3) {
		print_line("Intermediate computation: Failed to apply action. The new state is not valid.");
		print_line("New state: " + String(new_state));
		print_line("Task: ");
		for (int i = 0; i < task1.size(); ++i) {
			print_line(String(task1[i]));
			print_line("State: " + _item_to_string(state));
		}

		if (verbose >= 2) {
			Array action_info = task1.slice(1);
			action_info.insert(0, action.get_method());
			print_line("Recursive call: Not applicable action: " + _item_to_string(action_info));
		}
	}
	return false;
}

Variant Plan::_refine_task_and_continue(const Dictionary p_state, const Array p_task1, const Array p_todo_list, const Array p_plan, const int p_depth) {
	Array relevant = current_domain->get_task_methods()[p_task1[0]];
	if (verbose >= 3) {
		print_line("Depth " + itos(p_depth) + ", Task " + _item_to_string(p_task1) + ", Todo List " + _item_to_string(p_todo_list) + ", Plan " + _item_to_string(p_plan));
	}
	for (int i = 0; i < relevant.size(); i++) {
		Callable method = relevant[i];
		Array arguments;
		arguments.push_back(p_state);
		arguments.append_array(p_task1.slice(1));
		if (verbose >= 2) {
			print_line("Depth " + itos(p_depth) + ", Trying method: " + _item_to_string(method));
		}
		Variant result = method.callv(arguments);
		if (result.is_array()) {
			Array subgoals = result;
			Array todo_list;
			if (!subgoals.is_empty()) {
				todo_list.append_array(subgoals);
			}
			if (!p_todo_list.is_empty()) {
				todo_list.append_array(p_todo_list);
			}
			Variant plan = seek_plan(p_state, todo_list, p_plan, p_depth + 1);
			if (plan.is_array()) {
				return plan;
			}
		} else {
			if (verbose >= 3) {
				print_line("Not applicable");
			}
		}
	}

	if (verbose >= 2) {
		print_line("Recursive call: Failed to accomplish task: " + _item_to_string(p_task1));
	}

	return false;
}

Variant Plan::_refine_multigoal_and_continue(const Dictionary p_state, const Ref<Multigoal> p_goal1, const Array p_todo_list, const Array p_plan, const int p_depth) {
	if (verbose >= 3) {
		print_line("Depth " + itos(p_depth) + ", Multigoal " + p_goal1->get_name() + ": " + _item_to_string(p_goal1));
	}

	Array relevant = current_domain->get_multigoal_methods();

	if (verbose >= 3) {
		Array string_array;
		for (int i = 0; i < relevant.size(); i++) {
			print_line(String("Methods ") + String(relevant[i].call("get_method")));
		}
	}
	Array todo_list = p_todo_list;
	for (int i = 0; i < relevant.size(); i++) {
		if (verbose >= 2) {
			print_line("Depth " + itos(p_depth) + ", Trying method " + String(relevant[i].call("get_method")) + ": ");
		}
		Callable callable = relevant[i];
		Array arguments;
		arguments.push_back(p_state);
		arguments.push_back(p_goal1);
		Variant result = callable.callv(arguments);
		if (result.is_array()) {
			Array subgoals = result;
			TypedArray<Array> subtodo_list;
			if (verbose >= 3) {
				print_line("Intermediate computation: Method applicable.");
				print_line("Depth " + itos(p_depth) + ", Subgoals: " + _item_to_string(subgoals));
			}
			if (!subgoals.is_empty()) {
				subtodo_list.append_array(subgoals);
			}
			if (verify_goals) {
				subtodo_list.push_back(varray("_verify_mg", callable.get_method(), p_goal1, p_depth, verbose));
			}
			if (!p_todo_list.is_empty()) {
				subtodo_list.append_array(p_todo_list);
			}
			todo_list.clear();
			todo_list = subtodo_list;
			Variant plan = seek_plan(p_state, todo_list, p_plan, p_depth + 1);
			if (plan.is_array()) {
				return plan;
			}
		}
	}

	if (verbose >= 2) {
		print_line("Recursive call: Failed to achieve multigoal: " + p_goal1->get_name());
	}

	return false;
}

Variant Plan::_refine_unigoal_and_continue(const Dictionary p_state, const Array p_goal1, const Array p_todo_list, const Array p_plan, const int p_depth) {
	if (verbose >= 3) {
		String goals_list = vformat("Depth %d, Goals: %s", p_depth, _item_to_string(p_goal1));
	}

	String state_variable_name = p_goal1[0];
	String argument = p_goal1[1];
	Variant value = p_goal1[2];

	Dictionary state_variable = p_state[state_variable_name];

	if (state_variable[argument] == value) {
		if (verbose >= 3) {
			print_line("Intermediate computation: Goal already achieved.");
		}
		return seek_plan(p_state, p_todo_list, p_plan, p_depth + 1);
	}

	Array relevant = current_domain->get_unigoal_methods()[state_variable_name];
	if (verbose >= 3) {
		print_line("Methods: " + _item_to_string(relevant));
	}
	Array todo_list = p_todo_list;
	for (int i = 0; i < relevant.size(); i++) {
		Callable method = relevant[i];
		if (verbose >= 2) {
			print_line("Depth " + itos(p_depth) + ", Trying method " + _item_to_string(method));
		}
		Variant result = method.call(p_state, argument, value);
		if (result.is_array()) {
			Array subgoals = result;
			TypedArray<Array> subtodo_list;
			if (!subgoals.is_empty()) {
				subtodo_list.append_array(subgoals);
			}
			if (verify_goals) {
				subtodo_list.push_back(varray("_verify_g", method.get_method(), state_variable_name, argument, value, p_depth, verbose));
			}
			if (!todo_list.is_empty()) {
				subtodo_list.append_array(todo_list);
			}
			todo_list.clear();
			todo_list = subtodo_list;
			if (verbose >= 3) {
				print_line("Depth " + itos(p_depth) + ", Seeking todo list " + _item_to_string(todo_list));
			}
			Variant plan = seek_plan(p_state, todo_list, p_plan, p_depth + 1);
			if (plan.is_array()) {
				return plan;
			}
		} else {
			if (verbose >= 3) {
				print_line("Not applicable.");
			}
		}
	}

	if (verbose >= 2) {
		print_line("Recursive call: Failed to achieve goal: " + _item_to_string(p_goal1));
	}

	return false;
}

Variant Plan::find_plan(Dictionary state, Array todo_list) {
	if (verbose >= 1) {
		print_line("FindPlan> find_plan, verbose=" + itos(verbose) + ":");
		print_line("    state = " + _item_to_string(state) + "\n    todo_list = " + _item_to_string(todo_list));
	}

	Variant result = seek_plan(state, todo_list, Array(), 0);

	if (verbose >= 1) {
		print_line("FindPlan> result = " + _item_to_string(result));
	}

	return result;
}

Variant Plan::seek_plan(Dictionary p_state, Array p_todo_list, Array p_plan, int p_depth) {
	if (verbose >= 2) {
		print_line("Depth " + itos(p_depth) + ", Todo List: " + _item_to_string(p_todo_list));
	}

	if (p_todo_list.is_empty()) {
		if (verbose >= 3) {
			print_line("Depth " + itos(p_depth) + " no more tasks or goals, return plan: " + _item_to_string(p_plan));
		}
		return p_plan;
	}
	Variant todo_item = p_todo_list.front();
	p_todo_list = p_todo_list.slice(1);
	if (Object::cast_to<Multigoal>(todo_item)) {
		return _refine_multigoal_and_continue(p_state, todo_item, p_todo_list, p_plan, p_depth);
	} else if (todo_item.is_array()) {
		Array item = todo_item;
		Dictionary actions = current_domain->get_actions();
		Dictionary tasks = current_domain->get_task_methods();
		Dictionary unigoals = current_domain->get_unigoal_methods();
		Variant item_name = item.front();
		bool is_action = actions.has(item_name);
		bool is_task = tasks.has(item_name);
		bool is_unigoal = unigoals.has(item_name);
		if (is_action) {
			return _apply_action_and_continue(p_state, item, p_todo_list, p_plan, p_depth);
		} else if (is_task) {
			return _refine_task_and_continue(p_state, item, p_todo_list, p_plan, p_depth);
		} else if (is_unigoal) {
			return _refine_unigoal_and_continue(p_state, item, p_todo_list, p_plan, p_depth);
		}
	}
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
				Callable action_name = current_domain->get_actions()[action[0]];
				if (verbose >= 1) {
					String action_arguments;
					Array actions = action.slice(1, action.size());
					for (Variant element : actions) {
						action_arguments += String(" ") + String(element);
					}
					print_line(vformat("RunLazyLookahead> Task: %s, %s", action_name.get_method(), action_arguments));
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

Variant Plan::_apply_task_and_continue(Dictionary p_state, Callable p_command, Array p_arguments) {
	if (verbose >= 3) {
		print_line(vformat("_apply_command_and_continue %s, args = %s", p_command.get_method(), _item_to_string(p_arguments)));
	}
	Array argument;
	argument.push_back(p_state);
	argument.append_array(p_arguments);
	Variant next_state = p_command.get_object()->callv(p_command.get_method(), argument);
	if (!next_state) {
		if (verbose >= 3) {
			print_line(vformat("Not applicable command %s", argument));
		}
		return false;
	}

	if (verbose >= 3) {
		print_line("Applied");
		print_line(next_state);
	}
	return next_state;
}

void Plan::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_verify_goals"), &Plan::get_verify_goals);
    ClassDB::bind_method(D_METHOD("set_verify_goals", "value"), &Plan::set_verify_goals);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "verify_goals"), "set_verify_goals", "get_verify_goals");

	ClassDB::bind_method(D_METHOD("get_verbose"), &Plan::get_verbose);
	ClassDB::bind_method(D_METHOD("set_verbose", "level"), &Plan::set_verbose);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "verbose"), "set_verbose", "get_verbose");

	ClassDB::bind_method(D_METHOD("get_domains"), &Plan::get_domains);
	ClassDB::bind_method(D_METHOD("set_domains", "domain"), &Plan::set_domains);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "domains", PROPERTY_HINT_RESOURCE_TYPE, "Domain"), "set_domains", "get_domains");

	ClassDB::bind_method(D_METHOD("get_current_domain"), &Plan::get_current_domain);
	ClassDB::bind_method(D_METHOD("set_current_domain", "current_domain"), &Plan::set_current_domain);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "current_domain", PROPERTY_HINT_RESOURCE_TYPE, "Domain"), "set_current_domain", "get_current_domain");

	ClassDB::bind_method(D_METHOD("declare_actions", "actions"), &Plan::declare_actions);
	ClassDB::bind_method(D_METHOD("declare_task_methods", "task_name", "methods"), &Plan::declare_task_methods);
	ClassDB::bind_method(D_METHOD("declare_unigoal_methods", "state_var_name", "methods"), &Plan::declare_unigoal_methods);
	ClassDB::bind_method(D_METHOD("declare_multigoal_methods", "methods"), &Plan::declare_multigoal_methods);

	ClassDB::bind_method(D_METHOD("method_split_multigoal", "state", "multigoal"), &Plan::method_split_multigoal);

	ClassDB::bind_method(D_METHOD("find_plan", "state", "todo_list"), &Plan::find_plan);
	ClassDB::bind_method(D_METHOD("seek_plan", "state", "todo_list", "plan", "depth"), &Plan::seek_plan);
	ClassDB::bind_method(D_METHOD("run_lazy_lookahead", "state", "todo_list", "max_tries"), &Plan::run_lazy_lookahead, DEFVAL(10));
}

bool Plan::get_verify_goals() const {
	return verify_goals;
}

void Plan::set_verify_goals(bool value) {
	verify_goals = value;
}
