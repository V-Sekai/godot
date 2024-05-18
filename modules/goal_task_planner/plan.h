/**************************************************************************/
/*  plan.h                                                                */
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

#ifndef PLAN_H
#define PLAN_H
// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// plan.h
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021
// Author: K. S. Ernest (iFire) Lee <ernest.lee@chibifire.com>, August 28, 2022

#include "core/io/resource.h"
#include "core/string/print_string.h"
#include "core/variant/callable.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"

#include "domain.h"
#include <cstdint>

class Plan : public Resource {
	GDCLASS(Plan, Resource);

private:
	int verbose;
	TypedArray<Domain> domains;
	Ref<Domain> current_domain;

	int64_t _next_state_number;
	int64_t _next_multigoal_number;
	int64_t _next_domain_number;

	// ##
	// ##If verify_goals is True, then whenever the planner uses a method m to refine
	// ##a unigoal or multigoal, it will insert a "verification" task into the
	// ##current partial plan. If verify_goals is False, the planner won't insert any
	// ##verification tasks into the plan.
	// ##
	// ##The purpose of the verification task is to raise an exception if the
	// ##refinement produced by m doesn't achieve the goal or multigoal that it is
	// ##supposed to achieve. The verification task won't insert anything into the
	// ##final plan; it just will verify whether m did what it was supposed to do.
	bool verify_goals = true;

public:
	int get_verbose() const { return verbose; }
	TypedArray<Domain> get_domains() const { return domains; }
	Ref<Domain> get_current_domain() const { return current_domain; }
	int get_next_state_number() const { return _next_state_number; }
	int get_next_multigoal_number() const { return _next_multigoal_number; }
	int get_next_domain_number() const { return _next_domain_number; }

	void set_verbose(int v) { verbose = v; }
	void set_domains(TypedArray<Domain> d) { domains = d; }
	void set_current_domain(Ref<Resource> cd) { current_domain = cd; }
	void set_next_state_number(int nsn) { _next_state_number = nsn; }
	void set_next_multigoal_number(int nmn) { _next_multigoal_number = nmn; }
	void set_next_domain_number(int ndn) { _next_domain_number = ndn; }

	void print_domain(Ref<Domain> domain) {
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
	void print_actions(Ref<Domain> domain) {
		if (domain.is_null()) {
			domain = current_domain;
		}

		if (!domain->get_action_dictionary().is_empty()) {
			String actions = "-- Actions: ";

			// Create an iterator for the action_dict
			Array keys = domain->get_action_dictionary().keys();
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
	void _print_task_methods(Ref<Domain> domain) {
		if (!domain->get_task_method_dictionary().is_empty()) {
			print_line("\nTask name:         Relevant task methods:");
			print_line("---------------    ----------------------");

			String string_array;
			Array keys = domain->get_task_method_dictionary().keys();
			for (int i = 0; i < keys.size(); ++i) {
				string_array += String(keys[i]) + ", ";
			}
			print_line(string_array.substr(0, string_array.length() - 2)); // Remove last comma and space

			print_line("");
		} else {
			print_line("-- There are no task methods --");
		}
	}

	void _print_unigoal_methods(Ref<Domain> domain) {
		if (!domain->get_unigoal_method_dictionary().is_empty()) {
			print_line("Blackboard var name:    Relevant unigoal methods:");
			print_line("---------------    -------------------------");

			Array keys = domain->get_unigoal_method_dictionary().keys();
			for (int j = 0; j < keys.size(); ++j) {
				String string_array;
				Array methods = domain->get_unigoal_method_dictionary()[keys[j]];
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
	void _print_multigoal_methods(Ref<Domain> domain) {
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

	void print_methods(Ref<Domain> domain) {
		if (domain == nullptr) {
			domain = current_domain;
		}

		_print_task_methods(domain);
		_print_unigoal_methods(domain);
		_print_multigoal_methods(domain);
	}

	Dictionary declare_actions(Array p_actions) {
		if (current_domain.is_null()) {
			print_line("Cannot declare actions until a domain has been created.");
			return Dictionary();
		}

		for (int i = 0; i < p_actions.size(); ++i) {
			Callable action = p_actions[i];
			if (action.is_null()) {
				continue;
			}
			current_domain->get_action_dictionary()[action.get_method()] = action;
		}

		return current_domain->get_action_dictionary();
	}

	Dictionary declare_task_methods(String p_task_name, Array p_methods) {
		if (current_domain == nullptr) {
			print_line("Cannot declare methods until a domain has been created.");
			return Dictionary();
		}

		if (current_domain->get_task_method_dictionary().has(p_task_name)) {
			// task_name is already in the dictionary
			Array existing_methods = current_domain->get_task_method_dictionary()[p_task_name];
			for (int i = 0; i < p_methods.size(); ++i) {
				Variant m = p_methods[i];
				// check if method is not already in the list
				if (existing_methods.find(m) == -1) {
					existing_methods.push_back(m);
				}
			}
			current_domain->get_action_dictionary()[p_task_name] = existing_methods;
		} else {
			// The task_name is not in the dictionary, so add it.
			current_domain->get_task_method_dictionary()[p_task_name] = p_methods;
		}

		return current_domain->get_task_method_dictionary();
	}

	Dictionary declare_unigoal_methods(StringName p_state_var_name, Array p_methods) {
		if (current_domain == nullptr) {
			print_line("Cannot declare methods until a domain has been created.");
			return Dictionary();
		}

		if (!current_domain->get_unigoal_method_dictionary().has(p_state_var_name)) {
			current_domain->get_unigoal_method_dictionary()[p_state_var_name] = p_methods;
		} else {
			Array existing_methods = current_domain->get_unigoal_method_dictionary()[p_state_var_name];
			Array method_array;
			for (int i = 0; i < p_methods.size(); ++i) {
				Variant m = p_methods[i];
				if (!existing_methods.has(m)) {
					method_array.push_back(m);
				}
			}
			Array existing_method_array = current_domain->get_unigoal_method_dictionary()[p_state_var_name];
			existing_method_array.append_array(method_array);
			current_domain->get_unigoal_method_dictionary()[p_state_var_name] = existing_method_array;
		}

		return current_domain->get_unigoal_method_dictionary();
	}
	Array declare_multigoal_methods(Array p_methods) {
		if (current_domain == nullptr) {
			print_line("Cannot declare methods until a domain has been created.");
			return Array();
		}

		Array method_array;
		for (int i = 0; i < p_methods.size(); ++i) {
			Variant m = p_methods[i];
			if (!current_domain->get_multigoal_method_list().has(m)) {
				method_array.push_back(m);
			}
		}

		current_domain->get_multigoal_method_list().append_array(method_array);

		return current_domain->get_multigoal_method_list();
	}

	Array m_split_multigoal(Dictionary p_state, Ref<Multigoal> p_multigoal) {
		Dictionary goal_dict = get_current_domain()->_goals_not_achieved(p_state, p_multigoal);
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

	Variant _apply_action_and_continue(Dictionary state, Array task1, Array todo_list, Array plan, int depth) {
		Callable action = current_domain->get_action_dictionary()[task1[0]];
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
			print_line("Task: ");
			for (int i = 0; i < task1.size(); ++i) {
				print_line(String(task1[i]));
				print_line("State: ");
				for (const Variant *key = state.next(NULL); key != NULL; key = state.next(key)) {
					print_line("Key: " + String(*key) + ", Value: " + String(state[*key]));
				}
			}

			if (verbose >= 2) {
				Array action_info = task1.slice(1);
				action_info.insert(0, action.get_method());
				print_line("Recursive call: Not applicable action: ");
				for (int i = 0; i < action_info.size(); ++i) {
					print_line(String(action_info[i]));
				}
			}
		}
		return false;
	}
	Variant _refine_task_and_continue(Dictionary state, Array task1, Array todo_list, Array plan, int depth) {
		Array relevant = current_domain->get_task_method_dictionary()[task1[0]];
		if (verbose >= 3) {
			Array string_array;
			for (int i = 0; i < relevant.size(); i++) {
				string_array.push_back(relevant[i].call("get_method"));
			}

			String task1_string;
			for (int i = 0; i < task1.size(); ++i) {
				task1_string += String(task1[i]) + " ";
			}

			String todo_list_string;
			for (int i = 0; i < todo_list.size(); ++i) {
				todo_list_string += String(todo_list[i]) + " ";
			}

			String plan_string;
			for (int i = 0; i < plan.size(); ++i) {
				plan_string += String(plan[i]) + " ";
			}

			String array_string;
			for (int i = 0; i < string_array.size(); ++i) {
				array_string += String(string_array[i]) + " ";
			}

			print_line("Depth " + itos(depth) + ", Task " + task1_string + ", Todo List " + todo_list_string + ", Plan " + plan_string + ", Methods " + array_string);
		}

		for (int i = 0; i < relevant.size(); i++) {
			if (verbose >= 2) {
				print_line("Depth " + itos(depth) + ", Trying method " + String(relevant[i].call("get_method")) + ": ");
			}

			Array args = task1.slice(1);
			args.insert(0, state);
			Variant subtasks = relevant[i].call("get_object").call("callv", relevant[i].call("get_method"), args);

			if (subtasks.get_type() == Variant::ARRAY) {
				if (verbose >= 3) {
					print_line("Intermediate computation: Method applicable.");
					print_line("Depth " + itos(depth) + ", Subtasks: " + String(subtasks));
				}
				Array new_todo_list;
				Array new_subtasks;
				new_subtasks.append(subtasks);
				new_subtasks.append(todo_list);
				new_todo_list.append(new_subtasks);
				Variant result = seek_plan(state, new_todo_list, plan, depth + 1);

				if (result.get_type() == Variant::ARRAY) {
					return result;
				}
			}
		}

		if (verbose >= 2) {
			print_line("Recursive call: Failed to accomplish task: ");
			for (Variant t : task1) {
				print_line(t);
			}
		}

		return false;
	}

	Variant _refine_multigoal_and_continue(Dictionary state, Ref<Multigoal> goal1, Array todo_list, Array plan, int depth) {
		if (verbose >= 3) {
			print_line("Depth " + itos(depth) + ", Multigoal " + goal1->get_name() + ": ");
		}

		Array relevant = current_domain->get_multigoal_method_list();

		if (verbose >= 3) {
			Array string_array;
			for (int i = 0; i < relevant.size(); i++) {
				print_line(String("Methods ") + String(relevant[i].call("get_method")));
			}
		}

		for (int i = 0; i < relevant.size(); i++) {
			if (verbose >= 2) {
				print_line("Depth " + itos(depth) + ", Trying method " + String(relevant[i].call("get_method")) + ": ");
			}

			Variant subgoals = relevant[i].call("get_object").call("callv", relevant[i].call("get_method"), varray(state, goal1));

			if (subgoals.get_type() == Variant::ARRAY) {
				if (verbose >= 3) {
					print_line("Intermediate computation: Method applicable.");
					print_line("Depth " + itos(depth) + ", Subgoals: " + String(subgoals));
				}

				Array verification;

				if (verify_goals) {
					verification.push_back(varray("_verify_mg", String(relevant[i].call("get_method")), goal1, depth));
				} else {
					verification.clear();
				}

				todo_list.clear();
				todo_list.append_array(subgoals);
				todo_list.append_array(verification);
				todo_list.append_array(todo_list);
				Variant result = seek_plan(state, todo_list, plan, depth + 1);

				if (result.get_type() == Variant::ARRAY) {
					return result;
				}
			} else {
				if (verbose >= 3) {
					print_line("Intermediate computation: Method not applicable: " + String(relevant[i]));
				}
			}
		}

		if (verbose >= 2) {
			print_line("Recursive call: Failed to achieve multigoal: " + goal1->get_name());
		}

		return false;
	}

	Variant _refine_unigoal_and_continue(Dictionary state, Array goal1, Array todo_list, Array plan, int depth) {
		if (verbose >= 3) {
			for (int i = 0; i < goal1.size(); ++i) {
				print_line("Depth " + itos(depth) + ", Goal " + String(goal1[i]) + ": ");
			}
		}

		String state_var_name = goal1[0];
		String arg = goal1[1];
		Variant val = goal1[2];

		if (state[state_var_name].get(arg) == val) {
			if (verbose >= 3) {
				print_line("Intermediate computation: Goal already achieved.");
			}
			return seek_plan(state, todo_list, plan, depth + 1);
		}

		Array relevant = current_domain->get_unigoal_method_dictionary()[state_var_name];

		if (verbose >= 3) {
			Array string_array;
			for (int i = 0; i < relevant.size(); i++) {
				string_array.push_back(relevant[i].call("get_method"));
			}
			for (int i = 0; i < string_array.size(); ++i) {
				print_line("Methods " + String(string_array[i]));
			}
		}

		for (int i = 0; i < relevant.size(); i++) {
			if (verbose >= 2) {
				print_line("Depth " + itos(depth) + ", Trying method " + String(relevant[i].call("get_method")) + ": ");
			}

			Variant subgoals = relevant[i].call("get_object").call("callv", relevant[i].call("get_method"), varray(state, arg, val));

			if (subgoals.get_type() == Variant::ARRAY) {
				if (verbose >= 3) {
					print_line("Depth " + itos(depth) + ", Subgoals: " + String(subgoals));
				}

				Array verification;

				if (verify_goals) {
					verification.push_back(varray("_verify_g", String(relevant[i].call("get_method")), state_var_name, arg, val, depth));
				} else {
					verification.clear();
				}

				todo_list.clear();
				todo_list.append_array(subgoals);
				todo_list.append_array(verification);
				todo_list.append_array(todo_list);
				Variant result = seek_plan(state, todo_list, plan, depth + 1);

				if (result.get_type() == Variant::ARRAY) {
					return result;
				}
			}
		}

		if (verbose >= 2) {
			for (int i = 0; i < goal1.size(); ++i) {
				print_line("Recursive call: Failed to achieve goal: " + String(goal1[i]));
			}
		}

		return false;
	}
	Variant find_plan(Dictionary state, Array todo_list) {
		if (verbose >= 1) {
			print_line("FindPlan> find_plan, verbose=" + itos(verbose) + ":");
			String todo_string;
			for (int i = 0; i < todo_list.size(); i++) {
				todo_string += itos(i) + String(": ") + String(todo_list[i]) + ", ";
			}

			if (!todo_string.is_empty()) {
				todo_string.erase(todo_string.size() - 2);
			}

			String state_string;
			Array keys = state.keys();
			for (int i = 0; i < keys.size(); ++i) {
				Variant key = keys[i];
				Variant value = state[key];
				state_string += String(key) + ": " + String(value) + ", ";
			}

			if (!state_string.is_empty()) {
				state_string.erase(state_string.size() - 2);
			}

			print_line("    state = " + state_string + "\n    todo_list = " + todo_string);
		}

		Variant result = seek_plan(state, todo_list, Array(), 0);

		if (verbose >= 1) {
			print_line("FindPlan> result = " + String(result));
		}

		return result;
	}

	Variant seek_plan(Dictionary state, Array todo_list, Array plan, int depth) {
		if (verbose >= 2) {
			Array todo_array;
			for (int i = 0; i < todo_list.size(); i++) {
				todo_array.push_back(_item_to_string(todo_list[i]));
			}
			String todo_string;
			for (int i = 0; i < todo_array.size(); ++i) {
				todo_string += "[" + String(todo_array[i]) + "]";
			}
			print_line("Depth " + itos(depth) + ", Todo List Item " + todo_string);
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
			if (current_domain->get_action_dictionary().has(item1_array[0])) {
				return _apply_action_and_continue(state, item1, todo_list, plan, depth);
			} else if (current_domain->get_task_method_dictionary().has(item1_array[0])) {
				return _refine_task_and_continue(state, item1, todo_list, plan, depth);
			} else if (current_domain->get_unigoal_method_dictionary().has(item1_array[0])) {
				return _refine_unigoal_and_continue(state, item1, todo_list, plan, depth);
			}
		}

		print_line("Depth " + itos(depth) + ": " + String(item1) + " isn't an action, task, unigoal, or multigoal\n");

		return false;
	}
	String _item_to_string(Variant item) {
		return String(item);
	}

	Dictionary run_lazy_lookahead(Dictionary state, Array todo_list, int max_tries = 10) {
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
					Callable action_name = current_domain->get_action_dictionary()[action[0]];
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

	Variant _apply_task_and_continue(Dictionary state, Callable command, Array args) {
		if (verbose >= 3) {
			print_line(vformat("_apply_command_and_continue %s, args = %s", command.get_method(), args));
		}
		Array argument;
		argument.push_back(state);
		argument.append_array(args);
		Variant next_state = command.get_object()->callv(command.get_method(), argument);
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

protected:
	static void _bind_methods() {}
};

#endif // PLAN_H
