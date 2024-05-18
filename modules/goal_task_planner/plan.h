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

	// 	void print_domain(Ref<Domain> domain);
	// 	// void print_simple_temporal_network(Ref<Domain> domain = nullptr);
	// 	void print_actions(Ref<Domain> domain = Ref<Resource>());
	// 	void _print_task_methods(Ref<Domain> domain = nullptr);
	// 	void _print_unigoal_methods(Ref<Domain> domain = nullptr);
	// 	void _print_multigoal_methods(Ref<Domain> domain = nullptr);
	// 	void print_methods(Ref<Domain> domain = nullptr);
	// 	Dictionary declare_actions(Array actions);
	// 	Dictionary declare_task_methods(StringName task_name, Array methods);
	// 	Dictionary declare_unigoal_methods(StringName state_var_name, Array methods);
	// 	Array declare_multigoal_methods(Array methods);
	// 	Array m_split_multigoal(Dictionary state, Ref<Multigoal> multigoal);
	// 	Variant _apply_action_and_continue(Dictionary state, Array task1, Array todo_list, Array plan, int depth);
	// 	Variant _refine_task_and_continue(Dictionary state, Array task1, Array todo_list, Array plan, int depth);
	// 	Variant _refine_unigoal_and_continue(Dictionary state, Array goal1, Array todo_list, Array plan, int depth);
	// 	Variant _refine_multigoal_and_continue(Dictionary state, Ref<Multigoal> goal1, Array todo_list, Array plan, int depth);
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
	Variant seek_plan(Dictionary state, Array todo_list, Array plan, int depth) { return Variant(); } // TODO: FIX
	// 	String _item_to_string(Variant item);

	// Dictionary run_lazy_lookahead(Dictionary state, Array todo_list, int max_tries = 10) {
	// 	if (verbose >= 1) {
	// 		print_line(vformat("RunLazyLookahead> run_lazy_lookahead, verbose = %s, max_tries = %s", verbose, max_tries));
	// 		print_line(vformat("RunLazyLookahead> initial state: %s", state.keys()));
	// 		print_line(vformat("RunLazyLookahead> To do: %s", todo_list));
	// 	}

	// 	Dictionary ordinals;
	// 	ordinals[1] = "st";
	// 	ordinals[2] = "nd";
	// 	ordinals[3] = "rd";

	// 	for (int tries = 1; tries <= max_tries; tries++) {
	// 		if (verbose >= 1) {
	// 			print_line(vformat("RunLazyLookahead> %sth call to find_plan:", tries, ordinals.get(tries, "")));
	// 		}

	// 		Variant plan = find_plan(state, todo_list);
	// 		if (
	// 				plan.is_null() || (plan.get_type() == Variant::ARRAY && ((Array)plan).is_empty()) || (plan.get_type() == Variant::DICTIONARY && ((Dictionary)plan).is_empty())) {
	// 			if (verbose >= 1) {
	// 				print_line("run_lazy_lookahead: find_plan has failed");
	// 			}
	// 			return state;
	// 		}

	// 		if (
	// 				plan.is_null() || (plan.get_type() == Variant::ARRAY && ((Array)plan).is_empty()) || (plan.get_type() == Variant::DICTIONARY && ((Dictionary)plan).is_empty())) {
	// 			if (verbose >= 1) {
	// 				print_line(vformat("RunLazyLookahead> Empty plan => success\nafter %s calls to find_plan.", tries));
	// 			}
	// 			if (verbose >= 2) {
	// 				print_line(vformat("> final state %s", state));
	// 			}
	// 			return state;
	// 		}

	// 		if (plan.get_type() != Variant::BOOL) {
	// 			Array action_list = plan;
	// 			for (int i = 0; i < action_list.size(); i++) {
	// 				Array action = action_list[i];
	// 				String action_name = current_domain->get_action_dict()[action[0]];
	// 				if (verbose >= 1) {
	// 					String action_arguments;
	// 					Array actions = action.slice(1, action.size());
	// 					for (Variant element : actions) {
	// 						action_arguments += String(" ") + String(element);
	// 					}
	// 					print_line(vformat("RunLazyLookahead> Task: %s, %s", action_name , action_arguments));
	// 				}

	// 				Dictionary new_state = _apply_task_and_continue(state, action_name, action.slice(1, action.size()));
	// 				if (!new_state.is_empty()) {
	// 					if (verbose >= 2) {
	// 						print_line(new_state);
	// 					}
	// 					state = new_state;
	// 				} else {
	// 					if (verbose >= 1) {
	// 						print_line(vformat("RunLazyLookahead> WARNING: action %s failed; will call find_plan.", action_name));
	// 					}
	// 					break;
	// 				}
	// 			}
	// 		}

	// 		if (verbose >= 1 && !state.is_empty()) {
	// 			print_line("RunLazyLookahead> Plan ended; will call find_plan again.");
	// 		}
	// 	}

	// 	if (verbose >= 1) {
	// 		print_line("RunLazyLookahead> Too many tries, giving up.");
	// 	}
	// 	if (verbose >= 2) {
	// 		print_line(vformat("RunLazyLookahead> final state %s", state));
	// 	}

	// 	return state;
	// }

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
