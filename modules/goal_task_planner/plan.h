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

#pragma once

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

#include "core/io/resource.h"
#include "core/variant/typed_array.h"

#include "modules/goal_task_planner/multigoal.h"

class Domain;
class Plan : public Resource {
	GDCLASS(Plan, Resource);

	int verbose = 0;
	TypedArray<Domain> domains;
	Ref<Domain> current_domain;

	// If verify_goals is True, then whenever the planner uses a method m to refine
	// unigoal or multigoal, it will insert a "verification" task into the
	// current partial plan. If verify_goals is False, the planner won't insert any
	// verification tasks into the plan.
	//
	// The purpose of the verification task is to raise an exception if the
	// refinement produced by m doesn't achieve the goal or multigoal that it is
	// supposed to achieve. The verification task won't insert anything into the
	// final plan; it just will verify whether m did what it was supposed to do.
	bool verify_goals = true;
	static String _item_to_string(Variant p_item);
	Variant _seek_plan(Dictionary p_state, Array p_todo_list, Array p_plan, int p_depth);
	Variant _apply_task_and_continue(Dictionary p_state, Callable p_command, Array p_arguments);
	Variant _apply_action_and_continue(Dictionary p_state, Array p_first_task, Array p_todo_list, Array p_plan, int p_depth);
	Variant _refine_task_and_continue(const Dictionary p_state, const Array p_first_task, const Array p_todo_list, const Array p_plan, const int p_depth);
	Variant _refine_multigoal_and_continue(const Dictionary p_state, const Ref<Multigoal> p_goal, const Array p_todo_list, const Array p_plan, const int p_depth);
	Variant _refine_unigoal_and_continue(const Dictionary p_state, const Array p_first_goal, const Array p_todo_list, const Array p_plan, const int p_depth);

public:
	int get_verbose() const;
	void set_verbose(int p_level);
	TypedArray<Domain> get_domains() const;
	void set_domains(TypedArray<Domain> p_domain);
	Ref<Domain> get_current_domain() const;
	void set_current_domain(Ref<Domain> p_current_domain) { current_domain = p_current_domain; }
	void set_verify_goals(bool p_value);
	bool get_verify_goals() const;
	Variant find_plan(Dictionary p_state, Array p_todo_list);
	Dictionary run_lazy_lookahead(Dictionary p_state, Array p_todo_list, int p_max_tries = 10);

protected:
	static void _bind_methods();
};
