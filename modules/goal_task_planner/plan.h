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

// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021

#ifndef PLAN_H
#define PLAN_H

#include "core/io/resource.h"
#include "core/variant/callable.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"

#include "domain.h"

class Plan : public Resource {
	GDCLASS(Plan, Resource);

private:
	int verbose;
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

public:
	int get_verbose() const;
	TypedArray<Domain> get_domains() const;
	Ref<Domain> get_current_domain() const;
	void set_verbose(int p_level);
	void set_domains(TypedArray<Domain> p_domain);
	void set_current_domain(Ref<Domain> p_current_domain) { current_domain = p_current_domain; }
	Dictionary declare_actions(Array p_actions);
	Dictionary declare_task_methods(String p_task_name, Array p_methods);
	Dictionary declare_unigoal_methods(StringName p_state_var_name, Array p_methods);
	Array declare_multigoal_methods(Array p_methods);
	Array m_split_multigoal(Dictionary p_state, Ref<Multigoal> p_multigoal);
	Variant find_plan(Dictionary state, Array todo_list);
	Variant seek_plan(Dictionary state, Array todo_list, Array plan, int depth);
	Dictionary run_lazy_lookahead(Dictionary state, Array todo_list, int max_tries = 10);

public:
	Variant _apply_action_and_continue(Dictionary state, Array task1, Array todo_list, Array plan, int depth);
	Variant _refine_task_and_continue(Dictionary state, Array task1, Array todo_list, Array plan, int depth);
	Variant _refine_multigoal_and_continue(Dictionary state, Ref<Multigoal> goal1, Array todo_list, Array plan, int depth);
	Variant _refine_unigoal_and_continue(Dictionary state, Array goal1, Array todo_list, Array plan, int depth);
	String _item_to_string(Variant item);
	Variant _apply_task_and_continue(Dictionary state, Callable command, Array args);

protected:
	static void _bind_methods();
};

#endif // PLAN_H
