// Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
// K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
// plan.h
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2021 University of Maryland
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Dana Nau <nau@umd.edu>, July 7, 2021
// Author: K. S. Ernest (iFire) Lee <ernest.lee@chibifire.com>, August 28, 2022

#ifndef PLAN_H
#define PLAN_H

#include "core/io/resource.h"
// #include "core/variant/array.h"
// #include "modules/goal_task_planner/domain.h"
// #include "modules/goal_task_planner/multigoal.h"

class Plan : public Resource {
	GDCLASS(Plan, Resource);

// protected:
// 	static void _bind_methods();

// public:
// 	Plan();

// private:
// 	int verbose;
// 	Array domains;
// 	Ref<Domain> current_domain;

// 	int _next_state_number;
// 	int _next_multigoal_number;
// 	int _next_domain_number;

// 	// ##
// 	// ##If verify_goals is True, then whenever the planner uses a method m to refine
// 	// ##a unigoal or multigoal, it will insert a "verification" task into the
// 	// ##current partial plan. If verify_goals is False, the planner won't insert any
// 	// ##verification tasks into the plan.
// 	// ##
// 	// ##The purpose of the verification task is to raise an exception if the
// 	// ##refinement produced by m doesn't achieve the goal or multigoal that it is
// 	// ##supposed to achieve. The verification task won't insert anything into the
// 	// ##final plan; it just will verify whether m did what it was supposed to do.
// 	bool verify_goals = true;

// public:
// 	int get_verbose() const { return verbose; }
// 	Array get_domains() const { return domains; }
// 	Ref<Domain> get_current_domain() const { return current_domain; }
// 	int get_next_state_number() const { return _next_state_number; }
// 	int get_next_multigoal_number() const { return _next_multigoal_number; }
// 	int get_next_domain_number() const { return _next_domain_number; }

// 	void set_verbose(int v) { verbose = v; }
// 	void set_domains(Array d) { domains = d; }
// 	void set_current_domain(Ref<Resource> cd) { current_domain = cd; }
// 	void set_next_state_number(int nsn) { _next_state_number = nsn; }
// 	void set_next_multigoal_number(int nmn) { _next_multigoal_number = nmn; }
// 	void set_next_domain_number(int ndn) { _next_domain_number = ndn; }

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
// 	Variant find_plan(Dictionary state, Array todo_list);
// 	Variant seek_plan(Dictionary state, Array todo_list, Array plan, int depth);
// 	String _item_to_string(Variant item);
// 	Dictionary run_lazy_lookahead(Dictionary state, Array todo_list, int max_tries = 10);
// 	Variant _apply_task_and_continue(Dictionary state, Callable command, Array args);
};

#endif // PLAN_H
