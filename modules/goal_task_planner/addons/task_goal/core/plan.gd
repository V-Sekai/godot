# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# plan.gd
# SPDX-License-Identifier: MIT

# SPDX-FileCopyrightText: 2021 University of Maryland
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Author: Dana Nau <nau@umd.edu>, July 7, 2021
# Author: K. S. Ernest (iFire) Lee <ernest.lee@chibifire.com>, August 28, 2022

extends Resource

## Task Goal is an automated planning system that can plan for both tasks and
## goals.

## How much information to print while the program is running
##
## verbose is a global value whose initial value is 1. Its value determines how
## much debugging information GTPyhop will print:
## - verbose = 0: print nothing
## - verbose = 1: print the initial parameters and the answer
## - verbose = 2: also print a message on each recursive call
## - verbose = 3: also print some info about intermediate computations
@export var verbose: int = 0

## A list of all domains that have been created
@export var domains: Array[Resource] = []

# #The Domain object that find_plan, run_lazy_lookahead, etc., will use.
@export var current_domain: Resource

## Sequence number to use when making copies of states.
var _next_state_number: int = 0

## Sequence number to use when making copies of multigoals.
var _next_multigoal_number: int = 0

## Sequence number to use when making copies of domains.
var _next_domain_number: int = 0

const _domain_const = preload("domain.gd")


##	Print domain's actions, commands, and methods. The optional 'domain'
##	argument defaults to the current domain
func print_domain(domain: Object = null) -> void:
	if domain == null:
		domain = current_domain
	print("Domain name: %s" % resource_name)
	print_actions(domain)
	print_methods(domain)
	print_simple_temporal_network(domain)


func print_simple_temporal_network(domain: Object = null) -> void:
	if domain == null:
		domain = current_domain
	if domain.stn:
		print("-- Simple Temporal Network: %s" % domain.stn.to_dictionary())
	else:
		print("-- There is no Simple Temporal Network --")


## Print the names of all the actions
func print_actions(domain: Object = null) -> void:
	if domain == null:
		domain = current_domain
	if domain._action_dict:
		print("-- Actions:", ", ".join(domain._action_dict.keys()))
	else:
		print("-- There are no actions --")


func _print_task_methods(domain: Object = null) -> void:
	if domain._task_method_dict:
		print("")
		print("Task name:         Relevant task methods:")
		print("---------------    ----------------------")
		var string_array: Array = Array()
		for task in domain._task_method_dict:
			string_array.append(task)
		print("{task:<19}" + ", ".join(string_array))
		print("")
	else:
		print("-- There are no task methods --")


func _print_unigoal_methods(domain: Object = null) -> void:
	if domain._unigoal_method_dict:
		print("Blackboard var name:    Relevant unigoal methods:")
		print("---------------    -------------------------")
		for v in domain._unigoal_method_dict:
			var string_array: PackedStringArray = PackedStringArray()
			for f in domain._unigoal_method_dict[v]:
				string_array.push_back(f.get_method())
			print("{var:<19}" + ", ".join(string_array))
		print("")
	else:
		print("-- There are no unigoal methods --")


func _print_multigoal_methods(domain: Object = null) -> void:
	if domain._multigoal_method_list:
		var string_array: PackedStringArray = PackedStringArray()
		for f in domain._multigoal_method_list:
			string_array.push_back(f.get_method())
		print(
			"-- Multigoal methods:",
			", ".join(string_array),
		)
	else:
		print("-- There are no multigoal methods --")


## Print tables showing what all the methods are
func print_methods(domain: Object = null) -> void:
	if domain == null:
		domain = current_domain
	_print_task_methods(domain)
	_print_unigoal_methods(domain)
	_print_multigoal_methods(domain)


##	declare_actions adds each member of 'actions' to the current domain's list
##	of actions. For example, this says that pickup and putdown are actions:
##		declare_actions(pickup,putdown)
##
##	declare_actions can be called multiple times to add more actions.
##
##	You can see the current domain's list of actions by executing
##		current_domain.display()
func declare_actions(actions):
	if current_domain == null:
		print("Cannot declare actions until a domain has been created.")
		return []
	for action in actions:
		current_domain._action_dict[action.get_method()] = action
	return current_domain._action_dict


##	'task_name' should be a character string, and 'methods' should be a list
##	of functions. declare_task_methods adds each member of 'methods' to the
##	current domain's list of methods to use for tasks of the form
##		[task_name, arg1, ..., argn].
##
##	Example:
##		declare_task_methods('travel', travel_by_car, travel_by_foot)
##	says that travel_by_car and travel_by_foot are methods and that GTPyhop
##	should try using them for any task whose task name is 'travel', e.g.,
##		['travel', 'alice', 'store']
##		['travel', 'alice', 'umd', 'ucla']
##		['travel', 'alice', 'umd', 'ucla', 'slowly']
##		['travel', 'bob', 'home', 'park', 'looking', 'at', 'birds']
##
##	This is like Pyhop's declare_methods function, except that it can be
##	called several times to declare more methods for the same task.
func declare_task_methods(task_name: StringName, methods: Array):
	if current_domain == null:
		print("Cannot declare methods until a domain has been created.")
		return []
	if task_name in current_domain._task_method_dict:
		var old_methods = current_domain._task_method_dict[task_name]
		# even though current_domain._task_method_dict[task_name] is a list,
		# we don't want to add any methods that are already in it
		var method_arrays: Array = []
		for m in methods:
			if not old_methods.has(m):
				method_arrays.push_back(m)
		current_domain._task_method_dict[task_name].append_array(method_arrays)
	else:
		current_domain._task_method_dict[task_name] = methods
	return current_domain._task_method_dict


##	'state_var_name' should be a character string, and 'methods' should be a
##	list of functions. declare_unigoal_method adds each member of 'methods'
##	to the current domain's list of relevant methods for goals of the form
##		(state_var_name, arg, value)
##	where 'arg' and 'value' are the state variable's argument and the desired
##	value. For example,
##		declare_unigoal_method('loc',travel_by_car)
##	says that travel_by_car is relevant for goals such as these:
##		('loc', 'alice', 'ucla')
##		('loc', 'bob', 'home')
##
##	The above kind of goal, i.e., a desired value for a single state
##	variable, is called a "unigoal". To achieve a unigoal, GTPyhop will go
##	through the unigoal's list of relevant methods one by one, trying each
##	method until it finds one that is successful.
##
##	To see each unigoal's list of relevant methods, use
##		current_domain.display()
func declare_unigoal_methods(state_var_name: StringName, methods: Array):
	if current_domain == null:
		print("Cannot declare methods until a domain has been created.")
		return []
	if not state_var_name in current_domain._unigoal_method_dict.keys():
		current_domain._unigoal_method_dict[state_var_name] = methods
	else:
		var old_methods = current_domain._unigoal_method_dict[state_var_name]
		var method_array: Array = []
		for m in methods:
			if not old_methods.has(m):
				method_array.push_back(m)
		current_domain._unigoal_method_dict[state_var_name].append_array(method_array)
	return current_domain._unigoal_method_dict


##	declare_multigoal_methods adds each method in 'methods' to the current
##	domain's list of multigoal methods. For example, this says that
##	stack_all_blocks and unstack_all_blocks are multigoal methods:
##		declare_multigoal_methods(stack_all_blocks, unstack_all_blocks)
##
##	When GTPyhop tries to achieve a multigoal, it will go through the list
##	of multigoal methods one by one, trying each method until it finds one
##	that is successful. You can see the list by executing
##		current_domain.display()
##
##	declare_multigoal_methods can be called multiple times to add more
##	multigoal methods to the list.
##
##	For more information, see the docstring for the Multigoal class.
func declare_multigoal_methods(methods: Array):
	if current_domain == null:
		print("Cannot declare methods until a domain has been created.")
		return []
	var method_array: Array = []
	for m in methods:
		if not m in current_domain._multigoal_method_list:
			method_array.push_back(m)
	current_domain._multigoal_method_list = current_domain._multigoal_method_list + method_array
	return current_domain._multigoal_method_list


##	m_split_multigoal is the only multigoal method that GTPyhop provides,
##	and GTPyhop won't use it unless the user declares it explicitly using
##		declare_multigoal_methods(m_split_multigoal)
##
##	The method's purpose is to try to achieve a multigoal by achieving each
##	of the multigoal's individual goals sequentially. Parameters:
##		- 'state' is the current state
##		- 'multigoal' is the multigoal to achieve
##	If multigoal is true in the current state, m_split_multigoal returns
##	[]. Otherwise, it returns a goal list
##		[g_1, ..., g_n, multigoal],
##
##	where g_1, ..., g_n are all of the goals in multigoal that aren't true
##	in the current state. This tells the planner to achieve g_1, ..., g_n
##	sequentially, then try to achieve multigoal again. Usually this means
##	m_split_multigal will be used repeatedly, until it succeeds in producing
##	a state in which all of the goals in multigoal are simultaneously true.
##
##	The main problem with m_split_multigoal is that it isn't smart about
##	choosing the order in which to achieve g_1, ..., g_n. Some orderings may
##	work much better than others. Thus, rather than using the method as it's
##	defined below, one might want to modify it to choose a good order, e.g.,
##	by using domain-specific information or a heuristic function.
func m_split_multigoal(state: Dictionary, multigoal: Multigoal):
	var goal_dict: Dictionary = _domain_const._goals_not_achieved(state, multigoal)
	var goal_list: Array = []
	for state_var_name in goal_dict.keys():
		for arg in goal_dict[state_var_name]:
			# print("state_var_name: ", state_var_name)
			# print("arg: ", arg)
			if goal_dict[state_var_name].has(arg) and goal_dict[state_var_name].size() > 0:
				var val = goal_dict[state_var_name][arg]
				goal_list.append([state_var_name, arg, val])
	if not goal_list.is_empty():
		# achieve goals, then check whether they're all simultaneously true
		return goal_list + [multigoal]
	return goal_list


##
##If verify_goals is True, then whenever the planner uses a method m to refine
##a unigoal or multigoal, it will insert a "verification" task into the
##current partial plan. If verify_goals is False, the planner won't insert any
##verification tasks into the plan.
##
##The purpose of the verification task is to raise an exception if the
##refinement produced by m doesn't achieve the goal or multigoal that it is
##supposed to achieve. The verification task won't insert anything into the
##final plan; it just will verify whether m did what it was supposed to do.
var verify_goals: bool = true


## Actions in HTN are atomic units of work, representing the simplest tasks that can't be further broken down. Actions often called primitives.
func _apply_action_and_continue(
	state: Dictionary, task1: Array, todo_list: Array, plan: Array, depth: int
) -> Variant:
	var action: Callable = current_domain._action_dict[task1[0]]
	if verbose >= 2:
		print("Depth %s, Action %s: " % [depth, str([action.get_method()] + task1.slice(1))])
	var new_state = action.get_object().callv(action.get_method(), [state] + task1.slice(1))
	if new_state:
		if verbose >= 3:
			print("Intermediate computation: Action applied successfully.")
			print("New state: ", new_state)
		return seek_plan(new_state, todo_list, plan + [task1], depth + 1)
	if verbose >= 3:
		print("Intermediate computation: Failed to apply action. The new state is not valid.")
		print("New state: ", new_state)
		print("Task: ", task1)
		print("State: ", state)
	if verbose >= 2:
		print(
			"Recursive call: Not applicable action: ", str([action.get_method()] + task1.slice(1))
		)
	return false


func _refine_task_and_continue(
	state: Dictionary, task1: Array, todo_list: Array, plan: Array, depth: int
) -> Variant:
	var relevant: Array = current_domain._task_method_dict[task1[0]]
	if verbose >= 3:
		var string_array: PackedStringArray = []
		for m in relevant:
			string_array.push_back(m.get_method())
		print("Depth %s, Task %s, Methods %s" % [depth, task1, string_array])
	for method in relevant:
		if verbose >= 2:
			print("Depth %s, Trying method %s: " % [depth, method.get_method()])
		var subtasks: Variant = method.get_object().callv(
			method.get_method(), [state] + task1.slice(1)
		)
		if subtasks is Array:
			if verbose >= 3:
				print("Intermediate computation: Method applicable.")
				print("Depth %s, Subtasks: %s" % [depth, subtasks])

			var result: Variant = seek_plan(state, subtasks + todo_list, plan, depth + 1)
			if result is Array:
				return result
	if verbose >= 2:
		print("Recursive call: Failed to accomplish task: ", task1)
	return false


func _refine_unigoal_and_continue(
	state: Dictionary, goal1: Array, todo_list: Array, plan: Array, depth: int
) -> Variant:
	if verbose >= 3:
		print("Depth %s, Goal %s: " % [depth, goal1])

	var state_var_name: String = goal1[0]
	var arg: String = goal1[1]
	var val: Variant = goal1[2]

	if state.get(state_var_name).get(arg) == val:
		if verbose >= 3:
			print("Intermediate computation: Goal already achieved.")
		return seek_plan(state, todo_list, plan, depth + 1)

	var relevant = current_domain._unigoal_method_dict[state_var_name]

	if verbose >= 3:
		var string_array: PackedStringArray = []
		for m in relevant:
			string_array.push_back(m.get_method())
		print("Methods %s " % string_array)

	for method in relevant:
		if verbose >= 2:
			print("Depth %s, Trying method %s: " % [depth, method.get_method()])

		var subgoals: Variant = method.get_object().callv(method.get_method(), [state] + [arg, val])

		if subgoals is Array:
			if verbose >= 3:
				print("Depth %s, Subgoals: %s" % [depth, subgoals])

			var verification = []

			if verify_goals:
				verification = [
					["_verify_g", str(method.get_method()), state_var_name, arg, val, depth]
				]
			else:
				verification = []

			todo_list = subgoals + verification + todo_list
			var result: Variant = seek_plan(state, todo_list, plan, depth + 1)

			if result is Array:
				return result

	if verbose >= 2:
		print("Recursive call: Failed to achieve goal: ", goal1)

	return false


func _refine_multigoal_and_continue(
	state: Dictionary, goal1: Multigoal, todo_list: Array, plan: Array, depth: int
) -> Variant:
	if verbose >= 3:
		print("Depth %s, Multigoal %s: " % [depth, goal1])

	var relevant: Array = current_domain._multigoal_method_list

	if verbose >= 3:
		var string_array: PackedStringArray = PackedStringArray()
		for m in relevant:
			string_array.push_back(m.get_method())
		print("Methods %s" % string_array)

	for method in relevant:
		if verbose >= 2:
			print("Depth %s, Trying method %s: " % [depth, method.get_method()])

		var subgoals: Variant = method.get_object().callv(method.get_method(), [state, goal1])

		if subgoals is Array:
			if verbose >= 3:
				print("Intermediate computation: Method applicable.")
				print("Depth %s, Subgoals: %s" % [depth, subgoals])

			var verification = []

			if verify_goals:
				verification = [["_verify_mg", str(method.get_method()), goal1, depth]]
			else:
				verification = []

			todo_list = subgoals + verification + todo_list
			var result: Variant = seek_plan(state, todo_list, plan, depth + 1)

			if result is Array:
				return result
		else:
			if verbose >= 3:
				print("Intermediate computation: Method not applicable: ", method)

	if verbose >= 2:
		print("Recursive call: Failed to achieve multigoal: ", goal1)

	return false


func find_plan(state: Dictionary, todo_list: Array) -> Variant:
	if verbose >= 1:
		var todo_string = todo_list
		print("FindPlan> find_plan, verbose=%s:" % verbose)
		print("    state = %s\n    todo_list = %s" % [state, todo_string])

	var result: Variant = seek_plan(state, todo_list, [], 0)

	if verbose >= 1:
		print("FindPlan> result = ", result, "\n")

	return result


func seek_plan(state: Dictionary, todo_list: Array, plan: Array, depth: int) -> Variant:
	if verbose >= 2:
		var todo_array: PackedStringArray = []
		for x in todo_list:
			todo_array.push_back(_item_to_string(x))
		var todo_string = "[" + ", ".join(todo_array) + "]"
		print("Depth %s todo_list %s" % [depth, todo_string])

	if todo_list.is_empty():
		if verbose >= 3:
			print("depth %s no more tasks or goals, return plan" % [depth])
		return plan

	var item1 = todo_list.front()
	todo_list.pop_front()

	if item1 is Multigoal:
		return _refine_multigoal_and_continue(state, item1, todo_list, plan, depth)
	elif item1 is Array:
		if item1[0] in current_domain._action_dict.keys():
			return _apply_action_and_continue(state, item1, todo_list, plan, depth)
		elif item1[0] in current_domain._task_method_dict.keys():
			return _refine_task_and_continue(state, item1, todo_list, plan, depth)
		elif item1[0] in current_domain._unigoal_method_dict.keys():
			return _refine_unigoal_and_continue(state, item1, todo_list, plan, depth)

	print("Depth %s: %s isn't an action, task, unigoal, or multigoal\n" % [depth, item1])

	return false


func _item_to_string(item):
	return str(item)


## An adaptation of the run_lazy_lookahead algorithm from Ghallab et al.
## (2016), Automated Planning and Acting. It works roughly like this:
##   loop:
##       plan = find_plan(state, todo_list)
##       if plan = [] then return state    // the new current state
##       for each action in plan:
##           try to execute the corresponding command
##           if the command fails, continue the outer loop
## Arguments:
##   - 'state' is a state;
##   - 'todo_list' is a list of tasks, goals, and multigoals;
##   - max_tries is a bound on how many times to execute the outer loop.
##
## Note: whenever run_lazy_lookahead encounters an action for which there is
## no corresponding command definition, it uses the action definition instead.
func run_lazy_lookahead(state: Dictionary, todo_list: Array, max_tries: int = 10) -> Dictionary:
	if verbose >= 1:
		print(
			(
				"RunLazyLookahead> run_lazy_lookahead, verbose = %s, max_tries = %s"
				% [verbose, max_tries]
			)
		)
		print("RunLazyLookahead> initial state: %s" % [state.keys()])
		print("RunLazyLookahead> To do:", todo_list)

	var ordinals = {1: "st", 2: "nd", 3: "rd"}

	for tries in range(1, max_tries + 1):
		if verbose >= 1:
			print("RunLazyLookahead> %s%s call to find_plan:" % [tries, ordinals.get(tries, "")])

		var plan = find_plan(state, todo_list)
		if (
			plan == null
			or (typeof(plan) == TYPE_ARRAY and plan.is_empty())
			or (typeof(plan) == TYPE_DICTIONARY and !plan)
		):
			if verbose >= 1:
				print("run_lazy_lookahead: find_plan has failed")
			return state

		if (
			plan == null
			or (typeof(plan) == TYPE_ARRAY and plan.is_empty())
			or (typeof(plan) == TYPE_DICTIONARY and !plan)
		):
			if verbose >= 1:
				print("RunLazyLookahead> Empty plan => success\nafter {tries} calls to find_plan.")
			if verbose >= 2:
				print("> final state %s" % [state])
			return state

		if typeof(plan) != TYPE_BOOL:
			for action in plan:
				var action_name = current_domain._action_dict.get(action[0])
				if verbose >= 1:
					print("RunLazyLookahead> Task: %s" % [[action_name] + action.slice(1)])

				var new_state = _apply_task_and_continue(state, action_name, action.slice(1))
				if new_state is Dictionary:
					if verbose >= 2:
						print(new_state)
					state = new_state
				else:
					if verbose >= 1:
						print(
							(
								"RunLazyLookahead> WARNING: action %s failed; will call find_plan."
								% [action_name]
							)
						)
					break

		if verbose >= 1 and state != null:
			print("RunLazyLookahead> Plan ended; will call find_plan again.")

	if verbose >= 1:
		print("RunLazyLookahead> Too many tries, giving up.")
	if verbose >= 2:
		print("RunLazyLookahead> final state %s" % state)

	return state


func _apply_task_and_continue(state: Dictionary, command: Callable, args: Array) -> Variant:
	if verbose >= 3:
		print("_apply_command_and_continue %s, args = %s" % [[str(command.get_method())] + [args]])

	var next_state = command.get_object().callv(command.get_method(), [state] + args)
	if not next_state:
		if verbose >= 3:
			print("Not applicable command %s" % [command.get_method(), args])
		return false

	if verbose >= 3:
		print("Applied")
		print(next_state)
	return next_state
