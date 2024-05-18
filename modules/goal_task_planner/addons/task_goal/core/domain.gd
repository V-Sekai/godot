# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# domain.gd
# SPDX-License-Identifier: MIT

# SPDX-FileCopyrightText: 2021 University of Maryland
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Author: Dana Nau <nau@umd.edu>, July 7, 2021
# Author: K. S. Ernest (iFire) Lee <ernest.lee@chibifire.com>, August 28, 2022
#
## A class for holding planning-and-acting domains.
##
## d = Domain(domain_name) creates an object to contain the actions, commands,
## and methods for a planning-and-acting domain. 'domain_name' is the name to
## use for the new domain.

extends Resource

const verbose: int = 0

var _action_dict: Dictionary = {}

var _task_method_dict: Dictionary = {
	"_verify_g": [Callable(self, "_m_verify_g")],
	"_verify_mg": [Callable(self, "_m_verify_mg")],
}

var _unigoal_method_dict: Dictionary = {}

var _multigoal_method_list: Array = []

var stn: SimpleTemporalNetwork = SimpleTemporalNetwork.new()

func _m_verify_g(
	state: Dictionary,
	method: String,
	state_var: String,
	arg: String,
	desired_val: Variant,
	depth: int
) -> Variant:
	if state[state_var][arg] != desired_val:
		if verbose >= 3:
			print(
				(
					"Depth %s: method %s didn't achieve\nGoal %s[%s] = %s"
					% [depth, method, state_var, arg, desired_val]
				)
			)
		return false

	if not stn.is_consistent():
		if verbose >= 3:
			print(
				(
					"Depth %s: method %s resulted in inconsistent STN for %s"
					% [depth, method]
				)
			)
		return false

	if verbose >= 3:
		print(
			(
				"Depth %s: method %s achieved\nGoal %s[%s] = %s"
				% [depth, method, state_var, arg, desired_val]
			)
		)
	return []


static func _goals_not_achieved(state: Dictionary, multigoal: Multigoal) -> Dictionary:
	var unachieved: Dictionary = {}
	for n in multigoal.state.keys():
		for arg in multigoal.state[n]:
			var val = multigoal.state[n][arg]
			if state[n].has(arg) and val != state[n][arg]:
				if not unachieved.has(n):
					unachieved[n] = {}
				unachieved[n][arg] = val
	return unachieved


func _m_verify_mg(state: Dictionary, method: String, multigoal: Multigoal, depth: int) -> Variant:
	var goal_dict = _goals_not_achieved(state, multigoal)
	if goal_dict:
		if verbose >= 3:
			print("Depth %s: method %s didn't achieve %s" % [depth, method, multigoal])
		return false

	if not stn.is_consistent():
		if verbose >= 3:
			print(
				(
					"Depth %s: method %s resulted in inconsistent STN for %s"
					% [depth, method]
				)
			)
		return false

	if verbose >= 3:
		print("Depth %s: method %s achieved %s" % [depth, method, multigoal])
	return []


func _init(domain_name: String) -> void:
	set_name(domain_name)


func display() -> void:
	print(self)
