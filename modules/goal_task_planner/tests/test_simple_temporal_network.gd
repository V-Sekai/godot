# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# test_simple_temporal_network.gd
# SPDX-License-Identifier: MIT

extends "res://addons/gut/test.gd"

var stn: SimpleTemporalNetwork = null


func before_each():
	stn = SimpleTemporalNetwork.new()


func test_add_temporal_constraint():
	var from_constraint = TemporalConstraint.new(
		1, 5, 3, TemporalConstraint.TemporalQualifier.AT_START, "resource"
	)
	var result = stn.add_temporal_constraint(from_constraint)
	assert_true(result, "add_temporal_constraint should return true when adding valid constraint")


func test_get_temporal_constraint_by_name():
	var constraint = TemporalConstraint.new(
		1, 5, 3, TemporalConstraint.TemporalQualifier.AT_START, "resource"
	)
	stn.constraints.append(constraint)
	var result = stn.get_temporal_constraint_by_name("resource")
	assert_eq(
		result, constraint, "get_temporal_constraint_by_name should return the correct constraint"
	)


func test_is_consistent():
	var result = stn.is_consistent()
	assert_true(result, "is_consistent should return true when constraints are consistent")


func test_update_state():
	var state = {
		"constraint":
		TemporalConstraint.new(1, 5, 3, TemporalConstraint.TemporalQualifier.AT_START, "resource")
	}
	stn.update_state(state)
	assert_eq(
		stn.constraints.size(), 1, "update_state should add the constraint to the constraints array"
	)


func test_print_constraints_and_check_consistency():
	var constraint = TemporalConstraint.new(
		1, 2, 3, TemporalConstraint.TemporalQualifier.AT_START, "resource"
	)
	stn.add_temporal_constraint(constraint)
	for i in range(stn.constraints.size()):
		gut.p("Constraint " + str(i) + ": " + str(stn.constraints[i]), gut.LOG_LEVEL_ALL_ASSERTS)
	var result = stn.is_consistent()
	assert_true(result, "is_consistent should return true when the network is consistent")


func test_add_constraints_to_list():
	var from_constraint = TemporalConstraint.new(
		1, 2, 3, TemporalConstraint.TemporalQualifier.AT_START, "resource"
	)
	stn.add_constraints_to_list(from_constraint, null)
	assert_eq(
		stn.constraints.size(), 1, "add_constraints_to_list should add one constraint to the list"
	)


func test_process_constraint():
	var from_constraint = TemporalConstraint.new(
		1, 2, 3, TemporalConstraint.TemporalQualifier.AT_START, "resource"
	)
	var node_index = stn.process_constraint(from_constraint)
	assert_true(node_index != null, "process_constraint should return a valid node")
