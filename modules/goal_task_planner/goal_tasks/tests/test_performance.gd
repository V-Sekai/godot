# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# test_performance.gd
# SPDX-License-Identifier: MIT

extends "res://addons/gut/test.gd"

var stn: SimpleTemporalNetwork = null


func before_each() -> void:
	stn = SimpleTemporalNetwork.new()


func calculate_time_interval(i, temporal_qualifier) -> Array:
	if temporal_qualifier == 1:
		return [i * 10, i * 10 + 5]
	else:
		return [(i * 10) + 5, (i + 1) * 10]


func test_performance_with_large_number_of_constraints_fixme() -> void:
	var start_time = Time.get_ticks_msec()

	for i in range(800):
		var qualifier_1 = TemporalConstraint.TemporalQualifier.AT_START
		var interval_1 = calculate_time_interval(i, qualifier_1)
		var from_constraint = TemporalConstraint.new(
			interval_1[0], interval_1[1], 5, qualifier_1, "from" + str(i)
		)
		var qualifier_2 = TemporalConstraint.TemporalQualifier.AT_END
		var interval_2 = calculate_time_interval(i + 1, qualifier_2)
		var to_constraint = TemporalConstraint.new(
			interval_2[0], interval_2[1], 5, qualifier_2, "to" + str(i)
		)

		# Add constraints and propagate them.
		stn.add_temporal_constraint(from_constraint, to_constraint, 0, 10)

	assert_true(stn.is_consistent(), "Consistency test failed: STN should be consistent")
	var end_time = Time.get_ticks_msec()
	var time_taken = end_time - start_time
	gut.p("Time taken for constraints: " + str(time_taken) + " ms")
	assert_false(
		time_taken < 500, "Performance test failed: Time taken should be faster than 0.5 seconds"
	)


func test_performance_with_large_number_of_constraints_failure_fixme() -> void:
	var start_time = Time.get_ticks_msec()

	for i in range(500):
		var qualifier_1 = TemporalConstraint.TemporalQualifier.AT_START
		var interval_1 = calculate_time_interval(i, qualifier_1)
		var from_constraint = TemporalConstraint.new(
			interval_1[0], interval_1[1], 5, qualifier_1, "from" + str(i)
		)
		var qualifier_2 = TemporalConstraint.TemporalQualifier.AT_END
		var interval_2 = calculate_time_interval(i + 1, qualifier_2)
		var to_constraint = TemporalConstraint.new(
			interval_2[0], interval_2[1], 5, qualifier_2, "to" + str(i + 1)
		)

		# Add constraints and propagate them.
		stn.add_temporal_constraint(from_constraint, to_constraint, 0, 10)

	var qualifier_3 = TemporalConstraint.TemporalQualifier.AT_START
	var interval_3 = calculate_time_interval(0, qualifier_3)
	var _from_constraint = TemporalConstraint.new(
		interval_3[0], interval_3[1], 5, qualifier_3, "from" + str(0)
	)
	var qualifier_4 = TemporalConstraint.TemporalQualifier.AT_END
	var interval_4 = calculate_time_interval(0 + 1, qualifier_4)
	var _to_constraint = TemporalConstraint.new(
		interval_4[0], interval_4[1], 5, qualifier_4, "to" + str(1)
	)
	assert_true(
		stn.check_overlap(_from_constraint), "Consistency test failed: STN should not be consistent"
	)
	assert_true(
		stn.check_overlap(_to_constraint), "Consistency test failed: STN should not be consistent"
	)
	var end_time = Time.get_ticks_msec()
	var time_taken = end_time - start_time
	gut.p("Time taken for constraints: " + str(time_taken) + " ms")
	assert_true(
		time_taken < 500, "Performance test failed: Time taken should be faster than 0.5 seconds"
	)
