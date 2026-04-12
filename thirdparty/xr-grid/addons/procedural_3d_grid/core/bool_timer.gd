# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# bool_timer.gd
# SPDX-License-Identifier: MIT

class_name BoolTimer extends Object

## Sets a bool to true for a certain amount of time, then resets

var reset_time: int = 0
var value: bool:
	get = _get_value


## Set the value to true for time seconds
func set_true(time: float) -> void:
	reset_time = max(reset_time, Time.get_ticks_msec() + int(time * 1000.0))


## Overwrite the value to true for time seconds
func overwrite(time: float) -> void:
	reset_time = Time.get_ticks_msec() + int(time * 1000.0)


func intersect(time: float) -> void:
	reset_time


## Set the value to false and reset timer
func reset() -> void:
	reset_time = Time.get_ticks_msec() - 1


# Value { get { return Time.time <= reset_time; } }
func _get_value():
	return Time.get_ticks_msec() <= reset_time
