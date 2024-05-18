# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# temporal_constraint.gd
# SPDX-License-Identifier: MIT

extends Resource

class_name TemporalConstraint

enum TemporalQualifier {
	AT_START,
	AT_END,
	OVERALL,
}

@export var time_interval: Vector2i
@export var duration: int
@export var temporal_qualifier: TemporalQualifier


func _init(start: int, end: int, duration: int, qualifier: TemporalQualifier, resource: String):
	time_interval = Vector2i(start, end)
	self.duration = duration
	temporal_qualifier = qualifier
	resource_name = resource


func overlaps_with(other: TemporalConstraint) -> bool:
	if self.resource_name != other.resource_name:
		return false

	var self_start: int
	var self_end: int
	var other_start: int
	var other_end: int
	
	# Calculate the effective start and end times based on the temporal qualifier
	match self.temporal_qualifier:
		TemporalQualifier.AT_START:
			self_start = time_interval.x
			self_end = time_interval.x + duration
		TemporalQualifier.AT_END:
			self_start = time_interval.y - duration
			self_end = time_interval.y
		TemporalQualifier.OVERALL:
			self_start = time_interval.x
			self_end = time_interval.y

	match other.temporal_qualifier:
		TemporalQualifier.AT_START:
			other_start = other.time_interval.x
			other_end = other.time_interval.x + other.duration
		TemporalQualifier.AT_END:
			other_start = other.time_interval.y - other.duration
			other_end = other.time_interval.y
		TemporalQualifier.OVERALL:
			other_start = other.time_interval.x
			other_end = other.time_interval.y

	# Check if intervals overlap
	return self_start < other_end and other_start < self_end


func _to_string() -> String:
	return str(
		{
			"resource_name": resource_name,
			"time_interval": time_interval,
			"duration": duration,
			"temporal_qualifier": temporal_qualifier
		}
	)


static func sort_func(a: TemporalConstraint, b: TemporalConstraint) -> bool:
	return (
		a.time_interval.x < b.time_interval.x
		or (a.time_interval.x == b.time_interval.x and a.time_interval.y < b.time_interval.y)
	)
