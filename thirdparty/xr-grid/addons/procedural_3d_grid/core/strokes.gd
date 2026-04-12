# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# strokes.gd
# SPDX-License-Identifier: MIT

extends MeshInstance3D

@export var hand_left: XRController3D
@export var hand_right: XRController3D

@onready var simple_sketch: SimpleSketch = SimpleSketch.new()


func _ready() -> void:
	simple_sketch.target_mesh = mesh


var prev_hand_left_transform: Transform3D
var prev_hand_right_transform: Transform3D
var prev_hand_left_pressed: float
var prev_hand_right_pressed: float


func _process(delta: float) -> void:
	var hand_left_pressed: float = hand_left.get_float("trigger") > 0.05
	var hand_right_pressed: float = hand_right.get_float("trigger") > 0.05
	var max_size: float = 0.01

	if not is_zero_approx(hand_left_pressed):
		var from: Vector3 = to_local(prev_hand_left_transform.origin)
		var to: Vector3 = to_local(hand_left.global_transform.origin)

		var hand_left_just_pressed: bool = is_zero_approx(hand_left_pressed)

		simple_sketch.add_line(from, to, prev_hand_left_pressed * max_size, hand_left_pressed * max_size, Color(1, 1, 1), Color(1, 1, 1), hand_left_just_pressed)

	if not is_zero_approx(hand_right_pressed):
		var from: Vector3 = to_local(prev_hand_right_transform.origin)
		var to: Vector3 = to_local(hand_right.global_transform.origin)

		var hand_right_just_pressed: bool = is_zero_approx(hand_right_pressed)

		simple_sketch.add_line(from, to, prev_hand_right_pressed * max_size, hand_right_pressed * max_size, Color(0, 0, 0), Color(0, 0, 0), hand_right_just_pressed)

	prev_hand_left_transform = hand_left.global_transform
	prev_hand_right_transform = hand_right.global_transform
	prev_hand_left_pressed = hand_left_pressed
	prev_hand_right_pressed = hand_right_pressed
