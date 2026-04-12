# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# xr_pinch.gd
# SPDX-License-Identifier: MIT

extends Node3D

const WorldGrab = preload("res://addons/procedural_3d_grid/core/world_grab.gd")
const BoolTimer = preload("res://addons/procedural_3d_grid/core/bool_timer.gd")

@export var hand_left: XRController3D = null
@export var hand_right: XRController3D = null

var prev_hand_left_transform: Transform3D
var prev_hand_right_transform: Transform3D
var prev_hand_left_grab: float = 0
var prev_hand_right_grab: float = 0

var _world_grab = WorldGrab.new()

var from_pivot: Vector3
var to_pivot: Vector3
var grab_pivot: Vector3
var delta_transform: Transform3D
var target_transform: Transform3D = transform

var damping: float = 6.0
const max_pinch_time: float = 0.2  # sensitivity?

enum Mode {
	NONE,
	GRAB,
	PINCH,
	ORBIT,
}
var state: Mode = Mode.NONE

var left_hand_just_grabbed := BoolTimer.new()
var right_hand_just_grabbed := BoolTimer.new()
var left_hand_just_ungrabbed := BoolTimer.new()
var right_hand_just_ungrabbed := BoolTimer.new()

var linear_velocity: Vector3 = Vector3.ZERO
var angular_velocity: Vector3 = Vector3.ZERO

@export var debounce_duration: float = 0.4
var hand_left_grab_debounce_timer: float = 0.0
var hand_right_grab_debounce_timer: float = 0.0
var last_hand_left_grab_state: bool = false
var last_hand_right_grab_state: bool = false

@export var linear_dampening: float = 0.45
@export var angular_dampening: float = 0.45

func _process(delta_time: float) -> void:
	var hand_left_grab: float = hand_left.get_float("grip")
	var hand_right_grab: float = hand_right.get_float("grip")

	handle_debounce(hand_left_grab, delta_time, true)
	handle_debounce(hand_right_grab, delta_time, false)

	# Dampening
	delta_transform = delta_transform.interpolate_with(Transform3D(), damping * delta_time)

	update_hand_grab_status(last_hand_left_grab_state, prev_hand_left_grab, left_hand_just_grabbed, left_hand_just_ungrabbed)
	update_hand_grab_status(last_hand_right_grab_state, prev_hand_right_grab, right_hand_just_grabbed, right_hand_just_ungrabbed)

	var both_hands_just_ungrabbed: bool = left_hand_just_ungrabbed.value and right_hand_just_ungrabbed.value

	if both_hands_just_grabbed():
		state = Mode.PINCH

	if not (hand_left_grab and hand_right_grab):
		if state != Mode.NONE:
			state = Mode.NONE
			delta_transform = Transform3D()  # Reset delta_transform when not grabbing
			apply_velocity(delta_time)  # Apply stored velocities when not grabbing

	match state:
		Mode.NONE:
			# Do nothing when no hands are grabbing
			pass

		Mode.PINCH:
			if not (hand_left_grab and hand_right_grab):
				state = Mode.NONE

			if hand_left_grab and hand_right_grab and hand_left_grab_debounce_timer >= debounce_duration and hand_right_grab_debounce_timer >= debounce_duration:
				set_pinch_pivot_and_transform(prev_hand_left_transform.origin, prev_hand_right_transform.origin, hand_left.transform.origin, hand_right.transform.origin)
				store_velocity(prev_hand_left_transform, hand_left.transform, delta_time)
				store_velocity(prev_hand_right_transform, hand_right.transform, delta_time)

		Mode.ORBIT:
			if not (hand_left_grab and hand_right_grab):
				state = Mode.NONE

			if hand_left_grab and hand_right_grab and hand_left_grab_debounce_timer >= debounce_duration and hand_right_grab_debounce_timer >= debounce_duration:
				set_orbit_pivot_and_transform(prev_hand_left_transform.origin, prev_hand_right_transform.origin, hand_left.transform.origin, hand_right.transform.origin)
				store_velocity(prev_hand_left_transform, hand_left.transform, delta_time)
				store_velocity(prev_hand_right_transform, hand_right.transform, delta_time)

	# Integrate motion
	target_transform = delta_transform * target_transform

	# Smoothing
	transform = _world_grab.split_blend(transform, target_transform, .3, .1, .1, transform * target_transform.affine_inverse() * to_pivot, to_pivot)

	# Pass data required for the next frame
	prev_hand_left_transform = hand_left.transform
	prev_hand_right_transform = hand_right.transform
	prev_hand_left_grab = hand_left_grab
	prev_hand_right_grab = hand_right_grab

func handle_debounce(current_grab_value: float, delta_time: float, is_left_hand: bool) -> void:
	var grab_debounce_timer: float = hand_right_grab_debounce_timer

	if is_left_hand:
		grab_debounce_timer = hand_left_grab_debounce_timer

	var last_grab_state = last_hand_right_grab_state
	if is_left_hand:
		last_grab_state = last_hand_left_grab_state

	var is_currently_grabbing = current_grab_value > 0
	if is_currently_grabbing != last_grab_state:
		grab_debounce_timer += delta_time
		if grab_debounce_timer >= debounce_duration:
			if is_left_hand:
				last_hand_left_grab_state = is_currently_grabbing
				hand_left_grab_debounce_timer = 0
			else:
				last_hand_right_grab_state = is_currently_grabbing
				hand_right_grab_debounce_timer = 0
	else:
		if is_left_hand:
			hand_left_grab_debounce_timer = 0
		else:
			hand_right_grab_debounce_timer = 0

	# Assign back the values to the class properties
	if is_left_hand:
		hand_left_grab_debounce_timer = grab_debounce_timer
	else:
		hand_right_grab_debounce_timer = grab_debounce_timer

func apply_velocity(delta_time: float) -> void:
	# Apply linear damping, reducing the velocity by the damping factor each frame
	linear_velocity *= (1.0 - linear_dampening)

	# Update the position based on the new velocity
	target_transform.origin += linear_velocity * delta_time

	# Handle the angular movement
	var angular_speed = angular_velocity.length()
	if angular_speed != 0:
		var rotation_axis = angular_velocity.normalized()
		# Rotate the target transform based on angular velocity
		target_transform = target_transform.rotated(rotation_axis, angular_speed * delta_time)

		# Apply angular damping to reduce angular speed over time
		angular_velocity *= (1.0 - angular_dampening)

func store_velocity(prev_hand_transform: Transform3D, hand_transform: Transform3D, delta_time: float) -> void:
	if delta_time > 0:
		var displacement = hand_transform.origin - prev_hand_transform.origin
		linear_velocity = displacement / delta_time

		# Use quaternions to avoid gimbal lock
		var prev_quat = Quaternion(prev_hand_transform.basis)
		var current_quat = Quaternion(hand_transform.basis)
		var quat_difference = current_quat * prev_quat.inverse()

		# Get the axis and the amount of rotation (radians).
		# Note that `get_axis()` returns a normalized vector.
		var rotation_axis = quat_difference.get_axis()
		var rotation_amount = quat_difference.get_angle()

		# Here the angular velocity will be the rotation axis scaled by the amount of rotation over time.
		angular_velocity = rotation_axis * (rotation_amount / delta_time)

func both_hands_just_grabbed() -> bool:
	return left_hand_just_grabbed.value and right_hand_just_grabbed.value

func update_hand_grab_status(hand_grab: float, prev_hand_grab: float, just_grabbed: BoolTimer, just_ungrabbed: BoolTimer) -> void:
	if hand_grab and not prev_hand_grab:
		just_grabbed.set_true(max_pinch_time)
	if not hand_grab and prev_hand_grab:
		just_ungrabbed.set_true(max_pinch_time)

func set_pivot_and_transform(hand_grab: float, prev_hand_transform: Transform3D, hand_transform: Transform3D) -> void:
	if hand_grab:
		from_pivot = prev_hand_transform.origin
		to_pivot = prev_hand_transform.origin
		delta_transform = _world_grab.get_grab_transform(prev_hand_transform, hand_transform)

func set_pinch_pivot_and_transform(prev_hand_left_origin: Vector3, prev_hand_right_origin: Vector3, hand_left_origin: Vector3, hand_right_origin: Vector3) -> void:
	from_pivot = (prev_hand_left_origin + prev_hand_right_origin) / 2.0
	to_pivot = (hand_left_origin + hand_right_origin) / 2.0
	delta_transform = _world_grab.get_pinch_transform(prev_hand_left_origin, prev_hand_right_origin, hand_left_origin, hand_right_origin)

func set_orbit_pivot_and_transform(prev_hand_left_origin: Vector3, prev_hand_right_origin: Vector3, hand_left_origin: Vector3, hand_right_origin: Vector3) -> void:
	from_pivot = prev_hand_left_origin
	to_pivot = prev_hand_right_origin
	delta_transform = _world_grab.get_orbit_transform(prev_hand_left_origin, prev_hand_right_origin, hand_left_origin, hand_right_origin)
