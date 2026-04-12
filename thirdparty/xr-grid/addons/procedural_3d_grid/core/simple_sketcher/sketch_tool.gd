# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# sketch_tool.gd
# SPDX-License-Identifier: MIT

#@tool
class_name SketchTool extends Node3D

@export var CANVAS: NodePath
@onready var canvas: Node3D = get_node(CANVAS)

@export var active: bool = false
@export var pressure: float = 0.0
@export var color: Color = Color.BLACK

@onready var simple_sketch = SimpleSketch.new()


func _ready() -> void:
	if canvas == null:
		printerr("Cannot make the canvas ready.")
		return

	simple_sketch.target_mesh = canvas.get_node("strokes").mesh


var prev_active: bool = false


func _process(_delta: float) -> void:
	if active and not prev_active:
		simple_sketch.stroke_begin()

	if active and prev_active:
		var point = canvas.to_local(global_transform.origin)
		simple_sketch.stroke_add(point, pressure / canvas.scale.x, color)

	if not active and prev_active:
		simple_sketch.stroke_end()

	prev_active = active
