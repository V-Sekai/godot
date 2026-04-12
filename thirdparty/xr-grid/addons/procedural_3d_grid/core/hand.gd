# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# hand.gd  
# SPDX-License-Identifier: MIT

extends XRController3D

@onready var sketch_tool: Node3D = $SketchTool

var prev_hand_transform: Transform3D
var prev_hand_pressed: float


func _process(_delta: float) -> void:
	var hand_pressed: float = get_float("trigger")
	var max_size: float = 0.01

	if hand_pressed <= 0.05:
		sketch_tool.active = false
	else:
		sketch_tool.active = true
		sketch_tool.pressure = hand_pressed * max_size

	# Basic save/load for debugging
	var mesh: ArrayMesh = sketch_tool.canvas.get_node("strokes").mesh
	if is_button_pressed("ax_button"):
		_save_mesh(mesh)
	if is_button_pressed("by_button"):
		_load_mesh(mesh)


func _save_mesh(mesh: ArrayMesh):
	print("saving mesh")
	var result = ResourceSaver.save(mesh, "res://test_save.mesh")
	if result != OK:
		print("failed!")


func _load_mesh(mesh: ArrayMesh):
	print("loading mesh")
	var result_mesh: ArrayMesh = ResourceLoader.load("res://test_save.mesh", "ArrayMesh")

	mesh.clear_surfaces()
	if result_mesh:
		for i in range(0, result_mesh.get_surface_count()):
			mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, result_mesh.surface_get_arrays(i))
