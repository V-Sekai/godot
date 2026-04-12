# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# simple_sketch.gd
# SPDX-License-Identifier: MIT

#@tool
class_name SimpleSketch extends RefCounted

var target_mesh: ArrayMesh

var is_beginning: bool = false


func stroke_begin() -> void:
	is_beginning = true


var prev_point: Vector3
var prev_size: float
var prev_color: Color
var prev_tangent: Vector3


func stroke_add(point: Vector3, size: float = .01, color: Color = Color(0, 0, 0)) -> void:
	if is_beginning:
		add_line(point, point, size, size, color, color, true)
		is_beginning = false
	else:
		add_line(prev_point, point, prev_size, size, prev_color, color)

	prev_point = point
	prev_size = size
	prev_color = color


func stroke_end() -> void:
	is_beginning = false


func add_line(from: Vector3, to: Vector3, from_size: float = .01, to_size: float = .01, from_color: Color = Color(0, 0, 0), to_color: Color = Color(0, 0, 0), begin_stroke: bool = false) -> void:
	if target_mesh == null:
		return

	var from_tangent = prev_tangent
	var to_tangent = to - from

	if begin_stroke:
		from_tangent = to_tangent

	var arrays = []
	arrays.resize(ArrayMesh.ARRAY_MAX)
	arrays[ArrayMesh.ARRAY_VERTEX] = PackedVector3Array()
	arrays[ArrayMesh.ARRAY_TANGENT] = PackedFloat32Array()
	arrays[ArrayMesh.ARRAY_TEX_UV] = PackedVector2Array()
	arrays[ArrayMesh.ARRAY_COLOR] = PackedColorArray()

	# Build on original mesh if possible
	if target_mesh.get_surface_count() > 0:
		arrays = target_mesh.surface_get_arrays(0)

	# A
	arrays[ArrayMesh.ARRAY_VERTEX].append(from)
	arrays[ArrayMesh.ARRAY_TANGENT] += PackedFloat32Array([from_tangent.x, from_tangent.y, from_tangent.z, 1.0])
	arrays[ArrayMesh.ARRAY_TEX_UV].append(Vector2(0, -from_size))
	arrays[ArrayMesh.ARRAY_COLOR].append(from_color)

	# B
	arrays[ArrayMesh.ARRAY_VERTEX].append(from)
	arrays[ArrayMesh.ARRAY_TANGENT] += PackedFloat32Array([from_tangent.x, from_tangent.y, from_tangent.z, 1.0])
	arrays[ArrayMesh.ARRAY_TEX_UV].append(Vector2(0, from_size))
	arrays[ArrayMesh.ARRAY_COLOR].append(from_color)

	# D
	arrays[ArrayMesh.ARRAY_VERTEX].append(to)
	arrays[ArrayMesh.ARRAY_TANGENT] += PackedFloat32Array([to_tangent.x, to_tangent.y, to_tangent.z, 1.0])
	arrays[ArrayMesh.ARRAY_TEX_UV].append(Vector2(0, to_size))
	arrays[ArrayMesh.ARRAY_COLOR].append(to_color)

	# A
	arrays[ArrayMesh.ARRAY_VERTEX].append(from)
	arrays[ArrayMesh.ARRAY_TANGENT] += PackedFloat32Array([from_tangent.x, from_tangent.y, from_tangent.z, 1.0])
	arrays[ArrayMesh.ARRAY_TEX_UV].append(Vector2(0, -from_size))
	arrays[ArrayMesh.ARRAY_COLOR].append(from_color)

	# D
	arrays[ArrayMesh.ARRAY_VERTEX].append(to)
	arrays[ArrayMesh.ARRAY_TANGENT] += PackedFloat32Array([to_tangent.x, to_tangent.y, to_tangent.z, 1.0])
	arrays[ArrayMesh.ARRAY_TEX_UV].append(Vector2(0, to_size))
	arrays[ArrayMesh.ARRAY_COLOR].append(to_color)

	# C
	arrays[ArrayMesh.ARRAY_VERTEX].append(to)
	arrays[ArrayMesh.ARRAY_TANGENT] += PackedFloat32Array([to_tangent.x, to_tangent.y, to_tangent.z, 1.0])
	arrays[ArrayMesh.ARRAY_TEX_UV].append(Vector2(0, -to_size))
	arrays[ArrayMesh.ARRAY_COLOR].append(to_color)

	target_mesh.clear_surfaces()
	target_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)

	prev_tangent = to_tangent
