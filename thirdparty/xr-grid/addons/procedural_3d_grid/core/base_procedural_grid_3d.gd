# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# base_procedural_grid_3d.gd
# SPDX-License-Identifier: MIT

@tool
extends MultiMeshInstance3D

@export var pulse_regenerate_mesh: bool:
	set = set_pulse_regenerate_mesh
@export var points_per_dimension: int = 6:
	set = set_points_per_dimension
@export var n_vertex_circle: int = 32
@export var fade_zone: float = 0.5
@export var far_fade: float = 1.0


func set_pulse_regenerate_mesh(_value: Variant):
	regenerate_mesh()


func set_points_per_dimension(value: int):
	points_per_dimension = value
	multimesh.instance_count = points_per_dimension * points_per_dimension * points_per_dimension
	regenerate_mesh()


func _process(_delta: float):
	material_override.set_shader_parameter("_GridCenter", global_position)


func set_points_per_dimension_from_fade():
	points_per_dimension = int((fade_zone + far_fade) * 2)


func regenerate_mesh():
	var arr_mesh: ArrayMesh = ArrayMesh.new()

	var arrays: Array = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = PackedVector3Array()
	arrays[Mesh.ARRAY_NORMAL] = PackedVector3Array()
	arrays[Mesh.ARRAY_TEX_UV] = PackedVector2Array()

	# Generate a circle.
	var circle_p: Vector3 = Vector3(1, 0, 0)
	var angle: float = TAU * 1.0 / n_vertex_circle
	for i in range(1, n_vertex_circle + 1):
		var onCirclePos := Vector3(cos(i * angle), sin(i * angle), 0.0)

		arrays[Mesh.ARRAY_VERTEX].append(circle_p)
		arrays[Mesh.ARRAY_NORMAL].append(Vector3())
		arrays[Mesh.ARRAY_TEX_UV].append(Vector2())

		arrays[Mesh.ARRAY_VERTEX].append(Vector3())
		arrays[Mesh.ARRAY_NORMAL].append(Vector3())
		arrays[Mesh.ARRAY_TEX_UV].append(Vector2())

		arrays[Mesh.ARRAY_VERTEX].append(onCirclePos)
		arrays[Mesh.ARRAY_NORMAL].append(Vector3())
		arrays[Mesh.ARRAY_TEX_UV].append(Vector2())

		circle_p = onCirclePos

	arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)

	const axes: Array[Vector3] = [Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)]
	for i in range(0, 3):
		arrays[Mesh.ARRAY_VERTEX] = PackedVector3Array()
		arrays[Mesh.ARRAY_NORMAL] = PackedVector3Array()
		arrays[Mesh.ARRAY_TEX_UV] = PackedVector2Array()

		arrays[Mesh.ARRAY_VERTEX].append(Vector3(0, -0.5, 0))
		arrays[Mesh.ARRAY_VERTEX].append(Vector3(0, 0.5, 0))
		arrays[Mesh.ARRAY_VERTEX].append(Vector3(1, -0.5, 0))
		arrays[Mesh.ARRAY_VERTEX].append(Vector3(1, 0.5, 0))
		for j in range(0, 4):
			arrays[Mesh.ARRAY_NORMAL].append(axes[i])
			arrays[Mesh.ARRAY_TEX_UV].append(Vector2(1, 0))

		arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLE_STRIP, arrays)

	multimesh.mesh = arr_mesh

	for i in range(0, multimesh.instance_count):
		var grid_transform := Transform3D()
		grid_transform.origin = Vector3(i % points_per_dimension, i / points_per_dimension % points_per_dimension, i / (points_per_dimension * points_per_dimension))
		grid_transform.origin -= Vector3(1, 1, 1) * (points_per_dimension / 2 - 1)

		multimesh.set_instance_transform(i, grid_transform)

	print_verbose("Grid mesh has been regenerated")
