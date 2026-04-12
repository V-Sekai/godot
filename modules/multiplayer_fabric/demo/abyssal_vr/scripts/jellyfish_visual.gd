# jellyfish_visual.gd
# Build phase: CSG bell + tentacles → bake each piece to MeshInstance3D, free CSG.
# Animation runs on MeshInstance3D transforms (scale/rotation), not CSG nodes.

extends Node3D

var entity_id: int = 0

const BELL_RADIUS    := 0.3
const TENTACLE_R     := 0.018
const TENTACLE_H     := 0.55
const TENTACLE_COUNT := 8
const PULSE_PERIOD   := 0.9375   # seconds — visual bell pulse cadence

var _bell_mi:      MeshInstance3D            = null
var _tentacle_mis: Array[MeshInstance3D]     = []
var _time:         float                     = 0.0
var _built:        bool                      = false


func _ready() -> void:
	_build()


func _process(delta: float) -> void:
	if not _built:
		return
	_time += delta

	# Bell pulse: compress vertically at PULSE_PERIOD
	if _bell_mi:
		var pulse := 1.0 + 0.2 * sin(TAU * _time / PULSE_PERIOD)
		_bell_mi.scale = Vector3(1.0 / pulse, pulse, 1.0 / pulse)

	# Tentacle wave ripple
	for i in range(_tentacle_mis.size()):
		var phase := float(i) / TENTACLE_COUNT * TAU
		_tentacle_mis[i].rotation.x = sin(_time * 3.0 + phase) * 0.18


func _build() -> void:
	var bloom := entity_id < 256
	var bell_col := Color(0.2, 0.9, 0.85, 0.65) if bloom else Color(0.1, 0.7, 0.95, 0.65)
	var tent_col := Color(0.1, 0.5, 0.75, 0.5)

	# Bell — single CSGSphere3D (no combiner needed, one shape)
	var bell_csg := CSGSphere3D.new()
	bell_csg.radius = BELL_RADIUS
	add_child(bell_csg)

	# Tentacles — individual CSGCylinder3D nodes, radially placed under bell
	var tent_csgs: Array[CSGCylinder3D] = []
	for i in range(TENTACLE_COUNT):
		var angle := TAU / TENTACLE_COUNT * i
		var arm   := CSGCylinder3D.new()
		arm.radius = TENTACLE_R
		arm.height = TENTACLE_H
		arm.position = Vector3(
			cos(angle) * BELL_RADIUS * 0.65,
			-BELL_RADIUS - TENTACLE_H * 0.5,
			sin(angle) * BELL_RADIUS * 0.65)
		add_child(arm)
		tent_csgs.append(arm)

	await get_tree().process_frame

	# Bake bell
	_bell_mi = _bake_csg(bell_csg, _mat(bell_col))

	# Bake tentacles — preserve individual transforms for per-tentacle rotation
	for arm in tent_csgs:
		var mi := _bake_csg(arm, _mat(tent_col))
		if mi:
			_tentacle_mis.append(mi)

	_built = true


## Bake a CSGShape3D into a MeshInstance3D, free the CSG node, return the instance.
func _bake_csg(csg: CSGShape3D, mat: StandardMaterial3D) -> MeshInstance3D:
	var meshes := csg.get_meshes()
	csg.queue_free()
	if meshes.size() < 2:
		return null
	var mi := MeshInstance3D.new()
	mi.mesh             = meshes[1] as Mesh
	mi.transform        = meshes[0] as Transform3D
	mi.material_override = mat
	add_child(mi)
	return mi


func _mat(color: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color     = color
	m.transparency     = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.emission_enabled = true
	m.emission         = color * 0.28
	m.roughness        = 0.4
	return m
