# whale_visual.gd
# Build phase: CSG whale body (combiner) or remora sphere → bake to MeshInstance3D, free CSG.
# No per-frame animation needed — whales translate via global_position from fabric_client.

extends Node3D

var entity_id: int  = 0
var is_whale:  bool = false

const WHALE_R  := 0.8
const WHALE_H  := 3.0
const REMORA_R := 0.15


func _ready() -> void:
	if is_whale:
		_build_whale()
	else:
		_build_remora()


func _build_whale() -> void:
	var combiner := CSGCombiner3D.new()
	combiner.operation        = CSGShape3D.OPERATION_UNION
	combiner.rotation_degrees.x = 90.0   # orient along -Z travel direction
	add_child(combiner)

	var body := CSGCylinder3D.new()
	body.radius = WHALE_R
	body.height = WHALE_H
	body.material = _mat(Color(0.12, 0.22, 0.48, 1.0))
	combiner.add_child(body)

	var nose := CSGSphere3D.new()
	nose.radius   = WHALE_R * 0.42
	nose.position = Vector3(0, WHALE_H * 0.5, 0)
	nose.material = _mat(Color(0.10, 0.18, 0.40, 1.0))
	combiner.add_child(nose)

	var tail := CSGSphere3D.new()
	tail.radius   = WHALE_R * 0.55
	tail.position = Vector3(0, -WHALE_H * 0.45, 0)
	tail.material = _mat(Color(0.08, 0.16, 0.38, 1.0))
	combiner.add_child(tail)

	await get_tree().process_frame
	_bake_and_free(combiner, _mat(Color(0.12, 0.22, 0.48, 1.0)))


func _build_remora() -> void:
	# Slot offset mirrors whaleWithSharksStep: (memberIdx-7)*300 000 μm, (memberIdx%3-1)*300 000 μm
	var member_idx := entity_id % 14
	position = Vector3(
		(member_idx - 7)       * 0.3,
		(member_idx % 3 - 1)   * 0.3,
		0.0)

	var s := CSGSphere3D.new()
	s.radius   = REMORA_R
	s.material = _mat(Color(0.28, 0.48, 0.60, 1.0))
	add_child(s)

	await get_tree().process_frame
	_bake_and_free(s, _mat(Color(0.28, 0.48, 0.60, 1.0)))


func _bake_and_free(csg: CSGShape3D, mat: StandardMaterial3D) -> void:
	var meshes := csg.get_meshes()
	csg.queue_free()
	if meshes.size() < 2:
		return
	var mi := MeshInstance3D.new()
	mi.mesh              = meshes[1] as Mesh
	mi.transform         = meshes[0] as Transform3D
	mi.material_override = mat
	add_child(mi)


func _mat(color: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color     = color
	m.metallic         = 0.1
	m.roughness        = 0.65
	m.emission_enabled = true
	m.emission         = color * 0.1
	return m
