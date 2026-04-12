# pen_tool.gd
# CSG segments are built live during drawing (user sees real-time feedback).
# On trigger release (_end_stroke) the whole stroke combiner is baked to a
# MeshInstance3D and the CSG nodes are freed.

extends Node3D

@export var stroke_color: Color = Color(0.98, 0.82, 0.07, 0.9)
@export var seg_radius:   float = 0.012
@export var min_seg_len:  float = 0.025   # meters between segment knots

const MAX_SEGMENTS_PER_STROKE := 200
const MAX_STROKES             := 40

var _active:         bool     = false
var _last_pos:       Vector3  = Vector3.ZERO
var _cur_stroke:     CSGCombiner3D = null
var _strokes:        Array    = []   # Array[Node3D] — CSGCombiner3D during draw, MeshInstance3D after bake
var _seg_count:      int      = 0
var _stroke_counter: int      = 0
var _stroke_id:      int      = 0
var _fabric_client:  Node     = null


func _ready() -> void:
	var root := get_tree().current_scene
	if root:
		_fabric_client = root.find_child("FabricClient", true, false)


func _process(_delta: float) -> void:
	var parent := get_parent() as XRController3D
	if not parent:
		return

	var pressure: float = parent.get_float("trigger")
	var pos: Vector3    = global_position

	if pressure > 0.05:
		if not _active:
			_begin_stroke(pos)
		else:
			_continue_stroke(pos)
	else:
		if _active:
			_end_stroke()


func _begin_stroke(pos: Vector3) -> void:
	_active    = true
	_last_pos  = pos
	_seg_count = 0

	var peer_id: int = multiplayer.get_unique_id() if multiplayer else 1
	_stroke_id      = ((peer_id & 0xFFFF) << 16) | (_stroke_counter & 0xFFFF)
	_stroke_counter = (_stroke_counter + 1) & 0xFFFF

	var combiner := CSGCombiner3D.new()
	combiner.operation = CSGShape3D.OPERATION_UNION
	get_tree().current_scene.add_child(combiner)
	_cur_stroke = combiner

	_strokes.append(combiner)
	if _strokes.size() > MAX_STROKES:
		var oldest: Node3D = _strokes.pop_front()
		oldest.queue_free()


func _continue_stroke(pos: Vector3) -> void:
	if _seg_count >= MAX_SEGMENTS_PER_STROKE:
		return
	if _last_pos.distance_to(pos) < min_seg_len:
		return

	_add_segment(_last_pos, pos)
	if _fabric_client and _fabric_client.has_method("spawn_stroke_knot"):
		_fabric_client.spawn_stroke_knot(pos, _stroke_id, stroke_color)
	_last_pos = pos
	_seg_count += 1


func _end_stroke() -> void:
	_active = false
	if _cur_stroke:
		_bake_stroke(_cur_stroke)
	_cur_stroke = null


func _add_segment(from: Vector3, to: Vector3) -> void:
	var length := from.distance_to(to)
	if length < 0.001 or not _cur_stroke:
		return

	var seg    := CSGCylinder3D.new()
	seg.radius  = seg_radius * (1.0 + get_parent().get_float("trigger") * 0.5)
	seg.height  = length

	var up    := (to - from).normalized()
	var fwd   := Vector3.UP if abs(up.dot(Vector3.FORWARD)) < 0.99 else Vector3.RIGHT
	var basis := Basis()
	basis.y   = up
	basis.x   = fwd.cross(up).normalized()
	basis.z   = up.cross(basis.x).normalized()
	seg.global_transform = Transform3D(basis, (from + to) * 0.5)
	seg.material = _mat()
	_cur_stroke.add_child(seg)


func _bake_stroke(combiner: CSGCombiner3D) -> void:
	await get_tree().process_frame

	var meshes := combiner.get_meshes()
	var idx    := _strokes.find(combiner)

	if meshes.size() >= 2:
		var mi := MeshInstance3D.new()
		mi.mesh              = meshes[1] as Mesh
		mi.transform         = meshes[0] as Transform3D
		mi.material_override = _mat()
		get_tree().current_scene.add_child(mi)
		if idx >= 0:
			_strokes[idx] = mi   # replace CSG ref with baked mesh ref

	combiner.queue_free()


func _mat() -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color     = stroke_color
	m.transparency     = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.emission_enabled = true
	m.emission         = stroke_color * 0.35
	m.roughness        = 0.3
	return m
