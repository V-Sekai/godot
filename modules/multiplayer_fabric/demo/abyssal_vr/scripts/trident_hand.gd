# trident_hand.gd
# Poseidon trident built entirely from CSG nodes.
# Attach as a child of hand_left or hand_right (XRController3D).
# SketchTool draws from the same position — trident is cosmetic.
#
# CSG tree:
#   CSGCombiner3D (root, Union)
#   ├── staff       CSGCylinder3D  r=0.015  h=0.50  (along -Z)
#   ├── crossguard  CSGCylinder3D  r=0.008  h=0.10  (perpendicular, at staff mid)
#   ├── tine_c      CSGCylinder3D  r=0.010  h=0.13  (center, tip)
#   ├── tine_c_tip  CSGSphere3D    r=0.012          (point cap on center tine)
#   ├── tine_l      CSGCylinder3D  r=0.008  h=0.11  (left, splayed 12°)
#   ├── tine_l_tip  CSGSphere3D    r=0.010
#   ├── tine_r      CSGCylinder3D  r=0.008  h=0.11  (right, splayed -12°)
#   └── tine_r_tip  CSGSphere3D    r=0.010

extends Node3D

@export var tine_color:  Color = Color(0.8, 0.85, 1.0, 1.0)
@export var staff_color: Color = Color(0.5, 0.55, 0.7, 1.0)

const STAFF_R   := 0.015
const STAFF_H   := 0.50
const GUARD_R   := 0.008
const GUARD_H   := 0.10
const TINE_R    := 0.010
const TINE_H    := 0.13
const SIDE_R    := 0.008
const SIDE_H    := 0.11
const TIP_R     := 0.012
const SIDE_TIP  := 0.010
const SPLAY_DEG := 12.0


func _ready() -> void:
	_build()


func _build() -> void:
	var body := CSGCombiner3D.new()
	body.operation = CSGShape3D.OPERATION_UNION
	# Orient so the tip points forward (-Z) from the controller.
	body.rotation_degrees.x = 90.0
	add_child(body)

	# ── Staff ────────────────────────────────────────────────────────────────
	var staff := _cyl(STAFF_R, STAFF_H, staff_color)
	staff.position = Vector3(0, 0, 0)
	body.add_child(staff)

	# ── Cross-guard (horizontal bar near the base of tines) ──────────────────
	var guard := _cyl(GUARD_R, GUARD_H, staff_color)
	guard.rotation_degrees.z = 90.0
	guard.position = Vector3(0, STAFF_H * 0.5 - 0.02, 0)
	body.add_child(guard)

	# ── Center tine ──────────────────────────────────────────────────────────
	var tc := _cyl(TINE_R, TINE_H, tine_color)
	tc.position = Vector3(0, STAFF_H * 0.5 + TINE_H * 0.5, 0)
	body.add_child(tc)

	var tc_tip := _sphere(TIP_R, tine_color)
	tc_tip.position = Vector3(0, STAFF_H * 0.5 + TINE_H, 0)
	body.add_child(tc_tip)

	# ── Left tine ─────────────────────────────────────────────────────────────
	var tl := _cyl(SIDE_R, SIDE_H, tine_color)
	tl.rotation_degrees.z = -SPLAY_DEG
	tl.position = Vector3(-0.042, STAFF_H * 0.5 + SIDE_H * 0.4, 0)
	body.add_child(tl)

	var tl_tip := _sphere(SIDE_TIP, tine_color)
	tl_tip.position = Vector3(-0.042 - sin(deg_to_rad(SPLAY_DEG)) * SIDE_H * 0.5,
	                          STAFF_H * 0.5 + cos(deg_to_rad(SPLAY_DEG)) * SIDE_H * 0.85, 0)
	body.add_child(tl_tip)

	# ── Right tine ────────────────────────────────────────────────────────────
	var tr := _cyl(SIDE_R, SIDE_H, tine_color)
	tr.rotation_degrees.z = SPLAY_DEG
	tr.position = Vector3(0.042, STAFF_H * 0.5 + SIDE_H * 0.4, 0)
	body.add_child(tr)

	var tr_tip := _sphere(SIDE_TIP, tine_color)
	tr_tip.position = Vector3(0.042 + sin(deg_to_rad(SPLAY_DEG)) * SIDE_H * 0.5,
	                          STAFF_H * 0.5 + cos(deg_to_rad(SPLAY_DEG)) * SIDE_H * 0.85, 0)
	body.add_child(tr_tip)


func _mat(color: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.metallic = 0.9
	m.roughness = 0.12
	m.emission_enabled = true
	m.emission = color * 0.25
	return m


func _cyl(radius: float, height: float, color: Color) -> CSGCylinder3D:
	var c := CSGCylinder3D.new()
	c.radius = radius
	c.height = height
	c.material = _mat(color)
	return c


func _sphere(radius: float, color: Color) -> CSGSphere3D:
	var s := CSGSphere3D.new()
	s.radius = radius
	s.material = _mat(color)
	return s
