# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# world_grab.gd
# SPDX-License-Identifier: MIT

class_name WorldGrab extends RefCounted

## The WorldGrab utility makes it easy to add world-grab navigation to your XR project!
## [br][color=purple]Made by celyk[/color]
##
## Right now, WorldGrab is designed for a single use case: art viewing.
## [br]It allows one to grab the world with both hands and move it around, viewing it from all angles; It is not restricted by any up direction.
## [br]
## [br]Example usage:
## [codeblock]
## wg = WorldGrab.new()
##	soon()
## [/codeblock]
## @tutorial(celyk's repo): https://github.com/celyk/godot-useful-stuff
## @tutorial(xr-grid): https://github.com/V-Sekai/V-Sekai.xr-grid


## The transform that takes one to the other. Intended for a one handed grab.
func get_grab_transform(from: Transform3D, to: Transform3D) -> Transform3D:
	return to * from.affine_inverse()


## For orbitting around a central point, without scale, like spinning a globe.
func get_orbit_transform(from_pivot: Vector3, from_b: Vector3, to_pivot: Vector3, to_b: Vector3) -> Transform3D:
	# Center the pivot
	from_b -= from_pivot
	to_b -= to_pivot

	# Gather information on the shortest rotation
	var axis: Vector3 = from_b.cross(to_b)
	if axis == Vector3():
		axis = Vector3.RIGHT
	var angle: float = from_b.angle_to(to_b)

	# Construct the transformation that orbits about the pivot, with no scale!
	return Transform3D().translated(-from_pivot).rotated(axis.normalized(), angle).translated(to_pivot)


## This is a transformation which takes line (from_a,from_b) to line (to_a,to_b). It is analagous to pinch gesture on a touch screen.
func get_pinch_transform(from_a: Vector3, from_b: Vector3, to_a: Vector3, to_b: Vector3) -> Transform3D:
	var delta_scale: float = sqrt((to_b - to_a).length_squared() / (from_b - from_a).length_squared())

	# Orbit around pivot point a, and scale so that b is fixed in place.
	# According to symmetry, it is the same as if a and b are swapped.
	return get_orbit_transform(from_a, from_b, to_a, to_b).translated(-to_a).scaled(Vector3.ONE * delta_scale).translated(to_a)


## Separable blending of position, rotation and scale. Fine tune smoothing for maximum comfort.
func split_blend(from: Transform3D, to: Transform3D, pos_weight: float = 0.0, rot_weight: float = 0.0, scale_weight: float = 0.0, from_pivot: Vector3 = Vector3(), to_pivot: Vector3 = Vector3()) -> Transform3D:
	var new_from: Transform3D = Transform3D(from.basis, from_pivot)
	var new_to: Transform3D = Transform3D(to.basis, to_pivot)
	new_to = new_from.interpolate_with(new_to, pos_weight)

	return get_grab_transform(new_from, new_to) * from  #from.interpolate_with(to, pos_weight)

	from.origin -= from_pivot
	to.origin -= from_pivot

	var src_scale: Vector3 = from.basis.get_scale()
	var src_rot: Quaternion = from.basis.get_rotation_quaternion()

	var dst_scale: Vector3 = to.basis.get_scale()
	var dst_rot: Quaternion = to.basis.get_rotation_quaternion()

	var target_basis = Basis(src_rot.slerp(dst_rot, rot_weight)) * Basis.from_scale(src_scale.lerp(dst_scale, scale_weight))
	#to.basis = Basis( src_rot.slerp(dst_rot, rot_weight) ) * Basis.from_scale(src_scale.lerp(dst_scale, scale_weight))

	from.origin = (target_basis * from.basis.inverse()) * from.origin
	from.basis = target_basis

	#var delta_rot_scale : Basis = Basis( Quaternion().slerp(dst_rot * src_rot.inverse(), rot_weight).normalized() ) * Basis.from_scale(Vector3.ONE.lerp(dst_scale/src_scale, scale_weight))
	#from = Transform3D(delta_rot_scale) * from
	from = from.translated((to.origin - from.origin) * pos_weight)

	#var delta_transform : Transform3D = Transform3D(delta_rot_scale) #Transform3D(delta_rot_scale, Vector3().lerp(to.origin - from.origin, pos_weight))

	from.origin += from_pivot
	to.origin += from_pivot

	#var basis_inv : Basis = from.basis.inverse()
	#from.basis = Basis(src_rot.slerp(dst_rot, rot_weight).normalized()) * Basis.from_scale(src_scale.lerp(dst_scale, scale_weight))

	#from.origin -= from_pivot
	#to.origin -= to_pivot
	#from.translated_local(-from_pivot)
	#from.origin = from.origin.lerp(to.origin, pos_weight)
	#from.origin = from_pivot.lerp(to_pivot, pos_weight) + from.origin.slerp(to.origin, pos_weight)

	#from.origin -= from_pivot
	#from.origin = from.basis * (basis_inv * from.origin)
	#from.origin += from_pivot.lerp(to_pivot, pos_weight)

	return from
