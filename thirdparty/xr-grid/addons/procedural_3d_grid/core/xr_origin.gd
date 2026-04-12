# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# xr_origin.gd
# SPDX-License-Identifier: MIT

extends Node3D

var interface: XRInterface = null
var vr_supported: bool = false


func _ready() -> void:
	# Find our interface and check if it was successfully initialised.
	# Note that Godot should initialise this automatically IF you've
	# enabled it in project settings!
	interface = XRServer.find_interface("OpenXR")
	if interface and interface.is_initialized():
		print_verbose("OpenXR initialised successfully")

		var vp: Viewport = get_viewport()
		vp.use_xr = true
		print_verbose(vp.size)
