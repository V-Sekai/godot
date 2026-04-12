# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# plugin.gd
# SPDX-License-Identifier: MIT

@tool
extends EditorPlugin


func _enter_tree() -> void:
	# Initialization of the plugin goes here.
	# Add the new type with a name, a parent type, a script and an icon.
	add_custom_type("GPUTrail3D", "GPUParticles3D", preload("GPUTrail3D.gd"), preload("icon.png"))


func _exit_tree() -> void:
	# Clean-up of the plugin goes here.
	# Always remember to remove it from the engine when deactivated.
	remove_custom_type("GPUTrail3D")
