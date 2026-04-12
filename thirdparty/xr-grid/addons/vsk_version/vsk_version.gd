# Copyright (c) 2023-present. This file is part of V-Sekai https://v-sekai.org/.
# K. S. Ernest (Fire) Lee & Contributors (see .all-contributorsrc).
# vsk_version.gd
# SPDX-License-Identifier: MIT

@tool
extends Node

var build_constants = null

func _ready():
	if ResourceLoader.exists("build_constants.gd"):
		build_constants = load("build_constants.gd").new()
	else:
		print("build_constants.gd does not exist")

func get_build_label() -> String:
	var build_label = "DEVELOPER_BUILD"
	var build_date_str = "Build Date"
	var build_unix_time = -1
	
	if build_constants and build_constants.has("BUILD_LABEL"):
		build_label = build_constants.BUILD_LABEL
	if build_constants and build_constants.has("BUILD_DATE_STR"):
		build_date_str = build_constants.BUILD_DATE_STR
	if build_constants and build_constants.has("BUILD_UNIX_TIME"):
		build_unix_time = build_constants.BUILD_UNIX_TIME
	
	return build_date_str + "\n" + build_label
