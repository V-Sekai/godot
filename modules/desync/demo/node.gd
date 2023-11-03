# Copyright (c) 2018-present. This file is part of V-Sekai https://v-sekai.org/.
# SaracenOne & K. S. Ernest (Fire) Lee & Lyuma & MMMaellon & Contributors
# run_split.gd
# SPDX-License-Identifier: MIT
extends Node

var thread = Thread.new()

func _ready():
	var callable: Callable = Callable(self, "_async_run")
	callable = callable.bind(null)
	print(thread.start(callable))

func _async_run(_userdata):
	var desync = Desync.new()
	var result = desync.untar("https://v-sekai.github.io/casync-v-sekai-game/store", 
				"https://github.com/V-Sekai/casync-v-sekai-game/raw/main/vsekai_game_windows_x86_64.caidx",
				"vsekai_game_windows_x86_64",
				String())
