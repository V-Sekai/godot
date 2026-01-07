#!/usr/bin/env -S godot --headless --script
# SPDX-FileCopyrightText: 2025-present K. S. Ernest (iFire) Lee
# SPDX-License-Identifier: MIT
#
# Simple syntax test - verifies all split files can be loaded
# This doesn't require the compiled module, just checks GDScript syntax

extends SceneTree

const Domain = preload("domain.gd")

func _init():
	print("=== Testing Split GDScript Files ===")
	call_deferred("test_all_files")

func test_all_files():
	var errors = []
	var files_to_test = [
		"helpers.gd",
		"actions.gd",
		"task_methods.gd",
		"unigoal_methods.gd",
		"multigoal_methods.gd",
		"domain.gd"
	]

	for file in files_to_test:
		print("\nTesting: %s" % file)
		var script_path = "res://modules/goal_task_planner/tests/domains/magical_girls_college/" + file
		var script = load(script_path)
		if script == null:
			errors.append("%s: Failed to load" % file)
			print("  ❌ Failed to load")
		else:
			print("  ✅ Loaded successfully")
			# Try to get the class name
			if script.has_method("get_script_path"):
				print("  📄 Path: %s" % script.get_script_path())

	# Test that domain.gd can access all modules
	print("\n=== Testing Domain Integration ===")
	# Try to call a helper function through domain class
	var test_state = {"study_points": {"yuki": 10}}
	var result = Domain.get_study_points(test_state, "yuki")
	if result == 10:
		print("✅ Domain.get_study_points() works")
	else:
		errors.append("Domain.get_study_points() returned %s, expected 10" % result)
		print("  ❌ Domain.get_study_points() failed")

	# Test that actions can be called
	var test_state2 = {"is_at": {"yuki": "dorm"}, "needs": {"yuki": {"hunger": 50}}, "money": {"yuki": 100}}
	var result2 = Domain.action_eat_mess_hall(test_state2, "yuki")
	if result2 != null and result2 is Dictionary:
		var hunger_after = Domain.get_need(result2, "yuki", "hunger")
		if hunger_after > 50:
			print("✅ Domain.action_eat_mess_hall() works")
		else:
			errors.append("Domain.action_eat_mess_hall() didn't increase hunger")
			print("  ❌ Domain.action_eat_mess_hall() failed")
	else:
		errors.append("Domain.action_eat_mess_hall() returned null or invalid")
		print("  ❌ Domain.action_eat_mess_hall() failed")

	print("\n=== Test Results ===")
	if errors.size() == 0:
		print("✅ All files loaded successfully!")
		print("✅ Domain integration works!")
		quit(0)
	else:
		print("❌ Found %d error(s):" % errors.size())
		for error in errors:
			print("  - %s" % error)
		quit(1)
