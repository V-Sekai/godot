#!/usr/bin/env -S godot --headless --script
# Test script to generate DOT graph from magical girls college domain

extends SceneTree

const Domain = preload("domain.gd")

func _init():
	print("=== Magical Girls College Graph Export Test ===\n")
	call_deferred("run_test")

func run_test():
	# Create domain
	var domain = Domain.create_planner_domain(true, true, true, true)

	# Create planner
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(2)  # Enable decision tracking
	plan.set_max_depth(50)

	# Initial state
	var state = {}
	var is_at = {"yuki": "dorm"}
	state["is_at"] = is_at
	var study_points = {"yuki": 0}
	state["study_points"] = study_points
	var needs = {
		"yuki": {
			"hunger": 50,
			"energy": 50,
			"social": 50,
			"fun": 50,
			"hygiene": 50
		}
	}
	state["needs"] = needs
	var money = {"yuki": 100}
	state["money"] = money

	# Goal: earn study points and socialize (more varied plan)
	var todo_list = [["task_earn_study_points", "yuki", 20], ["task_socialize", "yuki", "maya", 2]]

	print("Initial State:")
	print("  Location: ", state["is_at"]["yuki"])
	print("  Study Points: ", state["study_points"]["yuki"])
	print("  Money: ", state["money"]["yuki"])
	print("\nGoal: ", todo_list)
	print("\n" + ("=".repeat(50)) + "\n")

	# Find plan
	var result = plan.find_plan(state, todo_list)

	if not result.get_success():
		print("ERROR: Planning failed!")
		quit(1)
		return

	print("✓ Plan found successfully!\n")

	# Extract and display plan
	var actions = result.extract_plan()
	print("Plan Actions:")
	for i in range(actions.size()):
		print("  ", i + 1, ". ", actions[i])
	print()

	# Export to DOT format
	var dot_graph = result.to_dot_graph()
	var dot_path = "user://magical_girls_college_graph.dot"
	var file = FileAccess.open(dot_path, FileAccess.WRITE)
	if file:
		file.store_string(dot_graph)
		file.close()
		print("✓ DOT graph saved to: ", dot_path)
		print("  Full path: ", OS.get_user_data_dir() + "/magical_girls_college_graph.dot")
	else:
		print("✗ Failed to save DOT graph")
		quit(1)
		return

	# Export to JSON format
	var json_graph = result.to_graph_json()
	var json_path = "user://magical_girls_college_graph.json"
	file = FileAccess.open(json_path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(json_graph, "\t"))
		file.close()
		print("✓ JSON graph saved to: ", json_path)

	# Show plan explanation
	var explanation = result.explain_plan()
	print("\nPlan Explanation:")
	print("  Total Nodes: ", explanation["total_nodes"])
	print("  Plan Length: ", explanation["plan_length"], " actions")
	print("  Decision Points: ", explanation["decision_points"].size())

	print("\n" + ("=".repeat(50)))
	print("Test complete!")
	print(("=".repeat(50)))
	quit(0)
