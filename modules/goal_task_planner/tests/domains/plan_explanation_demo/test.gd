extends Node

# Test script demonstrating plan explanation and debugging features
# Run this script to see plan explanation, debugging, and visualization features

const Domain = preload("res://modules/goal_task_planner/tests/domains/plan_explanation_demo/domain.gd")

func _ready():
	print("=== Plan Explanation and Debugging Demo ===\n")
	
	# Create domain
	var domain = PlanExplanationDemoDomain.create_planner_domain()
	
	# Create planner
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_verbose(2)  # Enable decision tracking
	plan.set_max_depth(10)
	
	# Initial state
	var state = {
		"location": "home",
		"money": 20,
		"inventory": []
	}
	
	# Goal: go to store and buy an apple
	var todo_list = [
		["go_to", "store"],
		["buy", "apple", "store"]
	]
	
	print("Initial State:")
	print("  Location: ", state["location"])
	print("  Money: ", state["money"])
	print("  Inventory: ", state["inventory"])
	print("\nGoal: ", todo_list)
	print("\n" + ("=".repeat(50)) + "\n")
	
	# Find plan
	var result = plan.find_plan(state, todo_list)
	
	if not result.get_success():
		print("ERROR: Planning failed!")
		return
	
	print("✓ Plan found successfully!\n")
	
	# Extract and display plan
	var actions = result.extract_plan()
	print("Plan Actions:")
	for i in range(actions.size()):
		print("  ", i + 1, ". ", actions[i])
	print()
	
	# === 1. Explain the plan ===
	print(("=".repeat(50)))
	print("1. PLAN EXPLANATION")
	print(("=".repeat(50)))
	var explanation = result.explain_plan()
	print("Success: ", explanation["success"])
	print("Total Nodes: ", explanation["total_nodes"])
	print("Plan Length: ", explanation["plan_length"], " actions")
	
	var node_counts = explanation["node_counts"]
	print("\nNode Counts:")
	print("  Actions: ", node_counts["actions"])
	print("  Tasks: ", node_counts["tasks"])
	print("  Failed: ", node_counts["failed"])
	print("  Closed: ", node_counts["closed"])
	
	var decision_points = explanation["decision_points"]
	print("\nDecision Points: ", decision_points.size())
	for i in range(decision_points.size()):
		var dp = decision_points[i]
		print("  ", i + 1, ". Node ", dp["node_id"], ": ", dp["info"])
		print("     Selected: ", dp["selected_method"])
		print("     Score: ", dp["selected_score"])
		print("     Alternatives: ", dp["alternatives_count"])
	print()
	
	# === 2. Get node explanations ===
	print(("=".repeat(50)))
	print("2. NODE EXPLANATIONS")
	print(("=".repeat(50)))
	var all_nodes = result.get_all_nodes()
	for i in range(min(all_nodes.size(), 5)):  # Show first 5 nodes
		var node = all_nodes[i]
		var node_id = node["node_id"]
		if node_id == 0:
			continue  # Skip root
		
		print("\nNode ", node_id, ":")
		var node_explanation = result.get_node_explanation(node_id)
		print(node_explanation)
	
	# === 3. Get alternative methods ===
	print("\n" + ("=".repeat(50)))
	print("3. ALTERNATIVE METHODS")
	print(("=".repeat(50)))
	for i in range(decision_points.size()):
		var dp = decision_points[i]
		var node_id = dp["node_id"]
		var alternatives = result.get_alternative_methods(node_id)
		
		if alternatives.size() > 0:
			print("\nNode ", node_id, " - Alternatives considered:")
			for j in range(alternatives.size()):
				var alt = alternatives[j]
				print("  ", j + 1, ". ", alt["method_id"])
				print("     Score: ", alt["score"])
				print("     Activity: ", alt["activity"])
				print("     Subtasks: ", alt["subtask_count"])
	
	# === 4. Get decision path ===
	print("\n" + ("=".repeat(50)))
	print("4. DECISION PATH")
	print(("=".repeat(50)))
	if all_nodes.size() > 1:
		var target_node = all_nodes[all_nodes.size() - 1]  # Last node
		var target_id = target_node["node_id"]
		var path = result.get_decision_path(target_id)
		
		print("Path to Node ", target_id, " (length: ", path["length"], "):")
		var path_nodes = path["path"]
		for i in range(path_nodes.size()):
			var path_node = path_nodes[i]
			print("  ", i + 1, ". Node ", path_node["node_id"], " (", path_node["type"], ")")
			if path_node.has("decision_info"):
				var di = path_node["decision_info"]
				if di.has("selected_method_id"):
					print("     Method: ", di["selected_method_id"])
	
	# === 5. Visualize plan graph ===
	print("\n" + ("=".repeat(50)))
	print("5. GRAPH VISUALIZATION")
	print(("=".repeat(50)))
	
	# Export to DOT format
	var dot_graph = result.to_dot_graph()
	# Save to user's home directory or project root
	var dot_path = "user://plan_graph.dot"
	var file = FileAccess.open(dot_path, FileAccess.WRITE)
	if file:
		file.store_string(dot_graph)
		file.close()
		print("✓ DOT graph saved to: ", dot_path)
		print("  Render with: dot -Tpng ", OS.get_user_data_dir() + "/plan_graph.dot", " -o plan_graph.png")
	else:
		print("✗ Failed to save DOT graph")
	
	# Export to JSON format
	var json_graph = result.to_graph_json()
	var json_path = "user://plan_graph.json"
	file = FileAccess.open(json_path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(json_graph, "\t"))
		file.close()
		print("✓ JSON graph saved to: ", json_path)
		print("  Can be used with visualization libraries (D3.js, Cytoscape.js)")
	else:
		print("✗ Failed to save JSON graph")
	
	# Print a sample of the DOT graph
	print("\nSample DOT graph (first 20 lines):")
	var dot_lines = dot_graph.split("\n")
	for i in range(min(dot_lines.size(), 20)):
		print("  ", dot_lines[i])
	if dot_lines.size() > 20:
		print("  ... (", dot_lines.size() - 20, " more lines)")
	
	print("\n" + ("=".repeat(50)))
	print("Demo complete!")
	print(("=".repeat(50)))

