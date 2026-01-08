#!/usr/bin/env -S godot --headless --script
# Generate SVG visualization of a Goal Task Planner plan

extends SceneTree

# Simple action: transfer a flag from one position to another
func action_transfer_flag(state: Dictionary, from_flag: int, to_flag: int) -> Dictionary:
	var new_state = state.duplicate(true)

	# Get flag dictionary
	var flag_dict = new_state.get("flag", {})
	if flag_dict.get(from_flag, false) == true and flag_dict.get(to_flag, false) == false:
		flag_dict[from_flag] = false
		flag_dict[to_flag] = true
		new_state["flag"] = flag_dict
		return new_state

	# If action can't be applied, return unchanged state (failure)
	return state

# Simple task: transfer flag from 0 to 1
func task_transfer_flag(state: Dictionary) -> Variant:
	var flag_dict = state.get("flag", {})
	if flag_dict.get(0, false) == true and flag_dict.get(1, false) == false:
		var result = []
		var action = ["action_transfer_flag", 0, 1]
		result.append(action)
		return result

	# Task cannot be applied
	return false

func _init():
	print("Generating Goal Task Planner SVG visualization...")

	# Create domain
	var domain = PlannerDomain.new()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)

	# Add action
	var actions = []
	actions.append(Callable(self, "action_transfer_flag"))
	domain.add_actions(actions)

	# Add task method
	var task_methods = []
	task_methods.append(Callable(self, "task_transfer_flag"))
	domain.add_task_methods("transfer_flag", task_methods)

	# Set up initial state
	var state = {}
	var flag_dict = {}
	flag_dict[0] = true  # Flag 0 is set
	flag_dict[1] = false # Flag 1 is not set
	state["flag"] = flag_dict

	# Create todo list
	var todo_list = []
	var task = ["transfer_flag"]
	todo_list.append(task)

	print("Initial state: ", state)
	print("Todo list: ", todo_list)

	# Find plan
	var result = plan.find_plan(state, todo_list)

	if result.get_success():
		print("Plan found successfully!")
		print("Plan: ", result.get_plan())

		# Get SVG visualization
		var svg_output = result.to_svg_graph()
		print("SVG generated successfully!")
		print("SVG length: ", svg_output.length(), " characters")

		# Save SVG to file
		var file = FileAccess.open("plan_visualization.svg", FileAccess.WRITE)
		if file:
			file.store_string(svg_output)
			file.close()
			print("SVG saved to: plan_visualization.svg")
			print("You can open this file in a web browser to view the plan visualization.")
		else:
			print("Failed to save SVG file")

		# Print first 500 characters for verification
		print("\nFirst 500 characters of SVG:")
		print(svg_output.substr(0, 500))
		print("...")
	else:
		print("Failed to find plan")
		print("Error: ", result.get_error_message())

	quit()
