extends SceneTree

const RescueDomain = preload("rescue_domain.gd")

func create_rescue_domain() -> PlannerDomain:
	var domain = PlannerDomain.new()

	# Add actions
	domain.add_actions([
		RescueDomain.action_free_robot,
		RescueDomain.action_die_update,
		RescueDomain.action_move_euclidean,
		RescueDomain.action_move_manhattan,
		RescueDomain.action_move_curved,
		RescueDomain.action_move_fly,
		RescueDomain.action_move_alt_fly,
		RescueDomain.action_inspect_person,
		RescueDomain.action_support_person,
		RescueDomain.action_support_person_2,
		RescueDomain.action_inspect_location,
		RescueDomain.action_clear_location,
		RescueDomain.action_clear_location_2,
		RescueDomain.action_replenish_supplies,
		RescueDomain.action_transfer,
		RescueDomain.action_capture_image,
		RescueDomain.action_change_altitude,
		RescueDomain.action_check_real,
		RescueDomain.action_engage_robot,
		RescueDomain.action_force_engage_robot
	])

	# Add task methods for move_task
	domain.add_task_methods("move_task", [
		RescueDomain.task_move_method_euclidean,
		RescueDomain.task_move_method_manhattan,
		RescueDomain.task_move_method_curved,
		RescueDomain.task_move_method_fly,
		RescueDomain.task_move_method_alt_fly
	])

	# Add task methods for new_robot_encap_task
	domain.add_task_methods("new_robot_encap_task", [RescueDomain.task_new_robot_encap])

	# Add task methods for rescue_task
	domain.add_task_methods("rescue_task", [
		RescueDomain.task_rescue_method_ground,
		RescueDomain.task_rescue_method_uav
	])

	# Add task methods for support_task
	domain.add_task_methods("support_task", [
		RescueDomain.task_support_person_method_1,
		RescueDomain.task_support_person_method_2,
		RescueDomain.task_support_person_method_3,
		RescueDomain.task_support_person_method_4,
		RescueDomain.task_support_person_method_5
	])

	# Add task methods for help_person_task
	domain.add_task_methods("help_person_task", [RescueDomain.task_help_person])

	# Add task methods for get_supplies_task
	domain.add_task_methods("get_supplies_task", [
		RescueDomain.task_get_supplies_method_robot,
		RescueDomain.task_get_supplies_method_base
	])

	# Add task methods for rescue_encap_task
	domain.add_task_methods("rescue_encap_task", [RescueDomain.task_rescue_encap])

	# Add task methods for survey_task
	domain.add_task_methods("survey_task", [
		RescueDomain.task_survey_method_front,
		RescueDomain.task_survey_method_bottom
	])

	# Add task methods for get_robot_task
	domain.add_task_methods("get_robot_task", [
		RescueDomain.task_get_robot_method_free,
		RescueDomain.task_get_robot_method_force
	])

	# Add task methods for adjust_altitude_task
	domain.add_task_methods("adjust_altitude_task", [
		RescueDomain.task_adjust_altitude_method_low,
		RescueDomain.task_adjust_altitude_method_high
	])

	return domain

func create_rescue_init_state() -> Dictionary:
	var state = {}

	# Rigid relations
	var rigid = {
		"obstacles": [],
		"wheeled_robots": ["r1", "w1"],
		"drones": ["a1"],
		"large_robots": []
	}
	state["rigid"] = rigid

	# Locations
	var loc = {
		"r1": [1, 1],
		"w1": [5, 5],
		"p1": [2, 2],
		"a1": [2, 1]
	}
	state["loc"] = loc

	# Robot types
	var robot_type = {
		"r1": "wheeled",
		"w1": "wheeled",
		"a1": "uav"
	}
	state["robot_type"] = robot_type

	# Medicine
	var has_medicine = {
		"a1": 0,
		"w1": 0,
		"r1": 0
	}
	state["has_medicine"] = has_medicine

	# Status
	var status = {
		"r1": "free",
		"w1": "free",
		"a1": "unk",
		"p1": "unk",
		"[2, 2]": "unk"
	}
	state["status"] = status

	# Altitude
	var altitude = {
		"a1": "high"
	}
	state["altitude"] = altitude

	# Current image
	var current_image = {
		"a1": null
	}
	state["current_image"] = current_image

	# Real status
	var real_status = {
		"r1": "free",
		"w1": "free",
		"a1": "free",
		"p1": "injured",
		"[2, 2]": "clear"
	}
	state["real_status"] = real_status

	# Real person
	var real_person = {
		"[2, 2]": "p1"
	}
	state["real_person"] = real_person

	# New robot
	var new_robot = {
		1: null
	}
	state["new_robot"] = new_robot

	# Weather
	var weather = {
		"[2, 2]": "clear"
	}
	state["weather"] = weather

	return state

func _init():
	var domain = create_rescue_domain()
	var plan = PlannerPlan.new()
	plan.set_current_domain(domain)
	plan.set_max_depth(100)
	plan.set_verbose(1) # Reduced verbosity for clarity
	print("Max Depth set to: ", plan.get_max_depth())

	var state = create_rescue_init_state()

	# Objective: Surprise task for a1 at p1_loc [2,2]
	var todo_list = [
		["survey_task", "a1", [2, 2]]
	]

	print("=== RESCUE DOMAIN - PORTED TO GDSCRIPT ===")
	print("Objective: Survey location [2, 2] with UAV a1")

	var result = plan.find_plan(state, todo_list)

	if result.get_success():
		print("=== PLANNING SUCCEEDED ===")
		var plan_actions = result.extract_plan()
		print("Plan has ", plan_actions.size(), " actions:")
		for i in range(plan_actions.size()):
			print("  ", i + 1, ". ", plan_actions[i])

		# Generate DOT graph
		var dot_content = result.to_dot_graph()
		var file = FileAccess.open("rescue_plan.dot", FileAccess.WRITE)
		file.store_string(dot_content)
		file.close()
		print("Saved plan to rescue_plan.dot")
	else:
		print("=== PLANNING FAILED ===")

	quit()
