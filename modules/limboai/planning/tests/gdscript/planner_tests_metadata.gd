# planner_tests_metadata.gd - Goal/HTN/Backtracking with metadata and entity_caps.

static func run_tests(h: RefCounted, domains: RefCounted) -> void:
	_test_metadata_entity_caps_goal(h, domains)
	_test_metadata_entity_caps_htn(h, domains)
	_test_metadata_entity_caps_backtracking(h, domains)

static func _test_metadata_entity_caps_goal(h: RefCounted, domains: RefCounted) -> void:
	var name := "Goal + metadata/entity_caps"
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	state.set_terrain_fact("loc1", "fact_key", 100)
	state.set_entity_capability("ent2", "health", 50)
	state.set_entity_capability_public("ent1", "speed", 5.0)
	var todo_list: Array = [["value", "value", 1]]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(domains.create_minimal_goal_domain())
	plan.set_verbose(1)
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	var expected: Array = [["action_increment", 1]]
	if not h.assert_plan_exact(name, plan_arr, expected):
		return
	if not state.has_terrain_fact("loc1", "fact_key") or int(state.get_terrain_fact("loc1", "fact_key")) != 100:
		h._fail(name, "terrain fact lost")
		return
	if not state.has_entity("ent2") or int(state.get_entity_capability("ent2", "health")) != 50:
		h._fail(name, "entity cap lost")
		return
	if not state.has_entity_capability_public("ent1", "speed"):
		h._fail(name, "entity cap public lost")
		return
	h._ok(name)

static func _test_metadata_entity_caps_htn(h: RefCounted, domains: RefCounted) -> void:
	var name := "HTN + metadata/entity_caps"
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	state.set_terrain_fact("loc1", "fact_key", 100)
	state.set_entity_capability("ent2", "health", 50)
	state.set_entity_capability_public("ent1", "speed", 5.0)
	var todo_list: Array = ["increment"]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(domains.create_minimal_htn_domain())
	plan.set_verbose(1)
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	var expected: Array = [["action_increment", 1]]
	if not h.assert_plan_exact(name, plan_arr, expected):
		return
	if not state.has_terrain_fact("loc1", "fact_key") or not state.has_entity("ent2") or not state.has_entity_capability_public("ent1", "speed"):
		h._fail(name, "metadata/entity_caps lost")
		return
	h._ok(name)

static func _test_metadata_entity_caps_backtracking(h: RefCounted, domains: RefCounted) -> void:
	var name := "Backtracking + metadata/entity_caps"
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	state.set_terrain_fact("loc1", "fact_key", 100)
	state.set_entity_capability("ent2", "health", 50)
	state.set_entity_capability_public("ent1", "speed", 5.0)
	var todo_list: Array = ["increment"]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(domains.create_minimal_backtracking_domain())
	plan.set_verbose(1)
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	var expected: Array = [["action_increment", 1]]
	if not h.assert_plan_exact(name, plan_arr, expected):
		return
	if not state.has_terrain_fact("loc1", "fact_key") or not state.has_entity("ent2") or not state.has_entity_capability_public("ent1", "speed"):
		h._fail(name, "metadata/entity_caps lost")
		return
	h._ok(name)
