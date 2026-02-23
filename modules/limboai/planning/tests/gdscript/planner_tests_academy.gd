# planner_tests_academy.gd - One-block Academy (get_archive_access, prepare_student) tests.
# Run via run_planner_tests.gd; requires PlannerTestHelpers and PlanningTestDomains.

static func run_tests(h: RefCounted, domains: RefCounted) -> void:
	_test_academy_get_archive_access(h, domains)
	_test_academy_prepare_student(h, domains)

static func _test_academy_get_archive_access(h: RefCounted, domains: RefCounted) -> void:
	var name := "One-block Academy get_archive_access"
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	var block := {
		"agent_locations": {"student_1": "start"},
		"key_at": "lounge",
		"archive_locked": true,
		"archive_accessible": false,
		"inventory": {},
		"outfit": {}
	}
	bb.set_var(&"block", block)
	state.set_entity_capability("student_1", "type", "student")
	state.set_entity_capability("student_1", "role", "bio_steward")
	var todo_list: Array = [["get_archive_access", "student_1"]]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_verbose(1)
	plan.set_current_domain(domains.create_academy_one_block_domain_get_archive_lounge_only())
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	var expected: Array = [
		["action_move", "student_1", "lounge"],
		["action_take", "student_1", "key"],
		["action_move", "student_1", "archive_door"],
		["action_use_object", "student_1", "key"]
	]
	if not h.assert_plan_exact(name, plan_arr, expected):
		return
	h._ok(name)

static func _test_academy_prepare_student(h: RefCounted, domains: RefCounted) -> void:
	var name := "One-block Academy prepare_student"
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	var block := {
		"agent_locations": {},
		"key_at": "lounge",
		"archive_locked": true,
		"archive_accessible": false,
		"inventory": {},
		"outfit": {}
	}
	bb.set_var("block", block)
	state.set_entity_capability("student_1", "type", "student")
	state.set_entity_capability("student_1", "role", "bio_steward")
	var todo_list: Array = [["prepare_student", "student_1"]]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_verbose(1)
	plan.set_current_domain(domains.create_academy_one_block_domain())
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	var expected: Array = [["action_equip_garment", "student_1", "apron_of_abundance"]]
	if not h.assert_plan_exact(name, plan_arr, expected):
		return
	h._ok(name)
