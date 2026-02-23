static func run_tests(h: RefCounted, domains: RefCounted) -> void:
	_test_htn(h, domains)
	_test_backtracking(h, domains)

static func _test_htn(h: RefCounted, domains: RefCounted) -> void:
	var name := "BTRunPlanner task HTN"
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	var todo_list: Array = ["increment"]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(domains.create_minimal_htn_domain())
	plan.set_verbose(1)
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	if not h.assert_plan_exact(name, res[1], [["action_increment", 1]]):
		return
	h._ok(name)

static func _test_backtracking(h: RefCounted, domains: RefCounted) -> void:
	var name := "BTRunPlanner backtracking"
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	var todo_list: Array = ["increment"]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(domains.create_minimal_backtracking_domain())
	plan.set_verbose(1)
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	if not h.assert_plan_exact(name, res[1], [["action_increment", 1]]):
		return
	h._ok(name)
