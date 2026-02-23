# planner_tests_goal.gd - Goal and two-step goal planner tests.
# Run via run_planner_tests.gd; requires PlannerTestHelpers and PlanningTestDomains.

static func run_tests(h: RefCounted, domains: RefCounted) -> void:
	_test_goal_planning(h, domains)
	_test_two_step_goal(h, domains)

static func _test_goal_planning(h: RefCounted, domains: RefCounted) -> void:
	var name := "BTRunPlanner goal planning"
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
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
	h._ok(name)

static func _test_two_step_goal(h: RefCounted, domains: RefCounted) -> void:
	var name := "BTRunPlanner two-step goal"
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	var todo_list: Array = [["value", "value", 2]]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(domains.create_two_step_goal_domain())
	plan.set_verbose(1)
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	var expected: Array = [["action_increment", 1], ["action_increment", 1]]
	if not h.assert_plan_exact(name, plan_arr, expected):
		return
	h._ok(name)
