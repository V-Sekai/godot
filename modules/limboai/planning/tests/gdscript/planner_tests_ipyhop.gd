# planner_tests_ipyhop.gd - Tests migrated from IPyHOP-temporal ipyhop_tests (backtracking_test.py).
# See: /FIRE202602/Incoming/Experiments/IPyHOP-temporal and V-Sekai-fire/IPyHOP-temporal.

static func run_tests(h: RefCounted, domains: RefCounted) -> void:
	_test_put_it_need0(h, domains)
	_test_put_it_need01(h, domains)
	_test_put_it_need10(h, domains)
	_test_put_it_need1(h, domains)

static func _run_ipyhop_backtracking(h: RefCounted, domains: RefCounted, name: String, todo_list: Array, expected_plan: Array) -> void:
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	bb.set_var(&"ipyhop", {"flag": -1})
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_verbose(2)
	plan.set_current_domain(domains.create_ipyhop_backtracking_domain())
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	if not h.assert_plan_exact(name, res[1], expected_plan):
		return
	h._ok(name)

static func _test_put_it_need0(h: RefCounted, domains: RefCounted) -> void:
	var name := "IPyHOP put_it + need0 (backtrack put_it once)"
	var todo_list: Array = [["put_it"], ["need0"]]
	var expected: Array = [["a_putv", 0], ["a_getv", 0], ["a_getv", 0]]
	_run_ipyhop_backtracking(h, domains, name, todo_list, expected)

static func _test_put_it_need01(h: RefCounted, domains: RefCounted) -> void:
	var name := "IPyHOP put_it + need01 (same backtrack as put_it+need0)"
	var todo_list: Array = [["put_it"], ["need01"]]
	var expected: Array = [["a_putv", 0], ["a_getv", 0], ["a_getv", 0]]
	_run_ipyhop_backtracking(h, domains, name, todo_list, expected)

static func _test_put_it_need10(h: RefCounted, domains: RefCounted) -> void:
	var name := "IPyHOP put_it + need10 (backtrack put_it and need10)"
	var todo_list: Array = [["put_it"], ["need10"]]
	var expected: Array = [["a_putv", 0], ["a_getv", 0], ["a_getv", 0]]
	_run_ipyhop_backtracking(h, domains, name, todo_list, expected)

static func _test_put_it_need1(h: RefCounted, domains: RefCounted) -> void:
	var name := "IPyHOP put_it + need1 (backtrack put_it until m1)"
	var todo_list: Array = [["put_it"], ["need1"]]
	var expected: Array = [["a_putv", 1], ["a_getv", 1], ["a_getv", 1]]
	_run_ipyhop_backtracking(h, domains, name, todo_list, expected)
