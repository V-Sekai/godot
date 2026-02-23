# planner_tests_ipyhop.gd - Tests migrated from IPyHOP-temporal ipyhop_tests.
# backtracking_test.py, sample_test_1.py, sample_test_2.py, sample_test_3.py.

static func run_tests(h: RefCounted, domains: RefCounted) -> void:
	_test_put_it_need0(h, domains)
	_test_put_it_need01(h, domains)
	_test_put_it_need10(h, domains)
	_test_put_it_need1(h, domains)
	_test_sample_1(h, domains)
	_test_sample_2(h, domains)
	_test_sample_3(h, domains)

static func _run_ipyhop_backtracking(h: RefCounted, domains: RefCounted, name: String, todo_list: Array, expected_plan: Array) -> void:
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	bb.set_var(&"ipyhop", {"flag": -1})
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
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

# ---- Sample tests (ipyhop_tests/sample_test_*.py, test_action_models, test_state_models) ----
# Initial state: flag[0]=true, flag[1..19]=false. State key "flag" from blackboard.

static func _make_sample_initial_flag() -> Dictionary:
	var flag: Dictionary = {}
	flag[0] = true
	for i in range(1, 20):
		flag[i] = false
	return flag

static func _run_ipyhop_sample(h: RefCounted, name: String, todo_list: Array, expected_plan: Array, domain: PlannerDomain, expect_failure: bool) -> void:
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	bb.set_var(&"flag", _make_sample_initial_flag())
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_verbose(3)
	plan.set_current_domain(domain)
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if expect_failure:
		if res[0]:
			h._fail(name, "expected planning to fail but got success")
			return
		if (res[1] as Array).size() != 0:
			h._fail(name, "expected empty plan on failure, got %d steps" % [(res[1] as Array).size()])
			return
		h._ok(name)
		return
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	if not h.assert_plan_exact(name, res[1], expected_plan):
		return
	h._ok(name)

static func _test_sample_1(h: RefCounted, domains: RefCounted) -> void:
	# sample_test_1.py: depth 2, backtrack to tm_1_3 and tm_2_2
	var name := "IPyHOP sample_1 (depth 2 backtrack tm_1 then tm_2)"
	var todo_list: Array = [["tm_1"], ["tm_2"]]
	var expected: Array = [
		["t_a", 0, 1], ["t_a", 1, 2], ["t_a", 2, 3], ["t_a", 3, 4],
		["t_a", 4, 5], ["t_a", 5, 6], ["t_a", 6, 7]
	]
	_run_ipyhop_sample(h, name, todo_list, expected, domains.create_ipyhop_sample_1_domain(), false)

static func _test_sample_2(h: RefCounted, domains: RefCounted) -> void:
	# sample_test_2.py: depth 3, tm_1 and tm_3
	var name := "IPyHOP sample_2 (depth 3 backtrack tm_1/tm_2)"
	var todo_list: Array = [["tm_1"], ["tm_3"]]
	var expected: Array = [
		["t_a", 0, 1], ["t_a", 1, 2], ["t_a", 2, 3], ["t_a", 3, 7],
		["t_a", 3, 4], ["t_a", 4, 5], ["t_a", 7, 8]
	]
	_run_ipyhop_sample(h, name, todo_list, expected, domains.create_ipyhop_sample_2_domain(), false)

static func _test_sample_3(h: RefCounted, domains: RefCounted) -> void:
	# sample_test_3.py: unsolvable (tm_3 needs t_a 9->10, flag 9 never set)
	var name := "IPyHOP sample_3 (unsolvable)"
	var todo_list: Array = [["tm_1"], ["tm_3"]]
	_run_ipyhop_sample(h, name, todo_list, [], domains.create_ipyhop_sample_3_domain(), true)
