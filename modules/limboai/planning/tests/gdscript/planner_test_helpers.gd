# planner_test_helpers.gd
# Shared helpers for planner GDScript tests (BTRunPlanner, exact-plan assert, state/bb).
# Used by run_planner_tests.gd and split test modules. Instantiate via preload(...).new().

extends RefCounted

var failed: int = 0
var passed: int = 0

func _ok(name: String) -> void:
	passed += 1
	print("[  OK  ] ", name)

func _fail(name: String, msg: String) -> void:
	failed += 1
	print("[ FAIL ] ", name, " — ", msg)

func make_state_and_bb() -> Array:
	var bb := Blackboard.new()
	var state := PlannerState.new()
	state.set_blackboard(bb)
	return [state, bb]

# Run planner via BTRunPlanner; returns [success: bool, plan_arr: Array].
func run_via_bt_run_planner(bb: Blackboard, state: PlannerState, plan: PlannerPlan) -> Array:
	var dummy := Node.new()
	var run_planner := BTRunPlanner.new()
	run_planner.set_planner_plan(plan)
	run_planner.set_planner_state(state)
	run_planner.initialize(dummy, bb, dummy)
	var status := run_planner.execute(0.0)
	var plan_arr: Array = bb.get_var(&"plan", Array(), false)
	dummy.free()
	return [status == BTTask.SUCCESS, plan_arr]

# Assert plan matches expected exactly; print plan. Returns true if match.
func assert_plan_exact(name: String, plan_arr: Array, expected_plan: Array) -> bool:
	print("  [%s] plan (%d steps): %s" % [name, plan_arr.size(), plan_arr])
	if plan_arr.size() != expected_plan.size():
		_fail(name, "expected exactly %d commands, got %d" % [expected_plan.size(), plan_arr.size()])
		return false
	for idx in range(plan_arr.size()):
		if plan_arr[idx] != expected_plan[idx]:
			_fail(name, "step %d: expected %s, got %s" % [idx, expected_plan[idx], plan_arr[idx]])
			return false
	return true
