# planner_tests_mznc.gd — mznc2025-style temporal + entity-capability test.
# Run via run_planner_tests.gd; requires PlannerTestHelpers and PlanningTestDomains.

const PLAN_DURATION_20MIN_USEC: int = 20 * 60 * 1_000_000  # 1_200_000_000

static func run_tests(h: RefCounted, domains: RefCounted) -> void:
	_test_mznc2025_temporal_entity(h, domains)

static func _test_mznc2025_temporal_entity(h: RefCounted, domains: RefCounted) -> void:
	var name := "mznc2025 temporal+entity (limboai_gtn)"
	var instance: Dictionary = {
		"state": { "value": { "value": 0 } },
		"entities": [{ "id": "worker_1", "capabilities": { "type": "worker", "skill": "A" } }],
		"todo_list_goal": ["value", "value", 5],
		"temporal_constraints": { "duration": PLAN_DURATION_20MIN_USEC },
		"entity_constraints": { "type": "worker", "capabilities": ["skill"] }
	}
	var state_bb: Array = h.make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	if instance.has("state") and instance.state is Dictionary:
		var state_dict: Dictionary = (instance.state as Dictionary)
		if state_dict.has("value"):
			bb.set_var(&"value", state_dict.get("value", Dictionary()) as Dictionary)
	for ent in instance.get("entities", []):
		if ent is Dictionary and ent.has("id") and ent.has("capabilities"):
			var ent_d: Dictionary = ent as Dictionary
			var caps: Dictionary = (ent_d.get("capabilities", Dictionary()) as Dictionary)
			for cap_name in caps.keys():
				state.set_entity_capability(str(ent_d.get("id")), str(cap_name), caps[cap_name])
	var plan := PlannerPlan.new()
	plan.set_current_domain(domains.create_minimal_goal_domain())
	plan.set_verbose(1)
	plan.set_max_depth(24)
	var start_usec: int = int(Time.get_unix_time_from_system() * 1_000_000)
	plan.set_time_range_dict({ "start_time": start_usec, "duration": PLAN_DURATION_20MIN_USEC })
	var goal_raw: Array = (instance.get("todo_list_goal", ["value", "value", 1]) as Array)
	var goal_val: int = int(goal_raw[2]) if goal_raw.size() >= 3 else 1
	var todo_list: Array = [["value", "value", goal_val]]
	bb.set_var(&"todo_list", todo_list)
	var res: Array = h.run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		h._fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	var expected_plan: Array = []
	for _i in range(goal_val):
		expected_plan.append(["action_increment", 1])
	if not h.assert_plan_exact(name, plan_arr, expected_plan):
		return
	h._ok(name)
