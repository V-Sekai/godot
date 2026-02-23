# run_planner_tests.gd
# Run planner tests via BTRunPlanner (modules/limboai/bt/tasks/planning).
# From repo root: godot --script modules/limboai/planning/tests/gdscript/run_planner_tests.gd

extends SceneTree

const PlanningTestDomains = preload("res://modules/limboai/planning/tests/gdscript/planning_test_domains.gd")

var _failed := 0
var _passed := 0
var _domains: RefCounted

func _init() -> void:
	call_deferred("_main")

func _main() -> void:
	_domains = PlanningTestDomains.new()
	_run_tests()
	_set_exit_code(1 if _failed > 0 else 0)
	quit()

func _set_exit_code(_code: int) -> void:
	pass

func _ok(name: String) -> void:
	_passed += 1
	print("[  OK  ] ", name)

func _fail(name: String, msg: String) -> void:
	_failed += 1
	print("[ FAIL ] ", name, " — ", msg)

func _run_tests() -> void:
	print("LimboAI planner tests (GDScript, BTRunPlanner)")
	print("------------------------------")

	_test_goal_planning()
	_test_two_step_goal()
	_test_htn()
	_test_backtracking()
	_test_metadata_entity_caps_goal()
	_test_metadata_entity_caps_htn()
	_test_metadata_entity_caps_backtracking()
	_test_academy_get_archive_access()
	_test_academy_prepare_student()
	_test_mznc2025_temporal_entity()

	print("------------------------------")
	print("Passed: ", _passed, "  Failed: ", _failed)

func _make_state_and_bb() -> Array:
	var bb := Blackboard.new()
	var state := PlannerState.new()
	state.set_blackboard(bb)
	return [state, bb]

# Run planner via BTRunPlanner (bt/tasks/planning); returns [success: bool, plan_arr: Array].
func _run_via_bt_run_planner(bb: Blackboard, state: PlannerState, plan: PlannerPlan) -> Array:
	var dummy := Node.new()
	var run_planner := BTRunPlanner.new()
	run_planner.set_planner_plan(plan)
	run_planner.set_planner_state(state)
	run_planner.initialize(dummy, bb, dummy)
	var status := run_planner.execute(0.0)
	var plan_arr: Array = bb.get_var(&"plan", Array(), false)
	dummy.free()
	return [status == BTTask.SUCCESS, plan_arr]

func _test_goal_planning() -> void:
	var name := "BTRunPlanner goal planning"
	var state_bb := _make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	var todo_list: Array = [["value", "value", 1]]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(_domains.create_minimal_goal_domain())
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	if plan_arr.size() < 1:
		_fail(name, "plan empty")
		return
	_ok(name)

func _test_two_step_goal() -> void:
	var name := "BTRunPlanner two-step goal"
	var state_bb := _make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	var todo_list: Array = [["value", "value", 2]]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(_domains.create_two_step_goal_domain())
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	if plan_arr.size() != 2:
		_fail(name, "expected 2 steps, got %d" % plan_arr.size())
		return
	for i in plan_arr.size():
		var cmd = plan_arr[i]
		if not cmd is Array or cmd.size() < 1 or str(cmd[0]) != "action_increment":
			_fail(name, "step %d not action_increment" % i)
			return
	_ok(name)

func _test_htn() -> void:
	var name := "BTRunPlanner task HTN"
	var state_bb := _make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	var todo_list: Array = ["increment"]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(_domains.create_minimal_htn_domain())
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	if plan_arr.size() < 1:
		_fail(name, "plan empty")
		return
	_ok(name)

func _test_backtracking() -> void:
	var name := "BTRunPlanner backtracking"
	var state_bb := _make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	var todo_list: Array = ["increment"]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(_domains.create_minimal_backtracking_domain())
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	if plan_arr.size() < 1:
		_fail(name, "plan empty")
		return
	_ok(name)

func _test_metadata_entity_caps_goal() -> void:
	var name := "Goal + metadata/entity_caps"
	var state_bb := _make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	state.set_terrain_fact("loc1", "fact_key", 100)
	state.set_entity_capability("ent2", "health", 50)
	state.set_entity_capability_public("ent1", "speed", 5.0)
	var todo_list: Array = [["value", "value", 1]]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(_domains.create_minimal_goal_domain())
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	if not state.has_terrain_fact("loc1", "fact_key") or int(state.get_terrain_fact("loc1", "fact_key")) != 100:
		_fail(name, "terrain fact lost")
		return
	if not state.has_entity("ent2") or int(state.get_entity_capability("ent2", "health")) != 50:
		_fail(name, "entity cap lost")
		return
	if not state.has_entity_capability_public("ent1", "speed"):
		_fail(name, "entity cap public lost")
		return
	_ok(name)

func _test_metadata_entity_caps_htn() -> void:
	var name := "HTN + metadata/entity_caps"
	var state_bb := _make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	state.set_terrain_fact("loc1", "fact_key", 100)
	state.set_entity_capability("ent2", "health", 50)
	state.set_entity_capability_public("ent1", "speed", 5.0)
	var todo_list: Array = ["increment"]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(_domains.create_minimal_htn_domain())
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	if not state.has_terrain_fact("loc1", "fact_key") or not state.has_entity("ent2") or not state.has_entity_capability_public("ent1", "speed"):
		_fail(name, "metadata/entity_caps lost")
		return
	_ok(name)

func _test_metadata_entity_caps_backtracking() -> void:
	var name := "Backtracking + metadata/entity_caps"
	var state_bb := _make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	state.set_terrain_fact("loc1", "fact_key", 100)
	state.set_entity_capability("ent2", "health", 50)
	state.set_entity_capability_public("ent1", "speed", 5.0)
	var todo_list: Array = ["increment"]
	bb.set_var(&"todo_list", todo_list)
	var plan := PlannerPlan.new()
	plan.set_current_domain(_domains.create_minimal_backtracking_domain())
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	if not state.has_terrain_fact("loc1", "fact_key") or not state.has_entity("ent2") or not state.has_entity_capability_public("ent1", "speed"):
		_fail(name, "metadata/entity_caps lost")
		return
	_ok(name)

func _test_academy_get_archive_access() -> void:
	var name := "One-block Academy get_archive_access"
	var state_bb := _make_state_and_bb()
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
	# Use lounge-only domain so the planner picks the method that works (key in lounge).
	plan.set_current_domain(_domains.create_academy_one_block_domain_get_archive_lounge_only())
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	if plan_arr.size() < 1:
		_fail(name, "plan empty")
		return
	var has_move := false
	var has_take_or_interact := false
	var has_use := false
	for el in plan_arr:
		if el is Array and el.size() >= 1:
			var cmd_name := str(el[0])
			if cmd_name == "action_move":
				has_move = true
			elif cmd_name == "action_take" or cmd_name == "action_interact_with":
				has_take_or_interact = true
			elif cmd_name == "action_use_object":
				has_use = true
	if not has_move or not has_take_or_interact or not has_use:
		_fail(name, "expected move, take/interact, use_object in plan")
		return
	_ok(name)

func _test_academy_prepare_student() -> void:
	var name := "One-block Academy prepare_student"
	var state_bb := _make_state_and_bb()
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
	plan.set_current_domain(_domains.create_academy_one_block_domain())
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	if plan_arr.size() != 1:
		_fail(name, "expected 1 step, got %d" % plan_arr.size())
		return
	var cmd = plan_arr[0]
	if not cmd is Array or cmd.size() < 3 or str(cmd[0]) != "action_equip_garment" or str(cmd[2]) != "apron_of_abundance":
		_fail(name, "expected action_equip_garment student_1 apron_of_abundance")
		return
	_ok(name)

func _test_mznc2025_temporal_entity() -> void:
	var name := "mznc2025 temporal+entity (limboai_gtn)"
	var instance: Dictionary
	var f := FileAccess.open("res://thirdparty/mznc2025_probs/limboai_gtn/instance_temporal_entity_01.json", FileAccess.READ)
	if f:
		var json := JSON.new()
		var err := json.parse(f.get_as_text())
		f.close()
		if err == OK:
			instance = json.get_data()
	# Fallback to embedded instance if file missing or parse failed
	if not instance.has("todo_list_goal"):
		instance = {
			"state": { "value": { "value": 0 } },
			"entities": [{ "id": "worker_1", "capabilities": { "type": "worker", "skill": "A" } }],
			"todo_list_goal": ["value", "value", 1],
			"temporal_constraints": { "duration": 1000000 },
			"entity_constraints": { "type": "worker", "capabilities": ["skill"] }
		}
	var state_bb := _make_state_and_bb()
	var state: PlannerState = state_bb[0]
	var bb: Blackboard = state_bb[1]
	# Seed state and entities from instance (temporal+entity problem format in limboai_gtn)
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
	plan.set_current_domain(_domains.create_minimal_goal_domain())
	# Temporal planning requires start horizon (microseconds since epoch)
	plan.set_time_range_dict({ "start_time": int(Time.get_unix_time_from_system() * 1_000_000) })
	# Goal from instance (use int for numeric value so planner state comparison succeeds)
	var goal_raw: Array = (instance.get("todo_list_goal", ["value", "value", 1]) as Array)
	var goal_val: int = int(goal_raw[2]) if goal_raw.size() >= 3 else 1
	var todo_list: Array = [["value", "value", goal_val]]
	bb.set_var(&"todo_list", todo_list)
	var res: Array = _run_via_bt_run_planner(bb, state, plan)
	if not res[0]:
		_fail(name, "BTRunPlanner execute failed")
		return
	var plan_arr: Array = res[1]
	if plan_arr.size() < 1:
		_fail(name, "plan empty")
		return
	_ok(name)
