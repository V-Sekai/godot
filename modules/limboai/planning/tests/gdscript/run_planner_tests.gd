# run_planner_tests.gd
# Run planner tests via BTRunPlanner (modules/limboai/bt/tasks/planning).
# From repo root: godot --script modules/limboai/planning/tests/gdscript/run_planner_tests.gd
# Tests are split into: goal, htn, metadata, academy, mznc (and optionally ipyhop-ported).

extends SceneTree

const PlanningTestDomains = preload("res://modules/limboai/planning/tests/gdscript/planning_test_domains.gd")
const PlannerTestHelpers = preload("res://modules/limboai/planning/tests/gdscript/planner_test_helpers.gd")
const TestsGoal = preload("res://modules/limboai/planning/tests/gdscript/planner_tests_goal.gd")
const TestsHtn = preload("res://modules/limboai/planning/tests/gdscript/planner_tests_htn.gd")
const TestsMetadata = preload("res://modules/limboai/planning/tests/gdscript/planner_tests_metadata.gd")
const TestsAcademy = preload("res://modules/limboai/planning/tests/gdscript/planner_tests_academy.gd")
const TestsMznc = preload("res://modules/limboai/planning/tests/gdscript/planner_tests_mznc.gd")
const TestsIpyhop = preload("res://modules/limboai/planning/tests/gdscript/planner_tests_ipyhop.gd")

var _domains: RefCounted
var _h: RefCounted

func _init() -> void:
	call_deferred("_main")

func _main() -> void:
	_domains = PlanningTestDomains.new()
	_h = PlannerTestHelpers.new()
	print("LimboAI planner tests (GDScript, BTRunPlanner)")
	print("------------------------------")
	TestsGoal.run_tests(_h, _domains)
	TestsHtn.run_tests(_h, _domains)
	TestsMetadata.run_tests(_h, _domains)
	TestsAcademy.run_tests(_h, _domains)
	TestsMznc.run_tests(_h, _domains)
	TestsIpyhop.run_tests(_h, _domains)
	print("------------------------------")
	print("Passed: ", _h.passed, "  Failed: ", _h.failed)
	_set_exit_code(1 if _h.failed > 0 else 0)
	quit()

func _set_exit_code(_code: int) -> void:
	pass
