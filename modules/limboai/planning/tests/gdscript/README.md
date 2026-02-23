# Planner tests (GDScript)

Planner tests run as the main script so you can use **godot --script** without running the full C++ test suite.

## Run

From the Godot engine/repo root (where the Godot binary and `modules/limboai` live):

```bash
godot --script modules/limboai/planning/tests/gdscript/run_planner_tests.gd
```

Or with headless (no display):

```bash
godot --headless --script modules/limboai/planning/tests/gdscript/run_planner_tests.gd
```

Exit code: `0` if all tests pass, `1` otherwise.

## Layout

- **run_planner_tests.gd** — Main entry; runs all test modules.
- **planner_test_helpers.gd** — Shared helpers (state/bb, BTRunPlanner run, exact-plan assert).
- **planner_tests_goal.gd** — Goal and two-step goal.
- **planner_tests_htn.gd** — HTN and backtracking.
- **planner_tests_metadata.gd** — Goal/HTN/Backtracking with metadata and entity_caps.
- **planner_tests_academy.gd** — One-block Academy (get_archive_access, prepare_student).
- **planner_tests_mznc.gd** — mznc2025-style temporal + entity-capability.
- **planning_test_domains.gd** — Domain definitions used by the tests.

## What is tested

- **Goal planning** — minimal unigoal domain (value predicate).
- **Two-step goal** — multi-step unigoal (value 2 → two actions).
- **HTN** — compound task "increment" with one method.
- **Backtracking** — same task with two methods (first fails, second succeeds).
- **Metadata / entity capabilities** — terrain facts and entity capabilities preserved with goal, HTN, and backtracking.
- **Academy one-block** — `get_archive_access` (move, take/interact, use_object) and `prepare_student` (equip by role).
- **mznc2025 temporal+entity** — 20-minute horizon, exact-plan assert.

## IPyHOP-temporal (reference tests)

Additional tests from [V-Sekai-fire/IPyHOP-temporal](https://github.com/V-Sekai-fire/IPyHOP-temporal) are available as a submodule for parity and reference:

- **Submodule:** `modules/limboai/thirdparty/IPyHOP-temporal`
- **Run their tests (Python):**  
  `cd modules/limboai/thirdparty/IPyHOP-temporal && python -m pytest ipyhop_tests/ -v`  
  (requires Python 3 and dependencies from their `requirements.txt`.)

IPyHOP is a re-entrant iterative GTN planner; the submodule provides backtracking, replanning, and temporal examples that align with LimboAI planning behavior.
