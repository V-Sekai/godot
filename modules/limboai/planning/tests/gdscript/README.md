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

## What is tested

- **Goal planning** — minimal unigoal domain (value predicate).
- **Two-step goal** — multi-step unigoal (value 2 → two actions).
- **HTN** — compound task "increment" with one method.
- **Backtracking** — same task with two methods (first fails, second succeeds).
- **Metadata / entity capabilities** — terrain facts and entity capabilities preserved with goal, HTN, and backtracking.
- **Academy one-block** — `get_archive_access` (move, take/interact, use_object) and `prepare_student` (equip by role).

Domains are implemented in `planning_test_domains.gd` (GDScript equivalents of the C++ test domains).
