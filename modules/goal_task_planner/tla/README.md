# TLA+ Models for Goal Task Planner

This directory contains TLA+ models used to analyze and verify the planning logic. These formal models helped identify and fix several critical bugs in the planner implementation.

## Files

- `GoalTaskPlanner.tla`: Main model of the planning system
- `GoalTaskPlanner.cfg`: TLC configuration for the main model
- `PlanningLoop.tla`: Focused model on the planning loop logic
- `TaskRecreation.tla`: Model for task recreation and method selection
- `BacktrackingTest4.tla`: Model for debugging Backtracking Test 4 failure
- `BacktrackingTest4Fixed.tla`: Refined model after fixes
- `EdgeCasesModel.tla`: Model for edge case analysis
- `TestCasesModel.tla`: Model for test case validation
- `run_tlc.sh`: Helper script to execute TLC

## Running TLA+ Models

### Prerequisites

1. **TLA+ Toolbox**: Install from https://github.com/tlaplus/tlaplus/releases
   - Location: `/Applications/TLA+ Toolbox.app`
   - Contains `tla2tools.jar` at `/Applications/TLA+ Toolbox.app/Contents/Eclipse/tla2tools.jar`

2. **Java**: Install Java runtime
   ```bash
   brew install java
   ```

### Running with TLC

Use the provided helper script:

```bash
# Run the main model
./run_tlc.sh GoalTaskPlanner.tla

# Run with specific configuration
./run_tlc.sh -config GoalTaskPlanner.cfg GoalTaskPlanner.tla

# Check for deadlocks
./run_tlc.sh -deadlock GoalTaskPlanner.tla
```

Or use TLC directly:
```bash
java -cp "/Applications/TLA+ Toolbox.app/Contents/Eclipse/tla2tools.jar" tlc2.TLC GoalTaskPlanner.tla
```

## Purpose

These TLA+ models were used to:

1. **Identify backtracking bugs**: The `BacktrackingTest4.tla` model helped identify that when a CLOSED node is reopened during backtracking, its `created_subtasks` (method array) must be blacklisted to ensure the planner tries alternative methods.

2. **Fix state restoration**: Models revealed that state restoration logic was incorrect, overwriting the current state with old snapshots. This led to fixes in `backtracking.cpp` to clear state and STN snapshots when reopening nodes.

3. **Validate task recreation**: The `TaskRecreation.tla` model helped verify that task recreation logic correctly handles failed root children and method selection.

4. **Verify edge cases**: Edge case models helped identify and fix issues with empty todo lists, invalid node IDs, and malformed graphs.

## Bugs Fixed Using TLA+

The following bugs were identified and fixed using TLA+ models:

1. **Blacklist persistence**: When a CLOSED node is reopened, its previously used method array must be blacklisted to prevent infinite loops.

2. **State restoration**: Reopened nodes should use the current state (with successful actions) rather than restoring old state snapshots.

3. **STN snapshot handling**: Added validation to check for snapshot existence before restoration to prevent crashes.

4. **Task recreation**: Fixed logic to correctly remove failed root children before attempting to recreate them.

5. **Individual action blacklisting**: Identified that individual actions should not be globally blacklisted, only method arrays.

## Model Structure

The models simplify the actual implementation but capture key behaviors:

- **State representation**: Dictionary-based state with flags/variables
- **Task method application**: Method selection and subtask creation
- **Action execution**: State transitions from action application
- **Backtracking**: Node reopening and method blacklisting
- **Task recreation**: Root-level task recreation when all tasks complete
- **Planning completion**: Conditions for successful plan completion

## Status

- ✅ TLA+ models parse correctly
- ✅ TLC can run all models
- ✅ Models helped identify and fix critical bugs
- ✅ All identified issues have been resolved in the C++ implementation
- ✅ All 1302 tests pass

## Notes

- TLA+ models are simplified abstractions of the actual C++ implementation
- Models focus on high-level behavior rather than implementation details
- Some models may deadlock or have incomplete specifications - this is expected as they were used for debugging specific issues
- The models served their purpose in identifying bugs and are maintained for reference
