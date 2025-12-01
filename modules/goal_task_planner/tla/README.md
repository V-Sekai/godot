# TLA+ Models for Goal Task Planner

This directory contains TLA+ models to help analyze and verify the planning logic.

## Files

- `GoalTaskPlanner.tla`: Main model of the planning system
- `GoalTaskPlanner.cfg`: TLC configuration
- `PlanningLoop.tla`: Focused model on the planning loop logic

## Running TLA+ Models

### Prerequisites

1. **TLA+ Toolbox**: Install from https://github.com/tlaplus/tlaplus/releases
   - Location: `/Applications/TLA+ Toolbox.app`

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

These models help identify:
1. Why the planner stops after 3 actions instead of 7
2. Whether the task recreation logic is correct
3. If there are deadlock or liveness issues in the planning loop

## Status

- ✅ TLA+ model parses correctly
- ✅ TLC can run the model
- ⚠️ Model currently deadlocks - needs refinement to match actual planner initialization
- The deadlock reveals that initial state references nodes that don't exist yet, which may be a clue about the planning issue

## Model Structure

The models simplify the actual implementation but capture key behaviors:
- State representation (flag dictionary)
- Task method application
- Action execution
- Task recreation logic
- Planning completion conditions

