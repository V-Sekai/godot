# Goal Task Planner

A Hierarchical Task Network (HTN) planner module for Godot Engine. Determines a [PlannerPlan] to accomplish a todo list from a provided state. The planner is compatible with IPyHOP and implements backtracking, temporal constraints, and entity requirements.

## Overview

The Goal Task Planner uses HTN planning to decompose high-level tasks into executable actions. It supports:

- **Three planning methods**: `find_plan()`, `run_lazy_lookahead()`, and `run_lazy_refineahead()`
- **Temporal constraints**: Simple Temporal Network (STN) solver for validating timing constraints
- **Entity requirements**: Match entities by type and capabilities during planning
- **Backtracking**: Automatic backtracking when plans fail, compatible with IPyHOP behavior
- **Plan simulation and replanning**: Simulate plan execution and replan from failure points

## Quick Start

```gdscript
# Create domain and planner
var domain = PlannerDomain.new()
var plan = PlannerPlan.new()
plan.set_current_domain(domain)

# Define actions and methods (see examples below)
# ...

# Plan
var state = {"location": "home"}
var todo_list = ["go_to", "store"]
var result = plan.find_plan(state, todo_list)

if result.get_success():
    var actions = result.extract_plan()
    print("Plan found: ", actions)
else:
    print("Planning failed")
```

## Core Concepts

### Todo List

The todo list is an `Array` of planner elements:
- **Goals**: Predicate-subject-value triples `[predicate, subject, value]`
- **[PlannerMultigoal]**: Arrays of unigoal arrays `[[predicate, subject, value], ...]`
- **Tasks**: Task names with arguments (e.g., `["go_to", "store"]`)
- **Actions**: Action arrays `[action_name, arg1, arg2, ...]`

### Actions

Actions are functions that modify the world state. They:
- Accept the current state `Dictionary` as the first argument, followed by action-specific arguments
- Return `false` if the action is not applicable, or a new state `Dictionary` representing the state after the action

```gdscript
func action_move(state, from, to):
    if state.get("location") != from:
        return false  # Not applicable
    var new_state = state.duplicate()
    new_state["location"] = to
    return new_state
```

### Methods

Methods decompose tasks, goals, and multigoals into subtasks. They:
- Accept the current state `Dictionary` as the first argument, followed by method-specific arguments
- Return `false` if the method is not applicable, or an `Array` of planner elements (goals, multigoals, tasks, actions)

```gdscript
func method_go_to(state, destination):
    var current = state.get("location")
    if current == destination:
        return []  # Already at destination
    return [["move", current, destination]]  # Return action to move
```

### Planning Methods

All three planning methods return `Ref<PlannerResult>`, which contains:
- `final_state`: The final state dictionary after planning
- `solution_graph`: The complete solution graph Dictionary
- `success`: Boolean indicating if planning succeeded
- `extract_plan()`: Method to extract the plan (Array of actions)

#### `find_plan(state, todo_list)`

Standard HTN planning that returns a `PlannerResult`. Does not execute actions. Use this when you only need the plan.

```gdscript
var result = plan.find_plan(state, todo_list)
if result.get_success():
    var actions = result.extract_plan()
```

#### `run_lazy_lookahead(state, todo_list, max_tries=10)`

Lazy lookahead search that attempts planning up to `max_tries` times, executing actions as it goes. Returns `PlannerResult` with the final state after execution. Use `result->extract_plan()` to get the partially executed plan.

#### `run_lazy_refineahead(state, todo_list)`

Graph-based lazy refinement planning with explicit backtracking and STN support. Returns `PlannerResult` with the final state. Use this when you need full graph-based planning with temporal constraints.

## Temporal Constraints

Actions, tasks, and goals can include optional temporal metadata specifying timing constraints. Temporal constraints are provided as a `Dictionary` with a `"temporal_constraints"` key containing:

- `start_time`: int64_t absolute time in microseconds since Unix epoch
- `end_time`: int64_t absolute time in microseconds since Unix epoch
- `duration`: int64_t duration in microseconds

Actions without temporal metadata can occur at any time and are not constrained by the Simple Temporal Network (STN). Actions with temporal metadata are added to the STN and their timing constraints are validated for consistency. If temporal constraints are inconsistent, planning fails.

```gdscript
var action_with_time = {
    "item": ["move", "home", "store"],
    "temporal_constraints": {
        "start_time": 1000000,
        "end_time": 2000000,
        "duration": 1000000
    }
}
```

## Entity Requirements

Entity requirements allow matching entities by type and capabilities during planning. Use `PlannerMetadata` and `PlannerEntityRequirement` to attach entity constraints to planner elements.

```gdscript
var entity_req = {
    "type": "robot",
    "capabilities": ["move", "grasp"]
}
var metadata = {
    "requires_entities": [entity_req]
}
```

## Public API

### PlannerPlan Methods

- **`blacklist_command(command)`**: Adds a command to the blacklist, preventing it from being used during planning
- **`get_iterations()`**: Returns the number of planning iterations from the last operation
- **`simulate(result, state, start_ind=0)`**: Simulates plan execution, returning an array of state dictionaries
- **`replan(result, state, fail_node_id)`**: Re-plans from a failure point in a previous plan

### PlannerResult Methods

- **`extract_plan()`**: Extracts the sequence of actions from the solution graph
- **`find_failed_nodes()`**: Returns all nodes with FAILED status
- **`get_all_nodes()`**: Returns all nodes in the solution graph
- **`get_node(node_id)`**: Returns a specific node by ID
- **`has_node(node_id)`**: Checks if a node exists

## Error Handling

The planner handles edge cases gracefully:

- **Empty todo_list**: Returns a failed `PlannerResult` with appropriate error message
- **Invalid node IDs**: Validates node IDs and skips invalid entries
- **Malformed graphs**: Validates graph structure before processing
- **Array bounds**: Checks array sizes before accessing elements
- **Missing dictionary keys**: Uses `has()` checks before accessing dictionary values

All errors are logged with appropriate verbosity levels and planning fails gracefully without crashing.

## Testing

Run all tests:
```bash
./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests
```

Run specific test:
```bash
./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests --test-name="<Test Name>"
```

## Documentation

- **AGENTS.md**: Comprehensive documentation for AI coding agents and developers working on the module
- **doc_classes/**: Godot class reference documentation
- **tla/README.md**: TLA+ models for formal verification

## IPyHOP Compatibility

This planner aims to be compatible with IPyHOP (Python HTN planner). The reference implementation is in `thirdparty/IPyHOP/`. Key compatibility features:

- Backtracking behavior matches IPyHOP's `_backtrack` method
- DFS traversal order matches IPyHOP's `dfs_preorder_nodes`
- CLOSED nodes are only reopened if they have descendants
- When backtracking from root, all nodes (including siblings) are considered in DFS

## License

See [LICENSE.md](LICENSE.md) for license information.
