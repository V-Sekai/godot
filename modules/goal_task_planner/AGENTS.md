# AGENTS.md

Instructions for AI coding agents working on the Goal Task Planner module.

## Project Overview

The Goal Task Planner is a Hierarchical Task Network (HTN) planner module for Godot Engine. It determines a `PlannerPlan` to accomplish a todo list from a provided state. The planner is compatible with IPyHOP and implements backtracking, temporal constraints, and entity requirements.

Key components:

-   `PlannerPlan`: Main planning interface with three planning methods
-   `PlannerDomain`: Defines actions, tasks, and methods
-   `PlannerState`: Represents world state
-   `PlannerResult`: Planning result containing final state, solution graph, and success status
-   `PlannerMultigoal`: Utility for working with multigoal arrays
-   `PlannerSolutionGraph`: Internal graph representation with node types and statuses
-   `PlannerBacktracking`: Handles backtracking when plans fail
-   `PlannerGraphOperations`: Graph manipulation utilities
-   `PlannerSTNSolver`: Simple Temporal Network solver for temporal constraints
-   `PlannerSTNConstraints`: STN constraint utilities
-   `PlannerMetadata`: Metadata for temporal and entity requirements
-   `PlannerEntityRequirement`: Entity matching requirements
-   `PlannerTimeRange`: Time range management
-   `PlannerTask` / `PlannerTaskMetadata`: Task metadata support

## Build Commands

-   Build editor: `godot-build-editor` (alias for `scons platform=macos arch=arm64 target=editor dev_build=yes debug_symbols=yes compiledb=yes tests=yes generate_bundle=yes cache_path=/Users/ernest.lee/.scons_cache`)
-   Build and run tests: `godot-build-editor && ./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests`
-   Run specific test: `./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests --test-name="<Test Name>"`

The module uses SCons build system. Source files are automatically discovered via `SCsub`.

## Code Style

-   Follow Godot Engine C++ coding conventions
-   Use `#pragma once` for header guards
-   Include Godot copyright header in all files
-   Use `Ref<>` for Godot object references
-   Use `memnew()` for object creation
-   Prefer `TypedArray<>` over raw arrays
-   Use `Dictionary` and `Array` for state representation

## Testing Instructions

### Test Structure

Tests are organized in `tests/` directory:

-   `unit/`: Unit tests for individual components
    -   `test_planner_components.h`: Component-level tests
    -   `test_comprehensive.h`: Comprehensive workflow tests
    -   `test_ipyhop_compatibility.h`: IPyHOP compatibility tests
-   `problems/`: Integration test scenarios
-   `domains/`: Test domain definitions (actions, methods)
-   `helpers/`: Test helper functions and utilities
-   `test_all.h`: Central include file (must include all test files)

### Running Tests

-   All tests: `./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests`
-   Specific test: `--test-name="IPyHOP Compatibility - Sample Test 1"`
-   After changes, always run tests to verify IPyHOP compatibility

### Test Conventions

-   Use `TEST_CASE("[Modules][Planner] <Description>")` format
-   Use `SUBCASE()` for test subdivisions
-   Use `CHECK()` for assertions
-   Test files must be included in `test_all.h` to be discovered
-   When adding new tests, update `test_all.h` accordingly

## IPyHOP Compatibility

This planner aims to be compatible with IPyHOP (Python HTN planner). Reference implementation is in `thirdparty/IPyHOP/`.

### Key Compatibility Requirements

-   Backtracking behavior must match IPyHOP's `_backtrack` method
-   DFS traversal order must match IPyHOP's `dfs_preorder_nodes`
-   CLOSED nodes are only reopened if they have descendants
-   When backtracking from root, all nodes (including siblings) are considered in DFS

### Testing IPyHOP Compatibility

-   Run IPyHOP Python tests: `cd thirdparty/IPyHOP && python -m pytest ipyhop_tests/`
-   Compare Godot planner output with IPyHOP output
-   Test files in `tests/unit/test_ipyhop_compatibility.h` verify compatibility

## Planning Methods

The `PlannerPlan` class provides three planning methods. **All three methods support STN (Simple Temporal Network) temporal constraints and entity-capability requirements.** All three methods return `Ref<PlannerResult>`, which contains:
- `final_state`: The final state dictionary after planning
- `solution_graph`: The complete solution graph Dictionary (from `PlannerSolutionGraph`)
- `success`: Boolean indicating if planning succeeded
- `extract_plan()`: Method to extract the plan (Array of actions) from the solution graph

1. **`find_plan(state, todo_list)`**: Standard HTN planning that returns a `PlannerResult`. Use `result->extract_plan()` to get the plan (Array of actions) or check `result->get_success()` for failure. Does not execute actions. Supports STN temporal constraints and entity-capability requirements. Use this when you only need the plan.
2. **`run_lazy_lookahead(state, todo_list, max_tries=10)`**: Lazy lookahead search that attempts planning up to `max_tries` times, executing actions as it goes. Returns `PlannerResult` with the final state after execution. Use `result->extract_plan()` to get the partially executed plan (actions that were successfully executed). Supports STN temporal constraints and entity-capability requirements. Use this when you want to execute the plan incrementally.
3. **`run_lazy_refineahead(state, todo_list)`**: Graph-based lazy refinement planning with explicit backtracking and STN support. Returns `PlannerResult` with the final state. Supports STN temporal constraints and entity-capability requirements. Use this when you need full graph-based planning with temporal constraints.

## File Organization

### Core Module Files

-   `plan.h/cpp`: Main planning logic and three planning methods
-   `domain.h/cpp`: Domain definition and management (actions, task methods, unigoal methods, multigoal methods)
-   `planner_state.h/cpp`: State representation
-   `planner_result.h/cpp`: Planning result containing final state, solution graph, and plan extraction
-   `multigoal.h/cpp`: Multigoal utility class
-   `backtracking.h/cpp`: Backtracking implementation
-   `graph_operations.h/cpp`: Graph manipulation utilities
-   `solution_graph.h`: Graph data structure with node types and statuses
-   `stn_solver.h/cpp`: Simple Temporal Network solver for temporal constraints
-   `stn_constraints.h/cpp`: STN constraint utilities
-   `planner_metadata.h`: Metadata system (temporal + entity requirements)
-   `entity_requirement.h`: Entity requirement matching
-   `planner_time_range.h`: Time range management

### Test Files

-   Domain definitions go in `tests/domains/`
-   Helper functions go in `tests/helpers/`
-   Test cases go in `tests/unit/` or `tests/problems/`
-   Always update `test_all.h` when adding new test files
-   New API tests are in `tests/unit/test_new_api.h` (blacklist, iterations, multigoal tags, node tags, replan, simulate, PlannerResult helpers)

### Include Paths

From test files:

-   Module headers: `../../<header>.h` (from `tests/unit/` or `tests/problems/`)
-   Test helpers: `../helpers/<helper>.h`
-   Test domains: `../domains/<domain>.h`
-   Other test files: `../<subdir>/<file>.h` or `./<file>.h` (same directory)

## Important Conventions

### Backtracking Logic

-   When a task fails, backtrack to find a CLOSED ancestor node with descendants
-   Only reopen CLOSED nodes that have descendants (checked via `is_retriable_closed_node`)
-   When backtracking from root (`p_parent_node_id == 0`), check for OPEN nodes at root first
-   Use reverse DFS from parent node to find retriable CLOSED nodes
-   `PlannerBacktracking::backtrack()` returns `BacktrackResult` with updated graph, state, and blacklisted commands
-   STN snapshots are restored during backtracking via `_restore_stn_from_node()`

### State Representation

-   States are `Dictionary` objects with nested structures
-   Actions return new state dictionaries (use `state.duplicate()`)
-   Methods (task, unigoal, multigoal) return `Array` of planner elements (goals, [PlannerMultigoal], tasks, and actions)
-   Goals are predicate-subject-value triples `[predicate, subject, value]`
-   Multigoals are `Array` of unigoal arrays: `[[predicate, subject, value], ...]`

### Temporal Constraints

-   Temporal metadata uses `PlannerTimeRange` and `PlannerMetadata`
-   Times are in microseconds (int64_t) since Unix epoch
-   STN solver validates temporal constraint consistency using Floyd-Warshall algorithm
-   Actions without temporal metadata are unconstrained and not added to STN
-   Temporal constraints include: `start_time`, `end_time`, `duration`
-   STN origin time point is anchored to current time at plan start (for `run_lazy_refineahead`)

### Entity Requirements

-   Entity requirements are specified via `PlannerEntityRequirement` (type + capabilities)
-   Entity matching occurs during planning when `PlannerMetadata` has entity requirements
-   Use `attach_metadata()` to add entity constraints to planner elements
-   Entity requirements can be specified as:
  - Convenience format: `{"type": String, "capabilities": Array}`
  - Full format: `{"requires_entities": Array}` with `PlannerEntityRequirement` dictionaries

### Solution Graph Structure

-   **Node Types**: `TYPE_ROOT`, `TYPE_ACTION`, `TYPE_TASK`, `TYPE_UNIGOAL`, `TYPE_MULTIGOAL`, `TYPE_VERIFY_GOAL`, `TYPE_VERIFY_MULTIGOAL`
-   **Node Status**: `STATUS_OPEN`, `STATUS_CLOSED`, `STATUS_FAILED`, `STATUS_NOT_APPLICABLE`
-   **Node Tags**: Nodes have a "tag" field ("new" or "old") used for replanning. New nodes default to "new", root is "old". During replanning, existing nodes are marked as "old" and new nodes are tagged "new".
-   Graph maintains parent-child relationships via `successors` arrays
-   Each node stores state snapshots, temporal metadata, and method information
-   CLOSED nodes are only reopened if they have descendants (checked via `is_retriable_closed_node`)
-   Use `solution_graph.set_node_tag(node_id, tag)` and `solution_graph.get_node_tag(node_id)` to manage node tags

### Metadata System

-   `PlannerMetadata`: Base class for temporal and entity requirements
-   `PlannerUnigoalMetadata`: Extends `PlannerMetadata` with predicate field
-   Use `attach_metadata()` to attach temporal and/or entity constraints to any planner element
-   Metadata is extracted during planning via `_extract_metadata()`

## Common Tasks

### Adding a New Test

1. Create test file in appropriate subdirectory (`unit/` or `problems/`)
2. Add `#include` to `tests/test_all.h`
3. Use `TEST_CASE("[Modules][Planner] <Name>")` format
4. Run tests to verify

### Adding a New Domain

1. Create domain file in `tests/domains/`
2. Define actions and methods as free functions
3. Create wrapper class with `static` methods for `Callable` creation
4. Use `callable_mp_static(&WrapperClass::method)` to register
5. Actions must return `false` or a new state `Dictionary`
6. Methods must return `false` or an `Array` of planner elements (goals, [PlannerMultigoal], tasks, and actions)
7. Register actions via `domain->add_actions()`
8. Register methods via `domain->add_task_methods()`, `add_unigoal_methods()`, or `add_multigoal_methods()`

### Debugging Planning Issues

-   Enable verbose logging: `plan->set_verbose(3)` (levels 0-3, 3 is most verbose)
-   Check backtracking behavior matches IPyHOP
-   Verify graph operations maintain correct structure
-   Use IPyHOP Python code as reference for expected behavior
-   Check STN consistency: `stn.is_consistent()` and `stn.check_consistency()`
-   Inspect solution graph nodes: `solution_graph.get_node(node_id)`
-   Verify entity requirements are being matched correctly
-   Check temporal constraints are properly attached and validated

### Debugging Infinite Loops and Inefficient Planning

When the planner generates excessively long plans or hits depth limits, determine if it's a **domain problem** or **planner problem**:

#### Domain Problems (Most Common)

Domain methods are responsible for:
-   **Progress tracking**: Methods should only recurse if they make progress toward the goal
-   **Termination conditions**: Methods must check if goals are already achieved before recursing
-   **State validation**: Methods should verify current state before making decisions (e.g., don't move a block that's already on the table)
-   **Idempotency**: Methods should avoid repeating the same operations

**Symptoms of domain problems:**
-   Planner repeatedly executes the same actions (e.g., `action_pickup(c)`, `action_putdown(c)` in a loop)
-   Methods return the same subtasks repeatedly
-   Methods don't check if goals are already achieved
-   Methods recurse unconditionally without progress checks

**Fixes:**
-   Add early termination checks: verify if goal is already achieved before recursing
-   Add state validation: check if operations are already done (e.g., block already on table)
-   Track progress: only recurse if actual work remains
-   Use TLA+ models to verify domain logic (see `tla/` directory)

#### Planner Problems (Less Common)

The planner is responsible for:
-   **State propagation**: Passing updated state to sibling nodes after actions execute
-   **Graph traversal**: Processing nodes in correct order (DFS)
-   **State snapshots**: Correctly saving/restoring state during backtracking

**Symptoms of planner problems:**
-   Methods are called with stale state (state doesn't reflect executed actions)
-   Sibling nodes don't receive updated state from previous siblings
-   State snapshots are incorrect during backtracking

**How to diagnose:**
1.   Check if actions update state correctly: `action->callv(args)` should return new state
2.   Verify state flows correctly: after action executes (line 1442 in `plan.cpp`), it returns to parent with `new_state`
3.   Check if sibling nodes receive updated state: when processing next sibling, verify `p_state` reflects previous actions
4.   Use verbose logging to trace state through the planning graph

**TLA+ Modeling**: Use TLA+ models in `tla/` directory to prototype and verify planning logic before implementing in C++.

## Reference Implementation

-   IPyHOP Python code: `thirdparty/IPyHOP/ipyhop/planner.py`
-   Key methods to reference:
    -   `_backtrack`: Backtracking logic
    -   `_planning`: Main planning loop
    -   `dfs_preorder_nodes`: DFS traversal order
-   IPyHOP examples: `thirdparty/IPyHOP/examples/` (blocks_world, rescue, robosub, simple_travel)
-   IPyHOP tests: `thirdparty/IPyHOP/ipyhop_tests/` (backtracking_test, sample_test_1-8, etc.)

## Public API Methods

### PlannerPlan Public Methods

-   **`blacklist_command(command)`**: Adds a command (action, task, or method result array) to the blacklist, preventing it from being used during planning. Useful for excluding known failing actions or method combinations.
-   **`get_iterations()`**: Returns the maximum number of planning iterations that occurred during the last planning operation. Useful for debugging and performance analysis. Reset at the start of each planning method call.
-   **`simulate(result, state, start_ind=0)`**: Simulates the execution of a plan from a `PlannerResult`, starting from the action at `start_ind`. Returns an `Array` of state `Dictionary` objects, one for each action execution. The first element is the initial state, and each subsequent element is the state after executing the corresponding action. Useful for previewing plan execution without modifying world state.
-   **`replan(result, state, fail_node_id)`**: Re-plans from a failure point in a previous plan. Loads the solution graph from the provided `PlannerResult`, marks nodes from root to the failure point as "old", reopens them, and continues planning from the failure point. Only actions tagged as "new" are included in the returned plan. Use `PlannerResult.find_failed_nodes()` to identify which nodes failed.
-   **`load_solution_graph(graph)`**: Loads a solution graph from a `Dictionary` into the planner's internal state. Used internally by `simulate()` and `replan()` to restore a solution graph from a `PlannerResult`.

### PlannerResult Helper Methods

-   **`extract_plan()`**: Extracts the sequence of actions from the solution graph. Returns an `Array` of action arrays, where each action array has the format `[action_name, arg1, arg2, ...]`. Only successfully completed (CLOSED) actions are included.
-   **`find_failed_nodes()`**: Returns an `Array` of `Dictionary` objects representing all nodes in the solution graph that have a FAILED status. Each dictionary contains "node_id", "type", and "info" keys. Useful for identifying which nodes failed during planning for use with `replan()`.
-   **`get_all_nodes()`**: Returns an `Array` of `Dictionary` objects representing all nodes in the solution graph. Each dictionary contains "node_id", "type", "status", "info", and "tag" keys. Provides a complete overview of the planning graph structure.
-   **`get_node(node_id)`**: Returns the node `Dictionary` for the specified node_id from the solution graph. The dictionary contains all node properties including "type", "status", "info", "tag", "successors", etc. Returns an empty dictionary if the node_id does not exist.
-   **`has_node(node_id)`**: Returns `true` if the solution graph contains a node with the specified node_id, `false` otherwise.

### Working with PlannerResult

When you have a `PlannerResult` from a planning operation:

1. **Check success**: `result->get_success()` to see if planning succeeded
2. **Get the plan**: `result->extract_plan()` to get the action sequence
3. **Inspect the graph**: `result->get_all_nodes()` to see all nodes, or `result->get_node(node_id)` for specific nodes
4. **Find failures**: `result->find_failed_nodes()` to identify failed nodes for replanning
5. **Simulate**: `plan->simulate(result, state, 0)` to simulate plan execution
6. **Replan**: `plan->replan(result, new_state, fail_node_id)` to replan from a failure point

## Additional Notes

### PlannerTask and PlannerTaskMetadata

-   `PlannerTask`: Resource class for tasks with metadata support
-   `PlannerTaskMetadata`: Resource class for task temporal metadata
-   Used for tasks that need temporal constraints attached

### PlannerMultigoal

-   Utility class for working with multigoal arrays
-   Static methods: `is_multigoal_array()`, `method_goals_not_achieved()`, `method_verify_multigoal()`
-   Multigoals are `Array` of unigoal arrays: `[[predicate, subject, value], ...]`
-   **Goal Tag Support**: `get_goal_tag(multigoal)` extracts the goal tag from a multigoal (returns empty string if no tag). `set_goal_tag(multigoal, tag)` attaches a goal tag to a multigoal, wrapping it in a Dictionary with "item" and "goal_tag" keys. Tags can be used to match multigoals to specific methods in the domain.

### STN Solver Details

-   Uses Floyd-Warshall algorithm for all-pairs shortest paths
-   Validates temporal constraint consistency
-   Supports time points, constraints (min/max distance), and snapshots for backtracking
-   Constants: `STN_INFINITY` (INT64_MAX), `STN_NEG_INFINITY` (INT64_MIN + 1)
-   Methods: `add_time_point()`, `add_constraint()`, `check_consistency()`, `create_snapshot()`, `restore_snapshot()`

### VSIDS Activity Tracking

The planner implements VSIDS (Variable State Independent Decaying Sum) style activity tracking for adaptive method selection:

-   **Activity scores**: Methods are scored based on their involvement in conflicts and successes
-   **Activity bumping**: Methods involved in backtracking paths get their activity scores increased
-   **Activity decay**: Activity scores decay periodically (decay factor: 0.95) to prevent stale scores from dominating
-   **Method selection**: When multiple methods are available, the one with the highest activity score is selected first

This optimization helps the planner learn from past planning attempts and prioritize methods that are more likely to succeed. The activity tracking is transparent to domain methods - they don't need to be aware of it.

**TLA+ Models**: See `tla/VSIDSActivityTracking.tla` for a formal model of the activity tracking logic.

## Commit Guidelines

-   Test all changes before committing
-   Ensure IPyHOP compatibility tests pass
-   Update `test_all.h` if adding new test files
-   Follow Godot commit message conventions
-   Keep commits focused on single changes
-   Test all three planning methods (`find_plan`, `run_lazy_lookahead`, `run_lazy_refineahead`) when making core changes
