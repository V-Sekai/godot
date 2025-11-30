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

The `PlannerPlan` class provides three planning methods. All three methods return `Ref<PlannerResult>`, which contains:
- `final_state`: The final state dictionary after planning
- `solution_graph`: The complete solution graph Dictionary (from `PlannerSolutionGraph`)
- `success`: Boolean indicating if planning succeeded
- `extract_plan()`: Method to extract the plan (Array of actions) from the solution graph

1. **`find_plan(state, todo_list)`**: Standard HTN planning that returns a `PlannerResult`. Use `result->extract_plan()` to get the plan (Array of actions) or check `result->get_success()` for failure. Does not execute actions. Use this when you only need the plan.
2. **`run_lazy_lookahead(state, todo_list, max_tries=10)`**: Lazy lookahead search that attempts planning up to `max_tries` times, executing actions as it goes. Returns `PlannerResult` with the final state after execution. Use this when you want to execute the plan incrementally.
3. **`run_lazy_refineahead(state, todo_list)`**: Graph-based lazy refinement planning with explicit backtracking and STN support. Returns `PlannerResult` with the final state. Use this when you need full graph-based planning with temporal constraints.

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
-   Graph maintains parent-child relationships via `successors` arrays
-   Each node stores state snapshots, temporal metadata, and method information
-   CLOSED nodes are only reopened if they have descendants (checked via `is_retriable_closed_node`)

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

## Reference Implementation

-   IPyHOP Python code: `thirdparty/IPyHOP/ipyhop/planner.py`
-   Key methods to reference:
    -   `_backtrack`: Backtracking logic
    -   `_planning`: Main planning loop
    -   `dfs_preorder_nodes`: DFS traversal order
-   IPyHOP examples: `thirdparty/IPyHOP/examples/` (blocks_world, rescue, robosub, simple_travel)
-   IPyHOP tests: `thirdparty/IPyHOP/ipyhop_tests/` (backtracking_test, sample_test_1-8, etc.)

## Additional Notes

### PlannerTask and PlannerTaskMetadata

-   `PlannerTask`: Resource class for tasks with metadata support
-   `PlannerTaskMetadata`: Resource class for task temporal metadata
-   Used for tasks that need temporal constraints attached

### PlannerMultigoal

-   Utility class for working with multigoal arrays
-   Static methods: `is_multigoal_array()`, `method_goals_not_achieved()`, `method_verify_multigoal()`
-   Multigoals are `Array` of unigoal arrays: `[[predicate, subject, value], ...]`

### STN Solver Details

-   Uses Floyd-Warshall algorithm for all-pairs shortest paths
-   Validates temporal constraint consistency
-   Supports time points, constraints (min/max distance), and snapshots for backtracking
-   Constants: `STN_INFINITY` (INT64_MAX), `STN_NEG_INFINITY` (INT64_MIN + 1)
-   Methods: `add_time_point()`, `add_constraint()`, `check_consistency()`, `create_snapshot()`, `restore_snapshot()`

## Commit Guidelines

-   Test all changes before committing
-   Ensure IPyHOP compatibility tests pass
-   Update `test_all.h` if adding new test files
-   Follow Godot commit message conventions
-   Keep commits focused on single changes
-   Test all three planning methods (`find_plan`, `run_lazy_lookahead`, `run_lazy_refineahead`) when making core changes
