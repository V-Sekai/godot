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
-   `PlannerPersona`: Persona entity for belief-immersed planning (human, AI, hybrid)
-   `PlannerBeliefManager`: Belief formation and updating across personas
-   `PlannerState`: Unified knowledge representation using triples with metadata

## Build Commands

See [docs/BUILD.md](docs/BUILD.md) for detailed build instructions.

Quick reference:

-   Build editor: `godot-build-editor`
-   Build and run tests: `godot-build-editor && ./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests`
-   Run specific test: `./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests --test-name="<Test Name>"`

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

-   Use `TEST_CASE("[Modules][Planner] <Description>")` format (or `[Modules][Planner][Domain]` for domain-specific tests)
-   Use `SUBCASE()` for test subdivisions
-   Use `CHECK()` for assertions
-   Test files must be included in `test_all.h` to be discovered
-   When adding new tests, update `test_all.h` accordingly
-   **Test Isolation**: Always call `plan->reset()` at the start of each test case that uses `PlannerPlan`, even though the constructor initializes state. This ensures complete isolation between tests.
-   **State Copying**: Use `init_state.duplicate(true)` before passing state to `find_plan()` to prevent state pollution. The `duplicate(true)` method correctly handles nested dictionaries and arrays in Godot.

## Temporal Planning

The planner supports temporal constraints using absolute time in microseconds (since Unix epoch). This differs from IPyHOP which uses ISO 8601 duration strings (e.g., `PT5M` for 5 minutes).

### Time Representation

-   **Absolute Time**: All times are in microseconds since Unix epoch (1970-01-01 00:00:00 UTC)
-   **Duration**: Action durations are specified in microseconds
-   **Conversion**: Use `PlannerTimeRange.unix_time_to_microseconds()` to convert from Unix time (seconds)

### Temporal Action Metadata

Actions can have temporal metadata attached:

```cpp
Dictionary temporal_metadata;
temporal_metadata["duration"] = static_cast<int64_t>(300000000LL); // 5 minutes
temporal_metadata["start_time"] = static_cast<int64_t>(1735732800000000LL); // Optional
temporal_metadata["end_time"] = static_cast<int64_t>(1735732805000000LL); // Optional
plan->attach_metadata(action_array, temporal_metadata);
```

### STN Solver

The planner uses a Simple Temporal Network (STN) solver to validate temporal constraints. The STN ensures all timing constraints are consistent and calculates feasible time windows for actions.

## IPyHOP Compatibility

This planner aims to be compatible with IPyHOP (Python HTN planner). Reference implementation is in `thirdparty/IPyHOP/`.

### Key Compatibility Requirements

-   Backtracking behavior must match IPyHOP's `_backtrack` method
-   DFS traversal order must match IPyHOP's `dfs_preorder_nodes`
-   CLOSED nodes are only reopened if they have descendants
-   When backtracking from root, all nodes (including siblings) are considered in DFS
-   **Task method pattern**: Methods check state conditions and return actions directly (no pre-execution)
-   **Action validation**: Actions validate themselves when executed, returning `None` (NIL) if they fail
-   **HTN decomposition**: Tasks should be decomposed into multiple methods, each handling specific cases
-   **State restoration**: When revisiting a node, always restore the saved state (matches IPyHOP behavior)
-   **Time representation**: Uses absolute microseconds instead of ISO 8601 durations (PT5M, PT10M, etc.)

### IPyHOP Task Method Pattern

Following the IPyHOP pattern (see `thirdparty/IPyHOP/examples/blocks_world/task_based/blocks_world_methods_1.py`):

1. **Task methods check state conditions**: Methods examine the current state to determine if they're applicable
2. **Return actions directly**: Based on state conditions, methods return actions without pre-executing them
3. **Actions validate themselves**: When actions are executed, they check preconditions and return `Variant()` (NIL) if they fail
4. **Use helper functions for safety**: Create helper functions like `would_action_be_safe()` that simulate state changes to check if actions would be safe, without actually calling the action

**Example from IPyHOP blocks_world:**

```python
def tm_get(state, b1):
    if state.clear[b1]:
        if state.pos[b1] == 'table':
            return [('a_pickup', b1)]
        else:
            return [('a_unstack', b1, state.pos[b1])]
    # Returns None implicitly if not applicable
```

**C++ equivalent:**

```cpp
// Method 1: Pickup from table
Variant task_get_method_pickup(Dictionary state, Variant b1) {
    Dictionary clear = state.get("clear", Dictionary());
    Dictionary pos = state.get("pos", Dictionary());
    if (clear.get(b1, false)) {
        Variant b1_pos = pos.get(b1, Variant());
        if (b1_pos.get_type() == Variant::STRING && String(b1_pos) == "table") {
            Array result;
            result.push_back(["action_pickup", b1]);
            return result;
        }
    }
    return Variant(); // Not applicable, try next method
}
```

### Testing IPyHOP Compatibility

-   Run IPyHOP Python tests: `cd thirdparty/IPyHOP && python -m pytest ipyhop_tests/`
-   Compare Godot planner output with IPyHOP output
-   Test files in `tests/unit/test_ipyhop_compatibility.h` verify compatibility
-   Reference IPyHOP examples: `thirdparty/IPyHOP/examples/blocks_world/task_based/blocks_world_methods_1.py` for decomposition patterns

## Planning Methods

The `PlannerPlan` class provides three planning methods. **All three methods support STN (Simple Temporal Network) temporal constraints and entity-capability requirements.** All three methods return `Ref<PlannerResult>`, which contains:

-   `final_state`: The final state dictionary after planning
-   `solution_graph`: The complete solution graph Dictionary (from `PlannerSolutionGraph`)
-   `success`: Boolean indicating if planning succeeded
-   `extract_plan()`: Method to extract the plan (Array of actions) from the solution graph

1. **`find_plan(state, todo_list)`**: Standard HTN planning that returns a `PlannerResult`. Use `result->extract_plan()` to get the plan (Array of actions) or check `result->get_success()` for failure. Does not execute actions. Supports STN temporal constraints and entity-capability requirements. Use this when you only need the plan.
2. **`run_lazy_lookahead(state, todo_list, max_tries=10)`**: Lazy lookahead search that attempts planning up to `max_tries` times, executing actions as it goes. Returns `PlannerResult` with the final state after execution. Use `result->extract_plan()` to get the partially executed plan (actions that were successfully executed). Supports STN temporal constraints and entity-capability requirements. Use this when you want to execute the plan incrementally.
3. **`run_lazy_refineahead(state, todo_list)`**: Graph-based lazy refinement planning with explicit backtracking and STN support. Returns `PlannerResult` with the final state. Supports STN temporal constraints and entity-capability requirements. Use this when you need full graph-based planning with temporal constraints.

## File Organization

### Core Module Files

All source files are located in `src/` directory:

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
-   `planner_persona.h/cpp`: Persona entity for belief-immersed planning
-   `planner_belief_manager.h/cpp`: Belief formation and updating across personas
-   `planner_state.h/cpp`: Unified knowledge representation using triples with metadata

### Test Files

-   Domain definitions go in `tests/domains/`
-   Helper functions go in `tests/helpers/`
-   Test cases go in `tests/unit/` or `tests/problems/`
-   Always update `test_all.h` when adding new test files
-   New API tests are in `tests/unit/test_new_api.h` (blacklist, iterations, multigoal tags, node tags, replan, simulate, PlannerResult helpers)

### Include Paths

From test files:

-   Module headers: `../../src/<header>.h` (from `tests/unit/` or `tests/problems/`)
-   Test helpers: `../helpers/<helper>.h`
-   Test domains: `../domains/<domain>.h`
-   Other test files: `../<subdir>/<file>.h` or `./<file>.h` (same directory)

## Important Conventions

### PlannerPlan Lifecycle and State Management

-   **Constructor**: `PlannerPlan()` automatically calls `reset()` to initialize all state to defaults
-   **Destructor**: `~PlannerPlan()` automatically calls `reset()` to clean up all state
-   **Test Isolation**: Always call `plan->reset()` at the start of each test case to ensure complete isolation, even though the constructor initializes state
-   **State Persistence**: Configuration (verbose, max_depth, domain) persists across `find_plan()` calls, but planning-specific state (solution_graph, blacklisted_commands, iterations, VSIDS activity) is reset at the start of each planning call
-   **State Copying**: The planner automatically deep-copies the input state using `duplicate(true)` to prevent state pollution. Tests should also use `state.duplicate(true)` before passing state to `find_plan()` for isolation.
-   **Temporal Time**: All temporal constraints use absolute time in microseconds (not ISO 8601 durations). Convert from Unix time using `PlannerTimeRange::unix_time_to_microseconds()`.

### Backtracking Logic

-   When a task fails, backtrack to find a CLOSED ancestor node with descendants
-   Only reopen CLOSED nodes that have descendants (checked via `is_retriable_closed_node`)
-   When backtracking from root (`p_parent_node_id == 0`), check for OPEN nodes at root first
-   Use reverse DFS from parent node to find retriable CLOSED nodes
-   `PlannerBacktracking::backtrack()` returns `BacktrackResult` with updated graph, state, and blacklisted commands
-   STN snapshots are restored during backtracking via `_restore_stn_from_node()`

### State Representation

-   States are `Dictionary` objects with nested structures or flat predicates
-   Actions return new state dictionaries (use `state.duplicate()`) or `Variant()` (NIL) on failure
-   Task methods return `Array` of planner elements (goals, [PlannerMultigoal], tasks, and actions) on success, or `Variant()` (NIL) on failure
-   Goals are predicate-subject-value triples `[predicate, subject, value]`
    -   **Subject must never be empty**: All goals must have a non-empty subject field
    -   **Standard goals**: `["study_points", "yuki", 10]` where state structure is `state["study_points"]["yuki"] = 10`
    -   **Flat predicate goals**: `["relationship_points", "relationship_points_yuki_maya", 3]` where:
        -   Predicate is `"relationship_points"` (used for method lookup)
        -   Subject is the full flat predicate `"relationship_points_yuki_maya"` (contains persona_id and companion_id)
        -   State structure is `state["relationship_points_yuki_maya"] = 3` (flat, not nested)
        -   The multigoal verification checks `state[subject] == value` for flat predicates
-   Multigoals are `Array` of unigoal arrays: `[[predicate, subject, value], ...]`
-   **Important**: Use `Variant()` (NIL) to signal failure, not empty `Array()` or empty `Dictionary()`. Empty arrays mean "success with no work", empty dictionaries are invalid.

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
-   Convenience format: `{"type": String, "capabilities": Array}`
-   Full format: `{"requires_entities": Array}` with `PlannerEntityRequirement` dictionaries

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
5. Actions must return `Variant()` (NIL) on failure or a new state `Dictionary` on success
6. Task methods must return `Variant()` (NIL) on failure or an `Array` of planner elements (goals, [PlannerMultigoal], tasks, and actions) on success
7. Register actions via `domain->add_actions()`
8. Register methods via `domain->add_task_methods()`, `add_unigoal_methods()`, or `add_multigoal_methods()`

### HTN Decomposition Pattern

HTN (Hierarchical Task Network) planning works by decomposing tasks into smaller subtasks. Follow the IPyHOP pattern:

**Key Principles:**

-   **Decompose tasks into multiple methods**: Instead of one large method with complex logic, create multiple methods that each handle a specific case
-   **Each method checks state conditions**: Methods should check if they're applicable based on state, return `Variant()` (NIL) if not applicable
-   **Methods return subtasks, not actions directly**: Higher-level tasks decompose into lower-level tasks, which eventually decompose into actions
-   **Check state conditions, don't pre-execute actions**: Methods should check if actions would be safe by simulating state changes, not by calling actions

**Example Pattern (from blocks_world and fox_geese_corn):**

```cpp
// Task: transport_all - decompose into multiple methods
// Method 1: Goal already achieved
Variant task_transport_all_method_done(Dictionary p_state) {
    if (goal_achieved(p_state)) {
        return Array(); // Success, no work needed
    }
    return Variant(); // Not applicable, try next method
}

// Method 2: Handle specific state condition
Variant task_transport_all_method_return_boat(Dictionary p_state) {
    if (boat_location == "east") {
        // Return action + recursive task
        Array result;
        result.push_back(["action_cross_west", 0, 0, 0]);
        result.push_back(["transport_all"]);
        return result;
    }
    return Variant(); // Not applicable, try next method
}

// Method 3: Decompose into smaller task
Variant task_transport_all_method_transport_one(Dictionary p_state) {
    if (boat_location == "west") {
        // Decompose into transport_one task + recursive transport_all
        Array result;
        result.push_back(["transport_one"]);
        result.push_back(["transport_all"]);
        return result;
    }
    return Variant(); // Not applicable, all methods failed
}

// Lower-level task: transport_one - also has multiple methods
Variant task_transport_one_method_geese(Dictionary p_state) {
    if (west_geese > 0 && would_action_be_safe(p_state, "east", 0, geese_count, 0)) {
        Array result;
        result.push_back(["action_cross_east", 0, geese_count, 0]);
        return result;
    }
    return Variant(); // Not applicable, try next method
}
```

**Registering Multiple Methods:**

```cpp
// Register multiple methods for the same task
TypedArray<Callable> transport_all_methods;
transport_all_methods.push_back(callable_mp_static(&DomainCallable::task_transport_all_method_done));
transport_all_methods.push_back(callable_mp_static(&DomainCallable::task_transport_all_method_return_boat));
transport_all_methods.push_back(callable_mp_static(&DomainCallable::task_transport_all_method_transport_one));
domain->add_task_methods("transport_all", transport_all_methods);
```

**IPyHOP Pattern for Task Methods:**

-   **Check state conditions directly**: Don't call actions to test them, check state variables
-   **Return actions directly**: Based on state conditions, return the appropriate action
-   **Use helper functions for safety checks**: Create `would_action_be_safe()` functions that simulate state changes without calling actions
-   **Return `Variant()` (NIL) on failure**: If no method is applicable, the planner will backtrack
-   **Methods are tried in order**: The planner tries each method until one returns a non-NIL value

**Benefits of HTN Decomposition:**

-   Better backtracking: If one method fails, the planner tries the next method
-   Clearer logic: Each method handles one specific case
-   Easier to debug: Methods are smaller and more focused
-   Matches IPyHOP pattern: Follows the same decomposition style as IPyHOP examples

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

1.  Check if actions update state correctly: `action->callv(args)` should return new state
2.  Verify state flows correctly: after action executes (line 1442 in `plan.cpp`), it returns to parent with `new_state`
3.  Check if sibling nodes receive updated state: when processing next sibling, verify `p_state` reflects previous actions
4.  Use verbose logging to trace state through the planning graph

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
-   **`get_method_activities()`**: Returns a `Dictionary` of VSIDS activity scores (method_id -> activity_score). Useful for inspecting which methods have been rewarded or penalized during planning. Activity scores persist across `find_plan()` calls to enable learning.
-   **`reset_vsids_activity()`**: Resets all VSIDS activity scores to zero, clearing accumulated learning. Useful when switching between different problem types or domains to prevent contamination. Activity persists by default to enable optimization across multiple attempts.
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

### PlannerMultigoal

-   Utility class for working with multigoal arrays
-   Static methods: `is_multigoal_array()`, `method_goals_not_achieved()`, `method_verify_multigoal()`, `get_goal_tag()`, `set_goal_tag()`
-   Multigoals are `Array` of unigoal arrays: `[[predicate, subject, value], ...]`
-   **Goal Tag Support**: `get_goal_tag(multigoal)` extracts the goal tag from a multigoal (returns empty string if no tag). `set_goal_tag(multigoal, tag)` attaches a goal tag to a multigoal, wrapping it in a Dictionary with "item" and "goal_tag" keys. Tags can be used to match multigoals to specific methods in the domain.

### Belief-Immersed Architecture

The planner supports belief-immersed planning through three key classes:

-   **`PlannerPersona`**: Represents a persona (human, AI, or hybrid) with capabilities and ego-centric beliefs. Personas are differentiated by capabilities (human: movable, inventory, craft, mine, build, interact; AI: movable, compute, optimize, predict, learn, navigate). Each persona maintains beliefs about other personas with confidence levels and timestamps. Use `PlannerPersona::create_human()`, `create_ai()`, `create_hybrid()`, or `create_basic()` to create personas.

-   **`PlannerBeliefManager`**: Handles belief formation, updating, and confidence management across personas. Manages the persona registry and enforces information asymmetry (personas cannot directly access each other's internal states). Use `register_persona()` to register personas, `get_beliefs_about()` to retrieve ego-centric beliefs, and `process_observation_for_persona()` / `process_communication_for_persona()` to update beliefs.

-   **`PlannerState`**: Unified knowledge representation using subject-predicate-object triples with metadata for information asymmetry. Stores facts, beliefs, terrain information, shared objects, public events, and entity data. Supports different knowledge types (facts, beliefs, states) with metadata including confidence, timestamps, and accessibility levels. Use `set_predicate()` with metadata to store beliefs, `observe_*()` methods to access observable information.

**Integration with PlannerPlan**: Set `current_persona` and `belief_manager` on a `PlannerPlan` instance to enable belief-immersed planning. The planner will:

1. **Merge observable facts** from the unified state into the planning state automatically (terrain, shared objects, public events, entity positions, public capabilities)
2. **Apply ego-centric perspective** by merging the persona's beliefs about others into the state (beliefs are accessible as state predicates like `belief_{target_persona_id}_{belief_key}`)
3. **Update beliefs** automatically when actions execute, processing observations through the belief manager

This enables multi-agent planning where each persona plans from their own perspective while sharing observable facts.

### STN Solver Details

-   Uses Floyd-Warshall algorithm for all-pairs shortest paths
-   Validates temporal constraint consistency
-   Supports time points, constraints (min/max distance), and snapshots for backtracking
-   Constants: `STN_INFINITY` (INT64_MAX), `STN_NEG_INFINITY` (INT64_MIN + 1)
-   Methods: `add_time_point()`, `add_constraint()`, `check_consistency()`, `create_snapshot()`, `restore_snapshot()`

### VSIDS Activity Tracking

The planner implements VSIDS (Variable State Independent Decaying Sum) style activity tracking, following Chuffed's proven approach:

-   **Activity scores**: Methods are scored based on their involvement in conflicts and successes
-   **Activity bumping**: Methods get their activity increased by a fixed increment (`activity_var_inc`)
-   **Activity inflation**: Instead of decaying activities directly, we increase `activity_var_inc` by 5% periodically (like Chuffed)
-   **Method selection**: When multiple methods are available, the one with the highest activity score (scaled by 10x) is selected first

**Implementation details (matching Chuffed):**

-   Fixed increment: `activity[v] += var_inc` (not calculated reward)
-   Activity inflation: `var_inc *= 1.05` (instead of `activity *= 0.95`)
-   Duplicate prevention: Each method rewarded once per solve (like Chuffed's `seen[]` array)
-   Normalization: If `var_inc > 1e100`, scale down all activities and `var_inc` by `1e-100`

This optimization helps the planner learn from past planning attempts and prioritize methods that are more likely to succeed. The activity tracking is transparent to domain methods - they don't need to be aware of it.

**VSIDS is triggered for all failure types:**

-   **TASK failures**: When all methods for a task fail, `_bump_conflict_path_activities()` is called
-   **UNIGOAL failures**: When all methods for a unigoal fail, `_bump_conflict_path_activities()` is called
-   **MULTIGOAL failures**: When all methods for a multigoal fail, `_bump_conflict_path_activities()` is called
-   **ACTION failures**: When an action fails, `_bump_conflict_path_activities()` is called (bumps parent method)

**Testing VSIDS:**

-   Use `plan->get_method_activities()` to inspect activity scores after planning
-   Run tests in `tests/unit/test_vsids.h` to verify VSIDS behavior
-   Enable verbose logging (`plan->set_verbose(3)`) to see activity scores when methods are selected
-   **Activity scores RESET at the start of each `find_plan()` call** - VSIDS learns during backtracking within a single solve, not across solves
-   **Reset activity**: Call `plan->reset_vsids_activity()` to clear all activity scores when switching problem types or domains
-   **Fixed increment rewards**: Methods are rewarded with `activity_var_inc` (following Chuffed's proven approach), not calculated rewards
-   **Activity inflation**: `activity_var_inc` grows by 5% periodically, making newer rewards worth more (efficient decay mechanism)
-   **Conflict-based bumping**: Methods in conflict paths get bumped (like Chuffed's conflict analysis)
-   **Success-based bumping**: Successful methods also get bumped (optimization for plan quality)

**TLA+ Models**: See `tla/VSIDSActivityTracking.tla` for a formal model of the activity tracking logic. Note: The TLA+ model is simplified; the actual implementation uses floating-point activities with growing increment values.

## Missing Features from aria-planner

The Godot planner is focused on core HTN planning functionality and does not include several features present in the Elixir aria-planner implementation.

### Major Missing Features

1. **Persona System**: ✅ **IMPLEMENTED** - The Godot planner now includes `PlannerPersona` with unified persona models (human, AI, hybrid) and capability-based differentiation. Personas can be created using `PlannerPersona::create_human()`, `create_ai()`, `create_hybrid()`, or `create_basic()`. The planner supports both domain-centric and persona-centric planning.

2. **Belief-Immersed Architecture**: ✅ **IMPLEMENTED** - The Godot planner now implements ego-centric planning with allocentric execution, information asymmetry, and belief formation through `PlannerPersona`, `PlannerBeliefManager`, and unified `PlannerState`. Personas maintain ego-centric beliefs about other personas, and the belief manager enforces information asymmetry. The unified state represents shared ground truth observable by all personas using triples with metadata.

3. **Plan Lifecycle Management**: aria-planner has comprehensive plan lifecycle tracking with execution status ("planned", "executing", "completed", "failed"), plan persistence, and performance metrics. The Godot planner has `PlannerResult` with success status but no execution lifecycle management.

4. **Domain Registry**: aria-planner has a domain registry system (GenServer) for dynamic domain registration and discovery. The Godot planner requires manual domain creation and management.

5. **External Constraint Solvers**: aria-planner includes Chuffed solver integration and FlatZinc generation. The Godot planner has `PlannerSTNSolver` for temporal constraints but no external constraint solvers.

6. **Database Persistence**: aria-planner uses Ecto schemas for predicates and plan storage. The Godot planner uses in-memory `Dictionary` structures (appropriate for game engines).

7. **MCP Integration**: aria-planner has Model Context Protocol tool handlers for external integration. The Godot planner has no MCP integration.

8. **Execution State Management**: aria-planner separates execution state from planning state with `AriaCore.ExecutionState`. The Godot planner uses the same `Dictionary` for both planning and execution.

### Design Differences (Not Missing Features)

-   **ISO 8601 vs Microseconds**: aria-planner uses ISO 8601 strings for temporal constraints; Godot uses integer microseconds (intentional design difference for performance)
-   **Database vs In-Memory**: aria-planner uses Ecto/PostgreSQL; Godot uses in-memory dictionaries (appropriate for game engine use)

## Commit Guidelines

-   Test all changes before committing
-   Ensure IPyHOP compatibility tests pass
-   Update `test_all.h` if adding new test files
-   Follow Godot commit message conventions
-   Keep commits focused on single changes
-   Test all three planning methods (`find_plan`, `run_lazy_lookahead`, `run_lazy_refineahead`) when making core changes
