# Goal Task Planner Specification

## 1. Introduction

The Goal Task Planner is a Hierarchical Task Network (HTN) planning module for the Godot Engine, heavily inspired by [IPyHOP](https://github.com/YashBansod/IPyHOP). It is designed to generate sequences of **Commands** (gameplay actions) to achieve a set of tasks or goals, given an initial state.

Undefined behavior in traditional planners (where actions are assumed to succeed) is replaced here with a **Situated Planning** approach:

-   **Commands** are fallible.
-   Execution failure triggers **Replanning** or **Backtracking**.
-   The system supports **Belief-Immersed Planning** (Personas) and **Temporal Constraints**.

## 2. Core Concepts

### 2.1. State Representation

The state of the world is represented as a Godot `Dictionary` managed by the **`PlannerState`** resource.

-   **Structure**: Unified Knowledge Representation using **Predicate-Subject-Object (PSO) Triples**.
    -   `Predicate` (String): The attribute or relationship identifier (e.g., "is_open").
    -   `Subject` (String): The entity being described (e.g., "door_1").
    -   `Object` (Variant): The value or target of the relationship (e.g., `true`, `Vector3(1,0,0)`).
    -   `Metadata` (Dictionary): Context info like `confidence`, `timestamp`, `source` (Persona ID), and `type` ("fact", "belief").
-   **Semantics**: The `PlannerState` API provides semantic accessors (`get_predicate`, `set_predicate`) abstracting the internal storage.

### 2.2. Domain Definition

The **`PlannerDomain`** defines the logic of the world using three main constructs. Methods are registered via `add_actions`, `add_task_methods`, etc.

#### A. Commands

To distinguish from engine-level "Actions" (animation/physics), we employ the term **Commands** to denote atomic plan operators that _attempt_ to transition the state.

-   **Non-deterministic Execution**: Unlike traditional HTN operators, Commands may fail during execution (runtime) or planning (preconditions).
-   **Signature**: `(state, args...) -> new_state | false | Variant()`
-   **Behavior**:
    -   Returns `new_state` on success.
    -   Returns `false` or `nil` on failure.
    -   **Blacklisting**: Failed operators are blacklisted during replanning to prevent infinite loops.

#### B. Compound Tasks

High-level objectives that cannot be executed directly but must be decomposed (e.g., `perform_heist`, `travel_to`).

#### C. Methods

Strategies to decompose a **Compound Task** into a list of **Subtasks** (Commands or Compound Tasks).

-   **Signature**: `(state, args...) -> [subtask1, subtask2, ...] | false`
-   **Ordering**: A Task can have multiple Methods. They are prioritized based on a **Scoring Heuristic** (VSIDS-like) rather than just declaration order.

### 2.3. Goals (Multigoals)

A goal represents a desired state or task list.

-   **Multigoal**: A sequence of tasks: `[task1, task2, ...]`.
-   **Goal Structure**: `[task_name, arg1, arg2, ...]`.

## 3. Planning Algorithm

### 3.1. Iterative Planning Loop

The planner uses an iterative **Depth-First Search (DFS)** with backtracking, protected by resource limits.

1.  **Input**: Initial State, Task Agenda (`todo_list`), Domain.
2.  **Constraints**:
    -   `max_depth`: Recursion limit (default: 10).
    -   `max_iterations`: Total step limit (default: 50000).
    -   `max_stack_size`: Memory safeguard.
3.  **Process**:
    -   Pop `current_task` from Task Agenda.
    -   **If Command**:
        -   Check preconditions.
        -   Validate against **Blacklist** (if re-entered from failure).
        -   Update State.
    -   **If Compound Task**:
        -   Select best **Method** (see 3.2).
        -   Decompose into subtasks.
        -   Push subtasks to Task Agenda.
4.  **Termination**:
    -   Success: Task Agenda empty.
    -   Failure: No applicable methods/commands found (Backtrack).

### 3.2. Method Selection (VSIDS-like)

To optimize search, methods are scored dynamically, inspired by SAT solvers (Chuffed/VSIDS):

-   **Activity**: Each method tracks an `activity` score.
-   **Decay**: Activity decays over time to prioritize recent effective strategies.
-   **Rewards**:
    -   Successful application increases activity.
    -   Participating in a successful plan yields a large reward.
-   **Selection**: Methods are sorted by `score = activity + history_bonus`.

### 3.3. Replanning & Failure Handling

Compatible with IPyHOP's "Replanning from the Middle":

-   **Solution Graph**: Maintains a tree of `(Task -> Method -> Subtasks)`.
-   **Failure**: When a node fails:
    1.  Mark node as **Failed/Blacklisted**.
    2.  Resume planning from the nearest open ancestor.
    3.  Exclude the path that led to failure.

## 4. Advanced Features

### 4.1. Temporal Planning (STN)

The planner integrates a **Simple Temporal Network (STN)** solver:

-   **Time Ranges**: Tracking `start_time` and `duration` for tasks.
-   **Constraints**: Preconditions can specify temporal relations (e.g., `task_A` must end before `task_B` starts).
-   **Consistency**: A Floyd-Warshall solver runs incrementally to ensure the schedule is valid.
-   **Backtracking**: STN state is snapshotted; backtracking restores the previous matrix state.

### 4.2. Belief-Immersed Architecture

Entities in the game have subjective views of the world ("Beliefs") distinct from the objective simulation state ("Facts").

-   **Personas (`PlannerPersona`)**: Defined agents with restricted capabilities and unique belief sets.
    -   _Human Capabilities_: `movable`, `inventory`, `craft`, `mine`, `build`, `interact`.
    -   _AI Capabilities_: `movable`, `compute`, `optimize`, `predict`, `learn`, `navigate`.
    -   _Hybrid_: Combines both sets (e.g., cyborgs).
-   **Belief Semantics**:
    -   **Confidence**: A normalized float (0.0 - 1.0) indicating certainty in a belief.
    -   **Timestamp**: Last time the belief was updated (game tick).
    -   **Information Asymmetry**: Agents plan based solely on their beliefs. If an agent believes a door is open, it might generate a plan to walk through it, even if the door is actually locked in the world state.
-   **Belief Manager**: The central coordinator that enforces asymmetry. It merges "Ego-Centric" beliefs with "Allocentric" facts only when resolving specific sensory commands.
-   **Capability Filtering**: Planning methods (HTN tasks) filter themselves based on the `current_persona`'s capabilities. For example, a `mine_resources` method is pruned from the search tree if the persona lacks the `mine` capability.

## 5. Interfaces

### 5.1. PlannerPlan

-   `find_plan(state, todo_list, domain)`: Main entry point.
-   `run_lazy_lookahead(state, todo_list, max_tries)`: Lazy execution where commands are executed as they are planned.
-   `attach_metadata(item, temporal_constraints, entity_constraints)`: Attach metadata (STN/Capabilities) to a task/command.
-   `blacklist_command(cmd)`: Block a specific command signature.
-   `replan(result, state, fail_node_id)`: Resume from a failure context using a previous `PlannerResult`.

### 5.2. PlannerResult

-   `success`: `bool`
-   `plan`: `Array<Command>`
-   `solution_graph`: `PlannerSolutionGraph` (Derivation Tree)
-   `time_allocations`: STN results.
-   `explain_plan()`: Returns a dictionary of decision logic and VSIDS scores.
-   `find_failed_nodes()`: Returns a list of failed nodes to feed into `replan`.

### 5.3. PlannerState (Resource)

-   `get_predicate(subject, predicate)`: Retrieve value from triple store.
-   `set_predicate(subject, predicate, value, metadata)`: atomic update of belief/fact.
-   `get_triples_as_array()`: Export state for debugging/serialization.

### 5.4. PlannerDomain (Resource)

-   `add_actions(methods)`: Register primitive commands.
-   `add_task_methods(task_name, methods)`: Register decomposition strategies.
-   `add_multigoal_methods(methods)`: Register goal-set handling strategies.

## 6. Verification

-   **Minizinc**: External validation for logical measuring.
