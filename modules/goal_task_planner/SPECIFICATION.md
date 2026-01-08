# Goal Task Planner Specification

## 1. Introduction

The Goal Task Planner is a Hierarchical Task Network (HTN) planning module for the Godot Engine. It is designed to generate sequences of commands (plans) to achieve a set of tasks or goals, given an initial state. It supports backtracking, temporal constraints, and entity-persona capabilities, maintaining compatibility with IPyHOP where applicable.

## 2. Core Concepts

### 2.1. State Representation

The state of the world is represented as a Godot `Dictionary`.

-   **Keys**: String identifiers for objects or concepts.
-   **Values**: Can be primitives (String, int, bool) or complex types (Dictionary, Array).
-   **Semantics**: The state represents the "current belief" of the planner about the world.

### 2.2. Domain Definition

The domain defines the "physics" and "logic" of the world.

-   **Commands (Primitive Tasks)**: Executable operations that modify the state.
    -   Also known as **Actions** in traditional HTN literature.
    -   In this game context, they are **Commands** that execute game logic (e.g., `move_to`, `open_door`).
    -   Signature: `(state, args...) -> new_state | false`
    -   Must return `false` or `Variant()` (nil) on failure.
    -   Must return a new `Dictionary` (state) on success.
-   **Tasks (Compound Tasks)**: High-level goals that need decomposition.
-   **Methods**: Strategies to decompose Tasks into Subtasks (Commands or other Tasks).
    -   Signature: `(state, args...) -> [subtask1, subtask2, ...] | false`
    -   A Task can have multiple Methods; they are tried in order.

### 2.3. Goals (Multigoals)

A goal (or task list) is an `Array` of tasks to be achieved.

-   **Multigoal**: An array of goals to be achieved in sequence. `[task1, task2, ...]`
-   **Goal Structure**: `[task_name, arg1, arg2, ...]`

## 3. Planning Algorithm

### 3.1. Main Loop

The planner uses a **Depth-First Search (DFS)** with backtracking.

1. **Input**: Initial State, Todo List (Multigoal).
2. **Process**:
    - Pop the first task from the Todo List.
    - If it's a **Command**:
        - Check preconditions (execute command logic).
        - If successful, update State, record Command, proceed to next task.
        - If failed, **Backtrack**.
    - If it's a **Task**:
        - Find matching **Methods** for the task.
        - For each Method (in define order):
            - Try to decompose Task into Subtasks.
            - Prepend Subtasks to the Todo List.
            - Recurse.
            - If recursion fails, try next Method.
        - If all Methods fail, **Backtrack**.
3. **Termination**:
    -   **Success**: Todo List is empty. Return Plan.
    -   **Failure**: All search paths exhausted. Return Failure.

### 3.2. Backtracking

When a path fails (Command precondition failed, or no Methods applicable), the planner must:

-   Revert the State to the snapshot before the decision.
-   Restore the Todo List.
-   Try the next alternative (Next Method).

### 3.3. Solution Graph

The planning process produces a `PlannerSolutionGraph` that records the derivation tree:

-   **Nodes**: Root, Tasks, Commands, Methods.
-   **Edges**: Decomposition (Task -> Method -> Subtasks).
-   **Status**: Open, Closed, Failed.
    The graph is useful for debugging, visualization, and incremental replanning.

## 4. Advanced Features

### 4.1. Temporal Planning (STN)

-   **Simple Temporal Network (STN)**: Manages temporal constraints (Start Time, End Time, Duration).
-   **Consistency**: The planner validates that all temporal constraints are satisfiable using a Floyd-Warshall solver.
-   **Backtracking**: STN state (constraints) must be snapshotted and restored during backtracking.

### 4.2. Entity-Persona Capabilities

-   **Persona**: Represents an agent (AI, Human).
-   **Beliefs**: Each persona has its own belief of the world state.
-   **Capabilities**: Personas have specific capabilities. The planner can filter methods/commands based on the capabilities of the executing persona.

## 5. Interfaces

### 5.1. PlannerPlan

-   `find_plan(state, todo_list, domain)`: Returns `PlannerResult`.
-   `yield_plan(...)`: Step-wise execution (coroutine style).

### 5.2. PlannerResult

-   `success`: Boolean.
-   `final_state`: Dictionary.
-   `plan`: Array of commands.
-   `solution_graph`: The derivation graph.

## 6. Verification

-   **Minizinc**: Used to verify strict logical correctness against an external constraint solver for supported domains.
