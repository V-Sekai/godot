# HTN Goal Temporal Planner Review

## Overview

The planner uses **HTN (Hierarchical Task Network)** planning with **STN (Simple Temporal Network)** for temporal constraint validation. This is a sophisticated combination that supports both hierarchical decomposition and temporal reasoning.

## Current Architecture

### 1. HTN Planning (Core)

-   **Hierarchical decomposition**: Tasks decompose into subtasks/actions
-   **Method selection**: VSIDS activity-based method selection
-   **Backtracking**: Explicit backtracking with solution graph
-   **Partial replanning**: `replan()` method for incremental planning

### 2. STN (Simple Temporal Network) Solver

-   **Algorithm**: Floyd-Warshall for all-pairs shortest paths
-   **Purpose**: Validates temporal constraint consistency
-   **Constraints**: Min/max distance between time points
-   **Time points**: Actions have start/end time points
-   **Consistency checking**: Detects negative cycles (inconsistent constraints)

### 3. Temporal Constraints

-   **Absolute times**: Microseconds since Unix epoch
-   **Relative constraints**: Duration, start_time, end_time
-   **Validation**: STN ensures all constraints are consistent
-   **Anchoring**: Origin time point anchors to plan start time

## Partial Order vs Total Order Planning

### What is Total Order Planning?

**Total Order**: Actions must be executed in a strict, linear sequence.

**Example**:

```
Action 1 → Action 2 → Action 3 → Action 4
```

-   Action 2 cannot start until Action 1 completes
-   Action 3 cannot start until Action 2 completes
-   Strict sequential ordering

**Characteristics**:

-   Simple to execute
-   Easy to understand
-   Less flexible
-   Cannot express parallel actions

### What is Partial Order Planning?

**Partial Order**: Actions can be executed in any order that respects constraints.

**Example**:

```
Action 1 ──┐
           ├──→ Action 4
Action 2 ──┤
           │
Action 3 ──┘
```

-   Actions 1, 2, 3 can execute in parallel or any order
-   Action 4 must wait for all three to complete
-   Flexible ordering

**Characteristics**:

-   More flexible
-   Can express parallelism
-   More complex to execute
-   Requires constraint satisfaction

## Current Implementation: **TOTAL ORDER**

### Evidence

1. **`extract_plan()` Implementation**:

    - Uses DFS (Depth-First Search) traversal
    - Visits nodes in a fixed order
    - Returns actions as a linear `Array`
    - No parallel execution support

2. **Action Execution**:

    - Actions executed sequentially in `extract_plan()`
    - State updated after each action
    - No concurrent action execution

3. **Solution Graph Structure**:
    - Tree structure (parent-child relationships)
    - Sequential decomposition
    - No explicit parallel branches

### Code Evidence

From `graph_operations.cpp`:

```cpp
// Extract plan using DFS - sequential order
Array PlannerGraphOperations::extract_plan(...) {
    // DFS traversal
    // Actions extracted in DFS order
    // Returns linear Array of actions
}
```

## STN: Enabling Partial Order (But Not Used)

### What STN Provides

-   **Temporal constraints**: Actions can have start/end times
-   **Constraint validation**: Ensures temporal constraints are consistent
-   **Flexible timing**: Actions don't need strict sequential ordering

### What's Missing

-   **No parallel execution**: Actions still executed sequentially
-   **No constraint-based ordering**: Plan extraction ignores temporal constraints
-   **STN only validates**: Doesn't change execution order

## Optimization Opportunities

### 1. **Partial Order Execution** ⭐ HIGH IMPACT

**Current**: All actions executed sequentially
**Opportunity**: Execute actions in parallel when temporal constraints allow

**Implementation**:

```cpp
// Group actions by temporal constraints
// Actions with overlapping time windows can execute in parallel
// Use STN to determine which actions can run concurrently
```

**Benefits**:

-   Faster plan execution
-   Better resource utilization
-   More realistic simulation (multiple actors can act simultaneously)

**Challenges**:

-   State updates need synchronization
-   Need to handle action failures in parallel context
-   More complex execution model

### 2. **STN-Based Plan Extraction** ⭐ MEDIUM IMPACT

**Current**: Plan extracted using DFS (ignores temporal constraints)
**Opportunity**: Extract plan based on temporal ordering (earliest start time first)

**Implementation**:

```cpp
// Sort actions by earliest start time (from STN)
// Extract plan in temporal order, not DFS order
// Respect temporal constraints during extraction
```

**Benefits**:

-   More efficient plan execution
-   Better temporal constraint utilization
-   Actions scheduled optimally

**Challenges**:

-   Need to query STN for earliest/latest times
-   May break some assumptions about plan ordering

### 3. **Incremental STN Updates** ⭐ LOW IMPACT

**Current**: STN rebuilt/revalidated on every constraint addition
**Opportunity**: Incremental updates using incremental Floyd-Warshall

**Implementation**:

-   Use incremental algorithms (e.g., incremental shortest paths)
-   Only recompute affected parts of distance matrix
-   Cache STN consistency state

**Benefits**:

-   Faster STN constraint checking
-   Reduced overhead during planning

**Challenges**:

-   More complex implementation
-   Need to maintain incremental state

### 4. **STN Constraint Tightening** ⭐ LOW IMPACT

**Current**: Constraints added independently
**Opportunity**: Tighten constraints based on action durations and dependencies

**Implementation**:

-   When action A must complete before action B
-   Tighten constraint: `B.start_time >= A.end_time`
-   Use action durations to infer tighter constraints

**Benefits**:

-   More precise temporal reasoning
-   Better constraint propagation
-   Fewer invalid plans generated

## Recommendations

### Priority 1: **Partial Order Execution** (If Needed)

**When to implement**:

-   If you need parallel action execution
-   If multiple actors need to act simultaneously
-   If plan execution time is a bottleneck

**Effort**: High (requires significant refactoring)
**Impact**: High (enables true parallelism)

### Priority 2: **STN-Based Plan Extraction** (Medium Priority)

**When to implement**:

-   If temporal constraints are important
-   If you want optimal action scheduling
-   If plan execution order matters

**Effort**: Medium (modify extract_plan to use STN)
**Impact**: Medium (better temporal utilization)

### Priority 3: **Incremental STN Updates** (Low Priority)

**When to implement**:

-   If STN operations are a bottleneck
-   If planning with many temporal constraints
-   If performance profiling shows STN overhead

**Effort**: High (complex incremental algorithms)
**Impact**: Low-Medium (performance improvement)

## Current Status

### ✅ What Works Well

1. **STN validation**: Correctly validates temporal constraints
2. **Temporal metadata**: Supports start_time, end_time, duration
3. **Constraint consistency**: Detects inconsistent constraints
4. **Backtracking**: STN snapshots support backtracking

### ⚠️ Limitations

1. **Total order execution**: Actions executed sequentially
2. **STN underutilized**: Only validates, doesn't optimize ordering
3. **No parallelism**: Cannot execute actions in parallel
4. **DFS-based extraction**: Ignores temporal constraints during extraction

## Conclusion

The planner is **currently total order** but has the infrastructure (STN) to support **partial order planning**. The STN solver correctly validates temporal constraints, but the plan extraction and execution are sequential.

**For 300 actors**: The current total order approach is fine if:

-   Each actor plans independently
-   Actions execute sequentially per actor
-   No need for inter-actor parallel execution

**Consider partial order if**:

-   You need actions to execute in parallel
-   Temporal constraints are critical
-   Plan execution time is a bottleneck

The STN infrastructure is solid and could be extended to support partial order execution if needed.
