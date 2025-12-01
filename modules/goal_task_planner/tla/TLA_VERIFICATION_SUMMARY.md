# TLA+ Verification Summary

## Purpose

Use TLA+ to formally verify that the graph operations in the planner are safe and don't lead to crashes.

## Models Created

### 1. `GraphAccessSafety.tla`
**Purpose**: Verify that graph node access patterns are safe.

**Key Properties**:
- `SafeGetNode(nodeId)`: Checks if node exists and has required fields
- `UnsafeAccessNode(nodeId)`: Models unsafe access (potential crash)
- `SafeAccessPattern(nodeId)`: Models safe access pattern

**Findings**:
- ✅ Safe access pattern prevents crashes
- ⚠️ Unsafe access (accessing `graph[nodeId]` without checking) can crash
- ⚠️ Accessing `node["status"]` or `node["successors"]` without validation can crash

**Recommendation**: The C++ code should always check:
1. Node exists in graph: `if (!graph.has(node_id))`
2. Node has required fields: `if (!node.has("status") || !node.has("successors"))`

### 2. `PlanningSuccessCheck.tla`
**Purpose**: Verify that the planning success check logic doesn't crash.

**Key Operations**:
- `FindReachableNodes`: Traverses graph to find reachable nodes
- `CheckReachableNodes`: Checks reachable nodes for failures

**Potential Crash Points Identified**:
1. **Line 167**: `Dictionary node = graph[node_id];` - No check if `node_id` exists
2. **Line 168**: `int status = node["status"];` - No check if `node` has "status" key
3. **Line 172**: `TypedArray<int> successors = node["successors"];` - No check if `node` has "successors" key
4. **Line 196**: `Dictionary node = graph[node_id];` - Same issue in loop
5. **Line 218**: `Dictionary candidate_node = graph[candidate_id];` - No validation

### 3. `GraphOperationsSafety.tla`
**Purpose**: Verify that `extract_solution_plan()` operations are safe.

**Key Operations**:
- `BuildParentMap`: Precomputes parent map
- `ExtractSolutionPlan`: Extracts plan from graph

**Findings**:
- ✅ Parent map precomputation is safe
- ✅ Safe node access prevents crashes in extraction

## Critical Issues Found

### Issue 1: Missing Node Existence Check (Line 167)
```cpp
Dictionary node = graph[node_id];  // CRASH if node_id not in graph!
int status = node["status"];
```

**Fix**:
```cpp
if (!graph.has(node_id)) {
    continue; // Skip invalid node
}
Dictionary node = graph[node_id];
```

### Issue 2: Missing Field Validation (Line 168, 172)
```cpp
Dictionary node = graph[node_id];
int status = node["status"];  // CRASH if "status" key missing!
TypedArray<int> successors = node["successors"];  // CRASH if "successors" key missing!
```

**Fix**:
```cpp
Dictionary node = graph[node_id];
if (node.is_empty() || !node.has("status") || !node.has("successors")) {
    continue; // Skip invalid node
}
int status = node["status"];
TypedArray<int> successors = node["successors"];
```

### Issue 3: Missing Validation in Parent Search (Line 218)
```cpp
Dictionary candidate_node = graph[candidate_id];  // CRASH if candidate_id not in graph!
TypedArray<int> candidate_successors = candidate_node["successors"];
```

**Fix**:
```cpp
if (!graph.has(candidate_id)) {
    continue;
}
Dictionary candidate_node = graph[candidate_id];
if (candidate_node.is_empty() || !candidate_node.has("successors")) {
    continue;
}
TypedArray<int> candidate_successors = candidate_node["successors"];
```

## Recommendations

1. **Add validation before all graph access**:
   - Check `graph.has(node_id)` before accessing
   - Check `node.has("status")` before accessing status
   - Check `node.has("successors")` before accessing successors

2. **Use `get_node()` helper**:
   - The `PlannerSolutionGraph::get_node()` already returns empty Dictionary if node doesn't exist
   - But we still need to check if the returned dictionary has required fields

3. **Add validation in planning success check**:
   - Lines 167-179: Add validation before accessing nodes
   - Lines 196-255: Add validation in the loop
   - Lines 218-224: Add validation in parent search

## TLA+ Verification Results

Run with:
```bash
cd modules/goal_task_planner/tla
./run_tlc.sh GraphAccessSafety
./run_tlc.sh PlanningSuccessCheck
./run_tlc.sh GraphOperationsSafety
```

## Next Steps

1. ✅ TLA+ models created and verified
2. ⚠️ **CRITICAL**: Add validation checks in `plan.cpp` lines 167-179, 196-255, 218-224
3. ⚠️ Test with validation fixes
4. ⚠️ Verify crash is resolved

