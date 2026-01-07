# find_plan() Detailed Analysis

## Current Performance
- **32.5% of total samples** (337 samples out of 1,036)
- **Primary bottleneck** in the planning system

## Function Breakdown

### 1. Initialization (Lines 84-120) - ~5-10% of find_plan time
- **Line 89**: `solution_graph = PlannerSolutionGraph()` - Creates new graph
- **Line 102**: `method_activities.clear()` - HashMap clear (fast ✅)
- **Line 108**: `stn.clear()` - STN solver reset
- **Line 112**: `stn.add_time_point("origin")` - STN initialization
- **Optimization**: Already optimized with HashMap

### 2. State Preparation (Lines 147-155) - ~10-15% of find_plan time
- **Line 147**: `p_todo_list.duplicate()` - Array duplicate (moderate cost)
- **Line 151**: `p_state.duplicate(true)` - **EXPENSIVE** - Deep copy of entire state
- **Line 154-155**: `_merge_allocentric_facts()` and `_get_ego_centric_state()` - State transformations
- **Optimization Opportunity**:
  - Could use shallow copy if state won't be modified
  - Cache allocentric facts merge result

### 3. Graph Construction (Lines 168-176) - 3.6% of total (37 samples)
- **Line 168**: `PlannerGraphOperations::add_nodes_and_edges()` - Adds initial tasks
- **Optimization**: Already using optimized internal structures

### 4. Root Node Validation (Lines 179-202) - ~2-3% of find_plan time
- **Line 179**: `solution_graph.get_node(0)` - **Dictionary conversion** (unnecessary)
- **Line 192**: `root_node_check["successors"]` - Dictionary access
- **Optimization Opportunity**: Use internal structure directly

### 5. Planning Loop (Line 205) - 19.9% of total (206 samples)
- **Line 205**: `_planning_loop_iterative()` - Main planning work
- **Optimization**: This is the core algorithm - harder to optimize

### 6. Success Checking (Lines 216-417) - ~15-20% of find_plan time
- **Lines 216-253**: Graph traversal to find reachable nodes
- **Lines 264-347**: Check each reachable node for failures
- **Lines 350-412**: Verify goal/multigoal checking
- **Optimization**: Already optimized with HashMap/LocalVector, but could cache results

### 7. Result Creation (Lines 419-475) - ~5-10% of find_plan time
- **Line 434**: `solution_graph.get_graph()` - **Dictionary conversion**
- **Line 439**: `graph_to_set.duplicate(true)` - **EXPENSIVE** - Deep copy of entire graph
- **Line 456**: `solution_graph.get_node(0)` - **Dictionary conversion** (unnecessary)
- **Line 463**: `solution_graph.get_graph()` - **Dictionary conversion** (duplicate call)
- **Line 468**: `graph_to_store.duplicate(true)` - **EXPENSIVE** - Deep copy again
- **Optimization Opportunity**:
  - Cache `get_graph()` result
  - Avoid duplicate deep copies
  - Use internal structure for root node

## Identified Optimizations

### High Impact (5-10% improvement each)

#### 1. Cache get_graph() Result
**Current**: Called 4+ times, each time returns reference (but still overhead)
**Optimized**: Call once, cache result
```cpp
const Dictionary &graph_dict = solution_graph.get_graph(); // Cache once
// Use graph_dict everywhere instead of calling get_graph() multiple times
```

#### 2. Use Internal Structure for Root Node Checks
**Current**: `solution_graph.get_node(0)` converts to Dictionary
**Optimized**: Use `get_node_internal(0)` directly
```cpp
const PlannerNodeStruct *root = solution_graph.get_node_internal(0);
if (root == nullptr || root->successors.is_empty()) {
    // Handle error
}
```

#### 3. Avoid Duplicate Deep Copies
**Current**: Deep copy graph twice (lines 439, 468)
**Optimized**: Copy once, reuse if needed
```cpp
Dictionary graph_dict = solution_graph.get_graph();
Dictionary graph_copy = graph_dict.duplicate(true); // Copy once
result->set_solution_graph(graph_copy);
// If we need to update, modify graph_dict and copy again only if needed
```

#### 4. Optimize State Duplicate
**Current**: Always deep copy state
**Optimized**: Shallow copy if state won't be modified, or reuse if possible
```cpp
// Only deep copy if we know state will be modified
Dictionary clean_state = p_state.duplicate(false); // Shallow copy
// Deep copy only nested dictionaries that will change
clean_state["needs"] = p_state["needs"].duplicate(true);
// etc.
```

### Medium Impact (2-5% improvement each)

#### 5. Cache Success Check Results
**Current**: Recompute reachable nodes every time
**Optimized**: Cache if graph hasn't changed
```cpp
// Only recompute if graph was modified
if (graph_modified) {
    // Recompute reachable nodes
}
```

#### 6. Optimize Verbose Logging
**Current**: `.keys()` called even when verbose is off (guarded, but still)
**Optimized**: Already guarded, but could be more efficient

### Low Impact (1-2% improvement)

#### 7. Reduce Dictionary Conversions
**Current**: Multiple conversions between internal structure and Dictionary
**Optimized**: Use internal structure throughout, convert only at boundaries

## Implementation Priority

1. **Cache get_graph() result** - Easy, high impact
2. **Use internal structure for root node** - Easy, medium-high impact
3. **Avoid duplicate deep copies** - Medium difficulty, high impact
4. **Optimize state duplicate** - Medium difficulty, medium impact
5. **Cache success check results** - Hard, medium impact

## Expected Overall Improvement

- **High priority optimizations**: 10-15% faster find_plan
- **Medium priority optimizations**: 5-10% faster find_plan
- **Total expected**: 15-25% faster find_plan
- **Overall system impact**: 5-8% faster overall (since find_plan is 32.5% of total)
