# Stack Overflow Analysis

## Current Implementation

### `extract_solution_plan()` - Already Iterative ✅
- Uses `while` loop with `to_visit` stack
- No recursion, so no stack overflow risk from this function

### `find_predecessor()` - Iterative but Expensive ⚠️
- Called **inside the loop** for each successor
- Iterates over **all nodes** in graph (O(n) per call)
- With 498+ nodes and many successors, this is O(n*m) complexity
- **Not recursive**, but very expensive

### `do_get_descendants()` - RECURSIVE! ❌
- **This is the problem!**
- Recursively calls itself for each successor
- If graph is deep (498+ nodes), this could cause stack overflow
- Called from `remove_descendants()` which is called during backtracking

## Stack Overflow Risk

### High Risk: `do_get_descendants()`
```cpp
void PlannerGraphOperations::do_get_descendants(...) {
    for (int i = 0; i < p_current_nodes.size(); i++) {
        // ...
        if (successors.size() > 0) {
            do_get_descendants(p_graph, successors, p_visited, p_result);  // RECURSIVE!
        }
    }
}
```

**Problem**: With a deep graph (498+ nodes), this recursive call could exceed stack limits.

### Medium Risk: `find_predecessor()` in loop
- Called for each successor in `extract_solution_plan()`
- With many successors, this creates many iterations
- Not recursive, but could be slow

## Solution: Convert to Iterative

### Fix `do_get_descendants()` - Convert to Iterative

**Current (Recursive)**:
```cpp
void do_get_descendants(..., TypedArray<int> p_current_nodes, ...) {
    for (int i = 0; i < p_current_nodes.size(); i++) {
        // ...
        if (successors.size() > 0) {
            do_get_descendants(p_graph, successors, p_visited, p_result);  // RECURSIVE
        }
    }
}
```

**Fixed (Iterative)**:
```cpp
void do_get_descendants(..., TypedArray<int> p_current_nodes, ...) {
    TypedArray<int> to_process = p_current_nodes;  // Use stack instead of recursion
    
    while (!to_process.is_empty()) {
        int node_id = to_process.pop_back();
        if (p_visited.has(node_id)) {
            continue;
        }
        p_visited.push_back(node_id);
        p_result.push_back(node_id);
        
        Dictionary node = p_graph.get_node(node_id);
        if (node.is_empty() || !node.has("successors")) {
            continue;
        }
        TypedArray<int> successors = node["successors"];
        // Add successors to stack instead of recursing
        for (int i = 0; i < successors.size(); i++) {
            if (!p_visited.has(successors[i])) {
                to_process.push_back(successors[i]);
            }
        }
    }
}
```

### Optimize `find_predecessor()` - Cache or Precompute

Instead of calling `find_predecessor()` for each successor, we could:
1. Build a reverse index (parent -> children) once
2. Or skip the check if not needed (it's just a validation)

## Test Case Analysis

- **498+ iterations** in planning loop
- **Many nodes** in solution graph
- **Deep graph structure** from backtracking

This suggests the graph could be very deep, making `do_get_descendants()` recursion dangerous.

## Recommendation

1. **Convert `do_get_descendants()` to iterative** (highest priority)
2. **Consider optimizing `find_predecessor()` calls** in `extract_solution_plan()`
3. **Add depth limit** to `do_get_descendants()` as safety measure

