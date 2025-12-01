# Nostradamus Distributor Analysis

## Key Insights from the Article

The [Nostradamus Distributor article](http://www.emulators.com/docs/nx25_nostradamus.htm) discusses optimizing interpreter dispatch loops to reduce branch mispredictions. While our issue is a crash (not performance), there are relevant patterns:

### 1. **Indirect Call Mispredictions**
- Indirect calls/jumps in tight loops cause branch mispredictions
- Each misprediction costs 10-40+ cycles depending on CPU
- Solution: Use software pipelining and explicit spacing

### 2. **Recursive to Iterative Conversion**
- Recursive patterns can cause stack overflow
- Iterative patterns with explicit stacks are safer
- ✅ **We've already done this for `do_get_descendants()`**

### 3. **Dispatch Loop Optimization**
- Tight loops with indirect calls are expensive
- Pattern: Reduce indirect calls, use direct calls when possible

## Application to Our Code

### Current Issue: `extract_solution_plan()` Loop

Our `extract_solution_plan()` has a potential performance issue:

```cpp
for (int i = successors.size() - 1; i >= 0; i--) {
    int succ_id = successors[i];
    if (!visited.has(succ_id)) {
        int parent_of_succ = find_predecessor(p_graph, succ_id);  // EXPENSIVE!
        // ...
    }
}
```

**Problem**: `find_predecessor()` is called **inside the loop** for each successor. This function:
- Iterates over **all nodes** in the graph (O(n))
- Called for each successor (potentially O(m) times)
- Total complexity: **O(n*m)** which could be very slow with 498+ nodes

### Optimization: Precompute Parent Map

Instead of calling `find_predecessor()` repeatedly, we could:

1. **Build a parent map once** at the start
2. **Lookup in O(1)** instead of O(n) per call

```cpp
// Build parent map once
Dictionary parent_map;  // child_id -> parent_id
Dictionary graph_dict = p_graph.get_graph();
Array keys = graph_dict.keys();
for (int i = 0; i < keys.size(); i++) {
    int parent_id = keys[i];
    Dictionary parent_node = p_graph.get_node(parent_id);
    if (!parent_node.is_empty() && parent_node.has("successors")) {
        TypedArray<int> successors = parent_node["successors"];
        for (int j = 0; j < successors.size(); j++) {
            parent_map[successors[j]] = parent_id;
        }
    }
}

// Then use O(1) lookup instead of O(n) search
int parent_of_succ = parent_map.get(succ_id, -1);
```

### But Wait - Is This Causing the Crash?

The crash is **SIGSEGV (segmentation fault)**, not a performance issue. However:

1. **Slow operations** can expose timing-related bugs
2. **Many iterations** might cause memory issues
3. **Graph corruption** might be more likely with many operations

## Recommendations

1. ✅ **Already done**: Convert `do_get_descendants()` to iterative
2. **Optimize `find_predecessor()` calls**: Precompute parent map
3. **Add cycle detection**: Ensure `visited` set prevents infinite loops
4. **Add depth limit**: Safety check for extremely deep graphs

## Next Steps

1. Optimize `extract_solution_plan()` to use parent map
2. Add explicit cycle detection
3. Add depth limit as safety measure
4. Test if optimization fixes crash (might be timing-related)

