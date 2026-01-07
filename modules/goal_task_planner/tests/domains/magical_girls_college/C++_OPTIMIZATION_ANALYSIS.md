# C++ Optimization Analysis: HashMap and LocalVector

## Current Data Structures

### Already Optimized ✅
1. **Planning Stack**: Uses `LocalVector<PlanningFrame>` (line 121 in plan.h)
   - Stack-allocated, no heap allocations
   - Fast push/pop operations

### Using Godot Variant Types (Can Be Optimized)
1. **Solution Graph**: `Dictionary graph` (solution_graph.h:65)
   - Currently: Godot's `Dictionary` (Variant-based, slower)
   - Potential: `HashMap<int, NodeStruct>` for internal operations
   - **Constraint**: Must remain `Dictionary` for GDScript API compatibility

2. **Method Activities**: `Dictionary method_activities` (plan.h:75)
   - Currently: `Dictionary` (String -> double)
   - Potential: `HashMap<String, double>` or `HashMap<int, double>` (if method IDs can be hashed)

3. **Successors Arrays**: `TypedArray<int>` (solution_graph.h:76, 103)
   - Currently: `TypedArray<int>` (Variant-based)
   - Potential: `LocalVector<int>` for internal operations

4. **Planning Loop Collections**: Various `Array` and `TypedArray<int>` (plan.cpp)
   - `Array failed_nodes` (line 217)
   - `Array open_nodes` (line 218)
   - `TypedArray<int> reachable_nodes` (line 219)
   - `TypedArray<int> to_visit` (line 219)
   - `TypedArray<int> visited` (line 222)
   - Potential: `LocalVector<int>` for all of these

5. **Graph Traversal**: `Array graph_keys` (plan.cpp:216, 299, etc.)
   - Currently: `graph.keys()` returns `Array`
   - Potential: Use iterator or `HashMap` keys directly

## Optimization Opportunities

### High Impact (Internal Operations)

#### 1. Solution Graph Internal Structure
**Current**: `Dictionary graph` (Variant-based, slower lookups)
```cpp
// solution_graph.h:65
Dictionary graph;  // O(log n) lookups, Variant overhead
```

**Optimized**: Internal HashMap with Dictionary conversion only for API
```cpp
// Internal structure (not exposed to GDScript)
HashMap<int, NodeStruct> graph_internal;

// NodeStruct - POD struct for fast access
struct NodeStruct {
    PlannerNodeType type;
    PlannerNodeStatus status;
    Variant info;
    LocalVector<int> successors;  // Fast vector
    Dictionary state;
    // ... other fields
};

// Only convert to Dictionary when needed for GDScript API
Dictionary get_graph() const {
    Dictionary result;
    for (const KeyValue<int, NodeStruct> &kv : graph_internal) {
        result[kv.key] = node_struct_to_dictionary(kv.value);
    }
    return result;
}
```

**Expected Impact**: 30-50% faster graph lookups and updates

#### 2. Successors as LocalVector
**Current**: `TypedArray<int>` (Variant overhead)
```cpp
// solution_graph.h:137
TypedArray<int> successors = parent["successors"];  // Variant conversion
successors.push_back(p_child_id);  // Variant overhead
```

**Optimized**: `LocalVector<int>` internally
```cpp
// In NodeStruct
LocalVector<int> successors;  // Fast, stack-allocated

// Only convert to TypedArray when needed for Dictionary storage
void add_successor(int p_parent_id, int p_child_id) {
    NodeStruct &node = graph_internal[p_parent_id];
    node.successors.push_back(p_child_id);
}
```

**Expected Impact**: 20-30% faster successor operations

#### 3. Planning Loop Collections
**Current**: `Array` and `TypedArray<int>` (plan.cpp:217-222)
```cpp
Array failed_nodes;
Array open_nodes;
TypedArray<int> reachable_nodes;
TypedArray<int> to_visit;
TypedArray<int> visited;
```

**Optimized**: `LocalVector<int>` for all
```cpp
LocalVector<int> failed_nodes;
LocalVector<int> open_nodes;
LocalVector<int> reachable_nodes;
LocalVector<int> to_visit;
HashSet<int> visited;  // O(1) lookups instead of O(n)
```

**Expected Impact**: 40-60% faster planning loop iterations

#### 4. Method Activities HashMap
**Current**: `Dictionary method_activities` (plan.h:75)
```cpp
Dictionary method_activities;  // String -> double, Variant overhead
```

**Optimized**: `HashMap<String, double>` or `HashMap<int, double>`
```cpp
// Option 1: String keys (if method names are needed)
HashMap<String, double> method_activities;

// Option 2: Integer method IDs (faster, if we can hash method IDs)
HashMap<int, double> method_activities;
```

**Expected Impact**: 20-30% faster activity lookups/updates

### Medium Impact (Graph Traversal)

#### 5. Graph Keys Iteration
**Current**: `Array graph_keys = graph.keys()` (plan.cpp:216)
```cpp
Array graph_keys = graph.keys();  // Creates new Array, Variant overhead
for (int i = 0; i < graph_keys.size(); i++) {
    int node_id = graph_keys[i];  // Variant conversion
}
```

**Optimized**: Direct HashMap iteration
```cpp
// With HashMap<int, NodeStruct>
for (const KeyValue<int, NodeStruct> &kv : graph_internal) {
    int node_id = kv.key;
    const NodeStruct &node = kv.value;
    // Direct access, no Variant conversion
}
```

**Expected Impact**: 30-40% faster graph traversal

### Low Impact (API Compatibility)

#### 6. State Dictionary Operations
**Current**: `Dictionary state` (passed around, duplicated)
```cpp
Dictionary state = frame.state;  // Variant copy
Dictionary clean_state = p_state.duplicate(true);  // Deep copy
```

**Note**: State must remain `Dictionary` for GDScript API compatibility.
**Optimization**: Minimize deep copies, use references where possible.

## Implementation Strategy

### Phase 1: Internal Structures (No API Changes)
1. Add `HashMap<int, NodeStruct>` to `PlannerSolutionGraph`
2. Keep `Dictionary graph` for GDScript API
3. Use internal HashMap for all operations
4. Convert to Dictionary only when `get_graph()` is called

### Phase 2: LocalVector for Collections
1. Replace `TypedArray<int>` with `LocalVector<int>` in NodeStruct
2. Replace `Array` with `LocalVector<int>` in planning loop
3. Use `HashSet<int>` for visited tracking

### Phase 3: Method Activities
1. Replace `Dictionary method_activities` with `HashMap<String, double>`
2. Or use integer method IDs if possible

## Performance Estimates

### Current Performance (48 actors)
- Average latency: ~1.4ms
- Max latency: ~2.7ms
- Planning time: ~5-20ms per plan (when it occurs)

### Expected Improvements
- **Graph lookups**: 30-50% faster → ~0.5-1ms saved per planning operation
- **Successor operations**: 20-30% faster → ~0.2-0.5ms saved
- **Planning loop**: 40-60% faster → ~2-5ms saved per planning operation
- **Overall planning**: 20-40% faster → ~1-4ms saved per plan

### Projected Performance (After Optimizations)
- Average latency: ~1.0-1.2ms (15-30% improvement)
- Max latency: ~2.0-2.3ms (15-25% improvement)
- Planning time: ~3-15ms per plan (20-40% faster)

## Code Changes Required

### 1. solution_graph.h
```cpp
// Add internal structure
struct NodeStruct {
    PlannerNodeType type;
    PlannerNodeStatus status;
    Variant info;
    LocalVector<int> successors;
    Dictionary state;
    Variant selected_method;
    TypedArray<Callable> available_methods;
    Callable action;
    int64_t start_time;
    int64_t end_time;
    int64_t duration;
    String tag;
};

class PlannerSolutionGraph {
private:
    HashMap<int, NodeStruct> graph_internal;  // Fast internal structure
    int next_node_id;

public:
    Dictionary graph;  // Keep for GDScript API (lazy conversion)

    // Internal methods use graph_internal
    NodeStruct &get_node_internal(int p_node_id);
    void update_node_internal(int p_node_id, const NodeStruct &p_node);

    // API methods convert when needed
    Dictionary get_graph() const;
    Dictionary get_node(int p_node_id) const;
};
```

### 2. plan.h
```cpp
// Replace Dictionary with HashMap
HashMap<String, double> method_activities;  // Instead of Dictionary
```

### 3. plan.cpp
```cpp
// Replace Array/TypedArray with LocalVector
LocalVector<int> failed_nodes;
LocalVector<int> open_nodes;
LocalVector<int> reachable_nodes;
LocalVector<int> to_visit;
HashSet<int> visited;  // O(1) lookups
```

## Compatibility Considerations

### GDScript API
- Must maintain `Dictionary` return types for `get_graph()`, `get_node()`
- Conversion overhead is acceptable (only happens on API calls, not during planning)
- Internal operations use fast structures

### Memory
- `LocalVector` is stack-allocated (faster, but limited size)
- `HashMap` is heap-allocated (slower allocation, but O(1) lookups)
- Trade-off: Use `LocalVector` for small collections, `HashMap` for large ones

## Conclusion

Yes, we can switch to `HashMap` and `LocalVector` for significant performance improvements:

1. **High Impact**: Internal graph structure, successors, planning loop collections
2. **Medium Impact**: Graph traversal, method activities
3. **Low Impact**: State operations (must remain Dictionary for API)

**Expected Overall Improvement**: 20-40% faster planning operations, 15-30% better latency.

The key is to use fast internal structures while maintaining `Dictionary` compatibility for the GDScript API.
