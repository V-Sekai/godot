# Temporal Planner Optimization Estimates

## Best Improvement: **STN-Based Plan Extraction** ⭐⭐⭐

### Estimated Impact

-   **Performance improvement**: 10-20% faster plan execution
-   **Latency reduction**: 0.1-0.3ms per plan (if temporal constraints are used)
-   **Effort**: Medium (2-3 days)
-   **Risk**: Low (doesn't change core planning logic)

### Why This is Best

1. **High ROI**: Medium effort, medium-high impact
2. **Low risk**: Only changes plan extraction, not planning logic
3. **Immediate benefit**: Works even with existing temporal constraints
4. **No architectural changes**: Can be implemented incrementally

### Implementation

```cpp
// Instead of DFS order, sort actions by earliest start time from STN
Array extract_plan_temporal_order(...) {
    // Get all actions
    // Query STN for earliest start time for each action
    // Sort by earliest start time
    // Return sorted actions
}
```

### Expected Results

-   **Current**: Actions executed in DFS order (may be suboptimal)
-   **After**: Actions executed in temporal order (optimal scheduling)
-   **Benefit**: Actions start as early as possible, reducing total plan duration

---

## Similar Optimizations

### 1. **Lazy STN Validation** ⭐⭐ (Similar to STN-Based Extraction)

**Concept**: Only validate STN when temporal constraints are actually used

**Current**: STN initialized and validated for every plan, even if no temporal constraints
**Optimization**: Skip STN operations if no temporal metadata present

**Estimated Impact**:

-   **Performance**: 2-5% faster planning when no temporal constraints
-   **Overhead reduction**: Eliminates STN initialization overhead (~0.01-0.05ms)
-   **Effort**: Low (1 day)
-   **Risk**: Very low

**Implementation**:

```cpp
// Check if any planner element has temporal constraints
bool has_temporal_constraints = false;
for (element in todo_list) {
    if (has_temporal_metadata(element)) {
        has_temporal_constraints = true;
        break;
    }
}

// Only initialize STN if needed
if (has_temporal_constraints) {
    stn.add_time_point("origin");
    // ... rest of STN setup
}
```

**Similarity**: Both optimize STN usage - one skips it entirely, one uses it better

---

### 2. **STN Constraint Caching** ⭐⭐ (Similar to Incremental Updates)

**Concept**: Cache STN consistency state and distance matrix

**Current**: STN recomputed/validated on every constraint addition
**Optimization**: Cache consistency state, only recompute when constraints change

**Estimated Impact**:

-   **Performance**: 5-10% faster when adding multiple constraints
-   **Overhead reduction**: Avoids redundant Floyd-Warshall runs
-   **Effort**: Medium (2 days)
-   **Risk**: Low-Medium (need to track constraint changes)

**Implementation**:

```cpp
class PlannerSTNSolver {
    bool consistency_cached = false;
    bool constraints_changed = true;

    void add_constraint(...) {
        constraints_changed = true;
        consistency_cached = false;
        // ... add constraint
    }

    bool is_consistent() {
        if (!consistency_cached || constraints_changed) {
            run_floyd_warshall();
            consistency_cached = true;
            constraints_changed = false;
        }
        return consistent;
    }
};
```

**Similarity**: Both reduce STN computation overhead - one incrementally, one via caching

---

### 3. **Early STN Failure Detection** ⭐ (Similar to Lazy Validation)

**Concept**: Detect inconsistent constraints immediately, don't wait for full validation

**Current**: All constraints added, then Floyd-Warshall validates all at once
**Optimization**: Check for obvious inconsistencies when adding constraints

**Estimated Impact**:

-   **Performance**: 5-15% faster when constraints are inconsistent (fail fast)
-   **Overhead reduction**: Avoids full Floyd-Warshall run for obviously invalid constraints
-   **Effort**: Low (1 day)
-   **Risk**: Very low

**Implementation**:

```cpp
bool add_constraint(...) {
    // Quick check: min > max is immediately invalid
    if (min_distance > max_distance) {
        consistent = false;
        return false; // Fail immediately
    }

    // Check against existing constraints for obvious conflicts
    // ... add constraint
}
```

**Similarity**: Both optimize STN validation - one skips it, one fails fast

---

## Comparison Table

| Optimization                  | Impact           | Effort      | Risk            | ROI        | Priority              |
| ----------------------------- | ---------------- | ----------- | --------------- | ---------- | --------------------- |
| **STN-Based Plan Extraction** | ⭐⭐⭐ High      | ⭐⭐ Medium | ⭐ Low          | ⭐⭐⭐⭐⭐ | **1st**               |
| Lazy STN Validation           | ⭐⭐ Medium      | ⭐ Low      | ⭐ Very Low     | ⭐⭐⭐⭐   | 2nd                   |
| STN Constraint Caching        | ⭐⭐ Medium      | ⭐⭐ Medium | ⭐⭐ Low-Medium | ⭐⭐⭐     | 3rd                   |
| Early STN Failure Detection   | ⭐ Low           | ⭐ Low      | ⭐ Very Low     | ⭐⭐⭐     | 4th                   |
| Partial Order Execution       | ⭐⭐⭐ Very High | ⭐⭐⭐ High | ⭐⭐⭐ High     | ⭐⭐       | 5th (if needed)       |
| Incremental STN Updates       | ⭐ Low           | ⭐⭐⭐ High | ⭐⭐ Medium     | ⭐         | 6th (not recommended) |

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)

1. **Lazy STN Validation** - Skip STN when not needed
2. **Early STN Failure Detection** - Fail fast on invalid constraints

**Expected**: 5-10% improvement for plans without temporal constraints

### Phase 2: Best Improvement (2-3 days)

3. **STN-Based Plan Extraction** - Optimal action scheduling

**Expected**: 10-20% improvement for plans with temporal constraints

### Phase 3: Advanced (2-3 days, if needed)

4. **STN Constraint Caching** - Reduce redundant validation

**Expected**: Additional 5-10% improvement for complex temporal plans

---

## Best Improvement Details: STN-Based Plan Extraction

### Current Behavior

```cpp
// Actions extracted in DFS order
extract_plan() {
    // DFS traversal
    // Actions: [A1, A2, A3, A4]
    // Execution: A1 → A2 → A3 → A4 (sequential)
}
```

### Optimized Behavior

```cpp
// Actions extracted in temporal order
extract_plan_temporal() {
    // Get all actions
    // Query STN for earliest start time
    // Sort: [A1 (t=0), A3 (t=5), A2 (t=10), A4 (t=15)]
    // Execution: A1 → A3 → A2 → A4 (temporally optimal)
}
```

### Benefits

1. **Optimal scheduling**: Actions start as early as possible
2. **Better resource utilization**: Actions can overlap if constraints allow
3. **Reduced plan duration**: Total plan time minimized
4. **Respects temporal constraints**: Uses STN information effectively

### Example Scenario

**Before (DFS order)**:

```
Action 1: start=0ms, duration=10ms
Action 2: start=10ms, duration=5ms
Action 3: start=15ms, duration=8ms
Total: 23ms
```

**After (Temporal order)**:

```
Action 1: start=0ms, duration=10ms
Action 3: start=5ms (can start earlier), duration=8ms
Action 2: start=10ms, duration=5ms
Total: 18ms (5ms faster!)
```

### Implementation Complexity

-   **Medium effort**: Need to modify `extract_plan()` to query STN
-   **Low risk**: Doesn't change planning logic, only extraction order
-   **Incremental**: Can be added as optional mode first

---

## Conclusion

**Best Improvement**: **STN-Based Plan Extraction** (10-20% improvement, medium effort)

**Similar Optimizations**:

1. Lazy STN Validation (skip when not needed)
2. STN Constraint Caching (reduce redundant computation)
3. Early STN Failure Detection (fail fast)

**Recommendation**: Start with Lazy STN Validation (quick win), then implement STN-Based Plan Extraction (best improvement).
