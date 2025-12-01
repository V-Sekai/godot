# VSIDS Implementation: Chuffed vs Our Implementation

## Chuffed's VSIDS Implementation

### Key Components (from `chuffed/core/conflict.cpp` and `sat.h`)

#### 1. Activity Bumping (`varBumpActivity`)
```cpp
inline void SAT::varBumpActivity(Lit p) {
    const int v = var(p);
    if (so.vsids) {
        activity[v] += var_inc;  // Simple addition of current var_inc
        if (order_heap.inHeap(v)) {
            order_heap.decrease(v);  // Update priority queue
        }
    }
}
```

**Key insight**: Chuffed bumps activity by adding `var_inc` (not a calculated reward)

#### 2. Activity Decay (`varDecayActivity`)
```cpp
inline void SAT::varDecayActivity() {
    var_inc *= 1.05;  // Increase var_inc by 5% (activity inflation)
    if (var_inc > 1e100) {
        // Normalize: scale down all activities and var_inc
        for (int i = 0; i < nVars(); i++) {
            activity[i] *= 1e-100;
        }
        var_inc *= 1e-100;
    }
}
```

**Key insight**: Chuffed uses **activity inflation** - increases `var_inc` instead of decaying activities directly

#### 3. When Activities Are Bumped
```cpp
void SAT::analyze(int nodeid, std::set<int>& contributingNogoods) {
    varDecayActivity();  // Decay FIRST (increase var_inc)
    claDecayActivity();
    getLearntClause(nodeid, contributingNogoods);
    // ...
}

void SAT::getLearntClause(...) {
    // During conflict analysis, bump variables in conflict path
    for (unsigned int j = (p == lit_Undef) ? 0 : 1; j < c.size(); j++) {
        const Lit q = c[j];
        if (seen[x] == 0) {
            varBumpActivity(q);  // Bump each variable once per conflict
            seen[x] = 1;  // Mark as seen to prevent duplicate bumps
        }
    }
}
```

**Key insight**: 
- Activities are bumped **during conflict analysis** (when backtracking)
- Each variable is bumped **once per conflict** (using `seen[]` to track)
- Decay happens **once per conflict** (at start of `analyze()`)

#### 4. Activity Selection
```cpp
DecInfo* SAT::branch() {
    const int next = order_heap.removeMin();  // Select variable with highest activity
    return new DecInfo(nullptr, 2 * next + static_cast<int>(polarity[next]));
}
```

**Key insight**: Uses a **priority heap** ordered by activity (highest first)

## Our Implementation vs Chuffed

| Aspect | Chuffed | Our Implementation | Status |
|--------|---------|-------------------|--------|
| **Bump amount** | `var_inc` (grows over time) | Calculated reward based on plan length | ❌ **Different** |
| **Decay method** | Activity inflation (`var_inc *= 1.05`) | Direct decay (`activity *= 0.95`) | ❌ **Different** |
| **When to bump** | During conflict analysis | When method succeeds | ❌ **Different** |
| **Bump frequency** | Once per conflict per variable | Once per solve per method | ❌ **Different** |
| **Selection** | Priority heap by activity | Score = activity * 10 + subtask bonus | ⚠️ **Similar** |

## Key Differences

### 1. **Bump Timing**
- **Chuffed**: Bumps variables **involved in conflicts** (failures)
- **Ours**: Rewards methods **that succeed** (successes)

**Analysis**: 
- Chuffed learns from **what went wrong** (conflict-driven)
- We're trying to learn from **what went right** (success-driven)
- **Both approaches are valid**, but serve different purposes

### 2. **Bump Amount**
- **Chuffed**: Fixed increment (`var_inc`) that grows over time
- **Ours**: Variable reward based on plan length (inverse relationship)

**Analysis**:
- Chuffed's approach is simpler and proven
- Our approach tries to optimize for shorter plans
- **Recommendation**: Consider using fixed increment like Chuffed

### 3. **Decay Strategy**
- **Chuffed**: Activity inflation (`var_inc *= 1.05`) - newer bumps are worth more
- **Ours**: Direct decay (`activity *= 0.95`) - older activities fade

**Analysis**:
- Activity inflation is more efficient (no need to iterate over all activities)
- Direct decay is more intuitive but less efficient
- **Recommendation**: Consider switching to activity inflation

### 4. **Duplicate Prevention**
- **Chuffed**: Uses `seen[]` array to prevent bumping same variable twice per conflict
- **Ours**: Uses `rewarded_methods_this_solve` array (just added)

**Analysis**: ✅ **Similar approach** - both prevent duplicate bumps

## Recommendations Based on Chuffed

### 1. Use Fixed Increment Instead of Calculated Reward
```cpp
// Instead of: 100.0 / (1.0 + action_count / 10.0)
// Use: activity_var_inc (which grows over time)
void PlannerPlan::_bump_method_activity(Callable p_method) {
    String method_id = _method_to_id(p_method);
    double current_activity = _get_method_activity(p_method);
    method_activities[method_id] = current_activity + activity_var_inc;
    activity_bump_count++;
}
```

### 2. Use Activity Inflation Instead of Direct Decay
```cpp
void PlannerPlan::_decay_method_activities() {
    // Activity inflation: increase var_inc instead of decaying activities
    activity_var_inc *= 1.05;
    
    // Normalize if var_inc gets too large
    if (activity_var_inc > 1e100) {
        Array keys = method_activities.keys();
        for (int i = 0; i < keys.size(); i++) {
            Variant key = keys[i];
            method_activities[key] = (double)method_activities[key] * 1e-100;
        }
        activity_var_inc *= 1e-100;
    }
    // No need to multiply all activities by decay factor!
}
```

### 3. Bump on Conflict, Not Just on Success
Currently we only reward successful methods. Chuffed bumps variables involved in conflicts. We should do both:
- **Bump on conflict**: When a method fails (already implemented via `_bump_conflict_path_activities`)
- **Bump on success**: When a method succeeds (currently implemented)

### 4. Use Priority Heap for Method Selection
Chuffed uses a heap for efficient selection. We use linear search. Consider:
- Using a heap for method selection (if we have many methods)
- Or keep linear search if method count is small

## Implementation Changes Needed

1. **Simplify reward calculation**: Use fixed `activity_var_inc` instead of calculated reward
2. **Switch to activity inflation**: Increase `var_inc` instead of decaying activities
3. **Keep conflict bumping**: Already implemented correctly
4. **Keep duplicate prevention**: Already added

## Why Chuffed's Approach Works

1. **Simplicity**: Fixed increment is easier to reason about
2. **Efficiency**: Activity inflation avoids iterating over all activities
3. **Proven**: Used in production SAT solvers for decades
4. **Scalability**: Works well with large numbers of variables

## Next Steps

1. Test Chuffed's approach in our codebase
2. Compare performance: fixed increment vs calculated reward
3. Measure: Does activity inflation work better than direct decay?
4. Verify: Does conflict-based bumping help more than success-based?

