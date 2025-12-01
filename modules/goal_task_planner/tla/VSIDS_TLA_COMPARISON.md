# VSIDS TLA+ Comparison: Theory vs Practice

## TLA+ Model Results (VSIDSActualBehavior)

### ✅ Model Shows VSIDS Works Correctly

**State Progression:**
```
State 1:  Activities = {inefficient: 0, efficient: 0}
State 2:  Activities = {inefficient: 0, efficient: 100,000}  ← Efficient selected, rewarded
State 3:  Activities = {inefficient: 0, efficient: 183,333}  ← Efficient selected again
State 4:  Activities = {inefficient: 0, efficient: 254,761}  ← Efficient selected again
...
State 10: Activities = {inefficient: 0, efficient: 548,397}  ← Efficient consistently preferred
```

**Key Observations:**
1. ✅ Efficient method is **consistently selected** after first reward
2. ✅ Activity **accumulates correctly** (100k → 183k → 254k → ...)
3. ✅ Inefficient method **never selected** (stays at 0)
4. ✅ Plan lengths increase linearly (2, 4, 6, 8, ...) - expected behavior

## Actual C++ Implementation Behavior

### ❌ Issues Observed in Tests

**From test logs:**
- Activity scores jump to **millions** (3,000,000+)
- Plan length still **300+ actions** (not decreasing)
- VSIDS doesn't seem to be helping

### Comparison: TLA+ vs C++

| Aspect | TLA+ Model | C++ Implementation | Status |
|--------|------------|-------------------|--------|
| **Activity accumulation** | ✅ Gradual (100k → 183k) | ❌ Explosive (millions) | **Mismatch** |
| **Method selection** | ✅ Prefers efficient | ❓ Unknown | **Needs verification** |
| **Plan length** | ✅ Linear increase (expected) | ❌ 300+ (too high) | **Mismatch** |
| **Reward application** | ✅ Every step | ❓ May not be called | **Needs verification** |

## Root Cause Analysis

### Hypothesis 1: Rewards Applied Too Frequently
**TLA+**: Rewards applied once per method selection
**C++**: May be rewarding methods multiple times per solve
**Fix**: Ensure `_reward_method_immediate()` only called once per method success

### Hypothesis 2: Activity Not Used in Selection
**TLA+**: Selection uses `activity * 10` as score
**C++**: May have bug where activity isn't actually used
**Fix**: Verify `_select_best_method()` uses activity scores

### Hypothesis 3: Reward Calculation Wrong
**TLA+**: `100000 / (10 + actions)` (scaled)
**C++**: `100.0 / (1.0 + actions/10.0)` - should be equivalent
**Fix**: Verify reward calculation matches TLA+ model

### Hypothesis 4: Activity Reset Issue
**TLA+**: Activity persists during solve
**C++**: May be resetting activity incorrectly
**Fix**: Verify activity only resets at start of `find_plan()`

## Recommendations

### 1. Add Debug Logging
```cpp
if (verbose >= 3) {
    print_line(vformat("VSIDS: Method '%s' activity: %.6f, score: %.6f",
        method_id, activity, score));
    print_line(vformat("VSIDS: Rewarding method '%s' with %.6f (actions: %d)",
        method_id, reward, action_count));
}
```

### 2. Verify Reward Application
- Check if `_reward_method_immediate()` is actually being called
- Verify it's not being called multiple times for same method
- Ensure `extract_solution_plan()` returns correct action count

### 3. Verify Method Selection
- Check if `_select_best_method()` actually uses activity scores
- Verify activity scaling (10x) is applied
- Ensure subtask bonus doesn't dominate activity

### 4. Compare with TLA+ Step-by-Step
Run C++ with verbose=3 and compare:
- Activity values at each step
- Method selection decisions
- Reward amounts
- Plan lengths

## Next Steps

1. ✅ TLA+ confirms algorithm works in theory
2. ⚠️ Need to debug why C++ doesn't match TLA+ behavior
3. ⚠️ Add more verbose logging to trace execution
4. ⚠️ Verify rewards are applied and used correctly

## TLA+ Model Files

- `VSIDSActualBehavior.tla`: Models current implementation
- `VSIDSComparison.tla`: Compares different reward strategies
- `VSIDSBlocksWorldLearning.tla`: Models learning during solve

Run with: `./run_tlc.sh VSIDSActualBehavior`

