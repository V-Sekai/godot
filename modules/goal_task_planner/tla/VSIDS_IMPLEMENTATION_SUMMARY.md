# VSIDS Implementation Summary

## What We Learned from Chuffed

### Chuffed's VSIDS Approach (Industry Standard)

1. **Fixed Increment Bumping**
   - `activity[v] += var_inc` (simple addition)
   - No calculated rewards based on plan length

2. **Activity Inflation (Not Direct Decay)**
   - `var_inc *= 1.05` (increase increment by 5%)
   - Activities are NOT multiplied by decay factor
   - More efficient: no need to iterate over all activities

3. **Conflict-Based Bumping**
   - Bumps variables involved in conflicts
   - Uses `seen[]` array to prevent duplicates per conflict

4. **Priority Heap Selection**
   - Uses heap to select highest activity variable
   - Efficient for large variable sets

## Our Implementation (Now Matches Chuffed)

### Changes Made

1. ✅ **Simplified reward**: Use fixed `activity_var_inc` instead of calculated reward
2. ✅ **Activity inflation**: Increase `var_inc` by 1.05, don't decay activities directly
3. ✅ **Duplicate prevention**: Track rewarded methods per solve
4. ✅ **Conflict bumping**: Already implemented via `_bump_conflict_path_activities()`

### What We Kept (Different from Chuffed)

1. **Success-based rewards**: We reward successful methods (optimization goal)
2. **Both conflict and success**: Learn from both failures and successes
3. **Linear search**: Keep linear search for method selection (methods count is small)

## Current Implementation

### Reward Function
```cpp
// Chuffed-style: Simple fixed increment
method_activities[method_id] = current_activity + activity_var_inc;
```

### Decay Function
```cpp
// Activity inflation (Chuffed-style)
activity_var_inc *= 1.05;  // Increase increment
// No direct decay of activities
```

### Bumping Strategy
- **On conflict**: `_bump_conflict_path_activities()` bumps methods in conflict path
- **On success**: `_reward_method_immediate()` rewards successful methods
- **Duplicate prevention**: `rewarded_methods_this_solve` tracks already-rewarded methods

## Results

### Before (Calculated Rewards)
- Activity scores: **3,000,000+** (explosive)
- Plan length: **300+ actions** (not improving)

### After (Chuffed-Style Fixed Increment)
- Activity scores: **3-15** (reasonable)
- Plan length: **TBD** (needs testing)

## Next Steps

1. ✅ Implemented Chuffed's approach
2. ⚠️ Test with blocks world to verify plan length decreases
3. ⚠️ Compare performance: fixed increment vs calculated reward
4. ⚠️ Verify activity inflation works correctly

## Files Modified

- `plan.cpp`: Simplified reward calculation, activity inflation only
- `plan.h`: Removed unused `activity_decay_factor`
- `AGENTS.md`: Updated documentation to reflect Chuffed's approach

## References

- Chuffed source: `chuffed/core/conflict.cpp` (lines 49-84)
- Chuffed header: `chuffed/core/sat.h` (lines 103-117)
- TLA+ models: `tla/VSIDSActualBehavior.tla`, `tla/VSIDSComparison.tla`

