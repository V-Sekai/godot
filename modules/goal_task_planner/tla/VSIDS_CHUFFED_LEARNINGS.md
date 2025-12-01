# Learning from Chuffed's VSIDS Implementation

## Key Insights from Chuffed

### 1. **Fixed Increment, Not Calculated Reward**
**Chuffed's approach:**
```cpp
activity[v] += var_inc;  // Simple addition
```

**Our old approach:**
```cpp
double reward = 100.0 / (1.0 + action_count / 10.0);
activity[method] += reward;  // Calculated reward
```

**Why Chuffed's is better:**
- Simpler and proven
- No need to calculate rewards based on plan length
- Activity inflation naturally handles decay

### 2. **Activity Inflation Instead of Direct Decay**
**Chuffed's approach:**
```cpp
var_inc *= 1.05;  // Increase increment (activity inflation)
// Activities are NOT multiplied by decay factor
```

**Our old approach:**
```cpp
activity_var_inc *= 1.05;
activity *= 0.95;  // Also decaying activities directly
```

**Why Chuffed's is better:**
- More efficient: No need to iterate over all activities
- Equivalent effect: Increasing `var_inc` makes newer bumps worth more
- Standard approach: Used in all major SAT solvers

### 3. **Bump on Conflict (Not Just Success)**
**Chuffed's approach:**
- Bumps variables **involved in conflicts** during conflict analysis
- Uses `seen[]` array to prevent duplicate bumps per conflict

**Our approach:**
- Bumps methods **that succeed** (optimization goal)
- Also bumps methods **in conflict paths** (learning from failures)
- Uses `rewarded_methods_this_solve` to prevent duplicates

**Both are valid:**
- Chuffed: Conflict-driven learning (what went wrong)
- Ours: Success + conflict learning (what went right + what went wrong)

### 4. **Duplicate Prevention**
**Chuffed:**
```cpp
if (seen[x] == 0) {
    varBumpActivity(q);
    seen[x] = 1;  // Mark as seen
}
```

**Ours (now):**
```cpp
if (!rewarded_methods_this_solve.has(method_id)) {
    _reward_method_immediate(method);
    rewarded_methods_this_solve.push_back(method_id);
}
```

**Status**: ✅ Both prevent duplicates correctly

## Changes Made to Match Chuffed

### 1. Simplified Reward Calculation
**Before:**
```cpp
double base_reward = 100.0 / (1.0 + p_current_action_count / 10.0);
double reward = base_reward * activity_var_inc;
method_activities[method_id] = current_activity + reward;
```

**After (matching Chuffed):**
```cpp
method_activities[method_id] = current_activity + activity_var_inc;
```

### 2. Activity Inflation Only
**Before:**
```cpp
activity_var_inc *= 1.05;
if (activity_var_inc > 1e100) {
    // normalize
} else {
    activity *= 0.95;  // Also decaying directly
}
```

**After (matching Chuffed):**
```cpp
activity_var_inc *= 1.05;  // Activity inflation
if (activity_var_inc > 1e100) {
    // normalize
}
// No direct decay - activity inflation handles it
```

### 3. Removed Unused Decay Factor
**Removed:**
```cpp
double activity_decay_factor = 0.95;  // No longer needed
```

## What We Kept (Different from Chuffed)

### 1. Success-Based Rewards
- **Chuffed**: Only bumps on conflicts
- **Ours**: Bumps on both conflicts AND successes
- **Rationale**: We want to optimize plan quality, not just learn from failures

### 2. Plan Length Consideration
- **Chuffed**: No plan length consideration (SAT solving)
- **Ours**: Could add plan length weighting in the future
- **Current**: Using fixed increment like Chuffed (simpler)

## Expected Improvements

1. **Simpler code**: Fixed increment is easier to understand
2. **Better performance**: Activity inflation is more efficient
3. **Proven approach**: Matches industry-standard SAT solver implementation
4. **No activity explosion**: Fixed increment prevents extreme values

## Testing

Run blocks world test to verify:
```bash
./bin/godot.macos.editor.dev.arm64 --test --test-path=modules/goal_task_planner/tests --test-name="Blocks World - Performance Test"
```

Check verbose logs for:
- Activity values should be reasonable (not millions)
- Methods should be rewarded with `var_inc` (not calculated rewards)
- Activity inflation should occur every 100 bumps

## References

- Chuffed source: `/Users/ernest.lee/Desktop/code/projects/chuffed/chuffed/core/conflict.cpp`
- Key functions: `varBumpActivity()`, `varDecayActivity()`
- Chuffed paper: "Chuffed: A lazy clause generation solver" (if available)

