# VSIDS Optimization for Blocks World

## Changes Made

### 1. Activity Resets Per Solve
- **Before**: Activity persisted across `find_plan()` calls
- **After**: Activity resets at start of each solve (line 80-84 in `plan.cpp`)
- **Rationale**: VSIDS learns during backtracking within a single solve, not across solves

### 2. Immediate Rewards
- **Added**: `_reward_method_immediate()` function
- **When**: Methods are rewarded immediately when they succeed (not at end of plan)
- **Applied to**: TASK, UNIGOAL, and MULTIGOAL methods

### 3. Quadratic Reward Scaling
- **Before**: `10000.0 / (1.0 + action_count)` - linear penalty
- **After**: `30000.0 / (1.0 + action_count * action_count)` - quadratic penalty
- **Impact**:
  - 30 actions: ~33 reward (was ~333)
  - 100 actions: ~3 reward (was ~100)
  - 300 actions: ~0.33 reward (was ~33)
- **Rationale**: Steeper penalty for longer plans ensures methods that lead to shorter plans get MUCH higher rewards

### 4. Activity Score Dominance
- **Before**: `score = activity * 10.0 + 100.0/(1+subtasks)`
- **After**: `score = activity * 100.0 + 10.0/(1+subtasks)`
- **Changes**:
  - Activity scaled by 100x (was 10x)
  - Subtask bonus reduced to 10.0 (was 100.0)
- **Rationale**: Ensure activity scores completely dominate method selection

### 5. Enhanced Verbose Logging
- **Test**: Blocks World Performance Test now uses `verbose=3`
- **Logs**:
  - Method selection with activity scores
  - Immediate rewards when methods succeed
  - Activity decay events
  - Method evaluation scores

## TLA+ Model

Created `tla/VSIDSBlocksWorldLearning.tla` to formally model:
- Immediate rewards based on current action count
- Activity-based method selection
- Learning during backtracking within a single solve

## Expected Behavior

1. **Initial State**: All methods start with activity = 0
2. **First Attempts**: Methods are tried in order (or by subtask bonus)
3. **Learning**: Methods that succeed with fewer actions get higher rewards
4. **Backtracking**: When backtracking occurs, methods with higher activity are preferred
5. **Optimization**: Over time, methods that lead to shorter plans accumulate higher activity and are selected first

## Testing

Run the blocks world performance test with verbose logging:
```cpp
plan->set_verbose(3);
Ref<PlannerResult> result = plan->find_plan(init_state, todo_list);
```

Check the logs for:
- VSIDS reward messages showing activity increases
- Method selection showing activity-based preference
- Plan length decreasing over backtracking attempts

## Key Metrics

- **Reward for 30-action plan**: ~33 points
- **Reward for 300-action plan**: ~0.33 points
- **Activity scaling**: 100x multiplier
- **Reward difference**: ~100x between efficient and inefficient methods

This should enable VSIDS to strongly prefer methods that lead to shorter plans, reducing action count from 300+ to ~30 within a single solve.

