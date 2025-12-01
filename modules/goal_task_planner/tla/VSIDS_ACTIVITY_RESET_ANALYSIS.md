# VSIDS Activity Reset Analysis

## Question: Should VSIDS activity reset with each solution graph?

### Current Implementation
- **Activity persists** across `find_plan()` calls
- Solution graph is reset each time (line 75: `solution_graph = PlannerSolutionGraph()`)
- Activity is stored in `PlannerPlan` member variable `method_activities`

## Option 1: Activity Persists (Current)

### Pros ✅
1. **Cross-attempt learning**: If solving the same problem multiple times, VSIDS learns which methods work better
2. **Optimization benefit**: Plan quality should improve over multiple attempts
3. **Efficient for repeated problems**: Same PlannerPlan object can accumulate knowledge
4. **Matches user's goal**: "number of actions should decrease" - requires persistence

### Cons ❌
1. **Problem contamination**: Activity from one problem type (e.g., blocks world) may influence a different problem type (e.g., isekai academy)
2. **Stale learning**: Methods that worked for Problem A may not work for Problem B
3. **No fresh start**: Can't reset learning when switching problem domains
4. **Test isolation concerns**: Tests may interfere with each other if using same PlannerPlan object

## Option 2: Activity Resets with Each Solution Graph

### Pros ✅
1. **Problem isolation**: Each `find_plan()` call starts fresh, no contamination
2. **Better for different problem types**: Blocks world learning doesn't affect isekai academy
3. **Test isolation**: Tests don't interfere with each other
4. **Clean slate**: Each problem gets unbiased method selection

### Cons ❌
1. **No cross-attempt learning**: Can't learn across multiple attempts on same problem
2. **Loses optimization benefit**: Plan quality won't improve over multiple attempts
3. **Wastes learning**: If solving same problem 10 times, each time starts from scratch
4. **Doesn't solve user's issue**: "number of actions not decreasing" - requires persistence

## Option 3: Hybrid - Reset Based on Problem Signature

### Concept
- Reset activity when problem characteristics change (e.g., different domain, different goal structure)
- Persist activity within the same problem type

### Implementation
```cpp
// Track problem signature
String current_problem_signature = _compute_problem_signature(p_state, p_todo_list);
if (current_problem_signature != last_problem_signature) {
    method_activities.clear();  // Reset for new problem type
    last_problem_signature = current_problem_signature;
}
```

### Pros ✅
- Best of both worlds: learns within problem type, resets between types
- Prevents contamination while preserving optimization

### Cons ❌
- More complex to implement
- Need to define "problem signature" (domain? goal structure? state size?)
- May reset too often or too rarely

## Recommendation

### For the Current Issue (Blocks World)
**Keep activity persistent** (Option 1) because:
- User wants plan quality to improve over multiple attempts
- Same problem type (blocks world) is being solved repeatedly
- The goal is optimization, not problem isolation

### For General Use
**Implemented Option 3 (Hybrid)** - Best of both worlds:
- ✅ **Default behavior**: Activity persists across `find_plan()` calls (enables learning)
- ✅ **Manual reset available**: Call `plan->reset_vsids_activity()` when switching problem types
- ✅ **Flexible**: Users can choose when to reset based on their use case

**When to reset:**
- Switching between different problem domains (e.g., blocks world → isekai academy)
- Starting a new problem type that shouldn't be influenced by previous learning
- Test isolation: Reset between test cases if needed
- Debugging: Reset to see fresh behavior without accumulated learning

**When to keep persistent:**
- Solving the same problem multiple times (optimization goal)
- Iterative planning on similar problems
- Learning which methods work best for a specific domain

## Implementation Notes

### Current Code Location
```cpp
// plan.cpp line 80-86
// VSIDS activity tracking: DO NOT clear - let it persist across planning attempts
// This allows VSIDS to learn and improve plan quality over time
// Only initialize if this is the first call (method_activities is empty)
if (method_activities.is_empty()) {
    activity_var_inc = 1.0;
    activity_bump_count = 0;
}
```

### Reset Method (Implemented)
Public method available:
```cpp
void PlannerPlan::reset_vsids_activity() {
    method_activities.clear();
    activity_var_inc = 1.0;
    activity_bump_count = 0;
}
```

**Usage:**
```cpp
// Reset when switching problem types
plan->reset_vsids_activity();
Ref<PlannerResult> result = plan->find_plan(new_state, new_todo_list);
```

## TLA+ Model

See `tla/VSIDSActivityResetAnalysis.tla` for formal comparison of strategies.

