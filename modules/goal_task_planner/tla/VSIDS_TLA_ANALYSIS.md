# VSIDS TLA+ Analysis and Comparison

## TLA+ Models Created

### 1. `VSIDSComparison.tla`
Compares three different reward strategies:
- **Linear**: `100.0 / (1.0 + actions)` - Simple inverse relationship
- **Quadratic**: `30000.0 / (1.0 + actions²)` - Steep penalty for long plans (original attempt)
- **Moderate**: `100.0 / (1.0 + actions/10.0)` capped at 100 - Current implementation

**Properties to check:**
- `LinearBest`: Does linear strategy produce shortest plans?
- `QuadBest`: Does quadratic strategy produce shortest plans?
- `ModBest`: Does moderate strategy produce shortest plans?
- `*LearnsFastest`: Which strategy learns to prefer efficient method fastest?

### 2. `VSIDSActualBehavior.tla`
Models the **current actual implementation**:
- Reward: `100.0 / (1.0 + action_count / 10.0)`, capped at 100.0
- Activity scaling: 10x
- Immediate rewards when methods succeed
- Activity resets at start of each solve

**Properties to check:**
- `EventuallyPrefersEfficient`: Does it eventually prefer efficient methods?
- `PlanLengthDecreases`: Does plan length decrease over time?
- `FinalPlanLength`: What's the final plan length?
- `IterationsToEfficient`: How many iterations until efficient method is preferred?

### 3. `VSIDSBlocksWorldLearning.tla`
Models VSIDS learning during a single blocks world solve.

## Running TLA+ Models

### Check Current Implementation Behavior
```bash
tlc VSIDSActualBehavior.cfg
```

### Compare Reward Strategies
```bash
tlc VSIDSComparison.cfg
```

### Analyze Learning Process
```bash
tlc VSIDSBlocksWorldLearning.cfg
```

## Key Insights from Models

### Reward Function Comparison

| Strategy | Reward at 0 actions | Reward at 30 actions | Reward at 300 actions | Ratio (0/300) |
|----------|---------------------|---------------------|----------------------|---------------|
| Linear | 100.0 | ~3.2 | ~0.33 | 300x |
| Quadratic | 30000.0 | ~33.0 | ~0.33 | 90,000x (too extreme) |
| Moderate | 100.0 | ~3.0 | ~0.3 | 333x |

### Activity Scaling Impact

| Scaling | Impact on Selection | Risk |
|---------|-------------------|------|
| 10x | Moderate preference for high activity | Balanced |
| 100x | Strong preference (original) | May dominate too much |
| 1x | Weak preference | May not help enough |

### Current Implementation Analysis

**Strengths:**
- Moderate rewards prevent activity explosion
- 10x scaling provides meaningful preference without over-dominance
- Capped rewards prevent extreme values

**Potential Issues:**
- Reward difference between 30 and 300 actions is only ~10x (may not be enough)
- Activity might not accumulate fast enough to influence selection
- Need to verify if rewards are actually being applied correctly

## Recommendations from TLA+ Analysis

1. **Verify reward application**: Check if `_reward_method_immediate()` is being called correctly
2. **Consider logarithmic scaling**: `log(1000 / (1 + actions))` might provide better gradient
3. **Track reward frequency**: Ensure methods aren't being rewarded too frequently
4. **Compare with baseline**: Run model with no VSIDS to see baseline plan length

## Next Steps

1. Run TLA+ models to verify properties
2. Compare model predictions with actual test results
3. Adjust reward function based on TLA+ findings
4. Test with different activity scaling factors

