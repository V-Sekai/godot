# Optimization Ranking by Highest Improvement Potential

## Ranking Criteria
- **Impact**: Expected improvement in overall system performance
- **Effort**: Implementation difficulty (Low/Medium/High)
- **Risk**: Risk of breaking functionality (Low/Medium/High)
- **ROI**: Return on investment (Impact / Effort)

## Top 10 Optimizations (Ranked by Impact)

### 🥇 1. Optimize `_process_node_iterative` (12.5% of total samples)
**Current Impact**: 129 samples (12.5% of total)
**Expected Improvement**: 20-30% faster node processing = **2.5-3.8% overall improvement**

**Why it's #1**:
- High sample count (12.5% of total)
- Called frequently during planning loop
- Room for optimization without changing algorithm

**Optimization Opportunities**:
- Reduce redundant state copies
- Cache node lookups
- Optimize Dictionary operations
- Use internal structures more directly

**Effort**: Medium | **Risk**: Low | **ROI**: ⭐⭐⭐⭐⭐

---

### 🥈 2. Optimize `_select_best_method` (6.5% of total samples)
**Current Impact**: 67 samples (6.5% of total)
**Expected Improvement**: 30-40% faster method selection = **2.0-2.6% overall improvement**

**Why it's #2**:
- Significant contributor (6.5%)
- Called for every method selection
- Already using HashMap (optimized), but can improve further

**Optimization Opportunities**:
- Cache method scores when state hasn't changed
- Pre-compute activity scores
- Reduce Variant conversions
- Early termination for obvious best methods

**Effort**: Medium | **Risk**: Low | **ROI**: ⭐⭐⭐⭐

---

### 🥉 3. Reduce Planning Depth (Algorithm-level)
**Current Impact**: Affects all planning operations (52.4% combined)
**Expected Improvement**: 20-30% faster planning = **10-15% overall improvement**

**Why it's #3**:
- Affects the entire planning process
- Easy to implement (just change max_depth)
- But may reduce plan quality

**Implementation**:
- Reduce `max_depth` from 10 to 7-8
- Test plan quality impact
- Use adaptive depth based on problem complexity

**Effort**: Low | **Risk**: Medium (plan quality) | **ROI**: ⭐⭐⭐⭐⭐

---

### 4. Cache Planning Results for Similar States
**Current Impact**: Eliminates redundant planning (could save 10-20% of planning calls)
**Expected Improvement**: 10-20% reduction in planning time = **3-5% overall improvement**

**Why it's #4**:
- Many actors have similar states
- Can reuse planning results
- Significant savings if cache hit rate is good

**Implementation**:
- Hash state to create cache key
- Store successful plans in cache
- Invalidate on state changes
- Limit cache size (LRU eviction)

**Effort**: High | **Risk**: Medium (cache invalidation) | **ROI**: ⭐⭐⭐

---

### 5. Optimize State Duplicate in `find_plan` (Line 151)
**Current Impact**: Called once per plan, but expensive
**Expected Improvement**: 5-10% faster find_plan = **1.6-3.2% overall improvement**

**Why it's #5**:
- Deep copy is expensive
- Called for every planning operation
- Can use shallow copy with selective deep copy

**Implementation**:
- Use `duplicate(false)` for shallow copy
- Only deep copy nested dictionaries that will change
- Similar to what we did in `execute_plan_helper`

**Effort**: Low | **Risk**: Low | **ROI**: ⭐⭐⭐⭐

---

### 6. Optimize Graph Success Checking (Lines 216-417)
**Current Impact**: ~15-20% of find_plan time
**Expected Improvement**: 20-30% faster success check = **1.0-1.5% overall improvement**

**Why it's #6**:
- Already optimized with HashMap/LocalVector
- But can cache results if graph hasn't changed
- Can early-terminate on first failure

**Implementation**:
- Cache reachable nodes if graph unchanged
- Early termination on first open node
- Skip verify goal checks if not needed

**Effort**: Medium | **Risk**: Low | **ROI**: ⭐⭐⭐

---

### 7. Reduce Dictionary Conversions in `find_plan`
**Current Impact**: Multiple conversions add overhead
**Expected Improvement**: 3-5% faster find_plan = **1.0-1.6% overall improvement**

**Why it's #7**:
- Already optimized some (cached get_graph())
- Can use internal structures more
- Reduce conversions at boundaries

**Implementation**:
- Use internal structures throughout
- Convert to Dictionary only at API boundaries
- Already partially done, can extend further

**Effort**: Low | **Risk**: Low | **ROI**: ⭐⭐⭐⭐

---

### 8. Optimize `add_nodes_and_edges` (3.6% of total)
**Current Impact**: 37 samples (3.6% of total)
**Expected Improvement**: 20-30% faster = **0.7-1.1% overall improvement**

**Why it's #8**:
- Moderate contributor
- Graph construction overhead
- Can optimize node creation

**Implementation**:
- Batch node creation
- Pre-allocate structures
- Reduce Dictionary operations

**Effort**: Medium | **Risk**: Low | **ROI**: ⭐⭐⭐

---

### 9. Early Termination Heuristics
**Current Impact**: Could save 10-20% of planning iterations
**Expected Improvement**: 5-10% faster planning = **2.6-5.2% overall improvement**

**Why it's #9**:
- High potential impact
- But requires careful heuristics
- May affect plan quality

**Implementation**:
- Detect impossible states early
- Skip planning if needs are satisfied
- Use domain-specific heuristics

**Effort**: High | **Risk**: Medium | **ROI**: ⭐⭐⭐

---

### 10. Optimize Verbose Logging
**Current Impact**: Minimal when verbose=0, but still some overhead
**Expected Improvement**: 1-2% faster = **0.3-0.6% overall improvement**

**Why it's #10**:
- Low impact
- Already mostly guarded
- Easy to optimize further

**Implementation**:
- More aggressive verbose guards
- Lazy string formatting
- Remove unnecessary checks

**Effort**: Low | **Risk**: None | **ROI**: ⭐⭐

---

## Summary by ROI (Return on Investment)

### Highest ROI (Easy + High Impact)
1. **Reduce Planning Depth** - Low effort, high impact (if plan quality allows)
2. **Optimize State Duplicate** - Low effort, medium-high impact
3. **Reduce Dictionary Conversions** - Low effort, medium impact

### Medium ROI (Medium Effort + Good Impact)
4. **Optimize `_process_node_iterative`** - Medium effort, high impact
5. **Optimize `_select_best_method`** - Medium effort, good impact
6. **Optimize Graph Success Checking** - Medium effort, medium impact

### Lower ROI (High Effort or Lower Impact)
7. **Cache Planning Results** - High effort, high impact (but complex)
8. **Early Termination Heuristics** - High effort, high impact (but risky)
9. **Optimize `add_nodes_and_edges`** - Medium effort, lower impact
10. **Optimize Verbose Logging** - Low effort, low impact

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. Reduce planning depth (test plan quality)
2. Optimize state duplicate
3. Reduce Dictionary conversions

**Expected**: 5-8% overall improvement

### Phase 2: Medium Effort (3-5 days)
4. Optimize `_process_node_iterative`
5. Optimize `_select_best_method`
6. Optimize graph success checking

**Expected**: 5-7% additional improvement

### Phase 3: Advanced (1-2 weeks)
7. Cache planning results
8. Early termination heuristics

**Expected**: 5-10% additional improvement

## Total Potential Improvement

- **Phase 1**: 5-8%
- **Phase 2**: 5-7%
- **Phase 3**: 5-10%
- **Total**: 15-25% overall performance improvement

**Current**: ~1.4ms average, ~3ms max
**After all optimizations**: ~1.0-1.2ms average, ~2.0-2.5ms max
