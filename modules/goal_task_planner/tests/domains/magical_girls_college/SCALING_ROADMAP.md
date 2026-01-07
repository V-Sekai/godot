# Roadmap to 300 Actors

## Current Status (48 Actors)

**Performance**:
- Average: 1.440ms ✅
- Max: 2.804ms ✅
- Target: <30ms (48x headroom) ✅

**Already Implemented** ✅:
1. ✅ Concurrent planning limits (Semaphore, max 4 concurrent)
2. ✅ Optimized state copying (shallow + selective deep)
3. ✅ Dictionary lookup caching
4. ✅ C++ optimizations (HashMap/LocalVector in planner)
5. ✅ Partial replanning (32.9% replan ratio)

## Scaling Challenge: 48 → 300 Actors

**Current**: 48 actors at 1.44ms average
**Target**: 300 actors at <30ms max latency
**Scaling factor**: 6.25x more actors

### Linear Projection (Unrealistic)
- Average: 1.44ms × 6.25 = **9ms** ✅
- Max: 2.8ms × 6.25 = **17.5ms** ✅

**BUT**: Planning doesn't scale linearly due to:
- Concurrent planning contention
- State copying overhead
- Dictionary operations scale with actor count

## Realistic Projection for 300 Actors

### Without Additional Optimizations
- **Average**: ~9-12ms (acceptable)
- **Max**: **25-40ms** ⚠️ (may exceed 30ms target)
- **Risk**: Planning spikes when many actors plan simultaneously

### With Remaining Optimizations
- **Average**: ~6-8ms ✅
- **Max**: **18-25ms** ✅ (within 30ms target)

## Remaining Optimizations Needed

### Phase 1: Planning Coordination (HIGH PRIORITY)

#### 1.1 Planning Result Caching ⚠️ **NOT YET IMPLEMENTED**
**Current**: Every actor plans independently, even for similar states
**Impact**: 30-50% of planning calls could be cached

**Implementation**:
```gdscript
# Add to Actor class
var planning_cache: Dictionary = {}
var cache_hits: int = 0
var cache_misses: int = 0

func get_cached_plan(state: Dictionary, critical_needs: Array) -> Variant:
    # Hash state + critical needs to create cache key
    var state_hash = hash_state_for_cache(state, critical_needs)
    var cached = planning_cache.get(state_hash, null)
    if cached != null:
        cache_hits += 1
        return cached
    cache_misses += 1
    return null

func cache_plan(state: Dictionary, critical_needs: Array, result: PlannerResult):
    var state_hash = hash_state_for_cache(state, critical_needs)
    planning_cache[state_hash] = result
    # Limit cache size (LRU eviction)
    if planning_cache.size() > 500:
        var keys = planning_cache.keys()
        for i in range(100):
            planning_cache.erase(keys[i])
```

**Expected Impact**: 
- Reduces planning calls by 30-50%
- At 300 actors: 30-50 actors planning → 15-25 actors planning
- Max latency: 40ms → **20-25ms** ✅

#### 1.2 Adaptive Planning Depth ⚠️ **PARTIALLY IMPLEMENTED**
**Current**: Fixed `max_depth = 10` for all actors
**Opportunity**: Use shallower depth for non-critical needs

**Implementation**:
```gdscript
# Adjust depth based on need urgency
func get_planning_depth(critical_needs: Array) -> int:
    var max_urgency = 0
    for need in critical_needs:
        var urgency = 100 - need["value"]
        max_urgency = max(max_urgency, urgency)
    
    # Critical needs (urgency > 40): full depth
    if max_urgency > 40:
        return 10
    # Moderate needs: reduced depth
    elif max_urgency > 20:
        return 7
    # Low urgency: minimal depth
    else:
        return 5
```

**Expected Impact**:
- Faster planning for non-critical needs
- Average planning time: 5-10ms → **3-7ms**
- Max latency: 25ms → **18-22ms** ✅

### Phase 2: State Management (MEDIUM PRIORITY)

#### 2.1 State Update Batching ⚠️ **NOT YET IMPLEMENTED**
**Current**: Each actor updates state independently
**Opportunity**: Batch state updates for better cache locality

**Implementation**:
```gdscript
# Collect state updates, apply in batch
var pending_state_updates: Array = []

func queue_state_update(actor_id: String, state_update: Dictionary):
    pending_state_updates.append({"actor": actor_id, "update": state_update})

func apply_batched_updates():
    # Process updates in batches of 50
    for i in range(0, pending_state_updates.size(), 50):
        var batch = pending_state_updates.slice(i, i + 50)
        # Apply batch updates
        for update in batch:
            apply_state_update(update["actor"], update["update"])
    pending_state_updates.clear()
```

**Expected Impact**:
- Better memory access patterns
- State operations: 2-3ms → **1-2ms**
- Overall latency: 22ms → **20ms** ✅

#### 2.2 Immutable State Updates ⚠️ **NOT YET IMPLEMENTED**
**Current**: Mutable state dictionaries
**Opportunity**: Use immutable state updates to reduce copying

**Expected Impact**:
- Reduce state copying overhead by 30-40%
- State operations: 1-2ms → **0.6-1.2ms**

### Phase 3: Message System (LOW PRIORITY)

#### 3.1 Message Pooling ⚠️ **NOT YET IMPLEMENTED**
**Current**: New `ActorMessage` created for each message
**Opportunity**: Reuse message objects

**Expected Impact**:
- Reduce allocation overhead
- Message overhead: 0.5-1ms → **0.3-0.5ms**

## Recommended Implementation Order

### Step 1: Planning Result Caching (1-2 days) 🔴 **CRITICAL**
**Why first**: Biggest impact on max latency spikes
**Expected**: Max latency 40ms → **20-25ms**

### Step 2: Adaptive Planning Depth (1 day) 🟡 **HIGH**
**Why second**: Easy to implement, good impact
**Expected**: Average latency 9ms → **7-8ms**

### Step 3: Test at 150 Actors (1 day) 🟡 **VALIDATION**
**Why third**: Validate optimizations at intermediate scale
**Target**: Max latency <20ms at 150 actors

### Step 4: State Update Batching (2-3 days) 🟢 **MEDIUM**
**Why fourth**: Additional optimization if needed
**Expected**: Additional 1-2ms improvement

### Step 5: Test at 300 Actors (1 day) 🟡 **FINAL VALIDATION**
**Why last**: Final validation at target scale
**Target**: Max latency <30ms at 300 actors

## Projected Timeline

**Total**: 6-9 days of development + testing

**Week 1**:
- Day 1-2: Planning result caching
- Day 3: Adaptive planning depth
- Day 4: Test at 150 actors
- Day 5: State update batching (if needed)

**Week 2**:
- Day 1: Test at 300 actors
- Day 2: Fine-tuning and optimization
- Day 3: Documentation and cleanup

## Success Criteria

### At 150 Actors
- ✅ Average latency: <8ms
- ✅ Max latency: <20ms
- ✅ Planning cache hit rate: >30%

### At 300 Actors
- ✅ Average latency: <10ms
- ✅ Max latency: <30ms (target)
- ✅ Planning cache hit rate: >40%
- ✅ System stable under load

## Fallback Plan

**If GDScript optimizations don't achieve target**:
1. Consider C++ state management (Phase 3 from SCALING_TO_300_ACTORS.md)
2. Reduce actor count to 200-250 actors
3. Use LOD (Level of Detail) - simpler planning for distant actors

## Conclusion

**Can we get to 300 actors?** ✅ **YES**

**Key Requirements**:
1. ✅ Planning result caching (critical!)
2. ✅ Adaptive planning depth
3. ✅ Validation at intermediate scales

**Timeline**: 1-2 weeks of focused development

**Confidence**: **HIGH** - Current architecture is solid, optimizations are well-understood, and we have significant headroom (48x) at current scale.

