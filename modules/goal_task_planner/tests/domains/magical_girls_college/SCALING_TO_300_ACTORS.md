# Scaling to 300 Actors (InZOI Scale) - Analysis

## Current Architecture

### Already in C++ ✅
- **HTN Planner (`PlannerPlan::find_plan`)**: Core planning algorithm is in C++
- **PlannerDomain**: Domain definition and method matching
- **Solution Graph**: Graph operations and backtracking
- **STN Solver**: Temporal constraint solving

### In GDScript (Simulation Layer)
- **Actor model**: Actor class, mailboxes, message passing
- **State management**: Dictionary operations, state copying
- **Action execution**: Plan execution helpers
- **Simulation loop**: WorkerThreadPool coordination

## Performance Analysis for 300 Actors

### Current Performance (48 actors, 4x oversubscription)
- **Average latency**: ~2.5ms
- **Max latency**: ~14ms
- **Per-actor average**: ~0.052ms (52 microseconds)
- **Per-actor max**: ~0.29ms (290 microseconds)

### Projected Performance (300 actors)
**Assumptions**: Linear scaling (optimistic), 4x oversubscription on 75 cores

#### Scenario 1: Linear Scaling (Unrealistic)
- **Average latency**: 2.5ms × (300/48) = **15.6ms** ✅ (within 30ms)
- **Max latency**: 14ms × (300/48) = **87.5ms** ❌ (exceeds 30ms by 2.9x)

#### Scenario 2: Realistic (Accounting for Overhead)
- **Average latency**: ~12-18ms (acceptable)
- **Max latency**: **50-80ms** (exceeds 30ms limit)
- **Problem**: Planning spikes don't scale linearly - concurrent planning causes exponential overhead

## Bottlenecks at 300 Actors Scale

### 1. HTN Planning (Already C++) - Still the Bottleneck
**Current**: Planning is in C++, but:
- Multiple actors planning simultaneously causes contention
- No planning result caching
- No concurrent planning limits
- Each planning call is independent (no batching)

**Impact at 300 actors**:
- If 10% plan per step: 30 actors planning simultaneously
- Each takes 5-20ms → total step time: 50-200ms ❌

**Solution**: Need planning coordination layer (can be GDScript)

### 2. State Dictionary Operations (GDScript)
**Current**:
- `state.duplicate(true)` - deep copy on every plan execution
- Multiple dictionary lookups per actor
- Nested dictionary access: `state["needs"][persona_id]`

**Impact at 300 actors**:
- 300 deep copies per step (if all execute plans)
- Dictionary operations: O(n) where n = number of actors
- Estimated overhead: 5-10ms per step

**Solution**: Optimize in GDScript OR move to C++

### 3. Message Processing (GDScript)
**Current**:
- Ring buffer operations (lockless, but still overhead)
- Message creation/processing per actor

**Impact at 300 actors**:
- 300 message creations per step
- Ring buffer operations: minimal overhead (already optimized)
- Estimated overhead: 1-2ms per step

**Solution**: Message pooling (can stay in GDScript)

### 4. WorkerThreadPool Coordination (GDScript)
**Current**:
- Waits for slowest actor
- No timeout mechanism

**Impact at 300 actors**:
- One slow actor (20ms planning) blocks entire step
- Estimated overhead: 0-20ms (depends on slowest actor)

**Solution**: Timeout mechanism, async processing (can stay in GDScript)

## Optimization Strategy: GDScript First, C++ if Needed

### Phase 1: GDScript Optimizations (High Impact, Low Effort)

#### 1.1 Limit Concurrent Planning
```gdscript
# Planning semaphore - limit to 4-8 concurrent planning operations
var planning_semaphore = Semaphore.new(4)  # Max 4 actors planning at once

func process_time_tick(current_time: float) -> void:
    # ... existing code ...
    if critical_needs.size() > 0:
        if planning_semaphore.try_acquire():
            # Planning allowed
            var result = plan.find_plan(state, critical_needs)
            planning_semaphore.release()
        else:
            # Skip planning this step, retry next interval
            return
```
**Expected impact**: Reduces max latency spikes by 60-80%

#### 1.2 Optimize State Copying
```gdscript
# Instead of: state.duplicate(true)
# Use: shallow copy with selective deep copy
func execute_plan_helper(state: Dictionary, plan_actions: Array, persona_id: String) -> Dictionary:
    var new_state = state.duplicate(false)  # Shallow copy
    # Only deep copy what changes
    new_state["needs"] = state["needs"].duplicate(true)
    new_state["money"] = state["money"].duplicate(true)
    # ... rest of execution ...
```
**Expected impact**: Reduces state copying overhead by 50-70%

#### 1.3 Cache Dictionary Lookups
```gdscript
func get_critical_needs_helper(state: Dictionary, persona_id: String) -> Array:
    var needs = state["needs"][persona_id]  # Cache this
    var threshold = 55
    # Use cached 'needs' instead of repeated lookups
```
**Expected impact**: Reduces lookup overhead by 20-30%

#### 1.4 Message Pooling
```gdscript
var message_pool: Array = []
const POOL_SIZE = 100

func get_message() -> ActorMessage:
    if message_pool.size() > 0:
        return message_pool.pop_back()
    return ActorMessage.new()

func return_message(msg: ActorMessage):
    msg.reset()  # Clear message data
    if message_pool.size() < POOL_SIZE:
        message_pool.append(msg)
```
**Expected impact**: Reduces allocation overhead by 30-40%

**Combined Phase 1 Impact**:
- **Max latency**: 87ms → **25-35ms** (still may exceed 30ms)
- **Average latency**: 15.6ms → **10-12ms** ✅

### Phase 2: Planning Coordination (Medium Impact, Medium Effort)

#### 2.1 Planning Result Caching
```gdscript
# Cache planning results for similar states
var planning_cache: Dictionary = {}  # state_hash -> plan_result

func get_cached_plan(state: Dictionary, critical_needs: Array) -> Variant:
    var state_hash = hash_state(state, critical_needs)
    return planning_cache.get(state_hash, null)

func cache_plan(state: Dictionary, critical_needs: Array, result: PlannerResult):
    var state_hash = hash_state(state, critical_needs)
    planning_cache[state_hash] = result
    # Limit cache size to prevent memory bloat
    if planning_cache.size() > 1000:
        # Remove oldest entries
        var keys = planning_cache.keys()
        for i in range(100):
            planning_cache.erase(keys[i])
```
**Expected impact**: Reduces redundant planning by 30-50%

#### 2.2 Batch Planning
```gdscript
# Collect planning requests, batch process them
var pending_planning: Array = []

func queue_planning(actor: Actor, state: Dictionary, critical_needs: Array):
    pending_planning.append({"actor": actor, "state": state, "needs": critical_needs})

func process_batched_planning():
    # Process in smaller batches (4-8 at a time)
    var batch_size = 4
    for i in range(0, pending_planning.size(), batch_size):
        var batch = pending_planning.slice(i, i + batch_size)
        # Process batch in parallel
        # ...
```
**Expected impact**: Better load distribution, reduces spikes

**Combined Phase 2 Impact**:
- **Max latency**: 25-35ms → **20-28ms** (closer to target)
- **Average latency**: 10-12ms → **8-10ms** ✅

### Phase 3: C++ Optimization (If Needed)

#### 3.1 State Management in C++
**What to move**:
- State dictionary operations
- State copying/updating
- Need decay calculations

**Why C++**:
- Dictionary operations are faster in C++
- Can use more efficient data structures (arrays, structs)
- Better memory management

**Expected impact**:
- State operations: 5-10ms → **1-2ms** (5x improvement)
- Overall latency: 20-28ms → **15-22ms** ✅

#### 3.2 Actor Message System in C++
**What to move**:
- Ring buffer operations
- Message creation/pooling
- Mailbox processing

**Why C++**:
- Lockless operations can be more efficient
- Better memory alignment
- Reduced GDScript overhead

**Expected impact**:
- Message overhead: 1-2ms → **0.3-0.5ms** (3-4x improvement)
- Overall latency: 15-22ms → **14-20ms** ✅

## Recommended Approach

### ✅ **Start with GDScript Optimizations (Phase 1 + 2)**
**Effort**: Medium (1-2 weeks)
**Impact**: High (should get to 20-28ms max latency)
**Risk**: Low (can test incrementally)

**Key optimizations**:
1. Limit concurrent planning (semaphore)
2. Optimize state copying (shallow + selective deep)
3. Cache dictionary lookups
4. Planning result caching
5. Message pooling

### ⚠️ **C++ Only If Needed (Phase 3)**
**Effort**: High (2-4 weeks)
**Impact**: Medium (additional 5-8ms improvement)
**Risk**: Medium (requires C++ expertise, more complex)

**When to consider C++**:
- If Phase 1+2 don't get us under 30ms consistently
- If we need to scale beyond 300 actors
- If we want to optimize for lower-end hardware

## Realistic Projection for 300 Actors

### With Phase 1 + 2 Optimizations (GDScript only)
- **Average latency**: 8-10ms ✅
- **Max latency**: 20-28ms ✅ (within 30ms with small buffer)
- **Feasible**: **YES** - should work with GDScript optimizations

### With Phase 3 (C++ optimizations)
- **Average latency**: 6-8ms ✅
- **Max latency**: 14-20ms ✅ (comfortable buffer)
- **Feasible**: **YES** - comfortable performance

## Conclusion

### Can we scale to 300 actors?
**YES**, but we need optimizations.

### Do we need C++?
**Not necessarily** - GDScript optimizations (Phase 1+2) should be sufficient:
- Concurrent planning limits (critical!)
- State copying optimization
- Planning result caching
- Message pooling

### When to use C++?
Only if GDScript optimizations don't achieve target latency, or if we need to scale beyond 300 actors.

### Recommended Path
1. ✅ **Implement Phase 1 optimizations** (concurrent planning limits, state copying)
2. ✅ **Test with 100-150 actors** (intermediate scale)
3. ✅ **Implement Phase 2 optimizations** (caching, batching)
4. ✅ **Test with 300 actors**
5. ⚠️ **Consider Phase 3 (C++)** only if needed

**Bottom line**: The HTN planner is already in C++ (the main bottleneck). The simulation layer can be optimized in GDScript first, with C++ as a fallback if needed.
