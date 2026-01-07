# Performance Analysis - Actor Model Simulation

## Current Performance Metrics
- **Max Latency**: ~22-24ms (within 15-30ms target with buffer)
- **Average Latency**: ~2.4-2.8ms
- **Actors**: 72 (6x oversubscription on 12 cores)
- **Thread Efficiency**: ~70-73%

## Identified Bottlenecks

### 1. **HTN Planning (`plan.find_plan()`) - PRIMARY BOTTLENECK** ⚠️
**Location**: `Actor.process_time_tick()` line 149

**Impact**: HIGH
- This is the most expensive operation in the simulation
- HTN planning involves recursive search through task networks
- When multiple actors plan simultaneously, this causes latency spikes
- Current max_depth is 10 (reduced from 15), but still expensive

**Evidence**:
- Peak latency (22-24ms) occurs when multiple actors trigger planning
- Adaptive throttling skips planning if >25ms, indicating planning can be slow
- Planning is the only operation with explicit timing and throttling

**Recommendations**:
1. **Further reduce planning depth** from 10 to 7-8 for faster planning
2. **Cache planning results** for similar states (state hashing)
3. **Defer planning** - batch planning operations across multiple steps
4. **Limit concurrent planning** - use a semaphore to limit how many actors plan simultaneously
5. **Use simpler planning heuristics** for non-critical needs

### 2. **State Dictionary Deep Copy (`state.duplicate(true)`)**
**Location**: `execute_plan_helper()` line 250

**Impact**: MEDIUM
- Creates a full deep copy of the entire state dictionary on every plan execution
- With 72 actors, this happens frequently
- Dictionary operations in GDScript can be slower than expected

**Recommendations**:
1. **Shallow copy with selective deep copy** - only deep copy nested dictionaries that change
2. **Immutable state updates** - use a more efficient state update pattern
3. **Pre-allocate state structures** to reduce allocation overhead

### 3. **Dictionary Lookups and Updates**
**Location**: Throughout `decay_needs_helper()`, `get_critical_needs_helper()`, `execute_plan_helper()`

**Impact**: MEDIUM
- Multiple dictionary lookups per actor per step:
  - `state["needs"][persona_id]` - accessed multiple times
  - `state["money"][persona_id]` - accessed multiple times
  - `state["location"][persona_id]` - accessed in get_critical_needs

**Recommendations**:
1. **Cache frequently accessed values** in local variables
2. **Flatten state structure** - use arrays instead of nested dictionaries where possible
3. **Use typed dictionaries** if available in GDScript

### 4. **Message Processing Overhead**
**Location**: `Actor.process_messages()` and `ActorMailbox` operations

**Impact**: LOW-MEDIUM
- Ring buffer operations are lockless but still have overhead
- Message creation (`ActorMessage.new()`) happens every step for every actor
- Message processing loop iterates through mailbox

**Recommendations**:
1. **Reuse message objects** - pool ActorMessage instances
2. **Batch message processing** - process multiple messages in one go
3. **Skip message creation** when possible - direct function calls for time ticks

### 5. **WorkerThreadPool Coordination**
**Location**: `simulation_step()` line 545-546

**Impact**: LOW-MEDIUM
- `wait_for_group_task_completion()` waits for the slowest actor
- If one actor takes 20ms to plan, the entire step waits

**Recommendations**:
1. **Timeout mechanism** - don't wait indefinitely for slow actors
2. **Process actors in batches** - smaller batches with timeouts
3. **Async processing** - allow actors to complete asynchronously

## Optimization Priority

### High Priority (Immediate Impact)
1. **Limit concurrent planning** - Use a semaphore/queue to limit simultaneous planning to 2-4 actors
2. **Reduce planning depth** - Further reduce from 10 to 7-8
3. **Cache state lookups** - Store frequently accessed values in local variables

### Medium Priority (Moderate Impact)
4. **Optimize state copying** - Use shallow copy with selective deep copy
5. **Message pooling** - Reuse ActorMessage objects
6. **Batch planning checks** - Stagger planning more aggressively

### Low Priority (Small Impact)
7. **Flatten state structure** - Use arrays instead of nested dictionaries
8. **Async actor processing** - Don't wait for all actors synchronously

## Expected Improvements

With the high-priority optimizations:
- **Max latency**: Should drop from ~22ms to ~15-18ms
- **Average latency**: Should drop from ~2.8ms to ~2.0ms
- **Planning overhead**: Reduced by 30-40%
- **Allow higher oversubscription**: Could potentially go from 6x to 8-10x

## Code Locations for Optimization

1. **Planning throttling**: `simulate_house_actors.gd:146-161`
2. **State copying**: `simulate_house_actors.gd:250`
3. **Dictionary lookups**: `simulate_house_actors.gd:206-248`
4. **Message creation**: `simulate_house_actors.gd:505-510`
5. **Worker coordination**: `simulate_house_actors.gd:543-546`
