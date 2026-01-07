# Scaling to 300+ Actors

## Current Status

**Performance at 1200 actors**:
- Average latency: ~15-18ms
- Max latency: ~26-30ms (within target)
- Tested maximum: 1286 actors
- Architecture: Actor model with lockless mailboxes

## Key Optimizations Implemented

### ✅ Planning Coordination
- Semaphore limits concurrent planning (max 4)
- Staggered planning intervals prevent spikes
- Adaptive throttling (skip if >25ms)

### ✅ Partial Replanning
- Uses `replan()` method for incremental planning
- Reduces full replan overhead
- ~30% replan ratio in practice

### ✅ C++ Optimizations
- HashMap/LocalVector for graph operations
- Internal node structures for direct access
- Optimized `_process_node_iterative` and `_select_best_method`

### ✅ Temporal Optimizations
- STN-based plan extraction (temporal ordering)
- Lazy STN validation (skip when no temporal constraints)

## Architecture

**C++ Layer** (Core):
- HTN Planner (`PlannerPlan::find_plan`)
- Solution Graph operations
- STN Solver
- Domain matching

**GDScript Layer** (Simulation):
- Actor model with mailboxes
- State management
- Action execution
- WorkerThreadPool coordination

## Scaling Path

### Current: 1200 actors ✅
- Max latency: ~26-30ms
- Comfortable buffer for 30ms target

### Future: 3000+ actors
If needed, consider:
- Planning result caching (LRU cache)
- Message pooling
- State update batching

## Conclusion

**Current system scales to 1200+ actors** with max latency under 30ms. The actor model with lockless mailboxes and C++ planner optimizations provide excellent scalability.
