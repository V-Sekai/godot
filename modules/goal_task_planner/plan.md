# Implementation Plan: Missing IPyHOP Features

This plan outlines the implementation of missing features from IPyHOP to improve the Goal Task Planner module.

## 8 Implementation Trajectories

### Trajectory 1: Incremental API-First (Sequential Quick Wins)
**Narrative**: Developer adds simple public APIs first (blacklist, iterations), then builds foundational data structures (node tags), enabling replanning and simulation. Each feature validates independently before moving to advanced probabilistic features.

**Phases**:
- **Short (0-2 days)**: Public blacklist API, iteration counter, multigoal tags
- **Mid (2-5 days)**: Node tagging system, replan API, simulate method
- **Long**: N/A (advanced features removed per core vs advanced rules)

**Pros**: Low risk, incremental validation, clear dependencies  
**Cons**: Sequential timeline, later features wait for foundations

**Judgment**:
- Feasibility: 10/10 (straightforward API additions, clear dependencies)
- Timeline: 8/10 (8 days total, sequential but predictable)
- Alignment: 10/10 (matches IPyHOP API, maintains backward compatibility)
- Risk: 10/10 (low risk, each step independently testable)
- **Total: 38/40**

### Trajectory 2: Test-Driven Development
**Narrative**: Developer writes tests for all features first, then implements to pass. Ensures robust implementation with comprehensive coverage from the start.

**Phases**:
- **Short (0-2 days)**: Write test cases for core features, set up test infrastructure
- **Mid (2-5 days)**: Implement features to pass tests (API first, then data structures)
- **Long (5-6 days)**: Integration testing, documentation, cleanup

**Pros**: High test coverage, robust implementation, clear success criteria  
**Cons**: Longer initial phase, potential over-engineering

**Judgment**:
- Feasibility: 9/10 (requires understanding all features upfront)
- Timeline: 7/10 (9 days, test writing adds overhead)
- Alignment: 9/10 (ensures compatibility but may miss edge cases initially)
- Risk: 9/10 (lower risk due to tests, but slower feedback)
- **Total: 34/40**

### Trajectory 3: Parallel Feature Development
**Narrative**: Developer splits work across three parallel tracks: API enhancements, data structures, and advanced features. Teams work simultaneously to maximize speed.

**Phases**:
- **Short (0-1 day)**: Split into 2 tracks (APIs, data structures), set up branches
- **Mid (1-4 days)**: Parallel implementation across tracks
- **Long (4-5 days)**: Integration, conflict resolution, testing

**Pros**: Fastest completion, parallel work  
**Cons**: Integration complexity, potential conflicts, coordination overhead

**Judgment**:
- Feasibility: 7/10 (requires careful coordination, dependency management)
- Timeline: 9/10 (7 days total, fastest approach)
- Alignment: 8/10 (may have integration issues, needs careful API design)
- Risk: 6/10 (higher risk of conflicts, integration problems)
- **Total: 30/40**

### Trajectory 4: Foundation-First (Data Structures Before APIs)
**Narrative**: Developer builds data structure enhancements first (node tags, multigoal tags), then implements APIs that depend on them. Ensures solid foundation before exposing functionality.

**Phases**:
- **Short (0-2 days)**: Node tagging system, multigoal tag support
- **Mid (2-5 days)**: Replan API (uses node tags), simulate method, public APIs
- **Long**: N/A (advanced features removed)

**Pros**: Solid foundation, no refactoring needed later  
**Cons**: Delayed API availability, longer feedback loop

**Judgment**:
- Feasibility: 9/10 (clear dependencies, but delayed API access)
- Timeline: 8/10 (8 days, similar to Trajectory 1)
- Alignment: 9/10 (good structure, but users wait for APIs)
- Risk: 9/10 (low risk, but slower user feedback)
- **Total: 35/40**

### Trajectory 5: Minimal Viable Product (MVP)
**Narrative**: Developer implements only essential features for IPyHOP compatibility: replan, simulate, and public blacklist. Skips advanced probabilistic features for now.

**Phases**:
- **Short (0-2 days)**: Node tagging, public blacklist API
- **Mid (2-4 days)**: Replan API, simulate method
- **Long (4-5 days)**: Testing, documentation, polish

**Pros**: Fastest to core functionality, focused scope  
**Cons**: Missing advanced features, may need to revisit later

**Judgment**:
- Feasibility: 10/10 (smaller scope, easier to complete)
- Timeline: 10/10 (5 days, fastest to value)
- Alignment: 7/10 (covers core IPyHOP features, but incomplete)
- Risk: 8/10 (low risk, but incomplete feature set)
- **Total: 35/40**

### Trajectory 6: Advanced Features First
**Narrative**: Developer implements probabilistic features first (action models, Monte Carlo), then builds APIs on top. Assumes advanced features are most valuable.

**Phases**:
- **Short (0-2 days)**: Node tagging, multigoal tags
- **Mid (2-5 days)**: Replan API, simulate method, public APIs
- **Long**: N/A (advanced features removed)

**Pros**: Advanced features available early  
**Cons**: Complex features first, harder to test incrementally

**Judgment**:
- Feasibility: 7/10 (complex features first, harder to validate)
- Timeline: 7/10 (8 days, similar timeline but riskier)
- Alignment: 8/10 (advanced features, but may not be most needed)
- Risk: 6/10 (higher risk, complex features without foundation)
- **Total: 28/40**

### Trajectory 7: API-Centric (User-Facing First)
**Narrative**: Developer prioritizes all user-facing APIs first (blacklist, iterations, replan, simulate, method replacement), then implements internal data structures and advanced features.

**Phases**:
- **Short (0-2 days)**: All public APIs (blacklist, iterations, replan, simulate)
- **Mid (2-4 days)**: Data structures (node tags, multigoal tags) to support APIs
- **Long**: N/A (advanced features removed)

**Pros**: Users get APIs immediately, can use features as they're built  
**Cons**: May need to refactor APIs when data structures are added

**Judgment**:
- Feasibility: 8/10 (APIs may need refactoring when data structures added)
- Timeline: 8/10 (8 days, but APIs available early)
- Alignment: 8/10 (good user experience, but may have technical debt)
- Risk: 7/10 (APIs may change when foundations are built)
- **Total: 31/40**

### Trajectory 8: Hybrid Incremental (Quick Wins + Foundation)
**Narrative**: Developer starts with quick wins (blacklist, iterations) for immediate value, then builds foundation (node tags), then implements dependent features (replan, simulate), finishing with advanced features.

**Phases**:
- **Short (0-1 day)**: Quick wins (blacklist, iterations, multigoal tags)
- **Mid (1-4 days)**: Foundation (node tags), then replan and simulate
- **Long**: N/A (advanced features removed)

**Pros**: Immediate value, then solid foundation, then advanced features  
**Cons**: Slightly longer than pure sequential

**Judgment**:
- Feasibility: 10/10 (balanced approach, clear progression)
- Timeline: 8/10 (8 days, good balance)
- Alignment: 10/10 (immediate value + solid foundation + completeness)
- Risk: 9/10 (low risk, incremental validation)
- **Total: 37/40**

## Ranking

1. **Trajectory 1: Incremental API-First** (38/40) - Best balance of feasibility, timeline, alignment, and risk. Clear dependencies, incremental validation.
2. **Trajectory 8: Hybrid Incremental** (37/40) - Similar to #1 but with immediate quick wins, slightly better user experience.
3. **Trajectory 4: Foundation-First** (35/40) - Solid approach, but delayed API availability.
4. **Trajectory 5: Minimal Viable Product** (35/40) - Fastest to core value, but incomplete.
5. **Trajectory 2: Test-Driven Development** (34/40) - Robust but slower.
6. **Trajectory 7: API-Centric** (31/40) - Good UX but technical debt risk.
7. **Trajectory 3: Parallel Development** (30/40) - Fast but risky.
8. **Trajectory 6: Advanced Features First** (28/40) - Complex features first, highest risk.

## Recommended: Trajectory 1 (Incremental API-First)

**Rationale**: Provides the best balance of safety, predictability, and completeness. Each feature builds on previous work, enabling incremental validation and testing. Clear dependencies minimize risk while maintaining IPyHOP compatibility.

## Feature Details

## Priority 1: Core API Enhancements

### 1.1 Public Blacklist API
**Status**: Not Started  
**Effort**: Low  
**Files**: `plan.h`, `plan.cpp`

- Add public method `blacklist_command(Variant p_command)` to `PlannerPlan`
- Expose via `_bind_methods()` for GDScript access
- Currently only available as private `_blacklist_command()`

### 1.2 Iteration Counter Access
**Status**: Not Started  
**Effort**: Low  
**Files**: `plan.h`, `plan.cpp`

- Add `int iterations` member to `PlannerPlan`
- Track iterations in `_planning_loop_recursive`
- Add `get_iterations()` public method
- Expose via `_bind_methods()`

### 1.3 Replanning API (`replan` method)
**Status**: Not Started  
**Effort**: Medium  
**Files**: `plan.h`, `plan.cpp`, `backtracking.h`, `backtracking.cpp`

- Add `Ref<PlannerResult> replan(Dictionary p_state, int p_fail_node_id)` method
- Implement `_post_failure_modify()` helper (similar to IPyHOP)
- Mark nodes as 'old' vs 'new' (requires node tagging)
- Reopen nodes from failure point up to root
- Clear descendants of reopened nodes
- Call `_planning_loop_recursive` from the failure point
- Extract only 'new' actions in result

### 1.4 Plan Simulation (`simulate` method)
**Status**: Not Started  
**Effort**: Low  
**Files**: `plan.h`, `plan.cpp`

- Add `Array simulate(Dictionary p_state, int p_start_ind = 0)` method
- Execute actions from `sol_plan` starting at `p_start_ind`
- Return array of state dictionaries (one per action execution)
- Use domain's action dictionary to execute actions
- Handle action failures gracefully

## Priority 2: Data Structure Enhancements

### 2.1 Node Tagging System
**Status**: Not Started  
**Effort**: Medium  
**Files**: `solution_graph.h`, `plan.cpp`, `backtracking.cpp`

- Add `tag` field to solution graph nodes (String: "new" or "old")
- Initialize all new nodes with tag "new"
- During replanning, mark existing nodes as "old"
- Use tags to filter which actions to extract in replan results

### 2.2 Multigoal Tag Support
**Status**: Not Started  
**Effort**: Low  
**Files**: `multigoal.h`, `multigoal.cpp`, `domain.h`, `domain.cpp`

- Add `goal_tag` support to multigoal arrays
- Store tag in metadata or wrapper dictionary
- Update `PlannerMultigoal` to handle tags
- Update domain to match multigoals by tag

## Implementation Order (Core Features Only)

1. **Phase 1** (Quick wins):
   - 1.1 Public Blacklist API
   - 1.2 Iteration Counter Access
   - 2.2 Multigoal Tag Support

2. **Phase 2** (Core functionality):
   - 2.1 Node Tagging System
   - 1.3 Replanning API
   - 1.4 Plan Simulation

**Note**: Advanced features (probabilistic models, Monte Carlo executor, method replacement, state convenience methods) have been removed per project rules:
- Focus on common use cases (80% scenarios)
- Avoid adding complex specialized features unless frequently needed
- Users can implement probabilistic execution externally if needed
- Simple features (method replacement, state methods) can be worked around

## Testing Strategy

- Add tests for each new feature in `test_planner_components.h` or new test files
- Verify IPyHOP compatibility for replanning and simulation
- Ensure backward compatibility with existing code

## Notes

- All new public methods should be exposed via `_bind_methods()` for GDScript access
- Follow Godot coding conventions and use `Ref<>` for object references
- Update `AGENTS.md` with new features
- Update documentation XML files for new methods

