# Sims-Style Game Scenarios

This document describes common scenarios from The Sims games and how the planner addresses them.

## Scenario 1: Starving and Exhausted

**Description**: Character has multiple critical needs (hunger: 20/100, energy: 15/100) and must satisfy both.

**Challenge**:

-   Multiple needs competing for attention
-   Limited resources (money, time)
-   Need to prioritize which need to address first

**Planner Solution**:

-   Tests multiple methods for each need
-   Backtracks if a method fails (e.g., not enough money for restaurant)
-   May need to satisfy one need before another (e.g., sleep before cooking)

**Code**: `test_scenario_starving_and_exhausted()`

## Scenario 2: Broke but Wants Fun

**Description**: Character has low fun (30/100) but no money (0). Wants to go to cinema (costs 15) but can't afford it.

**Challenge**:

-   Resource constraint (no money)
-   Must find alternative methods that don't require money
-   Tests method selection with constraints

**Planner Solution**:

-   Tries expensive method first (cinema) → fails due to money
-   Backtracks to free methods (games, streaming in dorm)
-   Selects methods that don't require money

**Code**: `test_scenario_broke_wants_fun()`

## Scenario 3: Cook but Wrong Location

**Description**: Character is hungry (30/100) and at library, but cooking requires being at dorm.

**Challenge**:

-   Location constraint (must be at dorm to cook)
-   Need to move before cooking
-   Alternative: use other methods that don't require movement

**Planner Solution**:

-   Tries cooking method → fails (wrong location)
-   Backtracks to either:
    -   Move to dorm then cook, OR
    -   Use other methods (mess hall, snack, restaurant) that don't require dorm
-   Selects most efficient path

**Code**: `test_scenario_cook_wrong_location()`

## Scenario 4: Dirty and Tired

**Description**: Character is dirty (25/100 hygiene) and tired (20/100 energy), currently at library.

**Challenge**:

-   Multiple needs with different location requirements
-   Shower requires dorm, nap can be at library
-   Must coordinate movement and actions

**Planner Solution**:

-   For hygiene: may need to move to dorm for shower, or use wash_hands (partial)
-   For energy: can nap at library (current location) or move to dorm for full sleep
-   Plans efficient sequence of actions

**Code**: `test_scenario_dirty_and_tired()`

## Scenario 5: Partial Money

**Description**: Character is hungry (25/100), has only 10 money. Can afford snack (5) but not restaurant (20).

**Challenge**:

-   Partial resource constraint
-   Must select methods within budget
-   May need multiple actions if one isn't enough

**Planner Solution**:

-   Tries restaurant → fails (not enough money)
-   Backtracks to snack or mess hall (free)
-   May need multiple snacks if one doesn't satisfy hunger fully
-   Tests recursive task satisfaction

**Code**: `test_scenario_partial_money()`

## Scenario 6: Social Need

**Description**: Character is lonely (20/100 social), needs interaction.

**Challenge**:

-   Social needs require other characters or activities
-   Multiple methods: talk, phone, club, group activities
-   Some methods may require location or money

**Planner Solution**:

-   Tries various social methods
-   May need to move to location (club)
-   May need money (phone call costs nothing, but group activities might)
-   Selects most appropriate method

**Code**: `test_scenario_social_need()`

## Scenario 7: Everything Low

**Description**: Character has ALL needs low (hunger: 30, energy: 25, social: 20, fun: 30, hygiene: 25), limited money (15), wrong location (library).

**Challenge**:

-   Complex multi-goal planning
-   Multiple constraints (money, location, time)
-   Deep backtracking required
-   May not be able to satisfy all needs with limited resources

**Planner Solution**:

-   Plans sequence to satisfy multiple needs
-   Prioritizes based on available resources
-   May need to move between locations
-   Tests deep backtracking when methods fail
-   May partially satisfy needs if resources run out

**Code**: `test_scenario_everything_low()`

## How the Planner Handles These

### Backtracking

When a method fails (e.g., not enough money, wrong location), the planner:

1. Marks the method as failed
2. Backtracks to try the next method
3. Continues until a working method is found or all methods exhausted

### Method Selection

The planner uses VSIDS (Variable State Independent Decaying Sum) heuristic to:

-   Prioritize methods that have worked before
-   Learn from successful plans
-   Adapt to constraints

### Resource Management

The planner:

-   Checks resource constraints before selecting methods
-   Backtracks if a method requires unavailable resources
-   Selects methods that fit within constraints

### Location Handling

The planner:

-   Checks location requirements for actions
-   Plans movement actions when needed
-   May choose methods that don't require movement if more efficient

## Running the Tests

```bash
cd /Users/ernest.lee/Desktop/godot
godot --headless --script modules/goal_task_planner/tests/domains/magical_girls_college/test_sims_scenarios.gd
```

Note: These tests require the compiled goal_task_planner module.
