# Magical Girls College Domain - Testing Guide

This directory contains a split GDScript domain implementation for testing the goal_task_planner module.

## File Structure

-   `helpers.gd` - Helper functions for state manipulation
-   `actions.gd` - All action functions
-   `task_methods.gd` - Task method functions
-   `unigoal_methods.gd` - Unigoal method functions
-   `multigoal_methods.gd` - Multigoal method functions
-   `domain.gd` - Main domain class (facade that re-exports all functions)
-   `test.gd` - Full test suite
-   `test_sims_scenarios.gd` - Sims-style scenario tests
-   `test_syntax.gd` - Syntax validation test
-   `simulate_house_actors.gd` - Multi-agent simulation (1200+ actors)

## Documentation

-   `SIMS_SCENARIOS.md` - Documentation of Sims scenarios
-   `LITERATURE_REVIEW.md` - HTN planning optimization literature review
-   `SCALING.md` - Scaling to 300+ actors analysis
-   `OPTIMIZATIONS.md` - Implemented optimizations summary
-   `PERFORMANCE.md` - Performance analysis and metrics

## How to Test

### Method 1: Run from Godot Project Root (Recommended)

From the Godot project root directory:

```bash
cd /Users/ernest.lee/Desktop/godot
godot --headless --script modules/goal_task_planner/tests/domains/magical_girls_college/test.gd
```

### Method 2: Run with Full Path

```bash
godot --headless --script /Users/ernest.lee/Desktop/godot/modules/goal_task_planner/tests/domains/magical_girls_college/test.gd
```

### Method 3: Run Specific Test Function

You can modify `test.gd` to run only specific tests by commenting out tests in `run_all_tests()`:

```gdscript
func run_all_tests():
    test_simple_planning()  # Only run this test
    # test_setup_personas()
    # test_setup_allocentric_facts()
    # ...
```

### Method 4: Test in Godot Editor

1. Open the Godot project
2. Open `modules/goal_task_planner/tests/domains/magical_girls_college/test.gd`
3. Run the script from the editor (F5 or Run button)

## Expected Output

When tests pass, you should see:

```
=== Magical Girls College GDScript Tests ===

Test: Setup personas with identity types and capabilities
Test: Setup allocentric facts - terrain, objects, events, positions
...

=== Test Results ===
Passed: 25
Failed: 0
Total: 25

✅ All tests passed!
```

## Troubleshooting

### Error: "Could not preload resource script"

-   Make sure you're running from the Godot project root
-   Check that all files are in the correct directory structure

### Error: "Could not find type PlannerDomain"

-   The test needs access to the goal_task_planner module
-   Make sure you're running from the project root where the module is compiled

### Module Not Found

-   Ensure the goal_task_planner module is compiled
-   Rebuild the Godot project if needed

## Quick Test

To quickly verify the split files work (syntax check only):

```bash
cd /Users/ernest.lee/Desktop/godot
godot --headless --script modules/goal_task_planner/tests/domains/magical_girls_college/test_syntax.gd
```

This will verify all split files can be loaded without errors.

## Full Test (Requires Compiled Module)

The full test suite requires the goal_task_planner module to be compiled into Godot:

```bash
cd /Users/ernest.lee/Desktop/godot
# First, compile Godot with the module
scons -j8

# Then run the full test
./bin/godot.linuxbsd.editor.dev.x86_64 --headless --script modules/goal_task_planner/tests/domains/magical_girls_college/test.gd
```

Or if using a pre-built Godot with the module:

```bash
godot --headless --script modules/goal_task_planner/tests/domains/magical_girls_college/test.gd
```

## Sims-Style Scenario Tests

Test common Sims game scenarios:

```bash
cd /Users/ernest.lee/Desktop/godot
godot --headless --script modules/goal_task_planner/tests/domains/magical_girls_college/test_sims_scenarios.gd
```

See `SIMS_SCENARIOS.md` for detailed descriptions of each scenario:

-   Starving and Exhausted (multiple critical needs)
-   Broke but Wants Fun (resource constraints)
-   Cook but Wrong Location (location requirements)
-   Dirty and Tired (multiple needs with constraints)
-   Partial Money (method selection with budget)
-   Social Need (interaction requirements)
-   Everything Low (complex multi-goal with all constraints)
