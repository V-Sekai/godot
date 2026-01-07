# Goal Task Planner

A [Hierarchical Task Network (HTN)](https://en.wikipedia.org/wiki/Hierarchical_task_network) planner for Godot Engine. Automatically generates plans to accomplish goals by decomposing high-level tasks into executable actions.

## What is This?

The Goal Task Planner helps you create AI that can figure out _how_ to accomplish goals. Instead of manually scripting every possible action sequence, you define:

-   **Actions**: Things that can be done (e.g., "move to location", "pick up item")
-   **Methods**: Ways to accomplish tasks (e.g., "to go to the store, first move to the door, then move to the store")
-   **Goals**: What you want to achieve (e.g., "be at the store", "have 5 apples")

The planner automatically figures out the sequence of actions needed to achieve your goals.

## Quick Start

Here's a complete example:

```gdscript
extends Node

func _ready():
    # Create the domain (defines what actions and methods are available)
    var domain = PlannerDomain.new()

    # Create the planner
    var plan = PlannerPlan.new()
    plan.set_current_domain(domain)

    # Register actions
    var actions = []
    actions.append(callable(self, "action_move"))
    domain.add_actions(actions)

    # Register task methods
    var methods = []
    methods.append(callable(self, "method_go_to"))
    domain.add_task_methods("go_to", methods)

    # Define initial state
    var state = {"location": "home"}

    # Define what we want to accomplish
    var todo_list = ["go_to", "store"]

    # Find a plan
    var result = plan.find_plan(state, todo_list)

    if result.get_success():
        var actions = result.extract_plan()
        print("Plan found: ", actions)
        # Output: Plan found: [["move", "home", "store"]]
    else:
        print("Could not find a plan")

# Action: Move from one location to another
func action_move(state: Dictionary, from: String, to: String):
    if state.get("location") != from:
        return false  # Can't move if not at 'from' location

    var new_state = state.duplicate()
    new_state["location"] = to
    return new_state

# Method: How to accomplish "go_to" task
func method_go_to(state: Dictionary, destination: String):
    var current = state.get("location", "")
    if current == destination:
        return []  # Already there, nothing to do
    return [["move", current, destination]]  # Return action to move
```

## Core Classes

### [PlannerPlan]

The main planning class. Use this to generate plans from your goals.

**Key Methods:**

-   `find_plan(state: Dictionary, todo_list: Array) -> PlannerResult`: Generate a plan without executing it
-   `run_lazy_lookahead(state: Dictionary, todo_list: Array, max_tries: int = 10) -> PlannerResult`: Generate and execute plan incrementally
-   `run_lazy_refineahead(state: Dictionary, todo_list: Array) -> PlannerResult`: Graph-based planning with temporal constraints
-   `simulate(result: PlannerResult, state: Dictionary, start_ind: int = 0) -> Array`: Simulate plan execution
-   `replan(result: PlannerResult, state: Dictionary, fail_node_id: int) -> PlannerResult`: Replan from a failure point

### [PlannerDomain]

Defines the "rules" of your world - what actions are possible and how tasks can be accomplished.

**Key Methods:**

-   `add_actions(actions: Array[Callable])`: Register action functions
-   `add_task_methods(task_name: String, methods: Array[Callable])`: Register methods for a task
-   `add_unigoal_methods(goal_name: String, methods: Array[Callable])`: Register methods for a goal
-   `add_multigoal_methods(methods: Array[Callable])`: Register methods for multigoals

### [PlannerResult]

The result of a planning operation. Contains the plan, final state, and success status.

**Key Methods:**

-   `get_success() -> bool`: Whether planning succeeded
-   `extract_plan() -> Array`: Get the sequence of actions
-   `get_final_state() -> Dictionary`: Get the final state after planning
-   `find_failed_nodes() -> Array`: Find nodes that failed (useful for replanning)

## Defining Actions

Actions are functions that modify the world state. They must:

1. Accept a `Dictionary` as the first parameter (the current state)
2. Accept action-specific parameters after that
3. Return `false` if the action cannot be applied, or a new state `Dictionary` if successful

```gdscript
# Action: Pick up an item
func action_pickup(state: Dictionary, item: String, location: String):
    # Check preconditions
    if state.get("location") != location:
        return false  # Not at the right location

    if state.get("holding") != null:
        return false  # Already holding something

    if not state.get("items", {}).get(location, []).has(item):
        return false  # Item not at location

    # Apply action - return new state
    var new_state = state.duplicate()
    new_state["holding"] = item
    var items = new_state.get("items", {}).duplicate()
    var location_items = items.get(location, []).duplicate()
    location_items.erase(item)
    items[location] = location_items
    new_state["items"] = items

    return new_state
```

## Defining Methods

Methods decompose tasks, goals, and multigoals into subtasks. They must:

1. Accept a `Dictionary` as the first parameter (the current state)
2. Accept method-specific parameters after that
3. Return `false` if the method doesn't apply, or an `Array` of planner elements (goals, tasks, actions)

```gdscript
# Method: How to accomplish "get_item" task
func method_get_item(state: Dictionary, item: String):
    # If already holding it, we're done
    if state.get("holding") == item:
        return []

    # Otherwise, need to go to where it is and pick it up
    var item_location = find_item_location(state, item)
    if item_location == null:
        return false  # Item doesn't exist

    return [
        ["go_to", item_location],  # First go to location
        ["pickup", item, item_location]  # Then pick it up
    ]
```

## Planning Methods

### `find_plan()` - Standard Planning

Use this when you just need a plan without executing it:

```gdscript
var result = plan.find_plan(state, todo_list)
if result.get_success():
    var actions = result.extract_plan()
    # Execute actions yourself
    for action in actions:
        execute_action(action)
```

### `run_lazy_lookahead()` - Incremental Execution

Use this when you want to execute the plan as it's being generated:

```gdscript
var result = plan.run_lazy_lookahead(state, todo_list, max_tries=10)
if result.get_success():
    var final_state = result.get_final_state()
    # State has been updated by executed actions
    var executed_actions = result.extract_plan()
```

### `run_lazy_refineahead()` - Temporal Planning

Use this when you need temporal constraints (timing requirements):

```gdscript
var result = plan.run_lazy_refineahead(state, todo_list)
# Automatically validates timing constraints
```

## Temporal Constraints

The planner supports temporal planning with action durations and timing constraints. **Time is represented in absolute microseconds** (since Unix epoch).

### Setting Initial Time

```gdscript
# Set initial time for temporal planning
# Get current time range, modify it, and set it back
var time_range = plan.get_time_range()
time_range.set_start_time(1735732800000000)  # microseconds since Unix epoch
plan.set_time_range(time_range)
```

### Action Durations and Constraints

Attach temporal metadata to actions:

```gdscript
# Action with duration
var temporal_constraints = {
    "duration": 300000000  # 5 minutes in microseconds
}
var action = ["walk", "person", "from", "to"]
plan.attach_metadata(action, temporal_constraints)

# Action with timing requirements
var temporal_constraints = {
    "start_time": 1735732800000000,  # Must start at this time
    "end_time": 1735732805000000,    # Must finish by this time
    "duration": 5000000              # Takes 5 seconds
}
plan.attach_metadata(action, temporal_constraints)
```

The planner uses a Simple Temporal Network (STN) to validate timing constraints.

### Converting Time Formats

```gdscript
# Convert Unix time (seconds) to microseconds
var microseconds = PlannerTimeRange.unix_time_to_microseconds(1735732800.0)
# Get current time in microseconds
var now = PlannerTimeRange.now_microseconds()
```

## Entity Requirements

Require specific entities with capabilities to perform actions:

```gdscript
# Require a robot with "grasp" capability
var entity_constraints = {
    "type": "robot",
    "capabilities": ["move", "grasp"]
}
var action = ["pickup", "item"]
plan.attach_metadata(action, Dictionary(), entity_constraints)
```

## Additional Methods

### `blacklist_command()`

Prevent certain actions or methods from being used:

```gdscript
plan.blacklist_command(["move", "dangerous_area", "safe_area"])
```

### `reset()`

Reset all planner state for test isolation:

```gdscript
plan.reset()
```

## Advanced Features

### Plan Simulation

Preview state changes without executing:

```gdscript
var result = plan.find_plan(state, todo_list)
var state_sequence = plan.simulate(result, state, 0)
# Returns array of states: [initial_state, after_action_1, after_action_2, ...]
```

### Replanning

Replan from a failure point:

```gdscript
var failed_nodes = result.find_failed_nodes()
if failed_nodes.size() > 0:
    var fail_node_id = failed_nodes[0]["node_id"]
    var new_result = plan.replan(result, current_state, fail_node_id)
```

## Tips

1. **State Management**: Always use `state.duplicate()` when modifying state in actions
2. **Callables**: Wrap functions in `Callable` objects when registering with the domain
3. **Error Checking**: Always check `result.get_success()` before using `extract_plan()`
4. **Performance**: Use `run_lazy_lookahead()` with a low `max_tries` for complex domains
5. **Debugging**: Use `plan.set_verbose(level)` to see detailed planning information

## Documentation

-   **[Class Reference](doc_classes/)**: Full API documentation for all classes
-   **[AGENTS.md](AGENTS.md)**: Detailed technical documentation for developers
-   **[TLA+ Models](tla/README.md)**: Formal verification models

## License

See [LICENSE.md](LICENSE.md) for license information.
