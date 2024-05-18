# Magical College Town Apartment Design Planner

## Special Rules

- State is represented as a two-layer nested dictionary.
- Only primitives (actions) can change the state.
- Tasks are a combination of [goals, other tasks, primitives].
- Goals represent the desired target state.
- Simple Temporal Networks (STNs) are used for scheduling and avoiding physical overlaps.
- An existing `path_find` function using A\* algorithm is available for spatial reasoning and layout optimization.
- Footprint data can be used to create Constructive Solid Geometry (CSG) shapes for collision detection and gameplay programming.

## State Variables

- `apartment_state`: Represents apartment's state with `layout`, `room_list` (rooms with properties), and `furniture_positions`.

## Spatial Reasoning with Path Finding

- `path_find` function optimizes movement paths and spatial configuration, ensuring unobstructed movement and practical design.

## Avoiding Physical Overlaps

- Constraints prevent object overlap and optimize spatial layout.

## Temporal Arrangement of Primitives using STNs

- STNs schedule primitives execution and ensure logical and efficient action ordering.

## Primitives (Actions)

### `create_room(apartment_state, room_name, size, style, function, pivot, footprint, time)`

```gdscript
["create_room", "bedroom1", "medium", "modern", "sleep", {"pivot": [0, 0, 0], "footprint": [5, 5, 3]}, 10]
```

### `place_furniture(apartment_state, item, room_name, position, orientation, pivot, footprint, time)`

```gdscript
["place_furniture", "bed", "bedroom1", "north-wall", "facing-south", {"pivot": [2, 2, 0], "footprint": [3, 2, 1]}, 15]
```

### `change_layout(apartment_state, new_layout)`

## Tasks

- `design_room(apartment_state, room_name, size, style, function)`
- `redesign_apartment(apartment_state, new_layout, room_specifications)`

## Implementation of STNs and Path Finding

- STNs and `path_find` function dynamically used throughout design process for feasible scheduling and spatial arrangements.

## Example Scenario

```gdscript
var plan = planner.find_plan(apartment_state.duplicate(true), [
	["design_room", "bedroom1", "medium", "modern", "sleep"],
	["design_room", "bedroom2", "small", "vintage", "study"]
]);

assert_eq(plan, [
	["create_room", "bedroom1", "medium", "modern", "sleep", {"pivot": [0, 0, 0], "footprint": [5, 5, 3]}, 10],
	["place_furniture", "bed", "bedroom1", "north-wall", "facing-south", {"pivot": [2, 2, 0], "footprint": [3, 2, 1]}, 15],
	["place_furniture", "wardrobe", "bedroom1", "east-wall", "facing-west", {"pivot": [4, 2, 0], "footprint": [1, 2, 2]}, 20],
	["create_room", "bedroom2", "small", "vintage", "study", {"pivot": [6, 0, 0], "footprint": [4, 4, 3]}, 30],
	["place_furniture", "desk", "bedroom2", "west-wall", "facing-east", {"pivot": [7, 2, 0], "footprint": [2, 1, 1]}, 35],
	["place_furniture", "bookshelf", "bedroom2", "south-wall", "facing-north", {"pivot": [8, 3, 0], "footprint": [1, 2, 2]}, 40]
]);
```

## References

1. [Skyrim's Modular Level Design GDC 2013](http://blog.joelburgess.com/2013/04/skyrims-modular-level-design-gdc-2013.html?m=1)
