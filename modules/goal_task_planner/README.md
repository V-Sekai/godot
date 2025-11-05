# Goal Task Planner
Determines a [PlannerPlan] to accomplish the todo list from the provided state.

The todo list is an array of goals, [PlannerMultigoal], tasks, and actions.

A goal is defined as one predicate-subject-object triple.

A [PlannerMultigoal] is a state [Dictionary] of predicate-subject-object triples.

Tasks can accept any number of arguments but only return either false or a series of goals, [PlannerMultigoal], tasks, and actions.

Actions can accept any number of arguments but only return the state of predicate-subject-object triples.

The return value is a [Variant], which means it could be of any type. In this case, it returns either false or an array of actions.

## Temporal Constraints

Actions, tasks, and goals can include optional temporal metadata specifying timing constraints. Temporal constraints are provided as a Dictionary with a "temporal_constraints" key containing:
- `start_time`: int64_t absolute time in microseconds since Unix epoch
- `end_time`: int64_t absolute time in microseconds since Unix epoch
- `duration`: int64_t duration in microseconds

Actions without temporal metadata can occur at any time and are not constrained by the Simple Temporal Network (STN). Actions with temporal metadata are added to the STN and their timing constraints are validated for consistency. If temporal constraints are inconsistent, planning fails.
