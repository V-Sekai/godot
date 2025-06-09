# Goal Task Planner
Determines a [PlannerPlan] to accomplish the todo list from the provided state.

The todo list is an array of goals, [PlannerMultigoal], tasks, and actions.

A goal is defined as one predicate-subject-object triple.

A [PlannerMultigoal] is a state [Dictionary] of predicate-subject-object triples.

Tasks can accept any number of arguments but only return either false or a series of goals, [PlannerMultigoal], tasks, and actions.

Actions can accept any number of arguments but only return the state of predicate-subject-object triples.

The return value is a [Variant], which means it could be of any type. In this case, it returns either false or an array of actions.
