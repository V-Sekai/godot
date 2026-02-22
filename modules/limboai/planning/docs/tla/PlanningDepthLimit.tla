---- MODULE PlanningDepthLimit ----

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxDepth, MaxNodes

VARIABLES
    iteration,        \* Current iteration count
    graph_size,       \* Number of nodes in graph
    planning_complete \* Whether planning completed successfully

\* Initialize
Init ==
    /\ iteration = 0
    /\ graph_size = 1  \* Start with root node
    /\ planning_complete = FALSE

\* Planning step - increment iteration and potentially add nodes
PlanningStep ==
    /\ iteration < MaxDepth
    /\ iteration' = iteration + 1
    /\ \/ /\ graph_size' = graph_size + 1  \* Add one node (efficient)
       /\ UNCHANGED planning_complete
       \/ /\ graph_size' = graph_size + 2  \* Add two nodes (less efficient)
          /\ UNCHANGED planning_complete
       \/ /\ graph_size' = graph_size + 3  \* Add three nodes (inefficient)
          /\ UNCHANGED planning_complete
       \/ /\ graph_size' = graph_size  \* No progress (stuck)
          /\ UNCHANGED planning_complete

\* Planning completes successfully
PlanningComplete ==
    /\ iteration < MaxDepth
    /\ planning_complete' = TRUE
    /\ UNCHANGED <<iteration, graph_size>>

\* Planning hits depth limit
PlanningDepthLimit ==
    /\ iteration = MaxDepth
    /\ planning_complete' = FALSE
    /\ UNCHANGED <<iteration, graph_size>>

\* Next state relation
Next ==
    \/ PlanningStep
    \/ PlanningComplete
    \/ PlanningDepthLimit

\* Safety: Planning should complete before hitting depth limit
PlanningCompletesBeforeLimit ==
    ~(iteration = MaxDepth /\ ~planning_complete)

\* Efficiency: Graph size should be reasonable (not exponential growth)
ReasonableGraphSize ==
    graph_size <= iteration * 2  \* At most 2 nodes per iteration on average

=============================================================================
