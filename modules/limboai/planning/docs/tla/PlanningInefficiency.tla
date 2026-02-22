---- MODULE PlanningInefficiency ----

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxDepth, MaxActions

VARIABLES
    iteration,        \* Current iteration count
    action_count,     \* Number of actions in plan
    planning_complete,\* Whether planning completed successfully
    method_selected   \* Which method was selected (for tracking)

\* Method types
Methods == {"EfficientMethod", "InefficientMethod"}

\* Initialize
Init ==
    /\ iteration = 0
    /\ action_count = 0
    /\ planning_complete = FALSE
    /\ method_selected = "EfficientMethod"

\* Efficient method: adds 1 action per iteration
EfficientStep ==
    /\ iteration < MaxDepth
    /\ method_selected = "EfficientMethod"
    /\ iteration' = iteration + 1
    /\ action_count' = action_count + 1
    /\ UNCHANGED planning_complete

\* Inefficient method: adds 3 actions per iteration (creates more nodes)
InefficientStep ==
    /\ iteration < MaxDepth
    /\ method_selected = "InefficientMethod"
    /\ iteration' = iteration + 1
    /\ action_count' = action_count + 3
    /\ UNCHANGED planning_complete

\* Planning completes successfully
PlanningComplete ==
    /\ action_count >= 6  \* Minimum actions needed
    /\ action_count <= 50  \* Reasonable upper bound
    /\ planning_complete' = TRUE
    /\ UNCHANGED <<iteration, action_count, method_selected>>

\* Planning hits depth limit
PlanningDepthLimit ==
    /\ iteration = MaxDepth
    /\ planning_complete' = FALSE
    /\ UNCHANGED <<iteration, action_count, method_selected>>

\* VSIDS should prefer efficient methods
\* If VSIDS works, method_selected should be "EfficientMethod" more often
VSIDSPrefersEfficient ==
    \/ method_selected = "EfficientMethod"
    \/ action_count < 10  \* Early in planning, both methods might be tried

\* Next state relation
Next ==
    \/ EfficientStep
    \/ InefficientStep
    \/ PlanningComplete
    \/ PlanningDepthLimit

\* Safety: Planning should complete before hitting depth limit
PlanningCompletesBeforeLimit ==
    ~(iteration = MaxDepth /\ ~planning_complete)

\* Efficiency: Action count should be reasonable
ReasonableActionCount ==
    action_count <= iteration * 2  \* At most 2 actions per iteration on average

\* VSIDS effectiveness: If VSIDS works, we should prefer efficient methods
VSIDSEffective ==
    \A i \in 0..iteration :
        IF i > 10  \* After initial exploration
        THEN method_selected = "EfficientMethod"
        ELSE TRUE  \* Early exploration is OK

=============================================================================
