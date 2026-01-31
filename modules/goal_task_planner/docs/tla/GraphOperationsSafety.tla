---- MODULE GraphOperationsSafety ----

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxNodes, MaxSuccessors

VARIABLES
    graph,           \* graph[nodeId] = [type |-> Type, status |-> Status, successors |-> Seq(nodeId)]
    visited,         \* Set of visited node IDs
    to_visit,        \* Sequence of node IDs to visit
    plan,            \* Sequence of extracted actions
    parent_map,      \* parent_map[childId] = parentId
    crash_point      \* Where the crash occurred (if any)

Type == {"Root", "Action", "Task", "Unigoal", "Multigoal", "VerifyGoal", "VerifyMultigoal"}
Status == {"Open", "Closed", "Failed", "NA"}

Node == [type : Type, status : Status, successors : Seq(Nat)]

\* Initialize a valid graph structure
Init ==
    /\ graph \in [1..MaxNodes -> Node]
    /\ visited = {}
    /\ to_visit = <<0>>
    /\ plan = <<>>
    /\ parent_map = [i \in {} |-> 0]
    /\ crash_point = "none"
    /\ graph[0].type = "Root"
    /\ graph[0].status = "NA"
    /\ graph[0].successors \in Seq(1..MaxNodes)

\* Build parent map - verify this is safe
BuildParentMap ==
    LET BuildStep ==
        /\ parent_map' = [childId \in UNION {graph[parentId].successors : parentId \in DOMAIN graph} |->
                          CHOOSE parentId \in DOMAIN graph : childId \in graph[parentId].successors]
        /\ UNCHANGED <<graph, visited, to_visit, plan, crash_point>>
    IN BuildStep

\* Safe graph access - verify node exists and has required fields
SafeGetNode(nodeId) ==
    /\ nodeId \in DOMAIN graph
    /\ "type" \in DOMAIN graph[nodeId]
    /\ "status" \in DOMAIN graph[nodeId]
    /\ "successors" \in DOMAIN graph[nodeId]

\* Extract solution plan - verify this doesn't crash
ExtractSolutionPlan ==
    LET ProcessNode ==
        IF to_visit = <<>>
        THEN /\ UNCHANGED <<graph, visited, to_visit, plan, parent_map, crash_point>>
        ELSE LET nodeId == Head(to_visit)
                 node == graph[nodeId]
             IN IF nodeId \in visited
                THEN /\ to_visit' = Tail(to_visit)
                     /\ UNCHANGED <<graph, visited, plan, parent_map, crash_point>>
                ELSE IF ~SafeGetNode(nodeId)
                     THEN /\ crash_point' = "Invalid node structure"
                          /\ UNCHANGED <<graph, visited, to_visit, plan, parent_map>>
                     ELSE /\ visited' = visited \cup {nodeId}
                          /\ IF /\ node.type = "Action"
                                 /\ node.status = "Closed"
                             THEN /\ plan' = Append(plan, nodeId)
                             ELSE /\ plan' = plan
                          /\ IF /\ (node.status = "Closed" \/ nodeId = 0)
                                 /\ node.successors /= <<>>
                             THEN LET ProcessSuccessor(succId) ==
                                      IF ~SafeGetNode(succId)
                                      THEN /\ crash_point' = "Invalid successor node"
                                           /\ UNCHANGED <<graph, visited, to_visit, plan, parent_map>>
                                      ELSE LET parentOfSucc == IF succId \in DOMAIN parent_map
                                                                 THEN parent_map[succId]
                                                                 ELSE -1
                                               succNode == graph[succId]
                                           IN IF /\ parentOfSucc = nodeId
                                                  /\ (succNode.status = Closed \/ succId = 0)
                                              THEN /\ to_visit' = Append(to_visit', succId)
                                                   /\ UNCHANGED <<graph, visited, plan, parent_map, crash_point>>
                                              ELSE /\ UNCHANGED <<graph, visited, to_visit, plan, parent_map, crash_point>>
                                  IN LET to_visit_temp == Tail(to_visit)
                                     IN /\ to_visit' = to_visit_temp
                                        /\ \A succId \in {node.successors[i] : i \in DOMAIN node.successors} :
                                           ProcessSuccessor(succId)
                             ELSE /\ to_visit' = Tail(to_visit)
                          /\ UNCHANGED <<graph, parent_map, crash_point>>
    IN ProcessNode

\* Check if planning succeeded - verify this doesn't crash
CheckPlanningSucceeded ==
    LET FindReachable ==
        LET ReachableStep ==
            LET to_visit_reachable == IF to_visit = <<>> THEN <<0>> ELSE to_visit
                visited_reachable == IF to_visit = <<>> THEN {} ELSE visited
            IN /\ visited_reachable' = visited_reachable \cup {Head(to_visit_reachable)}
               /\ IF Head(to_visit_reachable) \in DOMAIN graph
                  THEN LET node == graph[Head(to_visit_reachable)]
                       IN IF ~SafeGetNode(Head(to_visit_reachable))
                          THEN /\ crash_point' = "Invalid node in reachable check"
                               /\ UNCHANGED <<graph, visited, to_visit, plan, parent_map>>
                          ELSE IF /\ (node.status = Closed \/ Head(to_visit_reachable) = 0)
                                     /\ node.successors /= <<>>
                                  THEN /\ to_visit_reachable' = Append(Tail(to_visit_reachable),
                                                                       [i \in DOMAIN node.successors |-> node.successors[i]])
                                  ELSE /\ to_visit_reachable' = Tail(to_visit_reachable)
               ELSE /\ crash_point' = "Node not in graph during reachable check"
                    /\ UNCHANGED <<graph, visited, to_visit, plan, parent_map>>
        IN ReachableStep
    IN FindReachable

\* Next state relation
Next ==
    \/ BuildParentMap
    \/ ExtractSolutionPlan
    \/ CheckPlanningSucceeded

\* Safety property: No crashes
NoCrash == crash_point = "none"

\* Liveness: Eventually extract plan completes
ExtractCompletes == <> (to_visit = <<>> /\ crash_point = "none")

\* Type correctness: All nodes have valid structure
ValidGraph ==
    \A nodeId \in DOMAIN graph : SafeGetNode(nodeId)

\* Parent map correctness: All children have valid parents
ValidParentMap ==
    \A childId \in DOMAIN parent_map :
        /\ childId \in DOMAIN graph
        /\ parent_map[childId] \in DOMAIN graph
        /\ childId \in graph[parent_map[childId]].successors

=============================================================================
