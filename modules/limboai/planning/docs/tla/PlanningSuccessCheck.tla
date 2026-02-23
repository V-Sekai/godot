---- MODULE PlanningSuccessCheck ----

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxNodes

VARIABLES
    graph,              \* graph[nodeId] = [type |-> Type, status |-> Status, successors |-> Seq(nodeId)]
    reachable_nodes,    \* Set of reachable node IDs
    to_visit,           \* Sequence of node IDs to visit
    visited,            \* Set of visited node IDs
    planning_succeeded, \* Boolean flag
    crash_point         \* Where the crash occurred (if any)

Type == {"Root", "Action", "Task", "Unigoal", "Multigoal", "VerifyGoal", "VerifyMultigoal"}
Status == {"Open", "Closed", "Failed", "NA"}

Node == [type : Type, status : Status, successors : Seq(Nat)]

\* Initialize graph
Init ==
    /\ graph \in [0..MaxNodes -> Node]
    /\ reachable_nodes = {}
    /\ to_visit = <<0>>
    /\ visited = {}
    /\ planning_succeeded = FALSE
    /\ crash_point = "none"
    /\ graph[0].type = "Root"
    /\ graph[0].status = "NA"
    /\ graph[0].successors \in Seq(1..MaxNodes)

\* Safe node access
SafeGetNode(nodeId) ==
    /\ nodeId \in DOMAIN graph
    /\ "type" \in DOMAIN graph[nodeId]
    /\ "status" \in DOMAIN graph[nodeId]
    /\ "successors" \in DOMAIN graph[nodeId]

\* Find reachable nodes - this is where the crash might occur
FindReachableNodes ==
    IF to_visit = <<>>
    THEN /\ UNCHANGED <<graph, reachable_nodes, to_visit, visited, planning_succeeded, crash_point>>
    ELSE LET nodeId == Head(to_visit)
         IN IF nodeId \in visited
            THEN /\ to_visit' = Tail(to_visit)
                 /\ UNCHANGED <<graph, reachable_nodes, visited, planning_succeeded, crash_point>>
            ELSE IF ~SafeGetNode(nodeId)
                 THEN /\ crash_point' = "Invalid node in FindReachableNodes"
                      /\ UNCHANGED <<graph, reachable_nodes, to_visit, visited, planning_succeeded>>
                 ELSE LET node == graph[nodeId]
                      IN /\ visited' = visited \cup {nodeId}
                         /\ reachable_nodes' = reachable_nodes \cup {nodeId}
                         /\ IF /\ (node.status = "Closed" \/ nodeId = 0)
                                /\ node.successors /= <<>>
                            THEN LET AddSuccessors ==
                                     LET ProcessSucc(succId) ==
                                         IF ~SafeGetNode(succId)
                                         THEN /\ crash_point' = "Invalid successor in FindReachableNodes"
                                              /\ UNCHANGED <<graph, reachable_nodes, to_visit, visited, planning_succeeded>>
                                         ELSE /\ to_visit' = Append(to_visit', succId)
                                              /\ UNCHANGED <<graph, reachable_nodes, visited, planning_succeeded, crash_point>>
                                     IN \A succId \in {node.successors[i] : i \in DOMAIN node.successors} :
                                        ProcessSucc(succId)
                                 IN /\ to_visit' = Tail(to_visit)
                                    /\ AddSuccessors
                            ELSE /\ to_visit' = Tail(to_visit)
                         /\ UNCHANGED <<graph, planning_succeeded, crash_point>>

\* Check reachable nodes for failures - potential crash point
CheckReachableNodes ==
    LET CheckNode(nodeId) ==
        IF nodeId = 0
        THEN /\ UNCHANGED <<graph, reachable_nodes, to_visit, visited, planning_succeeded, crash_point>>
        ELSE IF ~SafeGetNode(nodeId)
             THEN /\ crash_point' = "Invalid node in CheckReachableNodes"
                  /\ UNCHANGED <<graph, reachable_nodes, to_visit, visited, planning_succeeded>>
             ELSE LET node == graph[nodeId]
                      status == node.status
                      nodeType == node.type
                  IN IF status = "Open"
                     THEN /\ planning_succeeded' = FALSE
                          /\ UNCHANGED <<graph, reachable_nodes, to_visit, visited, crash_point>>
                     ELSE IF status = "Failed"
                          THEN IF /\ nodeType /= "VerifyGoal"
                                  /\ nodeType /= "VerifyMultigoal"
                              THEN /\ planning_succeeded' = FALSE
                                   /\ UNCHANGED <<graph, reachable_nodes, to_visit, visited, crash_point>>
                              ELSE /\ UNCHANGED <<graph, reachable_nodes, to_visit, visited, planning_succeeded, crash_point>>
                          ELSE /\ UNCHANGED <<graph, reachable_nodes, to_visit, visited, planning_succeeded, crash_point>>
    IN /\ to_visit = <<>>  \* Only check when traversal is complete
       /\ \A nodeId \in reachable_nodes : CheckNode(nodeId)

\* Next state relation
Next ==
    \/ FindReachableNodes
    \/ CheckReachableNodes

\* Safety property: No crashes
NoCrash == crash_point = "none"

\* Type correctness
ValidGraph ==
    \A nodeId \in DOMAIN graph : SafeGetNode(nodeId)

\* All successors are valid nodes
ValidSuccessors ==
    \A nodeId \in DOMAIN graph :
        \A succId \in {graph[nodeId].successors[i] : i \in DOMAIN graph[nodeId].successors} :
            succId \in DOMAIN graph

=============================================================================
