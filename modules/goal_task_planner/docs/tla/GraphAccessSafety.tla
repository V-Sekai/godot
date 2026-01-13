---- MODULE GraphAccessSafety ----

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxNodes

VARIABLES
    graph,           \* graph[nodeId] = [type |-> Type, status |-> Status, successors |-> Seq(nodeId)]
    crash_point      \* Where the crash occurred (if any)

Type == {"Root", "Action", "Task", "Unigoal", "Multigoal", "VerifyGoal", "VerifyMultigoal"}
Status == {"Open", "Closed", "Failed", "NA"}

Node == [type : Type, status : Status, successors : Seq(Nat)]

\* Initialize a valid graph structure
Init ==
    /\ graph \in [0..MaxNodes -> Node]
    /\ crash_point = "none"
    /\ graph[0].type = "Root"
    /\ graph[0].status = "NA"
    /\ graph[0].successors \in Seq(1..MaxNodes)

\* Safe node access - this is what the C++ code should do
SafeGetNode(nodeId) ==
    /\ nodeId \in DOMAIN graph
    /\ "type" \in DOMAIN graph[nodeId]
    /\ "status" \in DOMAIN graph[nodeId]
    /\ "successors" \in DOMAIN graph[nodeId]

\* UNSAFE access - accessing node without checking if it exists (CRASH POINT)
UnsafeAccessNode(nodeId) ==
    LET node == graph[nodeId]
        status == node["status"]
        nodeType == node["type"]
        successors == node["successors"]
    IN IF ~SafeGetNode(nodeId)
       THEN /\ crash_point' = "Accessing invalid node"
            /\ UNCHANGED graph
       ELSE /\ UNCHANGED <<graph, crash_point>>

\* UNSAFE access - accessing node["status"] without checking node exists (CRASH POINT from line 168)
UnsafeAccessStatus(nodeId) ==
    IF nodeId \in DOMAIN graph
    THEN LET node == graph[nodeId]
             status == node["status"]  \* This could crash if node is invalid
         IN IF ~("status" \in DOMAIN node)
            THEN /\ crash_point' = "Node missing status field"
                 /\ UNCHANGED graph
            ELSE /\ UNCHANGED <<graph, crash_point>>
    ELSE /\ crash_point' = "Node not in graph"
         /\ UNCHANGED graph

\* UNSAFE access - accessing successors without checking (CRASH POINT from line 172)
UnsafeAccessSuccessors(nodeId) ==
    IF nodeId \in DOMAIN graph
    THEN LET node == graph[nodeId]
             successors == node["successors"]  \* This could crash if node is invalid
         IN IF ~("successors" \in DOMAIN node)
            THEN /\ crash_point' = "Node missing successors field"
                 /\ UNCHANGED graph
            ELSE /\ UNCHANGED <<graph, crash_point>>
    ELSE /\ crash_point' = "Node not in graph"
         /\ UNCHANGED graph

\* SAFE access pattern - what the code should do
SafeAccessPattern(nodeId) ==
    IF nodeId \in DOMAIN graph
    THEN LET node == graph[nodeId]
         IN IF /\ "type" \in DOMAIN node
                /\ "status" \in DOMAIN node
                /\ "successors" \in DOMAIN node
            THEN LET status == node["status"]
                     nodeType == node["type"]
                     successors == node["successors"]
                 IN /\ UNCHANGED <<graph, crash_point>>
            ELSE /\ crash_point' = "Node missing required fields"
                 /\ UNCHANGED graph
    ELSE /\ crash_point' = "Node not in graph"
         /\ UNCHANGED graph

\* Next state relation
Next ==
    \/ \E nodeId \in 0..MaxNodes : UnsafeAccessNode(nodeId)
    \/ \E nodeId \in 0..MaxNodes : UnsafeAccessStatus(nodeId)
    \/ \E nodeId \in 0..MaxNodes : UnsafeAccessSuccessors(nodeId)
    \/ \E nodeId \in 0..MaxNodes : SafeAccessPattern(nodeId)

\* Safety property: No crashes when using safe access
NoCrashWithSafeAccess ==
    \A nodeId \in 0..MaxNodes :
        IF nodeId \in DOMAIN graph
        THEN LET node == graph[nodeId]
             IN IF /\ "type" \in DOMAIN node
                    /\ "status" \in DOMAIN node
                    /\ "successors" \in DOMAIN node
                THEN crash_point = "none"
                ELSE TRUE  \* We detect the issue, don't crash
        ELSE TRUE  \* We detect the issue, don't crash

\* Type correctness: All nodes have valid structure
ValidGraph ==
    \A nodeId \in DOMAIN graph : SafeGetNode(nodeId)

=============================================================================
