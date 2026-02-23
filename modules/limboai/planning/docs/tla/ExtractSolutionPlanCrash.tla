---- MODULE ExtractSolutionPlanCrash ----
\* TLA+ model to identify crash points in extract_solution_plan()
\* Simplified model focusing on node access validation

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxNodes

VARIABLES
    graph,              \* Function: node_id -> node
    toVisit,            \* Sequence of node IDs
    visited,            \* Set of visited node IDs
    plan,               \* Sequence of actions
    crashPoint          \* Where crash occurs

\* Node structure
Node == [
    type |-> Nat,
    status |-> Nat,
    info |-> STRING,
    successors |-> Seq(Nat)
]

\* Constants
NodeTypeROOT == 0
NodeTypeACTION == 1
NodeStatusCLOSED == 1

\* Get node (with validation like our fixed get_node())
GetNode(nodeId) ==
    IF nodeId \in DOMAIN graph
    THEN graph[nodeId]
    ELSE [type |-> 999, status |-> 999, info |-> "", successors |-> <<>>]

\* Check if node is valid
IsValidNode(node) ==
    /\ node.type \in {0, 1, 2}
    /\ node.status \in {0, 1, 2}
    /\ node.successors \in Seq(Nat)

\* Initial state
Init ==
    /\ graph \in [Nat -> Node]
    /\ toVisit = <<0>>
    /\ visited = {}
    /\ plan = <<>>
    /\ crashPoint = "none"

\* Extract step
ExtractStep ==
    /\ toVisit # <<>>
    /\ LET nodeId == Head(toVisit)
           node == GetNode(nodeId)
       IN
       IF nodeId \in visited
       THEN /\ toVisit' = Tail(toVisit)
            /\ UNCHANGED <<graph, visited, plan, crashPoint>>
       ELSE /\ visited' = visited \cup {nodeId}
            /\ IF ~IsValidNode(node)
               THEN /\ crashPoint' = "invalid_node"
                    /\ UNCHANGED <<toVisit, plan>>
               ELSE /\ LET nodeType == node.type
                         nodeStatus == node.status
                    IN
                    IF nodeType = NodeTypeACTION /\ nodeStatus = NodeStatusCLOSED
                    THEN /\ plan' = Append(plan, node.info)
                         /\ IF nodeStatus = NodeStatusCLOSED \/ nodeId = NodeTypeROOT
                            THEN /\ LET successors == node.successors
                                 IN
                                 IF successors = <<>>
                                 THEN /\ toVisit' = Tail(toVisit)
                                 ELSE /\ toVisit' = Append(Tail(toVisit), Reverse(successors))
                            ELSE /\ toVisit' = Tail(toVisit)
                    ELSE /\ plan' = plan
                         /\ IF nodeStatus = NodeStatusCLOSED \/ nodeId = NodeTypeROOT
                            THEN /\ LET successors == node.successors
                                 IN
                                 IF successors = <<>>
                                 THEN /\ toVisit' = Tail(toVisit)
                                 ELSE /\ toVisit' = Append(Tail(toVisit), Reverse(successors))
                            ELSE /\ toVisit' = Tail(toVisit)
                    /\ crashPoint' = crashPoint

ExtractComplete ==
    /\ toVisit = <<>>
    /\ UNCHANGED <<graph, visited, plan, crashPoint>>

Next == ExtractStep \/ ExtractComplete

Spec == Init /\ [][Next]_<<graph, toVisit, visited, plan, crashPoint>>

NoCrash == crashPoint = "none"

====
