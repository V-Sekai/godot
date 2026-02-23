---- MODULE PlanningLoop ----
\* TLA+ model focusing on the planning loop logic
\* Specifically models why the planner stops after 3 actions instead of 7

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxDepth

VARIABLES
    state,
    solutionGraph,
    originalTodoList,
    iteration

\* Node structure (simplified)
Node == [
    type: {"ROOT", "TASK", "ACTION"},
    status: {"OPEN", "CLOSED", "FAILED"},
    info: STRING,
    successors: Seq(Int)
]

\* Initial state
InitState == [flag |-> [i \in 0..19 |-> IF i = 0 THEN TRUE ELSE FALSE]]

Init ==
    /\ state = InitState
    /\ solutionGraph = [
        0 |-> [type |-> "ROOT", status |-> "CLOSED", info |-> "root", successors |-> <<1, 2>>],
        1 |-> [type |-> "TASK", status |-> "OPEN", info |-> "task_method_1", successors |-> <<>>],
        2 |-> [type |-> "TASK", status |-> "OPEN", info |-> "task_method_2", successors |-> <<>>]
      ]
    /\ originalTodoList = <<<<"task_method_1">>, <<"task_method_2">>>>
    /\ iteration = 0

\* Check if all root children are CLOSED
AllRootChildrenClosed ==
    LET rootNode == solutionGraph[0]
        rootSuccessors == rootNode.successors
    IN \A i \in DOMAIN rootSuccessors:
        LET childId == rootSuccessors[i]
        IN /\ childId \in DOMAIN solutionGraph
           /\ solutionGraph[childId].status = "CLOSED"

\* Count closed root children
ClosedRootChildrenCount ==
    LET rootNode == solutionGraph[0]
        rootSuccessors == rootNode.successors
    IN Cardinality({i \in DOMAIN rootSuccessors:
        LET childId == rootSuccessors[i]
        IN /\ childId \in DOMAIN solutionGraph
           /\ solutionGraph[childId].status = "CLOSED"})

\* Check if we need to recreate tasks
NeedToRecreateTasks ==
    /\ AllRootChildrenClosed
    /\ ClosedRootChildrenCount < Len(originalTodoList)

\* Process task_method_1: creates actions 0->1, 1->2, 2->3
ProcessTask1 ==
    /\ solutionGraph[1].status = "OPEN"
    /\ LET updatedNode == [solutionGraph[1] EXCEPT !.status = "CLOSED", !.successors = <<3, 4, 5>>]
           newNodes == [
            3 |-> [type |-> "ACTION", status |-> "OPEN", info |-> "action_transfer_flag(0,1)", successors |-> <<>>],
            4 |-> [type |-> "ACTION", status |-> "OPEN", info |-> "action_transfer_flag(1,2)", successors |-> <<>>],
            5 |-> [type |-> "ACTION", status |-> "OPEN", info |-> "action_transfer_flag(2,3)", successors |-> <<>>]
           ]
       IN /\ solutionGraph' = [solutionGraph EXCEPT ![1] = updatedNode] \cup newNodes
          /\ state' = state
          /\ originalTodoList' = originalTodoList
          /\ iteration' = iteration + 1

\* Process task_method_2: creates actions 4->5, 5->6, 6->7
ProcessTask2 ==
    /\ solutionGraph[2].status = "OPEN"
    /\ LET updatedNode == [solutionGraph[2] EXCEPT !.status = "CLOSED", !.successors = <<6, 7, 8>>]
           newNodes == [
            6 |-> [type |-> "ACTION", status |-> "OPEN", info |-> "action_transfer_flag(4,5)", successors |-> <<>>],
            7 |-> [type |-> "ACTION", status |-> "OPEN", info |-> "action_transfer_flag(5,6)", successors |-> <<>>],
            8 |-> [type |-> "ACTION", status |-> "OPEN", info |-> "action_transfer_flag(6,7)", successors |-> <<>>]
           ]
       IN /\ solutionGraph' = [solutionGraph EXCEPT ![2] = updatedNode] \cup newNodes
          /\ state' = state
          /\ originalTodoList' = originalTodoList
          /\ iteration' = iteration + 1

\* Execute action: transfer_flag(i, j)
ExecuteAction(nodeId, i, j) ==
    LET node == solutionGraph[nodeId]
        actionInfo == node.info
    IN IF state.flag[i] = TRUE
       THEN /\ state' = [state EXCEPT !.flag[j] = TRUE]
            /\ LET updatedNode == [node EXCEPT !.status = "CLOSED"]
               IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
            /\ originalTodoList' = originalTodoList
            /\ iteration' = iteration + 1
       ELSE /\ state' = state
            /\ LET updatedNode == [node EXCEPT !.status = "FAILED"]
               IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
            /\ originalTodoList' = originalTodoList
            /\ iteration' = iteration + 1

\* Recreate failed task
RecreateFailedTask ==
    /\ NeedToRecreateTasks
    /\ LET rootNode == solutionGraph[0]
           rootSuccessors == rootNode.successors
           \* Find which task from originalTodoList is missing or failed
           taskToRecreate == IF \A i \in DOMAIN rootSuccessors:
                                LET childId == rootSuccessors[i]
                                    childNode == solutionGraph[childId]
                                IN childNode.info # "task_method_2"
                             THEN <<"task_method_2">>
                             ELSE <<"task_method_1">>
           newTaskId == CHOOSE n \in Nat: n \notin DOMAIN solutionGraph
           newTaskNode == [type |-> "TASK", status |-> "OPEN", info |-> taskToRecreate[1], successors |-> <<>>]
           updatedRoot == [rootNode EXCEPT !.successors = rootSuccessors \o <<newTaskId>>]
       IN /\ solutionGraph' = [solutionGraph EXCEPT ![0] = updatedRoot] \cup [newTaskId |-> newTaskNode]
          /\ state' = state
          /\ originalTodoList' = originalTodoList
          /\ iteration' = iteration + 1

\* Next state relation
Next ==
    \/ ProcessTask1
    \/ ProcessTask2
    \/ \E nodeId \in DOMAIN solutionGraph:
       /\ solutionGraph[nodeId].type = "ACTION"
       /\ solutionGraph[nodeId].status = "OPEN"
       /\ \E i, j \in 0..19:
          /\ solutionGraph[nodeId].info = "action_transfer_flag(" \o ToString(i) \o "," \o ToString(j) \o ")"
          /\ ExecuteAction(nodeId, i, j)
    \/ RecreateFailedTask

\* Specification
Spec == Init /\ [][Next]_<<state, solutionGraph, originalTodoList, iteration>>

\* Property: Eventually we should have 7 actions executed
\* This is a simplified check - in reality we'd extract the plan from the graph

====
