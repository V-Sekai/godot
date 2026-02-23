---- MODULE TaskRecreation ----
\* TLA+ model to debug task recreation issue
\* Problem: task_method_1 completes (sets flag[3]=true), but task_method_2 keeps failing

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxDepth

VARIABLES
    state,
    solutionGraph,
    originalTodoList,
    iteration,
    currentTaskMethod \* Track which method is being tried for each task

\* Node structure
Node == [
    type: {"ROOT", "TASK", "ACTION"},
    status: {"OPEN", "CLOSED", "FAILED"},
    info: STRING,
    successors: Seq(Nat),
    methodIndex: Nat  \* Which method index was used (0, 1, 2, ...)
]

\* Initial state: flag[0] = true, all others false
InitState == [flag |-> [i \in 0..19 |-> IF i = 0 THEN TRUE ELSE FALSE]]

Init ==
    /\ state = InitState
    /\ solutionGraph = [i \in {0, 1, 2} |->
        IF i = 0 THEN [type |-> "ROOT", status |-> "CLOSED", info |-> "root", successors |-> <<1, 2>>, methodIndex |-> 0]
        ELSE IF i = 1 THEN [type |-> "TASK", status |-> "OPEN", info |-> "task_method_1", successors |-> <<>>, methodIndex |-> 0]
        ELSE [type |-> "TASK", status |-> "OPEN", info |-> "task_method_2", successors |-> <<>>, methodIndex |-> 0]]
    /\ originalTodoList = <<<<"task_method_1">>, <<"task_method_2">>>>
    /\ iteration = 0
    /\ currentTaskMethod = [i \in {1, 2} |-> 0]  \* Both tasks start with method 0

\* task_method_1 has 3 methods:
\* method 0: [0->1, 1->2, 3->4]  (doesn't get flag[3])
\* method 1: [0->1, 1->2, 2->3]  (gets flag[3]) ✓
\* method 2: [0->1, 1->2, 2->3, 3->4]  (gets flag[3] and flag[4])

\* task_method_2 has 2 methods:
\* method 0: [3->4, 4->5, 6->7]  (needs flag[3]) ✓
\* method 1: [4->5, 5->6, 6->7]  (needs flag[4])

\* Apply task_method_1 with method index
ApplyTaskMethod1(nodeId, methodIdx) ==
    LET node == solutionGraph[nodeId]
        methodActions == IF methodIdx = 0 THEN <<<<0, 1>>, <<1, 2>>, <<3, 4>>>>
                        ELSE IF methodIdx = 1 THEN <<<<0, 1>>, <<1, 2>>, <<2, 3>>>>
                        ELSE <<<<0, 1>>, <<1, 2>>, <<2, 3>>, <<3, 4>>>>
        numActions == Len(methodActions)
        newSuccessors == [i \in 1..numActions |-> nodeId * 10 + i]
        updatedNode == [node EXCEPT !.status = "CLOSED", !.successors = newSuccessors, !.methodIndex = methodIdx]
        newNodes == [i \in 1..numActions |->
            LET actionId == nodeId * 10 + i
                actionArgs == methodActions[i]
            IN [type |-> "ACTION", status |-> "OPEN", info |-> actionArgs, successors |-> <<>>, methodIndex |-> 0]]
    IN /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode] \cup newNodes
       /\ state' = state
       /\ currentTaskMethod' = [currentTaskMethod EXCEPT ![nodeId] = methodIdx]
       /\ UNCHANGED <<originalTodoList, iteration>>

\* Apply task_method_2 with method index
ApplyTaskMethod2(nodeId, methodIdx) ==
    LET node == solutionGraph[nodeId]
        methodActions == IF methodIdx = 0 THEN <<<<3, 4>>, <<4, 5>>, <<6, 7>>>>
                        ELSE <<<<4, 5>>, <<5, 6>>, <<6, 7>>>>
        numActions == Len(methodActions)
        newSuccessors == [i \in 1..numActions |-> nodeId * 10 + i]
        updatedNode == [node EXCEPT !.status = "CLOSED", !.successors = newSuccessors, !.methodIndex = methodIdx]
        newNodes == [i \in 1..numActions |->
            LET actionId == nodeId * 10 + i
                actionArgs == methodActions[i]
            IN [type |-> "ACTION", status |-> "OPEN", info |-> actionArgs, successors |-> <<>>, methodIndex |-> 0]]
    IN /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode] \cup newNodes
       /\ state' = state
       /\ currentTaskMethod' = [currentTaskMethod EXCEPT ![nodeId] = methodIdx]
       /\ UNCHANGED <<originalTodoList, iteration>>

\* Execute action: transfer_flag(from, to)
ExecuteAction(nodeId, from, to) ==
    LET node == solutionGraph[nodeId]
    IN IF state.flag[from] = TRUE
       THEN /\ state' = [state EXCEPT !.flag[to] = TRUE]
            /\ LET updatedNode == [node EXCEPT !.status = "CLOSED"]
               IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
            /\ UNCHANGED <<originalTodoList, iteration, currentTaskMethod>>
       ELSE /\ state' = state
            /\ LET updatedNode == [node EXCEPT !.status = "FAILED"]
               IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
            /\ UNCHANGED <<originalTodoList, iteration, currentTaskMethod>>

\* Process task node - try methods in order
ProcessTask(nodeId) ==
    LET node == solutionGraph[nodeId]
        taskName == node.info
        methodIdx == currentTaskMethod[nodeId]
    IN IF taskName = "task_method_1"
       THEN IF methodIdx < 3  \* 3 methods available
            THEN ApplyTaskMethod1(nodeId, methodIdx)
            ELSE /\ state' = state
                 /\ LET updatedNode == [node EXCEPT !.status = "FAILED"]
                    IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
                 /\ UNCHANGED <<originalTodoList, iteration, currentTaskMethod>>
       ELSE IF taskName = "task_method_2"
            THEN IF methodIdx < 2  \* 2 methods available
                 THEN ApplyTaskMethod2(nodeId, methodIdx)
                 ELSE /\ state' = state
                      /\ LET updatedNode == [node EXCEPT !.status = "FAILED"]
                         IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
                      /\ UNCHANGED <<originalTodoList, iteration, currentTaskMethod>>
            ELSE UNCHANGED <<state, solutionGraph, originalTodoList, iteration, currentTaskMethod>>

\* Check if all root children are CLOSED
AllRootChildrenClosed ==
    LET rootNode == solutionGraph[0]
        rootSuccessors == rootNode.successors
    IN \A i \in DOMAIN rootSuccessors:
        LET childId == rootSuccessors[i]
        IN /\ childId \in DOMAIN solutionGraph
           /\ solutionGraph[childId].status = "CLOSED"

\* Count closed root children
ClosedCount ==
    LET rootNode == solutionGraph[0]
        rootSuccessors == rootNode.successors
    IN Cardinality({i \in DOMAIN rootSuccessors:
        LET childId == rootSuccessors[i]
        IN /\ childId \in DOMAIN solutionGraph
           /\ solutionGraph[childId].status = "CLOSED"})

\* Recreate failed task - KEY ISSUE: state should have flag[3] = true from task_method_1
RecreateFailedTask ==
    /\ AllRootChildrenClosed
    /\ ClosedCount < Len(originalTodoList)
    /\ iteration < MaxDepth
    /\ LET rootNode == solutionGraph[0]
           rootSuccessors == rootNode.successors
           \* Find which task is missing
           missingTask == IF \A i \in DOMAIN rootSuccessors:
                             LET childId == rootSuccessors[i]
                                 childNode == solutionGraph[childId]
                             IN childNode.info # "task_method_2"
                          THEN <<"task_method_2">>
                          ELSE <<"task_method_1">>
           newTaskId == CHOOSE n \in Nat: n \notin DOMAIN solutionGraph /\ n > 0
           newTaskNode == [type |-> "TASK", status |-> "OPEN", info |-> missingTask[1], successors |-> <<>>, methodIndex |-> 0]
           updatedRoot == [rootNode EXCEPT !.successors = rootSuccessors \o <<newTaskId>>]
       IN /\ solutionGraph' = [solutionGraph EXCEPT ![0] = updatedRoot] \cup [newTaskId |-> newTaskNode]
          /\ state' = state  \* CRITICAL: Use current state (should have flag[3]=true)
          /\ originalTodoList' = originalTodoList
          /\ iteration' = iteration + 1
          /\ currentTaskMethod' = currentTaskMethod \cup [newTaskId |-> 0]  \* Start with method 0

\* Find first OPEN node (DFS order)
FindOpenNode ==
    LET rootNode == solutionGraph[0]
        rootSuccessors == rootNode.successors
        openNodeId == CHOOSE n \in DOMAIN solutionGraph:
            /\ solutionGraph[n].status = "OPEN"
            /\ \E i \in DOMAIN rootSuccessors:
               LET path == <<rootSuccessors[i]>>
               IN n \in path \/ \E j \in DOMAIN path: solutionGraph[path[j]].successors
    IN openNodeId

\* Process one step
ProcessStep ==
    /\ iteration < MaxDepth
    /\ LET openNodeId == FindOpenNode
           node == solutionGraph[openNodeId]
       IN IF node.type = "TASK"
          THEN ProcessTask(openNodeId)
          ELSE IF node.type = "ACTION"
               THEN LET actionInfo == node.info
                        from == actionInfo[1]
                        to == actionInfo[2]
                    IN ExecuteAction(openNodeId, from, to)
               ELSE UNCHANGED <<state, solutionGraph, originalTodoList, iteration, currentTaskMethod>>
    /\ iteration' = iteration + 1

\* Next state relation
Next ==
    \/ ProcessStep
    \/ RecreateFailedTask

\* Specification
Spec == Init /\ [][Next]_<<state, solutionGraph, originalTodoList, iteration, currentTaskMethod>>

\* Property: Eventually flag[7] should be true
Flag7Property == <> (state.flag[7] = TRUE)

\* Property: Eventually both tasks are completed
BothTasksCompleted ==
    /\ AllRootChildrenClosed
    /\ ClosedCount >= Len(originalTodoList)

====
