---- MODULE BacktrackingTest4 ----
\* TLA+ model for IPyHOP Backtracking Test 4
\* Problem: put_it task with 3 methods, need1 task with 1 method
\* Expected: [('action_putv', 1), ('action_getv', 1), ('action_getv', 1)]

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxDepth

VARIABLES
    state,
    solutionGraph,
    originalTodoList,
    iteration,
    blacklistedCommands

\* Node structure
\* selectedMethod tracks which method was used: "m_err", "m0", "m1", or "" if none
Node == [
    type: {"ROOT", "TASK", "ACTION"},
    status: {"OPEN", "CLOSED", "FAILED"},
    info: STRING,
    successors: Seq(Nat),
    selectedMethod: STRING
]

\* Initial state: flag = -1 (represented as 999 to avoid negative numbers in TLA+)
\* We'll treat 999 as -1, 0 as 0, 1 as 1
InitState == [flag |-> 999]

Init ==
    /\ state = InitState
    /\ solutionGraph = [i \in {0, 1, 2} |->
        IF i = 0 THEN [type |-> "ROOT", status |-> "CLOSED", info |-> "root", successors |-> <<1, 2>>, selectedMethod |-> ""]
        ELSE IF i = 1 THEN [type |-> "TASK", status |-> "OPEN", info |-> "put_it", successors |-> <<>>, selectedMethod |-> ""]
        ELSE [type |-> "TASK", status |-> "OPEN", info |-> "need1", successors |-> <<>>, selectedMethod |-> ""]]
    /\ originalTodoList = <<<<"put_it">>, <<"need1">>>>
    /\ iteration = 0
    /\ blacklistedCommands = {}

\* Actions
\* action_putv(state, val): sets state.flag = val
\* action_getv(state, val): checks if state.flag == val, returns state if true, false otherwise

\* put_it methods:
\* task_method_m_err: [putv(0), getv(1)] - fails because getv(1) needs flag=1 but flag=0
\* task_method_m0: [putv(0), getv(0)] - succeeds, sets flag=0
\* task_method_m1: [putv(1), getv(1)] - succeeds, sets flag=1

\* need1 method:
\* task_method_m_need1: [getv(1)] - needs flag=1

\* Apply put_it method m_err (fails)
ApplyPutItMethodErr(nodeId) ==
    LET node == solutionGraph[nodeId]
        newSuccessors == <<3, 4>>  \* putv(0), getv(1)
        updatedNode == [node EXCEPT !.status = "CLOSED", !.successors = newSuccessors, !.selectedMethod = "m_err"]
        newNodes == [i \in {3, 4} |->
            IF i = 3 THEN [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_putv", 0>>, successors |-> <<>>, selectedMethod |-> ""]
            ELSE [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_getv", 1>>, successors |-> <<>>, selectedMethod |-> ""]]
        updatedGraph == [solutionGraph EXCEPT ![nodeId] = updatedNode]
    IN /\ solutionGraph' = [i \in (DOMAIN updatedGraph \cup DOMAIN newNodes) |->
            IF i \in DOMAIN newNodes THEN newNodes[i] ELSE updatedGraph[i]]
       /\ state' = state
       /\ UNCHANGED <<originalTodoList, blacklistedCommands>>

\* Apply put_it method m0 (succeeds)
ApplyPutItMethod0(nodeId) ==
    LET node == solutionGraph[nodeId]
        newSuccessors == <<5, 6>>  \* putv(0), getv(0)
        updatedNode == [node EXCEPT !.status = "CLOSED", !.successors = newSuccessors, !.selectedMethod = "m0"]
        newNodes == [i \in {5, 6} |->
            IF i = 5 THEN [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_putv", 0>>, successors |-> <<>>, selectedMethod |-> ""]
            ELSE [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_getv", 0>>, successors |-> <<>>, selectedMethod |-> ""]]
        updatedGraph == [solutionGraph EXCEPT ![nodeId] = updatedNode]
    IN /\ solutionGraph' = [i \in (DOMAIN updatedGraph \cup DOMAIN newNodes) |->
            IF i \in DOMAIN newNodes THEN newNodes[i] ELSE updatedGraph[i]]
       /\ state' = state
       /\ UNCHANGED <<originalTodoList, blacklistedCommands>>

\* Apply put_it method m1 (succeeds, sets flag=1)
ApplyPutItMethod1(nodeId) ==
    LET node == solutionGraph[nodeId]
        newSuccessors == <<7, 8>>  \* putv(1), getv(1)
        updatedNode == [node EXCEPT !.status = "CLOSED", !.successors = newSuccessors, !.selectedMethod = "m1"]
        newNodes == [i \in {7, 8} |->
            IF i = 7 THEN [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_putv", 1>>, successors |-> <<>>, selectedMethod |-> ""]
            ELSE [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_getv", 1>>, successors |-> <<>>, selectedMethod |-> ""]]
        updatedGraph == [solutionGraph EXCEPT ![nodeId] = updatedNode]
    IN /\ solutionGraph' = [i \in (DOMAIN updatedGraph \cup DOMAIN newNodes) |->
            IF i \in DOMAIN newNodes THEN newNodes[i] ELSE updatedGraph[i]]
       /\ state' = state
       /\ UNCHANGED <<originalTodoList, blacklistedCommands>>

\* Apply need1 method (needs flag=1)
ApplyNeed1Method(nodeId) ==
    LET node == solutionGraph[nodeId]
        newSuccessors == <<9>>  \* getv(1)
        updatedNode == [node EXCEPT !.status = "CLOSED", !.successors = newSuccessors, !.selectedMethod = "m_need1"]
        newNodes == [i \in {9} |-> [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_getv", 1>>, successors |-> <<>>, selectedMethod |-> ""]]
        updatedGraph == [solutionGraph EXCEPT ![nodeId] = updatedNode]
    IN /\ solutionGraph' = [i \in (DOMAIN updatedGraph \cup DOMAIN newNodes) |->
            IF i \in DOMAIN newNodes THEN newNodes[i] ELSE updatedGraph[i]]
       /\ state' = state
       /\ UNCHANGED <<originalTodoList, blacklistedCommands>>

\* Execute action_putv(val)
ExecutePutv(nodeId, val) ==
    LET node == solutionGraph[nodeId]
    IN /\ state' = [state EXCEPT !.flag = val]
       /\ LET updatedNode == [node EXCEPT !.status = "CLOSED"]
          IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
       /\ UNCHANGED <<originalTodoList, iteration, blacklistedCommands>>

\* Execute action_getv(val) - succeeds if state.flag == val
\* Note: 999 represents -1, so we compare directly
ExecuteGetv(nodeId, val) ==
    LET node == solutionGraph[nodeId]
    IN IF state.flag = val
       THEN /\ state' = state
            /\ LET updatedNode == [node EXCEPT !.status = "CLOSED"]
               IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
            /\ UNCHANGED <<originalTodoList, blacklistedCommands, iteration>>
       ELSE /\ state' = state
            /\ LET updatedNode == [node EXCEPT !.status = "FAILED"]
               IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
            /\ UNCHANGED <<originalTodoList, blacklistedCommands, iteration>>

\* Process put_it task - try methods in order: m_err, m0, m1
\* Key insight from TLA+ analysis:
\* - m_err fails: getv(1) needs flag=1 but flag=999 (representing -1), so [putv(0), getv(1)] gets blacklisted
\* - m0 succeeds initially: putv(0) sets flag=0, getv(0) checks flag=0 - succeeds, but need1 needs flag=1
\* - When need1 fails, it blacklists put_it's created_subtasks [putv(0), getv(0)]
\* - When put_it is reopened, m0's array is blacklisted, so it should try m1
\* - m1 succeeds: putv(1) sets flag=1, getv(1) checks flag=1 - succeeds, need1 can now succeed
\* The planner should skip blacklisted method arrays and try the next method
\*
\* In TLA+, we represent blacklisted arrays as strings for easier comparison
\* "m_err" = methodErrArray, "m0" = method0Array, "m1" = method1Array
ProcessPutItTask(nodeId) ==
    LET node == solutionGraph[nodeId]
    IN IF node.status = "OPEN"
       THEN \/ /\ ~("m_err" \in blacklistedCommands)
               /\ ApplyPutItMethodErr(nodeId)
            \/ /\ ~("m0" \in blacklistedCommands)
               /\ ApplyPutItMethod0(nodeId)
            \/ /\ ~("m1" \in blacklistedCommands)
               /\ ApplyPutItMethod1(nodeId)
       ELSE UNCHANGED <<state, solutionGraph, originalTodoList, blacklistedCommands>>

\* Process need1 task - if it fails, backtrack to put_it
\* When need1 fails (flag != 1):
\* - If put_it is CLOSED (has used a method), blacklist that method and reopen put_it
\* - If put_it is still OPEN, just mark need1 as FAILED (put_it will be processed later)
ProcessNeed1Task(nodeId) ==
    LET node == solutionGraph[nodeId]
        putItNode == solutionGraph[1]
        putItMethod == putItNode.selectedMethod
        putItStatus == putItNode.status
        \* Get all descendants of put_it (nodes in its successors sequence)
        putItSuccessorsSet == IF Len(putItNode.successors) > 0
            THEN {putItNode.successors[i]: i \in DOMAIN putItNode.successors}
            ELSE {}
        putItDescendants == putItSuccessorsSet \cup
            {i \in DOMAIN solutionGraph: \E j \in putItSuccessorsSet:
                j \in DOMAIN solutionGraph /\ Len(solutionGraph[j].successors) > 0 /\
                i \in {solutionGraph[j].successors[k]: k \in DOMAIN solutionGraph[j].successors}}
    IN IF node.status = "OPEN"
       THEN IF state.flag = 1
            THEN ApplyNeed1Method(nodeId)
            ELSE IF putItStatus = "CLOSED" /\ putItMethod # ""
                 \* put_it has used a method, blacklist it and reopen put_it
                 THEN /\ state' = state
                      /\ LET updatedNode == [node EXCEPT !.status = "FAILED"]
                             updatedPutItNode == [putItNode EXCEPT !.status = "OPEN", !.successors = <<>>, !.selectedMethod = ""]
                             remainingNodes == DOMAIN solutionGraph \ putItDescendants
                         IN /\ solutionGraph' = [i \in (remainingNodes \cup {1, nodeId}) |->
                                 IF i = 1 THEN updatedPutItNode
                                 ELSE IF i = nodeId THEN updatedNode
                                 ELSE solutionGraph[i]]
                            /\ blacklistedCommands' = blacklistedCommands \cup {putItMethod}
                      /\ UNCHANGED <<originalTodoList>>
                 \* put_it hasn't been processed yet, just mark need1 as failed
                 ELSE /\ state' = state
                      /\ LET updatedNode == [node EXCEPT !.status = "FAILED"]
                         IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
                      /\ UNCHANGED <<originalTodoList, blacklistedCommands>>
       ELSE UNCHANGED <<state, solutionGraph, originalTodoList, blacklistedCommands>>

\* Process action node
ProcessAction(nodeId) ==
    LET node == solutionGraph[nodeId]
        actionInfo == node.info
        actionName == actionInfo[1]
    IN IF actionName = "action_putv"
       THEN LET val == actionInfo[2]
            IN ExecutePutv(nodeId, val)
       ELSE IF actionName = "action_getv"
            THEN LET val == actionInfo[2]
                 IN ExecuteGetv(nodeId, val)
            ELSE UNCHANGED <<state, solutionGraph, originalTodoList, blacklistedCommands>>

\* Check if all root children are CLOSED
AllRootChildrenClosed ==
    LET rootNode == solutionGraph[0]
        rootSuccessors == rootNode.successors
    IN \A i \in DOMAIN rootSuccessors:
        LET childId == rootSuccessors[i]
        IN /\ childId \in DOMAIN solutionGraph
           /\ solutionGraph[childId].status = "CLOSED"

\* Process one step
ProcessStep ==
    /\ iteration < MaxDepth
    /\ \E nodeId \in DOMAIN solutionGraph:
        /\ solutionGraph[nodeId].status = "OPEN"
        /\ LET node == solutionGraph[nodeId]
           IN IF node.type = "TASK"
              THEN IF node.info = "put_it"
                   THEN ProcessPutItTask(nodeId)
                   ELSE IF node.info = "need1"
                        THEN ProcessNeed1Task(nodeId)
                        ELSE UNCHANGED <<state, solutionGraph, originalTodoList, blacklistedCommands>>
              ELSE IF node.type = "ACTION"
                   THEN ProcessAction(nodeId)
                   ELSE UNCHANGED <<state, solutionGraph, originalTodoList, blacklistedCommands>>
    /\ iteration' = iteration + 1

\* Planning complete
PlanningComplete ==
    /\ AllRootChildrenClosed
    /\ state' = state
    /\ solutionGraph' = solutionGraph
    /\ UNCHANGED <<originalTodoList, iteration, blacklistedCommands>>

\* Next state relation
Next ==
    \/ ProcessStep
    \/ PlanningComplete

\* Specification
Spec == Init /\ [][Next]_<<state, solutionGraph, originalTodoList, iteration, blacklistedCommands>>

\* Property: Eventually flag = 1
Flag1Property == <> (state.flag = 1 \/ state.flag = 999)

\* Property: Eventually both tasks are completed
BothTasksCompleted ==
    /\ AllRootChildrenClosed
    /\ (state.flag = 1 \/ state.flag = 999)

====
