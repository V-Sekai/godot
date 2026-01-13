---- MODULE GoalTaskPlanner ----
\* TLA+ model of the Goal Task Planner (HTN planner)
\* This models the core planning logic to help identify issues

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxDepth, MaxActions

VARIABLES
    state,           \* Current state (Dictionary-like: flag -> value)
    solutionGraph,   \* Solution graph: nodeId -> node
    currentNodeId,   \* Current node being processed
    parentNodeId,    \* Parent node ID
    iteration,       \* Iteration counter
    originalTodoList,\* Original todo list to track completion
    blacklistedCommands \* Blacklisted commands/actions

\* Node types
NodeType == {"ROOT", "ACTION", "TASK", "UNIGOAL", "MULTIGOAL", "VERIFY_GOAL", "VERIFY_MULTIGOAL"}

\* Node status
NodeStatus == {"OPEN", "CLOSED", "FAILED", "NOT_APPLICABLE"}

\* Node structure
Node == [
    type: NodeType,
    status: NodeStatus,
    info: STRING,  \* Simplified: action/task name
    successors: Seq(Nat),
    state: state  \* State snapshot
]

\* Initial state for IPyHOP Sample Test 1
InitState == [flag |-> [i \in 0..19 |-> IF i = 0 THEN TRUE ELSE FALSE]]

\* Action: transfer_flag(i, j) - transfers flag from i to j if flag[i] is true
ActionTransferFlag(i, j, s) ==
    /\ i \in DOMAIN s.flag
    /\ s.flag[i] = TRUE
    /\ LET newFlag == [s.flag EXCEPT ![j] = TRUE]
       IN [s EXCEPT !.flag = newFlag]

\* Task methods for task_method_1
TaskMethod1_1 == <<"action_transfer_flag", 0, 1, "action_transfer_flag", 1, 2, "action_transfer_flag", 3, 4>>
TaskMethod1_2 == <<"action_transfer_flag", 0, 1, "action_transfer_flag", 1, 2, "action_transfer_flag", 2, 3>>
TaskMethod1_3 == <<"action_transfer_flag", 0, 1, "action_transfer_flag", 1, 2, "action_transfer_flag", 2, 3, "action_transfer_flag", 3, 4>>

\* Task methods for task_method_2
TaskMethod2_1 == <<"action_transfer_flag", 3, 4, "action_transfer_flag", 4, 5, "action_transfer_flag", 6, 7>>
TaskMethod2_2 == <<"action_transfer_flag", 4, 5, "action_transfer_flag", 5, 6, "action_transfer_flag", 6, 7>>

\* Expected plan: 7 actions to get flag[7] = true
ExpectedPlan == <<
    <<"action_transfer_flag", 0, 1>>,
    <<"action_transfer_flag", 1, 2>>,
    <<"action_transfer_flag", 2, 3>>,
    <<"action_transfer_flag", 3, 4>>,
    <<"action_transfer_flag", 4, 5>>,
    <<"action_transfer_flag", 5, 6>>,
    <<"action_transfer_flag", 6, 7>>
>>

\* Initial predicate
Init ==
    /\ state = InitState
    /\ solutionGraph = [i \in {0} |-> IF i = 0 THEN [type |-> "ROOT", status |-> "NOT_APPLICABLE", info |-> "root", successors |-> <<1, 2>>, state |-> InitState] ELSE [type |-> "ROOT", status |-> "OPEN", info |-> "", successors |-> <<>>, state |-> InitState]]
    /\ currentNodeId = 0
    /\ parentNodeId = 0
    /\ iteration = 0
    /\ originalTodoList = <<<<"task_method_1">>, <<"task_method_2">>>>
    /\ blacklistedCommands = {}

\* Check if all root children are CLOSED
AllRootChildrenClosed ==
    LET rootNode == solutionGraph[0]
        rootSuccessors == rootNode.successors
    IN \A i \in DOMAIN rootSuccessors:
        LET childId == rootSuccessors[i]
        IN /\ childId \in DOMAIN solutionGraph
           /\ solutionGraph[childId].status = "CLOSED"

\* Check if we've completed all tasks from original todo list
AllTasksCompleted ==
    LET rootNode == solutionGraph[0]
        rootSuccessors == rootNode.successors
        closedCount == Cardinality({i \in DOMAIN rootSuccessors:
            LET childId == rootSuccessors[i]
            IN /\ childId \in DOMAIN solutionGraph
               /\ solutionGraph[childId].status = "CLOSED"})
    IN closedCount >= Len(originalTodoList)

\* Apply task_method_1 (simplified: use method 1_2 which gives 0->1->2->3)
ApplyTaskMethod1(nodeId) ==
    LET node == solutionGraph[nodeId]
        newSuccessors == <<3, 4, 5>>  \* Three action nodes
        updatedNode == [node EXCEPT !.status = "CLOSED", !.successors = newSuccessors]
        newNodes == [i \in {3, 4, 5} |->
            IF i = 3 THEN [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_transfer_flag", 0, 1>>, successors |-> <<>>, state |-> state]
            ELSE IF i = 4 THEN [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_transfer_flag", 1, 2>>, successors |-> <<>>, state |-> state]
            ELSE [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_transfer_flag", 2, 3>>, successors |-> <<>>, state |-> state]]
    IN /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode] \cup newNodes
       /\ state' = state

\* Apply task_method_2 (simplified: use method 2_2 which gives 4->5->6->7)
ApplyTaskMethod2(nodeId) ==
    LET node == solutionGraph[nodeId]
        newSuccessors == <<6, 7, 8>>  \* Three action nodes
        updatedNode == [node EXCEPT !.status = "CLOSED", !.successors = newSuccessors]
        newNodes == [i \in {6, 7, 8} |->
            IF i = 6 THEN [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_transfer_flag", 4, 5>>, successors |-> <<>>, state |-> state]
            ELSE IF i = 7 THEN [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_transfer_flag", 5, 6>>, successors |-> <<>>, state |-> state]
            ELSE [type |-> "ACTION", status |-> "OPEN", info |-> <<"action_transfer_flag", 6, 7>>, successors |-> <<>>, state |-> state]]
    IN /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode] \cup newNodes
       /\ state' = state

\* Process an action node
ProcessAction(nodeId) ==
    LET node == solutionGraph[nodeId]
        actionInfo == node.info
        actionName == actionInfo[1]
        fromFlag == actionInfo[2]
        toFlag == actionInfo[3]
    IN IF actionName = "action_transfer_flag"
       THEN /\ IF state.flag[fromFlag] = TRUE
                 THEN /\ state' = ActionTransferFlag(fromFlag, toFlag, state)
                      /\ LET updatedNode == [node EXCEPT !.status = "CLOSED"]
                         IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
                 ELSE /\ state' = state
                      /\ LET updatedNode == [node EXCEPT !.status = "FAILED"]
                         IN solutionGraph' = [solutionGraph EXCEPT ![nodeId] = updatedNode]
            /\ iteration' = iteration + 1
            /\ UNCHANGED <<currentNodeId, parentNodeId, originalTodoList, blacklistedCommands>>
       ELSE UNCHANGED <<state, solutionGraph, currentNodeId, parentNodeId, iteration, originalTodoList, blacklistedCommands>>

\* Process a task node
ProcessTask(nodeId) ==
    LET node == solutionGraph[nodeId]
        taskName == node.info[1]
    IN IF taskName = "task_method_1"
       THEN /\ ApplyTaskMethod1(nodeId)
            /\ iteration' = iteration + 1
            /\ UNCHANGED <<currentNodeId, parentNodeId, originalTodoList, blacklistedCommands>>
       ELSE IF taskName = "task_method_2"
            THEN /\ ApplyTaskMethod2(nodeId)
                 /\ iteration' = iteration + 1
                 /\ UNCHANGED <<currentNodeId, parentNodeId, originalTodoList, blacklistedCommands>>
            ELSE UNCHANGED <<state, solutionGraph, currentNodeId, parentNodeId, iteration, originalTodoList, blacklistedCommands>>

\* Process a node (simplified)
ProcessNode(nodeId) ==
    LET node == solutionGraph[nodeId]
    IN IF node.type = "TASK"
       THEN ProcessTask(nodeId)
       ELSE IF node.type = "ACTION"
            THEN ProcessAction(nodeId)
            ELSE UNCHANGED <<state, solutionGraph, currentNodeId, parentNodeId, iteration, originalTodoList, blacklistedCommands>>

\* Planning step: process one node
PlanningStep ==
    /\ iteration < MaxDepth
    /\ \E nodeId \in DOMAIN solutionGraph:
        /\ solutionGraph[nodeId].status = "OPEN"
        /\ ProcessNode(nodeId)

\* Planning complete
PlanningComplete ==
    /\ AllRootChildrenClosed
    /\ AllTasksCompleted
    /\ state' = state
    /\ solutionGraph' = solutionGraph
    /\ UNCHANGED <<currentNodeId, parentNodeId, iteration, originalTodoList, blacklistedCommands>>

\* Next state relation
Next ==
    \/ PlanningStep
    \/ PlanningComplete

\* Specification
Spec == Init /\ [][Next]_<<state, solutionGraph, currentNodeId, parentNodeId, iteration, originalTodoList, blacklistedCommands>>

\* Property: Eventually flag[7] should be true
Flag7Property == <> (state.flag[7] = TRUE)

\* Property: Eventually all tasks are completed
AllTasksCompletedProperty == <> AllTasksCompleted

====
