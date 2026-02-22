---- MODULE BacktrackingTest4Detailed ----

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS MaxNodes

(*
  Backtracking Test 4:
  - Initial state: flag = -1
  - Tasks: [put_it, need1]
  - put_it has 3 methods: m_err, m0, m1
  - need1 has 1 method: m_need1

  Methods:
  - m_err: [action_putv(0), action_getv(1)]  // Sets flag=0, then checks flag==1 (FAILS)
  - m0: [action_putv(0), action_getv(0)]     // Sets flag=0, then checks flag==0 (SUCCEEDS)
  - m1: [action_putv(1), action_getv(1)]     // Sets flag=1, then checks flag==1 (SUCCEEDS)
  - m_need1: [action_getv(1)]                // Checks flag==1 (SUCCEEDS if flag==1)

  Expected plan: [action_putv(1), action_getv(1), action_getv(1)]
  - put_it uses m1: [action_putv(1), action_getv(1)]
  - need1 uses m_need1: [action_getv(1)]
*)

VARIABLES
    state,
    solutionGraph,
    blacklistedCommands,
    currentNodeId,
    iteration

vars == <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>

TypeOK ==
    /\ state \in [-1 .. 1]  (* flag value *)
    /\ solutionGraph \in [0..MaxNodes -> [status: {"OPEN", "CLOSED", "FAILED"},
                                          type: {"ROOT", "TASK", "ACTION"},
                                          info: Seq(STRING),
                                          successors: Seq(Nat),
                                          selectedMethod: STRING,
                                          createdSubtasks: Seq(Seq(STRING))]]
    /\ blacklistedCommands \in Seq(Seq(Seq(STRING)))
    /\ currentNodeId \in 0..MaxNodes
    /\ iteration \in Nat

Init ==
    /\ state = -1
    /\ solutionGraph = [0 |-> [status |-> "OPEN",
                               type |-> "ROOT",
                               info |-> <<"root">>,
                               successors |-> <<1, 2>>,
                               selectedMethod |-> "",
                               createdSubtasks |-> <<>>],
                       1 |-> [status |-> "OPEN",
                              type |-> "TASK",
                              info |-> <<"put_it">>,
                              successors |-> <<>>,
                              selectedMethod |-> "",
                              createdSubtasks |-> <<>>],
                       2 |-> [status |-> "OPEN",
                              type |-> "TASK",
                              info |-> <<"need1">>,
                              successors |-> <<>>,
                              selectedMethod |-> "",
                              createdSubtasks |-> <<>>]]
    /\ blacklistedCommands = <<>>
    /\ currentNodeId = 0
    /\ iteration = 0

(* Actions *)
ActionPutv(flagVal) ==
    state' = flagVal

ActionGetv(flagVal) ==
    /\ state = flagVal
    /\ state' = state

(* Methods return subtask arrays *)
MethodMErr == <<<<"action_putv", "0">>, <<"action_getv", "1">>>>
MethodM0 == <<<<"action_putv", "0">>, <<"action_getv", "0">>>>
MethodM1 == <<<<"action_putv", "1">>, <<"action_getv", "1">>>>
MethodMNeed1 == <<<<"action_getv", "1">>>>

(* Check if a command array is blacklisted *)
IsBlacklisted(cmd) ==
    \E i \in 1..Len(blacklistedCommands):
        /\ Len(blacklistedCommands[i]) = Len(cmd)
        /\ \A j \in 1..Len(cmd):
            /\ Len(blacklistedCommands[i][j]) = Len(cmd[j])
            /\ \A k \in 1..Len(cmd[j]):
                blacklistedCommands[i][j][k] = cmd[j][k]

(* Process put_it task *)
ProcessPutIt(nodeId) ==
    LET node == solutionGraph[nodeId]
        availableMethods == <<MethodMErr, MethodM0, MethodM1>>
    IN IF node.status = "OPEN"
       THEN LET foundMethod == CHOOSE m \in 1..Len(availableMethods):
                                    /\ ~IsBlacklisted(availableMethods[m])
                                    /\ \A i \in 1..m-1: IsBlacklisted(availableMethods[i])
            IN IF foundMethod \in DOMAIN availableMethods
               THEN /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                        [node EXCEPT !.status = "CLOSED",
                                              !.selectedMethod = IF foundMethod = 1 THEN "m_err"
                                                                 ELSE IF foundMethod = 2 THEN "m0"
                                                                 ELSE "m1",
                                              !.createdSubtasks = availableMethods[foundMethod],
                                              !.successors = <<>>]]
                    /\ state' = state
                    /\ blacklistedCommands' = blacklistedCommands
                    /\ currentNodeId' = nodeId
                    /\ iteration' = iteration + 1
               ELSE /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                        [node EXCEPT !.status = "FAILED"]]
                    /\ state' = state
                    /\ blacklistedCommands' = blacklistedCommands \cup {availableMethods[1], availableMethods[2], availableMethods[3]}
                    /\ currentNodeId' = nodeId
                    /\ iteration' = iteration + 1
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>

(* Process need1 task *)
ProcessNeed1(nodeId) ==
    LET node == solutionGraph[nodeId]
        availableMethods == <<MethodMNeed1>>
    IN IF node.status = "OPEN"
       THEN LET method == availableMethods[1]
            IN IF ~IsBlacklisted(method)
               THEN /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                        [node EXCEPT !.status = "CLOSED",
                                              !.selectedMethod = "m_need1",
                                              !.createdSubtasks = method,
                                              !.successors = <<>>]]
                    /\ state' = state
                    /\ blacklistedCommands' = blacklistedCommands
                    /\ currentNodeId' = nodeId
                    /\ iteration' = iteration + 1
               ELSE /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                        [node EXCEPT !.status = "FAILED"]]
                    /\ state' = state
                    /\ blacklistedCommands' = blacklistedCommands \cup {method}
                    /\ currentNodeId' = nodeId
                    /\ iteration' = iteration + 1
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>

(* Process action *)
ProcessAction(nodeId, actionInfo) ==
    LET node == solutionGraph[nodeId]
        actionName == actionInfo[1]
        actionArg == IF Len(actionInfo) > 1 THEN actionInfo[2] ELSE ""
    IN IF node.status = "OPEN"
       THEN IF actionName = "action_putv"
            THEN /\ state' = IF actionArg = "0" THEN 0 ELSE 1
                 /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                     [node EXCEPT !.status = "CLOSED"]]
                 /\ blacklistedCommands' = blacklistedCommands
                 /\ currentNodeId' = nodeId
                 /\ iteration' = iteration + 1
            ELSE IF actionName = "action_getv"
            THEN IF state = IF actionArg = "0" THEN 0 ELSE 1
                 THEN /\ state' = state
                      /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                          [node EXCEPT !.status = "CLOSED"]]
                      /\ blacklistedCommands' = blacklistedCommands
                      /\ currentNodeId' = nodeId
                      /\ iteration' = iteration + 1
                 ELSE /\ state' = state
                      /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                           [node EXCEPT !.status = "FAILED"]]
                      /\ blacklistedCommands' = blacklistedCommands
                      /\ currentNodeId' = nodeId
                      /\ iteration' = iteration + 1
            ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>

(* Add subtasks to graph *)
AddSubtasks(parentId, subtasks) ==
    LET node == solutionGraph[parentId]
        newNodes == [i \in 1..Len(subtasks) |->
                     [status |-> "OPEN",
                      type |-> IF subtasks[i][1] = "action_putv" \/ subtasks[i][1] = "action_getv"
                               THEN "ACTION"
                               ELSE "TASK",
                      info |-> subtasks[i],
                      successors |-> <<>>,
                      selectedMethod |-> "",
                      createdSubtasks |-> <<>>]]
        newSuccessors == [j \in 1..Len(subtasks) |-> MaxNodes + j]
    IN /\ solutionGraph' = [solutionGraph @@
                          [i \in DOMAIN newNodes |-> newNodes[i]] @@
                          [parentId |-> [node EXCEPT !.successors = node.successors \o <<newSuccessors[1]>> \o <<newSuccessors[2]>> \o ...]]]
       /\ state' = state
       /\ blacklistedCommands' = blacklistedCommands
       /\ currentNodeId' = parentId
       /\ iteration' = iteration

(* Planning step *)
Next ==
    \/ \E nodeId \in DOMAIN solutionGraph:
         /\ solutionGraph[nodeId].status = "OPEN"
         /\ solutionGraph[nodeId].type = "TASK"
         /\ solutionGraph[nodeId].info[1] = "put_it"
         /\ ProcessPutIt(nodeId)
    \/ \E nodeId \in DOMAIN solutionGraph:
         /\ solutionGraph[nodeId].status = "OPEN"
         /\ solutionGraph[nodeId].type = "TASK"
         /\ solutionGraph[nodeId].info[1] = "need1"
         /\ ProcessNeed1(nodeId)
    \/ \E nodeId \in DOMAIN solutionGraph:
         /\ solutionGraph[nodeId].status = "OPEN"
         /\ solutionGraph[nodeId].type = "ACTION"
         /\ ProcessAction(nodeId, solutionGraph[nodeId].info)

Spec == Init /\ [][Next]_vars

(* Properties *)
PlanningSucceeds ==
    /\ solutionGraph[1].status = "CLOSED"
    /\ solutionGraph[2].status = "CLOSED"
    /\ state = 1

PlanningFails ==
    \/ solutionGraph[1].status = "FAILED"
    \/ solutionGraph[2].status = "FAILED"

====
