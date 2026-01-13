---- MODULE TestCasesModel ----

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS MaxNodes

(*
  This model captures the execution flow of the failing test cases to identify edge cases.

  Backtracking Test 4:
  - Initial: flag = -1
  - Tasks: [put_it, need1]
  - put_it methods: m_err, m0, m1
  - need1 method: m_need1

  Sample Test 1:
  - Initial: flag[0] = True, flag[1..7] = False
  - Tasks: [task_method_1, task_method_2]
  - task_method_1 methods: m1_1, m1_2, m1_3
  - task_method_2 methods: m2_1, m2_2
*)

VARIABLES
    state,
    solutionGraph,
    blacklistedCommands,
    currentNodeId,
    iteration,
    nextNodeId

vars == <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration, nextNodeId>>

(* State representation *)
(* For Backtracking Test 4: state is a record with flag field *)
StateType == [flag: Int]

TypeOK ==
    /\ state \in [flag: Int]
    /\ solutionGraph \in [0..MaxNodes -> [status: {"OPEN", "CLOSED", "FAILED"},
                                          type: {"ROOT", "TASK", "ACTION"},
                                          info: Seq(STRING),
                                          successors: Seq(Nat),
                                          selectedMethod: STRING,
                                          createdSubtasks: Seq(Seq(STRING))]]
    /\ blacklistedCommands \in Seq(Seq(Seq(STRING)))
    /\ currentNodeId \in 0..MaxNodes
    /\ iteration \in Nat
    /\ nextNodeId \in Nat

(* Backtracking Test 4 Initial State *)
(* Use 999 to represent -1 *)
InitBT4 ==
    /\ state = [flag |-> 999]
    /\ solutionGraph = [i \in {0, 1, 2} |->
        IF i = 0 THEN [status |-> "OPEN",
                       type |-> "ROOT",
                       info |-> <<"root">>,
                       successors |-> <<1, 2>>,
                       selectedMethod |-> "",
                       createdSubtasks |-> <<>>]
        ELSE IF i = 1 THEN [status |-> "OPEN",
                             type |-> "TASK",
                             info |-> <<"put_it">>,
                             successors |-> <<>>,
                             selectedMethod |-> "",
                             createdSubtasks |-> <<>>]
        ELSE [status |-> "OPEN",
              type |-> "TASK",
              info |-> <<"need1">>,
              successors |-> <<>>,
              selectedMethod |-> "",
              createdSubtasks |-> <<>>]]
    /\ blacklistedCommands = <<>>
    /\ currentNodeId = 0
    /\ iteration = 0
    /\ nextNodeId = 3

(* Method definitions *)
MethodMErr == <<<<"action_putv", "0">>, <<"action_getv", "1">>>>
MethodM0 == <<<<"action_putv", "0">>, <<"action_getv", "0">>>>
MethodM1 == <<<<"action_putv", "1">>, <<"action_getv", "1">>>>
MethodMNeed1 == <<<<"action_getv", "1">>>>

(* Check if command array is blacklisted - with proper nested comparison *)
IsBlacklisted(cmd) ==
    \E i \in 1..Len(blacklistedCommands):
        LET bl == blacklistedCommands[i]
        IN /\ Len(bl) = Len(cmd)
           /\ \A j \in 1..Len(cmd):
               LET cmdElem == cmd[j]
                   blElem == bl[j]
               IN /\ Len(cmdElem) = Len(blElem)
                  /\ \A k \in 1..Len(cmdElem):
                      cmdElem[k] = blElem[k]

(* Process put_it task - try methods in order *)
ProcessPutItTask(nodeId) ==
    LET node == solutionGraph[nodeId]
        methods == <<MethodMErr, MethodM0, MethodM1>>
        methodNames == <<"m_err", "m0", "m1">>
    IN IF node.status = "OPEN"
       THEN LET TryMethod(methodIdx) ==
                 IF methodIdx > Len(methods)
                 THEN (* All methods failed *)
                     /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                         [node EXCEPT !.status = "FAILED"]]
                     /\ state' = state
                     /\ blacklistedCommands' = blacklistedCommands
                     /\ currentNodeId' = nodeId
                     /\ iteration' = iteration + 1
                     /\ nextNodeId' = nextNodeId
                 ELSE LET method == methods[methodIdx]
                      IN IF IsBlacklisted(method)
                         THEN TryMethod(methodIdx + 1)
                         ELSE (* Use this method *)
                             /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                                 [node EXCEPT !.status = "CLOSED",
                                                       !.selectedMethod = methodNames[methodIdx],
                                                       !.createdSubtasks = method,
                                                       !.successors = <<nextNodeId, nextNodeId + 1>>]]
                             /\ state' = state
                             /\ blacklistedCommands' = blacklistedCommands
                             /\ currentNodeId' = nodeId
                             /\ iteration' = iteration + 1
                             /\ nextNodeId' = nextNodeId + 2
            IN TryMethod(1)
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration, nextNodeId>>

(* Process need1 task *)
ProcessNeed1Task(nodeId) ==
    LET node == solutionGraph[nodeId]
    IN IF node.status = "OPEN"
       THEN IF IsBlacklisted(MethodMNeed1)
            THEN /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                     [node EXCEPT !.status = "FAILED"]]
                 /\ state' = state
                 /\ blacklistedCommands' = blacklistedCommands
                 /\ currentNodeId' = nodeId
                 /\ iteration' = iteration + 1
                 /\ nextNodeId' = nextNodeId
            ELSE /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                      [node EXCEPT !.status = "CLOSED",
                                            !.selectedMethod = "m_need1",
                                            !.createdSubtasks = MethodMNeed1,
                                            !.successors = <<nextNodeId>>]]
                 /\ state' = state
                 /\ blacklistedCommands' = blacklistedCommands
                 /\ currentNodeId' = nodeId
                 /\ iteration' = iteration + 1
                 /\ nextNodeId' = nextNodeId + 1
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration, nextNodeId>>

(* Process action_putv *)
ProcessPutvAction(nodeId, arg) ==
    LET node == solutionGraph[nodeId]
        flagVal == IF arg = "0" THEN 0 ELSE 1
    IN IF node.status = "OPEN"
       THEN /\ state' = [state EXCEPT !.flag = flagVal]
            /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                 [node EXCEPT !.status = "CLOSED"]]
            /\ blacklistedCommands' = blacklistedCommands
            /\ currentNodeId' = nodeId
            /\ iteration' = iteration + 1
            /\ nextNodeId' = nextNodeId
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration, nextNodeId>>

(* Process action_getv *)
ProcessGetvAction(nodeId, arg) ==
    LET node == solutionGraph[nodeId]
        flagVal == IF arg = "0" THEN 0 ELSE 1
        currentFlag == IF state.flag = 999 THEN -1 ELSE state.flag
    IN IF node.status = "OPEN"
       THEN IF currentFlag = flagVal
            THEN /\ state' = state
                 /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                      [node EXCEPT !.status = "CLOSED"]]
                 /\ blacklistedCommands' = blacklistedCommands
                 /\ currentNodeId' = nodeId
                 /\ iteration' = iteration + 1
                 /\ nextNodeId' = nextNodeId
            ELSE /\ state' = state
                 /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                      [node EXCEPT !.status = "FAILED"]]
                 /\ blacklistedCommands' = blacklistedCommands
                 /\ currentNodeId' = nodeId
                 /\ iteration' = iteration + 1
                 /\ nextNodeId' = nextNodeId
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration, nextNodeId>>

(* Process action *)
ProcessAction(nodeId) ==
    LET node == solutionGraph[nodeId]
        actionInfo == node.info
        actionName == actionInfo[1]
        actionArg == IF Len(actionInfo) > 1 THEN actionInfo[2] ELSE ""
    IN IF actionName = "action_putv"
       THEN ProcessPutvAction(nodeId, actionArg)
       ELSE IF actionName = "action_getv"
       THEN ProcessGetvAction(nodeId, actionArg)
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration, nextNodeId>>

(* Backtracking: when a node fails, blacklist its parent's method and reopen parent *)
Backtrack(nodeId) ==
    LET node == solutionGraph[nodeId]
        parentId == CHOOSE p \in DOMAIN solutionGraph:
                      nodeId \in solutionGraph[p].successors
        parentNode == IF parentId \in DOMAIN solutionGraph THEN solutionGraph[parentId] ELSE [status |-> "OPEN", type |-> "ROOT", info |-> <<>>, successors |-> <<>>, selectedMethod |-> "", createdSubtasks |-> <<>>]
        parentMethod == parentNode.createdSubtasks
    IN IF parentId \in DOMAIN solutionGraph /\ parentNode.status = "CLOSED" /\ parentMethod # <<>>
       THEN /\ blacklistedCommands' = IF ~IsBlacklisted(parentMethod)
                                        THEN Append(blacklistedCommands, parentMethod)
                                        ELSE blacklistedCommands
            /\ solutionGraph' = [solutionGraph EXCEPT ![parentId] =
                                 [parentNode EXCEPT !.status = "OPEN",
                                       !.selectedMethod = "",
                                       !.createdSubtasks = <<>>,
                                       !.successors = <<>>]]
            /\ state' = state
            /\ currentNodeId' = parentId
            /\ iteration' = iteration + 1
            /\ nextNodeId' = nextNodeId
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration, nextNodeId>>

(* Planning step *)
Next ==
    \/ \E nodeId \in DOMAIN solutionGraph:
         /\ solutionGraph[nodeId].status = "OPEN"
         /\ solutionGraph[nodeId].type = "TASK"
         /\ solutionGraph[nodeId].info[1] = "put_it"
         /\ ProcessPutItTask(nodeId)
    \/ \E nodeId \in DOMAIN solutionGraph:
         /\ solutionGraph[nodeId].status = "OPEN"
         /\ solutionGraph[nodeId].type = "TASK"
         /\ solutionGraph[nodeId].info[1] = "need1"
         /\ ProcessNeed1Task(nodeId)
    \/ \E nodeId \in DOMAIN solutionGraph:
         /\ solutionGraph[nodeId].status = "OPEN"
         /\ solutionGraph[nodeId].type = "ACTION"
         /\ ProcessAction(nodeId)
    \/ \E nodeId \in DOMAIN solutionGraph:
         /\ solutionGraph[nodeId].status = "FAILED"
         /\ Backtrack(nodeId)

Spec == InitBT4 /\ [][Next]_vars

(* Properties *)
PlanningSucceeds ==
    /\ solutionGraph[1].status = "CLOSED"
    /\ solutionGraph[2].status = "CLOSED"
    /\ (state.flag = 1 \/ state.flag = 999)  (* 1 or -1 represented as 999 *)

PlanningFails ==
    \/ solutionGraph[1].status = "FAILED"
    \/ solutionGraph[2].status = "FAILED"

====
