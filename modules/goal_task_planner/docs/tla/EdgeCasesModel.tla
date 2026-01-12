---- MODULE EdgeCasesModel ----

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS MaxNodes

(*
  Simplified model to identify edge cases in the failing tests.
  Focuses on the key issue: array comparison in blacklist checking.
*)

VARIABLES
    state,
    solutionGraph,
    blacklistedCommands,
    currentNodeId,
    iteration

vars == <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>

(* State: flag value, using 999 to represent -1 *)
TypeOK ==
    /\ state \in {999, 0, 1}  (* 999 = -1 *)
    /\ solutionGraph \in [Nat -> [status: {"OPEN", "CLOSED", "FAILED"},
                                  type: {"ROOT", "TASK", "ACTION"},
                                  info: Seq(STRING),
                                  successors: Seq(Nat),
                                  selectedMethod: STRING,
                                  createdSubtasks: Seq(Seq(STRING))]]
    /\ blacklistedCommands \in Seq(Seq(Seq(STRING)))
    /\ currentNodeId \in Nat
    /\ iteration \in Nat

(* Method arrays *)
MethodMErr == <<<<"action_putv", "0">>, <<"action_getv", "1">>>>
MethodM0 == <<<<"action_putv", "0">>, <<"action_getv", "0">>>>
MethodM1 == <<<<"action_putv", "1">>, <<"action_getv", "1">>>>
MethodMNeed1 == <<<<"action_getv", "1">>>>

(* Check if command array is blacklisted - KEY EDGE CASE *)
(* This is where the bug likely is: nested array comparison *)
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

Init ==
    /\ state = 999  (* -1 *)
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

(* Process put_it: try m_err, then m0, then m1 *)
ProcessPutIt(nodeId) ==
    LET node == solutionGraph[nodeId]
    IN IF node.status = "OPEN"
       THEN \/ (* Try m_err *)
               /\ ~IsBlacklisted(MethodMErr)
               /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                    [node EXCEPT !.status = "CLOSED",
                                          !.selectedMethod = "m_err",
                                          !.createdSubtasks = MethodMErr,
                                          !.successors = <<3, 4>>]]
               /\ state' = state
               /\ blacklistedCommands' = blacklistedCommands
               /\ currentNodeId' = nodeId
               /\ iteration' = iteration + 1
           \/ (* Try m0 *)
               /\ ~IsBlacklisted(MethodM0)
               /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                    [node EXCEPT !.status = "CLOSED",
                                          !.selectedMethod = "m0",
                                          !.createdSubtasks = MethodM0,
                                          !.successors = <<5, 6>>]]
               /\ state' = state
               /\ blacklistedCommands' = blacklistedCommands
               /\ currentNodeId' = nodeId
               /\ iteration' = iteration + 1
           \/ (* Try m1 *)
               /\ ~IsBlacklisted(MethodM1)
               /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                    [node EXCEPT !.status = "CLOSED",
                                          !.selectedMethod = "m1",
                                          !.createdSubtasks = MethodM1,
                                          !.successors = <<7, 8>>]]
               /\ state' = state
               /\ blacklistedCommands' = blacklistedCommands
               /\ currentNodeId' = nodeId
               /\ iteration' = iteration + 1
           \/ (* All methods blacklisted - fail *)
               /\ IsBlacklisted(MethodMErr)
               /\ IsBlacklisted(MethodM0)
               /\ IsBlacklisted(MethodM1)
               /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                    [node EXCEPT !.status = "FAILED"]]
               /\ state' = state
               /\ blacklistedCommands' = blacklistedCommands
               /\ currentNodeId' = nodeId
               /\ iteration' = iteration + 1
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>

(* Process need1 *)
ProcessNeed1(nodeId) ==
    LET node == solutionGraph[nodeId]
    IN IF node.status = "OPEN"
       THEN IF ~IsBlacklisted(MethodMNeed1)
            THEN /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                     [node EXCEPT !.status = "CLOSED",
                                           !.selectedMethod = "m_need1",
                                           !.createdSubtasks = MethodMNeed1,
                                           !.successors = <<9>>]]
                 /\ state' = state
                 /\ blacklistedCommands' = blacklistedCommands
                 /\ currentNodeId' = nodeId
                 /\ iteration' = iteration + 1
            ELSE /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                      [node EXCEPT !.status = "FAILED"]]
                 /\ state' = state
                 /\ blacklistedCommands' = blacklistedCommands
                 /\ currentNodeId' = nodeId
                 /\ iteration' = iteration + 1
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>

(* Execute action_putv *)
ExecutePutv(nodeId, val) ==
    LET node == solutionGraph[nodeId]
        flagVal == IF val = "0" THEN 0 ELSE 1
    IN IF node.status = "OPEN"
       THEN /\ state' = flagVal
            /\ solutionGraph' = [solutionGraph EXCEPT ![nodeId] =
                                 [node EXCEPT !.status = "CLOSED"]]
            /\ blacklistedCommands' = blacklistedCommands
            /\ currentNodeId' = nodeId
            /\ iteration' = iteration + 1
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>

(* Execute action_getv *)
ExecuteGetv(nodeId, val) ==
    LET node == solutionGraph[nodeId]
        flagVal == IF val = "0" THEN 0 ELSE 1
        currentFlag == IF state = 999 THEN 999 ELSE state  (* 999 represents -1 *)
    IN IF node.status = "OPEN"
       THEN IF currentFlag = flagVal
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

(* Backtrack: when action fails, blacklist parent's method and reopen parent *)
Backtrack(nodeId) ==
    LET node == solutionGraph[nodeId]
        parentId == CHOOSE p \in DOMAIN solutionGraph:
                      nodeId \in solutionGraph[p].successors
        parentNode == IF parentId \in DOMAIN solutionGraph THEN solutionGraph[parentId]
                      ELSE [status |-> "OPEN", type |-> "ROOT", info |-> <<>>, successors |-> <<>>, selectedMethod |-> "", createdSubtasks |-> <<>>]
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
       ELSE UNCHANGED <<state, solutionGraph, blacklistedCommands, currentNodeId, iteration>>

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
         /\ solutionGraph[nodeId].info[1] = "action_putv"
         /\ ExecutePutv(nodeId, solutionGraph[nodeId].info[2])
    \/ \E nodeId \in DOMAIN solutionGraph:
         /\ solutionGraph[nodeId].status = "OPEN"
         /\ solutionGraph[nodeId].type = "ACTION"
         /\ solutionGraph[nodeId].info[1] = "action_getv"
         /\ ExecuteGetv(nodeId, solutionGraph[nodeId].info[2])
    \/ \E nodeId \in DOMAIN solutionGraph:
         /\ solutionGraph[nodeId].status = "FAILED"
         /\ Backtrack(nodeId)

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
