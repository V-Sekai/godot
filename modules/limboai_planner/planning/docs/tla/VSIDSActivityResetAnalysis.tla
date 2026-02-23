---- MODULE VSIDSActivityResetAnalysis ----
EXTENDS Naturals, Integers, Sequences, FiniteSets, Reals

(*
 * Analysis: Should VSIDS activity reset with each solution graph?
 *
 * This model compares two strategies:
 * 1. Activity persists across find_plan() calls
 * 2. Activity resets with each solution graph
 *)

CONSTANTS Methods, PlanningAttempts, ProblemTypes

VARIABLES
    (* Strategy 1: Persistent activity *)
    persistentActivities,    (* Map: Method -> Real *)
    persistentBumpCount,
    persistentPlanLengths,   (* Sequence of plan lengths *)

    (* Strategy 2: Reset activity *)
    resetActivities,         (* Map: Method -> Real *)
    resetBumpCount,
    resetPlanLengths,        (* Sequence of plan lengths *)

    attemptCount,
    currentProblemType       (* Which problem we're solving *)

BadMethod == CHOOSE m \in Methods : TRUE
GoodMethod == CHOOSE m \in Methods \ {BadMethod} : TRUE

Init ==
    /\ persistentActivities = [m \in Methods |-> 0.0]
    /\ persistentBumpCount = 0
    /\ persistentPlanLengths = <<>>
    /\ resetActivities = [m \in Methods |-> 0.0]
    /\ resetBumpCount = 0
    /\ resetPlanLengths = <<>>
    /\ attemptCount = 0
    /\ currentProblemType = CHOOSE pt \in ProblemTypes : TRUE

(* Strategy 1: Activity persists *)
PersistentReward(m, planLength) ==
    LET reward == 100.0 / (1.0 + planLength)
        newActivity == persistentActivities[m] + reward
    IN  /\ persistentActivities' = [persistentActivities EXCEPT ![m] = newActivity]
        /\ persistentBumpCount' = persistentBumpCount + 1
        /\ UNCHANGED <<resetActivities, resetBumpCount, resetPlanLengths, attemptCount, currentProblemType>>

PersistentSelect ==
    LET scores == [m \in Methods |-> persistentActivities[m]]
        maxScore == CHOOSE s \in {scores[m] : m \in Methods} :
                     \A s2 \in {scores[m] : m \in Methods} : s >= s2
        selectedMethod == CHOOSE m \in Methods : scores[m] = maxScore
        planLength == IF selectedMethod = BadMethod THEN 100 ELSE 10
    IN  /\ persistentPlanLengths' = Append(persistentPlanLengths, planLength)
        /\ attemptCount' = attemptCount + 1
        /\ PersistentReward(selectedMethod, planLength)
        /\ UNCHANGED <<resetActivities, resetBumpCount, resetPlanLengths, currentProblemType>>

(* Strategy 2: Activity resets each time *)
ResetActivity ==
    /\ resetActivities' = [m \in Methods |-> 0.0]
    /\ resetBumpCount' = 0
    /\ UNCHANGED <<persistentActivities, persistentBumpCount, persistentPlanLengths, attemptCount, currentProblemType>>

ResetReward(m, planLength) ==
    LET reward == 100.0 / (1.0 + planLength)
        newActivity == resetActivities[m] + reward
    IN  /\ resetActivities' = [resetActivities EXCEPT ![m] = newActivity]
        /\ resetBumpCount' = resetBumpCount + 1
        /\ UNCHANGED <<persistentActivities, persistentBumpCount, persistentPlanLengths, resetPlanLengths, attemptCount, currentProblemType>>

ResetSelect ==
    /\ ResetActivity  (* Reset at start of each planning attempt *)
    /\ LET scores == [m \in Methods |-> resetActivities[m]]
           maxScore == CHOOSE s \in {scores[m] : m \in Methods} :
                        \A s2 \in {scores[m] : m \in Methods} : s >= s2
           selectedMethod == CHOOSE m \in Methods : scores[m] = maxScore
           planLength == IF selectedMethod = BadMethod THEN 100 ELSE 10
       IN  /\ resetPlanLengths' = Append(resetPlanLengths, planLength)
           /\ attemptCount' = attemptCount + 1
           /\ ResetReward(selectedMethod, planLength)
           /\ UNCHANGED <<persistentActivities, persistentBumpCount, persistentPlanLengths, currentProblemType>>

PlanningAttempt ==
    /\ attemptCount < PlanningAttempts
    /\ \/ PersistentSelect
       \/ ResetSelect

Next == PlanningAttempt

Spec == Init /\ [][Next]_<<persistentActivities, persistentBumpCount, persistentPlanLengths,
                        resetActivities, resetBumpCount, resetPlanLengths,
                        attemptCount, currentProblemType>>

(* Analysis Properties *)

(* Strategy 1: Should improve over time (later attempts better) *)
PersistentImproves ==
    \A i \in DOMAIN persistentPlanLengths \ {1} :
        persistentPlanLengths[i] <= persistentPlanLengths[i-1] \/ i = 1

(* Strategy 2: Each attempt starts fresh, so no cross-attempt learning *)
ResetStartsFresh ==
    \A i \in DOMAIN resetPlanLengths :
        resetPlanLengths[i] >= 10  (* Always starts from scratch, may pick bad method first *)

(* Key insight: Strategy 1 learns across attempts, Strategy 2 doesn't *)
====
