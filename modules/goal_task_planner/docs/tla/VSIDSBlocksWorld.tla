---- MODULE VSIDSBlocksWorld ----
EXTENDS Naturals, Integers, Sequences, FiniteSets, Reals

(*
 * TLA+ model to verify VSIDS should persist across planning attempts
 * and improve plan quality (reduce action count) over time.
 *
 * Key insights:
 * 1. VSIDS activity should NOT be cleared between find_plan() calls
 * 2. VSIDS should REWARD successful methods (bump on success), not punish failures
 * 3. Methods that produce shorter plans should get higher rewards
 *)

CONSTANTS Methods, PlanningAttempts

VARIABLES
    methodActivities,    (* Map: Method -> Real (activity score) *)
    bumpCount,           (* Number of bumps since last decay *)
    planLengths,         (* Sequence of plan lengths from each attempt *)
    attemptCount         (* Current planning attempt number *)

(* Method that generates long plans (bad) *)
BadMethod == CHOOSE m \in Methods : TRUE

(* Method that generates short plans (good) *)
GoodMethod == CHOOSE m \in Methods \ {BadMethod} : TRUE

Init ==
    /\ methodActivities = [m \in Methods |-> 0.0]
    /\ bumpCount = 0
    /\ planLengths = <<>>
    /\ attemptCount = 0

(* Bump activity when method succeeds (reward success) *)
(* Reward shorter plans more: inverse of plan length + bonus for fewer subtasks *)
BumpActivityOnSuccess(m, planLength, subtaskCount) ==
    LET currentActivity == methodActivities[m]
        (* Reward shorter plans: higher reward for shorter plans *)
        planReward == 100.0 / (1.0 + planLength)
        (* Bonus for methods with fewer subtasks (more direct) *)
        subtaskBonus == 50.0 / (1.0 + subtaskCount)
        totalReward == planReward + subtaskBonus
        newActivity == currentActivity + totalReward
    IN  /\ methodActivities' = [methodActivities EXCEPT ![m] = newActivity]
        /\ bumpCount' = bumpCount + 1
        /\ UNCHANGED <<planLengths, attemptCount>>

(* Decay activities periodically *)
DecayActivities ==
    LET decayFactor == 0.95
        newActivities == [m \in Methods |-> methodActivities[m] * decayFactor]
    IN  /\ methodActivities' = newActivities
        /\ bumpCount' = 0
        /\ UNCHANGED <<planLengths, attemptCount>>

ShouldDecay == bumpCount >= 100

(* Select method with highest activity *)
SelectMethod ==
    LET scores == [m \in Methods |-> methodActivities[m]]
        maxScore == CHOOSE s \in {scores[m] : m \in Methods} :
                     \A s2 \in {scores[m] : m \in Methods} : s >= s2
        selectedMethod == CHOOSE m \in Methods : scores[m] = maxScore
        (* Plan length: bad method = 100 actions, good method = 10 actions *)
        planLength == IF selectedMethod = BadMethod THEN 100 ELSE 10
        (* Subtask count: bad method has more subtasks *)
        subtaskCount == IF selectedMethod = BadMethod THEN 10 ELSE 2
    IN  /\ planLengths' = Append(planLengths, planLength)
        /\ attemptCount' = attemptCount + 1
        /\ BumpActivityOnSuccess(selectedMethod, planLength, subtaskCount)  (* Reward the method that was used *)

(* Planning attempt *)
PlanningAttempt ==
    /\ attemptCount < PlanningAttempts
    /\ SelectMethod
    /\ IF ShouldDecay THEN DecayActivities ELSE UNCHANGED methodActivities

Next ==
    \/ PlanningAttempt
    \/ (ShouldDecay /\ DecayActivities /\ UNCHANGED <<planLengths, attemptCount>>)

Spec == Init /\ [][Next]_<<methodActivities, bumpCount, planLengths, attemptCount>>

TypeOK ==
    /\ methodActivities \in [Methods -> Real]
    /\ bumpCount \in Nat
    /\ planLengths \in Seq(Nat)
    /\ attemptCount \in Nat

ActivityNonNegative == \A m \in Methods : methodActivities[m] >= 0.0

(* VSIDS should improve over time: later plans should be shorter *)
PlanImproves ==
    \A i \in DOMAIN planLengths \ {1} :
        planLengths[i] <= planLengths[i-1] \/ i = 1

(* After bad method fails and gets bumped, good method should be selected *)
GoodMethodSelectedAfterFailure ==
    \A i \in DOMAIN planLengths :
        IF i > 1 /\ planLengths[i-1] = 100  (* Previous attempt used bad method *)
        THEN planLengths[i] = 10  (* Next attempt should use good method *)
        ELSE TRUE

Invariant == TypeOK /\ ActivityNonNegative

(* Property: VSIDS should eventually prefer good method *)
EventuallyPrefersGoodMethod ==
    <>(\A i \in DOMAIN planLengths : planLengths[i] = 10)

====
