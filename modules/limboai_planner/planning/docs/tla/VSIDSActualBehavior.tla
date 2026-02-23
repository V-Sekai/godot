---- MODULE VSIDSActualBehavior ----
EXTENDS Naturals, Integers, Sequences, FiniteSets

(*
 * TLA+ model of the ACTUAL current implementation behavior.
 *
 * Current implementation:
 * - Reward: 100 / (1 + action_count / 10), capped at 100
 * - Activity scaling: 10x
 * - Activity resets at start of each solve
 * - Rewards applied immediately when methods succeed
 *
 * Note: Using integers scaled by 1000 to represent decimals (e.g., 100.0 = 100000)
 *)

CONSTANTS Methods, MaxActions

VARIABLES
    methodActivities,    (* Map: Method -> Int (scaled by 1000) *)
    actionCount,          (* Current number of actions *)
    selectedMethods,      (* Sequence of selected methods *)
    planLengths,          (* Plan length at each method selection *)
    bumpCount            (* Bump count for decay *)

InefficientMethod == CHOOSE m \in Methods : TRUE
EfficientMethod == CHOOSE m \in Methods \ {InefficientMethod} : TRUE

Init ==
    /\ methodActivities = [m \in Methods |-> 0]
    /\ actionCount = 0
    /\ selectedMethods = <<>>
    /\ planLengths = <<>>
    /\ bumpCount = 0

(* Current implementation reward (scaled by 1000) *)
(* Reward: 100 / (1 + actions/10) = 1000 / (10 + actions) *)
RewardMethod(m, currentActions) ==
    LET denominator == 10 + currentActions
        baseReward == IF denominator = 0 THEN 100000 ELSE (1000000 \div denominator)  (* 100.0 * 1000, scaled *)
        reward == IF baseReward > 100000 THEN 100000 ELSE baseReward  (* Cap at 100.0 * 1000 *)
        currentActivity == methodActivities[m]
        newActivity == currentActivity + reward
    IN  /\ methodActivities' = [methodActivities EXCEPT ![m] = newActivity]
        /\ actionCount' = actionCount + IF m = InefficientMethod THEN 10 ELSE 2
        /\ selectedMethods' = Append(selectedMethods, m)
        /\ planLengths' = Append(planLengths, actionCount')
        /\ bumpCount' = bumpCount + 1

(* Current implementation selection (10x scaling) *)
(* When scores are equal, prefer efficient method (deterministic tie-breaker) *)
SelectMethod ==
    LET scores == [m \in Methods |-> methodActivities[m] * 10]
        maxScore == CHOOSE s \in {scores[m] : m \in Methods} :
                     \A s2 \in {scores[m] : m \in Methods} : s >= s2
        candidates == {m \in Methods : scores[m] = maxScore}
        selectedMethod == IF EfficientMethod \in candidates THEN EfficientMethod ELSE InefficientMethod
    IN  /\ RewardMethod(selectedMethod, actionCount)
        /\ UNCHANGED <<>>

(* Decay every 100 bumps (95% = multiply by 950/1000) *)
DecayActivities ==
    LET newActivities == [m \in Methods |-> (methodActivities[m] * 950) \div 1000]
    IN  /\ methodActivities' = newActivities
        /\ bumpCount' = 0
        /\ UNCHANGED <<actionCount, selectedMethods, planLengths>>

ShouldDecay == bumpCount >= 100

PlanningStep ==
    /\ actionCount < MaxActions
    /\ SelectMethod

Next ==
    \/ PlanningStep
    \/ (ShouldDecay /\ DecayActivities /\ UNCHANGED <<actionCount, selectedMethods, planLengths>>)

Spec == Init /\ [][Next]_<<methodActivities, actionCount, selectedMethods, planLengths, bumpCount>>

TypeOK ==
    /\ methodActivities \in [Methods -> Int]
    /\ actionCount \in Nat
    /\ selectedMethods \in Seq(Methods)
    /\ planLengths \in Seq(Nat)
    /\ bumpCount \in Nat

ActivityNonNegative == \A m \in Methods : methodActivities[m] >= 0

(* Analysis properties *)

(* Does it eventually prefer efficient method? *)
EventuallyPrefersEfficient ==
    <>(\E i \in DOMAIN selectedMethods :
        selectedMethods[i] = EfficientMethod /\
        \A j \in i+1..Len(selectedMethods) : selectedMethods[j] = EfficientMethod)

(* Does plan length decrease over time? *)
PlanLengthDecreases ==
    \A i \in DOMAIN planLengths \ {1} :
        planLengths[i] <= planLengths[i-1] \/ i = 1

(* What's the final plan length? *)
FinalPlanLength ==
    IF Len(planLengths) > 0
    THEN planLengths[Len(planLengths)]
    ELSE 0

(* How many iterations until efficient method is preferred? *)
IterationsToEfficient ==
    CHOOSE i \in DOMAIN selectedMethods :
        selectedMethods[i] = EfficientMethod /\
        \A j \in i+1..Len(selectedMethods) : selectedMethods[j] = EfficientMethod

Invariant == TypeOK /\ ActivityNonNegative

====
