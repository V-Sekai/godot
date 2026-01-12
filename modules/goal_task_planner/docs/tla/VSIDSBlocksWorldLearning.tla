---- MODULE VSIDSBlocksWorldLearning ----
EXTENDS Naturals, Integers, Sequences, FiniteSets, Reals

(*
 * TLA+ model of VSIDS learning during a single blocks world solve.
 *
 * Key insight: VSIDS learns during backtracking within one solve.
 * Methods that lead to fewer actions get higher rewards.
 *)

CONSTANTS Methods, MaxActions

VARIABLES
    methodActivities,    (* Map: Method -> Real (activity score) *)
    actionCount,          (* Current number of actions in plan *)
    selectedMethods,      (* Sequence of methods selected so far *)
    planLengths,          (* Sequence of plan lengths at each method selection *)
    bumpCount             (* Number of bumps for decay tracking *)

(* Method that produces many actions (inefficient) *)
InefficientMethod == CHOOSE m \in Methods : TRUE

(* Method that produces few actions (efficient) *)
EfficientMethod == CHOOSE m \in Methods \ {InefficientMethod} : TRUE

Init ==
    /\ methodActivities = [m \in Methods |-> 0.0]
    /\ actionCount = 0
    /\ selectedMethods = <<>>
    /\ planLengths = <<>>
    /\ bumpCount = 0

(* Reward method immediately when it succeeds *)
(* Reward uses quadratic scaling: 30000 / (1 + actions^2) *)
(* This creates a steep penalty for longer plans *)
RewardMethod(m, currentActions) ==
    LET reward == 30000.0 / (1.0 + currentActions * currentActions)
        currentActivity == methodActivities[m]
        newActivity == currentActivity + reward
    IN  /\ methodActivities' = [methodActivities EXCEPT ![m] = newActivity]
        /\ actionCount' = actionCount + IF m = InefficientMethod THEN 10 ELSE 2
        /\ selectedMethods' = Append(selectedMethods, m)
        /\ planLengths' = Append(planLengths, actionCount')
        /\ bumpCount' = bumpCount + 1

(* Select method with highest activity *)
(* Activity is scaled by 100x to dominate over other factors *)
SelectMethod ==
    LET scores == [m \in Methods |-> methodActivities[m] * 100.0]
        maxScore == CHOOSE s \in {scores[m] : m \in Methods} :
                     \A s2 \in {scores[m] : m \in Methods} : s >= s2
        selectedMethod == CHOOSE m \in Methods : scores[m] = maxScore
    IN  /\ RewardMethod(selectedMethod, actionCount)
        /\ UNCHANGED <<>>

(* Decay activities periodically *)
DecayActivities ==
    LET decayFactor == 0.95
        newActivities == [m \in Methods |-> methodActivities[m] * decayFactor]
    IN  /\ methodActivities' = newActivities
        /\ bumpCount' = 0
        /\ UNCHANGED <<actionCount, selectedMethods, planLengths>>

ShouldDecay == bumpCount >= 100

(* Planning step *)
PlanningStep ==
    /\ actionCount < MaxActions
    /\ SelectMethod
    /\ IF ShouldDecay THEN DecayActivities ELSE UNCHANGED methodActivities

Next ==
    \/ PlanningStep
    \/ (ShouldDecay /\ DecayActivities /\ UNCHANGED <<actionCount, selectedMethods, planLengths>>)

Spec == Init /\ [][Next]_<<methodActivities, actionCount, selectedMethods, planLengths, bumpCount>>

TypeOK ==
    /\ methodActivities \in [Methods -> Real]
    /\ actionCount \in Nat
    /\ selectedMethods \in Seq(Methods)
    /\ planLengths \in Seq(Nat)
    /\ bumpCount \in Nat

ActivityNonNegative == \A m \in Methods : methodActivities[m] >= 0.0

(* VSIDS should prefer efficient method after learning *)
EventuallyPrefersEfficient ==
    <>(\E i \in DOMAIN selectedMethods :
        selectedMethods[i] = EfficientMethod /\
        \A j \in i+1..Len(selectedMethods) : selectedMethods[j] = EfficientMethod)

(* Plan length should decrease over time *)
PlanLengthDecreases ==
    \A i \in DOMAIN planLengths \ {1} :
        planLengths[i] <= planLengths[i-1] \/ i = 1

Invariant == TypeOK /\ ActivityNonNegative

====
