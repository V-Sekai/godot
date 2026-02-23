---- MODULE VSIDSComparison ----
EXTENDS Naturals, Integers, Sequences, FiniteSets

(*
 * TLA+ model comparing different VSIDS reward strategies for blocks world.
 *
 * Compares:
 * 1. Linear reward: 100 / (1 + actions)
 * 2. Quadratic reward: 30000 / (1 + actions^2)
 * 3. Moderate reward: 100 / (1 + actions/10) with cap
 *
 * Goal: Find which strategy best reduces plan length during backtracking.
 *
 * Note: Using integers scaled by 1000 to represent decimals
 *)

CONSTANTS Methods, MaxIterations

VARIABLES
    (* Strategy 1: Linear reward *)
    linearActivities,
    linearActionCount,
    linearPlanLengths,

    (* Strategy 2: Quadratic reward *)
    quadActivities,
    quadActionCount,
    quadPlanLengths,

    (* Strategy 3: Moderate reward *)
    modActivities,
    modActionCount,
    modPlanLengths,

    iteration

(* Method that produces many actions (inefficient) *)
InefficientMethod == CHOOSE m \in Methods : TRUE

(* Method that produces few actions (efficient) *)
EfficientMethod == CHOOSE m \in Methods \ {InefficientMethod} : TRUE

Init ==
    /\ linearActivities = [m \in Methods |-> 0]
    /\ linearActionCount = 0
    /\ linearPlanLengths = <<>>
    /\ quadActivities = [m \in Methods |-> 0]
    /\ quadActionCount = 0
    /\ quadPlanLengths = <<>>
    /\ modActivities = [m \in Methods |-> 0]
    /\ modActionCount = 0
    /\ modPlanLengths = <<>>
    /\ iteration = 0

(* Strategy 1: Linear reward (scaled by 1000) *)
LinearReward(m, currentActions) ==
    LET reward == 100000 \div (1 + currentActions)  (* 100.0 * 1000 *)
        currentActivity == linearActivities[m]
        newActivity == currentActivity + reward
    IN  /\ linearActivities' = [linearActivities EXCEPT ![m] = newActivity]
        /\ linearActionCount' = linearActionCount + IF m = InefficientMethod THEN 10 ELSE 2
        /\ linearPlanLengths' = Append(linearPlanLengths, linearActionCount')
        /\ UNCHANGED <<quadActivities, quadActionCount, quadPlanLengths,
                       modActivities, modActionCount, modPlanLengths, iteration>>

LinearSelect ==
    LET scores == [m \in Methods |-> linearActivities[m] * 10]
        maxScore == CHOOSE s \in {scores[m] : m \in Methods} :
                     \A s2 \in {scores[m] : m \in Methods} : s >= s2
        candidates == {m \in Methods : scores[m] = maxScore}
        selectedMethod == IF EfficientMethod \in candidates THEN EfficientMethod ELSE InefficientMethod
    IN  /\ LinearReward(selectedMethod, linearActionCount)
        /\ UNCHANGED <<quadActivities, quadActionCount, quadPlanLengths,
                       modActivities, modActionCount, modPlanLengths, iteration>>

(* Strategy 2: Quadratic reward (scaled by 1000) *)
QuadReward(m, currentActions) ==
    LET squared == currentActions * currentActions
        reward == IF squared > 100000 THEN 0 ELSE (30000000 \div (1 + squared))  (* 30000.0 * 1000 *)
        currentActivity == quadActivities[m]
        newActivity == currentActivity + reward
    IN  /\ quadActivities' = [quadActivities EXCEPT ![m] = newActivity]
        /\ quadActionCount' = quadActionCount + IF m = InefficientMethod THEN 10 ELSE 2
        /\ quadPlanLengths' = Append(quadPlanLengths, quadActionCount')
        /\ UNCHANGED <<linearActivities, linearActionCount, linearPlanLengths,
                       modActivities, modActionCount, modPlanLengths, iteration>>

QuadSelect ==
    LET scores == [m \in Methods |-> quadActivities[m] * 100]
        maxScore == CHOOSE s \in {scores[m] : m \in Methods} :
                     \A s2 \in {scores[m] : m \in Methods} : s >= s2
        candidates == {m \in Methods : scores[m] = maxScore}
        selectedMethod == IF EfficientMethod \in candidates THEN EfficientMethod ELSE InefficientMethod
    IN  /\ QuadReward(selectedMethod, quadActionCount)
        /\ UNCHANGED <<linearActivities, linearActionCount, linearPlanLengths,
                       modActivities, modActionCount, modPlanLengths, iteration>>

(* Strategy 3: Moderate reward (scaled by 1000) *)
ModReward(m, currentActions) ==
    LET denominator == 10 + currentActions
        baseReward == IF denominator = 0 THEN 100000 ELSE (1000000 \div denominator)  (* 100.0 * 1000 *)
        reward == IF baseReward > 100000 THEN 100000 ELSE baseReward  (* Cap at 100.0 * 1000 *)
        currentActivity == modActivities[m]
        newActivity == currentActivity + reward
    IN  /\ modActivities' = [modActivities EXCEPT ![m] = newActivity]
        /\ modActionCount' = modActionCount + IF m = InefficientMethod THEN 10 ELSE 2
        /\ modPlanLengths' = Append(modPlanLengths, modActionCount')
        /\ UNCHANGED <<linearActivities, linearActionCount, linearPlanLengths,
                       quadActivities, quadActionCount, quadPlanLengths, iteration>>

ModSelect ==
    LET scores == [m \in Methods |-> modActivities[m] * 10]
        maxScore == CHOOSE s \in {scores[m] : m \in Methods} :
                     \A s2 \in {scores[m] : m \in Methods} : s >= s2
        candidates == {m \in Methods : scores[m] = maxScore}
        selectedMethod == IF EfficientMethod \in candidates THEN EfficientMethod ELSE InefficientMethod
    IN  /\ ModReward(selectedMethod, modActionCount)
        /\ UNCHANGED <<linearActivities, linearActionCount, linearPlanLengths,
                       quadActivities, quadActionCount, quadPlanLengths, iteration>>

(* Planning step - all strategies run in parallel *)
PlanningStep ==
    /\ iteration < MaxIterations
    /\ LinearSelect
    /\ QuadSelect
    /\ ModSelect
    /\ iteration' = iteration + 1

Next == PlanningStep

Spec == Init /\ [][Next]_<<linearActivities, linearActionCount, linearPlanLengths,
                        quadActivities, quadActionCount, quadPlanLengths,
                        modActivities, modActionCount, modPlanLengths,
                        iteration>>

TypeOK ==
    /\ linearActivities \in [Methods -> Int]
    /\ linearActionCount \in Nat
    /\ linearPlanLengths \in Seq(Nat)
    /\ quadActivities \in [Methods -> Int]
    /\ quadActionCount \in Nat
    /\ quadPlanLengths \in Seq(Nat)
    /\ modActivities \in [Methods -> Int]
    /\ modActionCount \in Nat
    /\ modPlanLengths \in Seq(Nat)
    /\ iteration \in Nat

(* Comparison properties *)

(* Which strategy has the shortest final plan? *)
LinearBest ==
    /\ Len(linearPlanLengths) > 0
    /\ Len(quadPlanLengths) > 0
    /\ Len(modPlanLengths) > 0
    /\ linearPlanLengths[Len(linearPlanLengths)] <= quadPlanLengths[Len(quadPlanLengths)]
    /\ linearPlanLengths[Len(linearPlanLengths)] <= modPlanLengths[Len(modPlanLengths)]

QuadBest ==
    /\ Len(linearPlanLengths) > 0
    /\ Len(quadPlanLengths) > 0
    /\ Len(modPlanLengths) > 0
    /\ quadPlanLengths[Len(quadPlanLengths)] <= linearPlanLengths[Len(linearPlanLengths)]
    /\ quadPlanLengths[Len(quadPlanLengths)] <= modPlanLengths[Len(modPlanLengths)]

ModBest ==
    /\ Len(linearPlanLengths) > 0
    /\ Len(quadPlanLengths) > 0
    /\ Len(modPlanLengths) > 0
    /\ modPlanLengths[Len(modPlanLengths)] <= linearPlanLengths[Len(linearPlanLengths)]
    /\ modPlanLengths[Len(modPlanLengths)] <= quadPlanLengths[Len(quadPlanLengths)]

(* Which strategy learns fastest (prefers efficient method earliest)? *)
LinearLearnsFastest ==
    /\ \E i \in DOMAIN linearPlanLengths :
        i <= 5 /\ \A j \in i..Len(linearPlanLengths) :
            linearPlanLengths[j] <= 20  (* Efficient method produces ~2 actions per step *)

QuadLearnsFastest ==
    /\ \E i \in DOMAIN quadPlanLengths :
        i <= 5 /\ \A j \in i..Len(quadPlanLengths) :
            quadPlanLengths[j] <= 20

ModLearnsFastest ==
    /\ \E i \in DOMAIN modPlanLengths :
        i <= 5 /\ \A j \in i..Len(modPlanLengths) :
            modPlanLengths[j] <= 20

Invariant == TypeOK

====
