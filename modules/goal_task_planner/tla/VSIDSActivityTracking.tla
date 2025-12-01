---- MODULE VSIDSActivityTracking ----
(*
  TLA+ model for VSIDS-style method activity tracking in HTN planner.
  
  This models:
  - Method activity scores (like VSIDS variable activities)
  - Activity bumping on conflicts (backtracking)
  - Activity decay over time
  - Method selection based on activity scores
*)

EXTENDS Naturals, Integers, Sequences, FiniteSets

CONSTANTS Methods, MaxBumpsBeforeDecay

VARIABLES 
    methodActivities,    (* Map: Method -> Int (activity score) *)
    activityVarInc,       (* Increment value (grows over time) *)
    bumpCount,           (* Number of bumps since last decay *)
    selectedMethods      (* Sequence of selected methods (for verification) *)

(* Initialize activities to 0 for all methods *)
Init ==
    /\ methodActivities = [m \in Methods |-> 0]
    /\ activityVarInc = 1
    /\ bumpCount = 0
    /\ selectedMethods = <<>>

(* Get activity score for a method *)
GetActivity(m) == methodActivities[m]

(* Bump activity of a method (called on conflict) *)
BumpActivity(m) ==
    LET newActivity == methodActivities[m] + activityVarInc
    IN  /\ methodActivities' = [methodActivities EXCEPT ![m] = newActivity]
        /\ bumpCount' = bumpCount + 1
        /\ activityVarInc' = activityVarInc
        /\ selectedMethods' = selectedMethods

(* Decay all activities and increase var_inc *)
(* Simplified: multiply by 95/100 (0.95) and increase var_inc by 5% *)
DecayActivities ==
    LET newVarInc == activityVarInc + (activityVarInc \div 20)  (* Increase by 5% *)
        newActivities == [m \in Methods |-> (methodActivities[m] * 95) \div 100]  (* Multiply by 0.95 *)
    IN  /\ methodActivities' = newActivities
        /\ activityVarInc' = newVarInc
        /\ bumpCount' = 0
        /\ selectedMethods' = selectedMethods

(* Check if decay should occur *)
ShouldDecay == bumpCount >= MaxBumpsBeforeDecay

(* Select method with highest activity from candidates *)
SelectBestMethod(candidates) ==
    LET scores == [m \in candidates |-> methodActivities[m]]
        maxScore == CHOOSE s \in {scores[m] : m \in candidates} : 
                     \A s2 \in {scores[m] : m \in candidates} : s >= s2
        bestMethod == CHOOSE m \in candidates : scores[m] = maxScore
    IN  /\ selectedMethods' = Append(selectedMethods, bestMethod)
        /\ UNCHANGED <<methodActivities, activityVarInc, bumpCount>>

(* Bump activities of all methods in conflict path *)
BumpConflictPath(conflictPath) ==
    LET newActivities == 
        [m \in Methods |-> 
         IF m \in conflictPath
         THEN methodActivities[m] + activityVarInc
         ELSE methodActivities[m]]
    IN  /\ methodActivities' = newActivities
        /\ bumpCount' = bumpCount + Cardinality(conflictPath)
        /\ activityVarInc' = activityVarInc
        /\ selectedMethods' = selectedMethods

(* Next state: either bump activity, decay, or select method *)
Next ==
    \/ \E m \in Methods : BumpActivity(m)
    \/ IF ShouldDecay THEN DecayActivities ELSE UNCHANGED <<methodActivities, activityVarInc, bumpCount, selectedMethods>>
    \/ \E candidates \in SUBSET Methods \ {{}} : SelectBestMethod(candidates)
    \/ \E conflictPath \in SUBSET Methods : BumpConflictPath(conflictPath)

(* Specification *)
Spec == Init /\ [][Next]_<<methodActivities, activityVarInc, bumpCount, selectedMethods>>

(* Properties *)
TypeOK ==
    /\ methodActivities \in [Methods -> Int]
    /\ activityVarInc \in Int
    /\ activityVarInc > 0
    /\ bumpCount \in Nat
    /\ selectedMethods \in Seq(Methods)

(* Activity should always be non-negative *)
ActivityNonNegative == \A m \in Methods : methodActivities[m] >= 0

(* VarInc should always be positive *)
VarIncPositive == activityVarInc > 0

(* Invariant *)
Invariant == TypeOK /\ ActivityNonNegative /\ VarIncPositive

====
