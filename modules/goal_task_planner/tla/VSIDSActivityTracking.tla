---- MODULE VSIDSActivityTracking ----
EXTENDS Naturals, Integers, Sequences, FiniteSets

CONSTANTS Methods, MaxBumpsBeforeDecay

VARIABLES
    methodActivities,    (* Map: Method -> Int *)
    bumpCount,           (* Number of bumps since last decay *)
    selectedMethods      (* Sequence of selected methods *)

Init ==
    /\ methodActivities = [m \in Methods |-> 0]
    /\ bumpCount = 0
    /\ selectedMethods = <<>>

(* Simplified bump - just increments by 1 *)
BumpActivity(m) ==
    LET newActivity == methodActivities[m] + 1
    IN  /\ methodActivities[m] < 3  (* Bound max activity *)
        /\ methodActivities' = [methodActivities EXCEPT ![m] = newActivity]
        /\ bumpCount' = bumpCount + 1
        /\ selectedMethods' = selectedMethods

(* Simplified decay - just resets bump count and decrements activities *)
DecayActivities ==
    /\ methodActivities' = [m \in Methods |-> IF methodActivities[m] > 0 THEN methodActivities[m] - 1 ELSE 0]
    /\ bumpCount' = 0
    /\ selectedMethods' = selectedMethods

ShouldDecay == bumpCount >= MaxBumpsBeforeDecay

SelectBestMethod(candidates) ==
    LET scores == [m \in candidates |-> methodActivities[m]]
        maxScore == CHOOSE s \in {scores[m] : m \in candidates} :
                     \A s2 \in {scores[m] : m \in candidates} : s >= s2
        bestMethod == CHOOSE m \in candidates : scores[m] = maxScore
    IN  /\ Len(selectedMethods) < 2  (* Strictly bound selection history length *)
        /\ selectedMethods' = Append(selectedMethods, bestMethod)
        /\ UNCHANGED <<methodActivities, bumpCount>>

BumpConflictPath(conflictPath) ==
    LET newActivities ==
        [m \in Methods |->
         IF m \in conflictPath /\ methodActivities[m] < 3
         THEN methodActivities[m] + 1
         ELSE methodActivities[m]]
    IN  /\ methodActivities' = newActivities
        /\ bumpCount' = bumpCount + 1
        /\ selectedMethods' = selectedMethods

Next ==
    \/ \E m \in Methods : BumpActivity(m)
    \/ (ShouldDecay /\ DecayActivities)
    \/ (Len(selectedMethods) < 2 /\ \E candidates \in SUBSET Methods \ {{}} : SelectBestMethod(candidates))
    \/ \E conflictPath \in SUBSET Methods : conflictPath # {} /\ BumpConflictPath(conflictPath)

Spec == Init /\ [][Next]_<<methodActivities, bumpCount, selectedMethods>>

TypeOK ==
    /\ methodActivities \in [Methods -> Int]
    /\ bumpCount \in Nat
    /\ selectedMethods \in Seq(Methods)

ActivityNonNegative == \A m \in Methods : methodActivities[m] >= 0

Invariant == TypeOK /\ ActivityNonNegative
====
