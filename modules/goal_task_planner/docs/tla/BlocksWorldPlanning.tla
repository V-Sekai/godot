---- MODULE BlocksWorldPlanning ----

EXTENDS Naturals, Sequences, FiniteSets

(* TLA+ model to investigate why Blocks World planner finds suboptimal plans *)

(* Blocks: a, b, c *)
CONSTANTS Blocks, Table
ASSUME Blocks = {"a", "b", "c"}
ASSUME Table = "table"

(* State representation *)
VARIABLES pos, clear, holding, plan, iterations

(* Initial state: a on b, b on table, c on table *)
InitState ==
    /\ pos = [x \in Blocks |-> IF x = "a" THEN "b" ELSE Table]
    /\ clear = [x \in Blocks |-> IF x = "a" THEN TRUE ELSE IF x = "b" THEN FALSE ELSE TRUE]
    /\ holding = [x \in {"hand"} |-> FALSE]
    /\ plan = <<>>
    /\ iterations = 0

(* Goal state: b on a, c on b, a on table *)
GoalState ==
    /\ pos["a"] = Table
    /\ pos["b"] = "a"
    /\ pos["c"] = "b"
    /\ clear["a"] = FALSE
    /\ clear["b"] = FALSE
    /\ clear["c"] = TRUE
    /\ holding["hand"] = FALSE

(* Action: pickup(b) *)
ActionPickup(b) ==
    /\ pos[b] = Table
    /\ clear[b] = TRUE
    /\ holding["hand"] = FALSE
    /\ pos' = pos @@ [b |-> "hand"]
    /\ clear' = clear @@ [b |-> FALSE]
    /\ holding' = holding @@ ["hand" |-> b]
    /\ plan' = Append(plan, ["action_pickup", b])
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<>>

(* Action: unstack(b, c) *)
ActionUnstack(b, c) ==
    /\ pos[b] = c
    /\ c # Table
    /\ clear[b] = TRUE
    /\ holding["hand"] = FALSE
    /\ pos' = pos @@ [b |-> "hand"]
    /\ clear' = clear @@ [b |-> FALSE, c |-> TRUE]
    /\ holding' = holding @@ ["hand" |-> b]
    /\ plan' = Append(plan, ["action_unstack", b, c])
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<>>

(* Action: putdown(b) *)
ActionPutdown(b) ==
    /\ pos[b] = "hand"
    /\ pos' = pos @@ [b |-> Table]
    /\ clear' = clear @@ [b |-> TRUE]
    /\ holding' = holding @@ ["hand" |-> FALSE]
    /\ plan' = Append(plan, ["action_putdown", b])
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<>>

(* Action: stack(b, c) *)
ActionStack(b, c) ==
    /\ pos[b] = "hand"
    /\ clear[c] = TRUE
    /\ pos' = pos @@ [b |-> c]
    /\ clear' = clear @@ [b |-> TRUE, c |-> FALSE]
    /\ holding' = holding @@ ["hand" |-> FALSE]
    /\ plan' = Append(plan, ["action_stack", b, c])
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<>>

(* All possible actions *)
Next ==
    \/ \E b \in Blocks : ActionPickup(b)
    \/ \E b, c \in Blocks : ActionUnstack(b, c)
    \/ \E b \in Blocks : ActionPutdown(b)
    \/ \E b, c \in Blocks : ActionStack(b, c)

(* Specification *)
Spec == InitState /\ [][Next]_<<pos, clear, holding, plan, iterations>>

(* Property: Plan should reach goal state *)
GoalReached ==
    /\ pos["a"] = Table
    /\ pos["b"] = "a"
    /\ pos["c"] = "b"
    /\ clear["a"] = FALSE
    /\ clear["b"] = FALSE
    /\ clear["c"] = TRUE
    /\ holding["hand"] = FALSE

(* Property: Optimal plan should have 6 actions *)
OptimalPlan == Len(plan) = 6

(* Expected optimal plan: unstack(a,b), putdown(a), pickup(b), stack(b,a), pickup(c), stack(c,b) *)
ExpectedPlan ==
    plan = <<["action_unstack", "a", "b"],
             ["action_putdown", "a"],
             ["action_pickup", "b"],
             ["action_stack", "b", "a"],
             ["action_pickup", "c"],
             ["action_stack", "c", "b"]>>

(* Check if current plan is a valid path to goal *)
ValidPlan ==
    /\ GoalReached
    /\ Len(plan) >= 6
    /\ Len(plan) <= 20  (* Reasonable upper bound *)

(* Check for inefficient planning (too many iterations) *)
InefficientPlanning == iterations > 50

====
