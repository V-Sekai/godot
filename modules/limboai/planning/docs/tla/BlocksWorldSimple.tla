---- MODULE BlocksWorldSimple ----

EXTENDS Naturals, Sequences, FiniteSets

(* Simplified TLA+ model to investigate Blocks World planning inefficiency *)

CONSTANTS Blocks, Table
ASSUME Blocks = {"a", "b", "c"}
ASSUME Table = "table"

VARIABLES state, plan, steps

InitState ==
    /\ state = [pos |-> [x \in Blocks |-> IF x = "a" THEN "b" ELSE IF x = "b" THEN Table ELSE Table],
                clear |-> [x \in Blocks |-> IF x = "a" THEN TRUE ELSE IF x = "b" THEN FALSE ELSE TRUE],
                holding |-> [x \in {"hand"} |-> FALSE]]
    /\ plan = <<>>
    /\ steps = 0

GoalState ==
    /\ state.pos["a"] = Table
    /\ state.pos["b"] = "a"
    /\ state.pos["c"] = "b"
    /\ state.clear["a"] = FALSE
    /\ state.clear["b"] = FALSE
    /\ state.clear["c"] = TRUE
    /\ state.holding["hand"] = FALSE

ActionPickup(b) ==
    /\ state.pos[b] = Table
    /\ state.clear[b] = TRUE
    /\ state.holding["hand"] = FALSE
    /\ LET new_pos == state.pos @@ [b |-> "hand"]
           new_clear == state.clear @@ [b |-> FALSE]
           new_holding == state.holding @@ ["hand" |-> b]
       IN state' = [pos |-> new_pos, clear |-> new_clear, holding |-> new_holding]
    /\ plan' = Append(plan, ["action_pickup", b])
    /\ steps' = steps + 1

ActionUnstack(b, c) ==
    /\ state.pos[b] = c
    /\ c # Table
    /\ state.clear[b] = TRUE
    /\ state.holding["hand"] = FALSE
    /\ LET new_pos == state.pos @@ [b |-> "hand"]
           new_clear == state.clear @@ [b |-> FALSE, c |-> TRUE]
           new_holding == state.holding @@ ["hand" |-> b]
       IN state' = [pos |-> new_pos, clear |-> new_clear, holding |-> new_holding]
    /\ plan' = Append(plan, ["action_unstack", b, c])
    /\ steps' = steps + 1

ActionPutdown(b) ==
    /\ state.pos[b] = "hand"
    /\ LET new_pos == state.pos @@ [b |-> Table]
           new_clear == state.clear @@ [b |-> TRUE]
           new_holding == state.holding @@ ["hand" |-> FALSE]
       IN state' = [pos |-> new_pos, clear |-> new_clear, holding |-> new_holding]
    /\ plan' = Append(plan, ["action_putdown", b])
    /\ steps' = steps + 1

ActionStack(b, c) ==
    /\ state.pos[b] = "hand"
    /\ state.clear[c] = TRUE
    /\ LET new_pos == state.pos @@ [b |-> c]
           new_clear == state.clear @@ [b |-> TRUE, c |-> FALSE]
           new_holding == state.holding @@ ["hand" |-> FALSE]
       IN state' = [pos |-> new_pos, clear |-> new_clear, holding |-> new_holding]
    /\ plan' = Append(plan, ["action_stack", b, c])
    /\ steps' = steps + 1

Next ==
    \/ \E b \in Blocks : ActionPickup(b)
    \/ \E b, c \in Blocks : ActionUnstack(b, c)
    \/ \E b \in Blocks : ActionPutdown(b)
    \/ \E b, c \in Blocks : ActionStack(b, c)

Spec == InitState /\ [][Next]_<<state, plan, steps>>

OptimalPlan == Len(plan) = 6

InefficientPlan == Len(plan) > 10

====
