---- MODULE BlocksWorldValidation ----

EXTENDS Naturals, Sequences

(* Validate that a blocks world plan achieves the goal state *)

(* State representation *)
VARIABLES pos, clear, holding

(* Initial state for small problem *)
InitState1 ==
    /\ pos = [a |-> b, b |-> "table", c |-> "table"]
    /\ clear = [a |-> TRUE, b |-> FALSE, c |-> TRUE]
    /\ holding = ["hand" |-> FALSE]

(* Goal state *)
Goal1a ==
    /\ pos = [a |-> "table", b |-> a, c |-> b]
    /\ clear = [a |-> FALSE, b |-> FALSE, c |-> TRUE]
    /\ holding = ["hand" |-> FALSE]

(* Action: pickup(b) *)
ActionPickup(b) ==
    /\ pos[b] = "table"
    /\ clear[b] = TRUE
    /\ holding["hand"] = FALSE
    /\ pos' = pos @@ [b |-> "hand"]
    /\ clear' = clear @@ [b |-> FALSE]
    /\ holding' = holding @@ ["hand" |-> b]
    /\ UNCHANGED <<>>

(* Action: unstack(b, c) *)
ActionUnstack(b, c) ==
    /\ pos[b] = c
    /\ c # "table"
    /\ clear[b] = TRUE
    /\ holding["hand"] = FALSE
    /\ pos' = pos @@ [b |-> "hand"]
    /\ clear' = clear @@ [b |-> FALSE, c |-> TRUE]
    /\ holding' = holding @@ ["hand" |-> b]
    /\ UNCHANGED <<>>

(* Action: putdown(b) *)
ActionPutdown(b) ==
    /\ pos[b] = "hand"
    /\ pos' = pos @@ [b |-> "table"]
    /\ clear' = clear @@ [b |-> TRUE]
    /\ holding' = holding @@ ["hand" |-> FALSE]
    /\ UNCHANGED <<>>

(* Action: stack(b, c) *)
ActionStack(b, c) ==
    /\ pos[b] = "hand"
    /\ clear[c] = TRUE
    /\ pos' = pos @@ [b |-> c]
    /\ clear' = clear @@ [b |-> TRUE, c |-> FALSE]
    /\ holding' = holding @@ ["hand" |-> FALSE]
    /\ UNCHANGED <<>>

(* Expected plan: 6 actions *)
ExpectedPlan == <<
    ["action_unstack", a, b],
    ["action_putdown", a],
    ["action_pickup", b],
    ["action_stack", b, a],
    ["action_pickup", c],
    ["action_stack", c, b]
>>

(* Execute a plan and verify it reaches the goal *)
ExecutePlan(plan) ==
    LET state == [pos |-> InitState1.pos, clear |-> InitState1.clear, holding |-> InitState1.holding]
    IN LET finalState == ExecutePlanRecursive(plan, state, 1)
       IN finalState.pos = Goal1a.pos /\ finalState.clear = Goal1a.clear /\ finalState.holding = Goal1a.holding

ExecutePlanRecursive(plan, state, idx) ==
    IF idx > Len(plan) THEN state
    ELSE LET action == plan[idx]
             actionName == action[1]
         IN IF actionName = "action_pickup" THEN
                ExecutePlanRecursive(plan, ApplyPickup(state, action[2]), idx + 1)
            ELSE IF actionName = "action_unstack" THEN
                ExecutePlanRecursive(plan, ApplyUnstack(state, action[2], action[3]), idx + 1)
            ELSE IF actionName = "action_putdown" THEN
                ExecutePlanRecursive(plan, ApplyPutdown(state, action[2]), idx + 1)
            ELSE IF actionName = "action_stack" THEN
                ExecutePlanRecursive(plan, ApplyStack(state, action[2], action[3]), idx + 1)
            ELSE state

ApplyPickup(state, b) ==
    [pos |-> state.pos @@ [b |-> "hand"],
     clear |-> state.clear @@ [b |-> FALSE],
     holding |-> state.holding @@ ["hand" |-> b]]

ApplyUnstack(state, b, c) ==
    [pos |-> state.pos @@ [b |-> "hand"],
     clear |-> state.clear @@ [b |-> FALSE, c |-> TRUE],
     holding |-> state.holding @@ ["hand" |-> b]]

ApplyPutdown(state, b) ==
    [pos |-> state.pos @@ [b |-> "table"],
     clear |-> state.clear @@ [b |-> TRUE],
     holding |-> state.holding @@ ["hand" |-> FALSE]]

ApplyStack(state, b, c) ==
    [pos |-> state.pos @@ [b |-> c],
     clear |-> state.clear @@ [b |-> TRUE, c |-> FALSE],
     holding |-> state.holding @@ ["hand" |-> FALSE]]

(* Property: Expected plan should achieve the goal *)
PlanCorrectness == ExecutePlan(ExpectedPlan)

====
