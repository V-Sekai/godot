---- MODULE BlocksWorldHTN ----

EXTENDS Naturals, Sequences, FiniteSets

(* TLA+ model to investigate HTN planning inefficiency in Blocks World *)

(* Blocks: a, b, c *)
CONSTANTS Blocks, Table
ASSUME Blocks = {"a", "b", "c"}
ASSUME Table = "table"

(* Planning state *)
VARIABLES state, todo_list, plan, iterations, max_depth

(* Initial state: a on b, b on table, c on table *)
InitState ==
    /\ state = [pos |-> [x \in Blocks |-> IF x = "a" THEN "b" ELSE IF x = "b" THEN Table ELSE Table],
                clear |-> [x \in Blocks |-> IF x = "a" THEN TRUE ELSE IF x = "b" THEN FALSE ELSE TRUE],
                holding |-> [x \in {"hand"} |-> FALSE]]
    /\ todo_list = <<["move_blocks", [pos |-> [x \in Blocks |-> IF x = "a" THEN Table ELSE IF x = "b" THEN "a" ELSE "b"]]]>>
    /\ plan = <<>>
    /\ iterations = 0
    /\ max_depth = 50

(* Goal state *)
GoalState ==
    /\ state.pos["a"] = Table
    /\ state.pos["b"] = "a"
    /\ state.pos["c"] = "b"
    /\ state.clear["a"] = FALSE
    /\ state.clear["b"] = FALSE
    /\ state.clear["c"] = TRUE
    /\ state.holding["hand"] = FALSE

(* Helper: Check if block is done (in correct position) *)
IsDone(b, s, goal) ==
    IF b = Table THEN TRUE
    ELSE /\ goal.pos[b] = s.pos[b]
         /\ IF s.pos[b] = Table THEN TRUE
            ELSE IsDone(s.pos[b], s, goal)

(* Helper: Get status of a block *)
Status(b, s, goal) ==
    IF IsDone(b, s, goal) THEN "done"
    ELSE IF ~s.clear[b] THEN "inaccessible"
    ELSE IF goal.pos[b] = Table THEN "move-to-table"
    ELSE IF IsDone(goal.pos[b], s, goal) /\ s.clear[goal.pos[b]] THEN "move-to-block"
    ELSE "waiting"

(* Action: pickup(b) *)
ActionPickup(b) ==
    /\ state.pos[b] = Table
    /\ state.clear[b] = TRUE
    /\ state.holding["hand"] = FALSE
    /\ state' = [state EXCEPT !.pos[b] = "hand",
                              !.clear[b] = FALSE,
                              !.holding["hand"] = b]
    /\ plan' = Append(plan, ["action_pickup", b])
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<todo_list, max_depth>>

(* Action: unstack(b, c) *)
ActionUnstack(b, c) ==
    /\ state.pos[b] = c
    /\ c # Table
    /\ state.clear[b] = TRUE
    /\ state.holding["hand"] = FALSE
    /\ state' = [state EXCEPT !.pos[b] = "hand",
                              !.clear[b] = FALSE,
                              !.clear[c] = TRUE,
                              !.holding["hand"] = b]
    /\ plan' = Append(plan, ["action_unstack", b, c])
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<todo_list, max_depth>>

(* Action: putdown(b) *)
ActionPutdown(b) ==
    /\ state.pos[b] = "hand"
    /\ state' = [state EXCEPT !.pos[b] = Table,
                              !.clear[b] = TRUE,
                              !.holding["hand"] = FALSE]
    /\ plan' = Append(plan, ["action_putdown", b])
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<todo_list, max_depth>>

(* Action: stack(b, c) *)
ActionStack(b, c) ==
    /\ state.pos[b] = "hand"
    /\ state.clear[c] = TRUE
    /\ state' = [state EXCEPT !.pos[b] = c,
                              !.clear[b] = TRUE,
                              !.clear[c] = FALSE,
                              !.holding["hand"] = FALSE]
    /\ plan' = Append(plan, ["action_stack", b, c])
    /\ iterations' = iterations + 1
    /\ UNCHANGED <<todo_list, max_depth>>

(* Task method: move_blocks - finds first block that needs moving *)
MethodMoveBlocks ==
    /\ todo_list[1][1] = "move_blocks"
    /\ LET goal == todo_list[1][2]
           blocks == Blocks
           FindMoveToTable(b) == Status(b, state, goal) = "move-to-table"
           FindMoveToBlock(b) == Status(b, state, goal) = "move-to-block"
           FindWaiting(b) == Status(b, state, goal) = "waiting"
       IN \/ \E b \in blocks : FindMoveToTable(b)
             /\ LET dest == Table
                    new_todo == <<["move_one", b, dest], ["move_blocks", goal]>>
                IN /\ todo_list' = new_todo
                   /\ UNCHANGED <<state, plan, iterations, max_depth>>
          \/ \E b \in blocks : FindMoveToBlock(b)
             /\ LET goal_pos == goal.pos
                    dest == goal_pos[b]
                    new_todo == <<["move_one", b, dest], ["move_blocks", goal]>>
                IN /\ todo_list' = new_todo
                   /\ UNCHANGED <<state, plan, iterations, max_depth>>
          \/ \E b \in blocks : FindWaiting(b)
             /\ LET new_todo == <<["move_one", b, Table], ["move_blocks", goal]>>
                IN /\ todo_list' = new_todo
                   /\ UNCHANGED <<state, plan, iterations, max_depth>>
          \/ (* All blocks done *)
             /\ todo_list' = Tail(todo_list)
             /\ UNCHANGED <<state, plan, iterations, max_depth>>

(* Task method: move_one(b, dest) - moves block b to destination *)
MethodMoveOne ==
    /\ todo_list[1][1] = "move_one"
    /\ LET b == todo_list[1][2]
           dest == todo_list[1][3]
       IN /\ todo_list' = <<["get", b], ["put", b, dest]>>
          /\ UNCHANGED <<state, plan, iterations, max_depth>>

(* Task method: get(b) - gets block b *)
MethodGet ==
    /\ todo_list[1][1] = "get"
    /\ \E b \in Blocks :
       /\ todo_list[1][2] = b
       /\ \/ /\ state.pos[b] = Table
             /\ state.clear[b]
             /\ todo_list' = <<["action_pickup", b]>>
             /\ UNCHANGED <<state, plan, iterations, max_depth>>
          \/ /\ state.pos[b] # Table
             /\ state.pos[b] # "hand"
             /\ state.clear[b]
             /\ \E c \in Blocks :
                /\ state.pos[b] = c
                /\ todo_list' = <<["action_unstack", b, c]>>
                /\ UNCHANGED <<state, plan, iterations, max_depth>>
          \/ /\ (* Cannot get block - method fails *)
             /\ UNCHANGED <<state, todo_list, plan, iterations, max_depth>>

(* Task method: put(b, dest) - puts block b on destination *)
MethodPut ==
    /\ todo_list[1][1] = "put"
    /\ \E b \in Blocks, dest \in Blocks \cup {Table} :
       /\ todo_list[1][2] = b
       /\ todo_list[1][3] = dest
       /\ \/ /\ dest = Table
             /\ todo_list' = <<["action_putdown", b]>>
             /\ UNCHANGED <<state, plan, iterations, max_depth>>
          \/ /\ dest # Table
             /\ state.clear[dest]
             /\ todo_list' = <<["action_stack", b, dest]>>
             /\ UNCHANGED <<state, plan, iterations, max_depth>>
          \/ /\ (* Cannot put block - method fails *)
             /\ UNCHANGED <<state, todo_list, plan, iterations, max_depth>>

(* Execute action *)
ExecuteAction ==
    /\ todo_list[1][1] \in {"action_pickup", "action_unstack", "action_putdown", "action_stack"}
    /\ LET action == todo_list[1]
       IN IF action[1] = "action_pickup"
          THEN /\ ActionPickup(action[2])
               /\ todo_list' = Tail(todo_list)
               /\ UNCHANGED <<max_depth>>
          ELSE IF action[1] = "action_unstack"
               THEN /\ ActionUnstack(action[2], action[3])
                    /\ todo_list' = Tail(todo_list)
                    /\ UNCHANGED <<max_depth>>
               ELSE IF action[1] = "action_putdown"
                    THEN /\ ActionPutdown(action[2])
                         /\ todo_list' = Tail(todo_list)
                         /\ UNCHANGED <<max_depth>>
                    ELSE IF action[1] = "action_stack"
                         THEN /\ ActionStack(action[2], action[3])
                              /\ todo_list' = Tail(todo_list)
                              /\ UNCHANGED <<max_depth>>
                         ELSE /\ UNCHANGED <<state, todo_list, plan, iterations, max_depth>>

(* Next step *)
Next ==
    \/ /\ iterations < max_depth
       /\ todo_list # <<>>
       /\ \/ MethodMoveBlocks
          \/ MethodMoveOne
          \/ MethodGet
          \/ MethodPut
          \/ ExecuteAction
    \/ /\ iterations >= max_depth
       /\ UNCHANGED <<state, todo_list, plan, iterations, max_depth>>

(* Specification *)
Spec == InitState /\ [][Next]_<<state, todo_list, plan, iterations, max_depth>>

(* Property: Planning succeeds *)
PlanningSucceeds == todo_list = <<>> /\ GoalState

(* Property: Optimal plan has 6 actions *)
OptimalPlan == Len(plan) = 6

(* Property: Planning is inefficient *)
InefficientPlanning == iterations > 20 \/ Len(plan) > 10

====
