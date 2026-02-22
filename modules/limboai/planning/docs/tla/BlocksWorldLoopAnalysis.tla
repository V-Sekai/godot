----MODULE BlocksWorldLoopAnalysis----

EXTENDS Naturals, Sequences

CONSTANTS Blocks, Table
ASSUME Table \notin Blocks

VARIABLES pos, clear, holding, goalPos, goalClear

TypeOK ==
    /\ pos \in [Blocks \cup {Table} -> Blocks \cup {Table}]
    /\ clear \in [Blocks -> BOOLEAN]
    /\ holding \in [{"hand"} -> Blocks \cup {FALSE}]
    /\ goalPos \in [Blocks -> Blocks \cup {Table}]
    /\ goalClear \in [Blocks -> BOOLEAN]

Init ==
    /\ pos = [b \in Blocks |-> Table]  (* All blocks on table initially *)
    /\ clear = [b \in Blocks |-> TRUE]  (* All blocks clear *)
    /\ holding = [{"hand"} |-> FALSE]  (* Hand empty *)
    /\ goalPos = [b \in Blocks |-> Table]  (* Goal: all on table *)
    /\ goalClear = [b \in Blocks |-> TRUE]

IsDone(block) ==
    IF block = Table THEN TRUE
    ELSE IF block \in DOMAIN goalPos THEN
        pos[block] = goalPos[block]
    ELSE TRUE

Status(block) ==
    IF IsDone(block) THEN "done"
    ELSE IF ~clear[block] THEN "inaccessible"
    ELSE IF block \notin DOMAIN goalPos \/ goalPos[block] = Table THEN "move-to-table"
    ELSE IF IsDone(goalPos[block]) /\ clear[goalPos[block]] THEN "move-to-block"
    ELSE "waiting"

AllBlocksDone ==
    \A b \in Blocks : IsDone(b)

TaskMoveBlocks ==
    /\ ~AllBlocksDone
    /\ \E b \in Blocks : Status(b) = "move-to-table"
    /\ pos' = pos
    /\ clear' = clear
    /\ holding' = holding
    /\ goalPos' = goalPos
    /\ goalClear' = goalClear

TaskMoveBlocksWaiting ==
    /\ ~AllBlocksDone
    /\ \A b \in Blocks : Status(b) \notin {"move-to-table", "move-to-block"}
    /\ \E b \in Blocks : Status(b) = "waiting" /\ pos[b] \neq Table
    /\ \A b \in Blocks : Status(b) = "waiting" => pos[b] \neq Table
    /\ pos' = pos
    /\ clear' = clear
    /\ holding' = holding
    /\ goalPos' = goalPos
    /\ goalClear' = goalClear

TaskMoveBlocksDone ==
    /\ AllBlocksDone
    /\ UNCHANGED <<pos, clear, holding, goalPos, goalClear>>

Next ==
    \/ TaskMoveBlocks
    \/ TaskMoveBlocksWaiting
    \/ TaskMoveBlocksDone

Spec == Init /\ [][Next]_<<pos, clear, holding, goalPos, goalClear>>

LoopInvariant ==
    \A b \in Blocks : pos[b] \in Blocks \cup {Table}

====
