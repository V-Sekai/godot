---- MODULE BlocksWorldRecursion ----

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxDepth, Blocks

VARIABLES
    iteration,
    task_stack,        \* Stack of tasks: [task_name, args...]
    pos,               \* Position state: block -> location
    clear,             \* Clear state: block -> boolean
    holding            \* Holding state: hand -> boolean

\* Task structure: [task_name, arg1, arg2, ...]
Task == Seq(STRING)

\* Initialize Blocks World state
InitState ==
    /\ pos = ["a" |-> "b", "b" |-> "table", "c" |-> "table"]
    /\ clear = ["a" |-> TRUE, "b" |-> FALSE, "c" |-> TRUE]
    /\ holding = ["hand" |-> FALSE]

\* Goal state
GoalState ==
    /\ pos = ["a" |-> "table", "b" |-> "a", "c" |-> "b"]
    /\ clear = ["c" |-> TRUE, "b" |-> FALSE, "a" |-> FALSE]
    /\ holding = ["hand" |-> FALSE]

\* Initialize
Init ==
    /\ iteration = 0
    /\ task_stack = <<<"move_blocks", GoalState>>>
    /\ pos = ["a" |-> "b", "b" |-> "table", "c" |-> "table"]
    /\ clear = ["a" |-> TRUE, "b" |-> FALSE, "c" |-> TRUE]
    /\ holding = ["hand" |-> FALSE]

\* move_blocks decomposes into move_one and move_blocks (RECURSIVE!)
\* This is the problem - move_blocks creates move_blocks infinitely
MoveBlocksDecompose ==
    /\ Head(Head(task_stack)) = "move_blocks"
    /\ iteration < MaxDepth
    /\ iteration' = iteration + 1
    /\ \* Decompose: move_blocks(goal) -> move_one(block, loc) + move_blocks(remaining_goal)
       LET goal == Head(task_stack)[2]  \* Goal state
           remaining_blocks == {b \in Blocks : pos[b] /= goal["pos"][b]}
       IN IF Cardinality(remaining_blocks) > 0
          THEN LET block == CHOOSE b \in remaining_blocks : TRUE
                   loc == goal["pos"][block]
               IN /\ task_stack' = Append(Tail(task_stack),
                                          <<<"move_one", block, loc>>,
                                           <<"move_blocks", goal>>>)  \* RECURSIVE!
                  /\ UNCHANGED <<pos, clear, holding>>
          ELSE /\ task_stack' = Tail(task_stack)  \* Base case: no blocks to move
               /\ UNCHANGED <<pos, clear, holding>>
    /\ UNCHANGED iteration

\* move_one decomposes into get and put
MoveOneDecompose ==
    /\ Head(Head(task_stack)) = "move_one"
    /\ iteration < MaxDepth
    /\ iteration' = iteration + 1
    /\ LET task == Head(task_stack)
           block == task[1]
           loc == task[2]
       IN /\ task_stack' = Append(Tail(task_stack),
                                  <<<"get", block>>,
                                   <<"put", block, loc>>>)
          /\ UNCHANGED <<pos, clear, holding, iteration>>

\* get executes action_pickup
GetExecute ==
    /\ Head(Head(task_stack)) = "get"
    /\ LET task == Head(task_stack)
           block == task[1]
       IN IF /\ pos[block] /= "hand"
             /\ clear[block] = TRUE
             /\ holding["hand"] = FALSE
          THEN /\ pos' = pos @@ [block |-> "hand"]
               /\ clear' = clear @@ [pos[block] |-> TRUE]
               /\ holding' = holding @@ ["hand" |-> TRUE]
               /\ task_stack' = Tail(task_stack)
               /\ UNCHANGED iteration
          ELSE /\ UNCHANGED <<pos, clear, holding, task_stack, iteration>>

\* put executes action_putdown
PutExecute ==
    /\ Head(Head(task_stack)) = "put"
    /\ LET task == Head(task_stack)
           block == task[1]
           loc == task[2]
       IN IF /\ pos[block] = "hand"
             /\ holding["hand"] = TRUE
             /\ (loc = "table" \/ clear[loc] = TRUE)
          THEN /\ pos' = pos @@ [block |-> loc]
               /\ clear' = clear @@ [block |-> TRUE, loc |-> IF loc = "table" THEN TRUE ELSE FALSE]
               /\ holding' = holding @@ ["hand" |-> FALSE]
               /\ task_stack' = Tail(task_stack)
               /\ UNCHANGED iteration
          ELSE /\ UNCHANGED <<pos, clear, holding, task_stack, iteration>>

\* Planning completes when stack is empty and goal is reached
PlanningComplete ==
    /\ task_stack = <<>>
    /\ pos = GoalState["pos"]
    /\ clear = GoalState["clear"]
    /\ holding = GoalState["holding"]
    /\ UNCHANGED iteration

\* Planning hits depth limit
PlanningDepthLimit ==
    /\ iteration = MaxDepth
    /\ task_stack /= <<>>  \* Still have tasks
    /\ UNCHANGED <<pos, clear, holding>>

\* Next state relation
Next ==
    \/ MoveBlocksDecompose
    \/ MoveOneDecompose
    \/ GetExecute
    \/ PutExecute
    \/ PlanningComplete
    \/ PlanningDepthLimit

\* Safety: Planning should complete before hitting depth limit
PlanningCompletesBeforeLimit ==
    ~(iteration = MaxDepth /\ task_stack /= <<>>)

\* The infinite recursion problem
InfiniteRecursionProblem ==
    \E i \in 0..iteration :
        /\ Head(task_stack[i]) = "move_blocks"
        /\ \E j \in i+1..iteration :
            Head(task_stack[j]) = "move_blocks"  \* move_blocks creates move_blocks

=============================================================================
