---- MODULE InfiniteRecursion ----

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxDepth

VARIABLES
    iteration,
    task_stack,        \* Stack of tasks to process
    current_task       \* Current task being processed

\* Task types
Tasks == {"move_blocks", "move_one", "get", "put"}

\* Initialize with move_blocks task
Init ==
    /\ iteration = 0
    /\ task_stack = <<"move_blocks">>
    /\ current_task = "move_blocks"

\* move_blocks can decompose into move_one and move_blocks (RECURSIVE!)
\* This creates infinite recursion if not handled properly
MoveBlocksDecompose ==
    /\ current_task = "move_blocks"
    /\ iteration < MaxDepth
    /\ iteration' = iteration + 1
    /\ task_stack' = Append(task_stack, <<"move_one", "move_blocks">>)  \* RECURSIVE!
    /\ current_task' = Head(task_stack')

\* move_one decomposes into get and put
MoveOneDecompose ==
    /\ current_task = "move_one"
    /\ iteration < MaxDepth
    /\ iteration' = iteration + 1
    /\ task_stack' = Append(Tail(task_stack), <<"get", "put">>)
    /\ current_task' = IF Len(task_stack') > 0 THEN Head(task_stack') ELSE ""

\* Complete a task (remove from stack)
CompleteTask ==
    /\ current_task /= ""
    /\ task_stack /= <<>>
    /\ task_stack' = Tail(task_stack)
    /\ current_task' = IF Len(task_stack') > 0 THEN Head(task_stack') ELSE ""
    /\ UNCHANGED iteration

\* Planning completes when stack is empty
PlanningComplete ==
    /\ task_stack = <<>>
    /\ current_task = ""
    /\ UNCHANGED iteration

\* Planning hits depth limit
PlanningDepthLimit ==
    /\ iteration = MaxDepth
    /\ task_stack /= <<>>  \* Still have tasks to process
    /\ UNCHANGED <<task_stack, current_task>>

\* Next state relation
Next ==
    \/ MoveBlocksDecompose
    \/ MoveOneDecompose
    \/ CompleteTask
    \/ PlanningComplete
    \/ PlanningDepthLimit

\* Safety: Planning should complete before hitting depth limit
PlanningCompletesBeforeLimit ==
    ~(iteration = MaxDepth /\ task_stack /= <<>>)

\* The problem: move_blocks keeps decomposing into move_blocks
\* This creates infinite recursion unless there's a base case
InfiniteRecursionProblem ==
    \E i \in 0..iteration :
        /\ task_stack[i] = "move_blocks"
        /\ \E j \in i+1..iteration :
            task_stack[j] = "move_blocks"  \* move_blocks creates move_blocks

=============================================================================
