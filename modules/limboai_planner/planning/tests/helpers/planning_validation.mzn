% MiniZinc model to validate the expected planning sequence for IPyHOP Sample Test 1
% This models the flag transfer problem where we need to transfer flags from 0->1->2->...->7

% State: flag[i] = true means flag i is set
% Initial state: flag[0] = true, flag[1..19] = false
% Goal: flag[7] = true

% Actions: action_transfer_flag(i, j) - transfers flag from i to j if flag[i] is true
% This sets flag[j] = true

% Tasks:
% - task_method_1: Should produce actions to get flag[3] = true
% - task_method_2: Should produce actions to get flag[7] = true

% Expected plan:
% [action_transfer_flag(0,1), action_transfer_flag(1,2), action_transfer_flag(2,3),
%  action_transfer_flag(3,4), action_transfer_flag(4,5), action_transfer_flag(5,6),
%  action_transfer_flag(6,7)]

int: num_flags = 20;
int: max_actions = 10;

% Decision variables
array[1..max_actions] of var 0..num_flags: from_flag;
array[1..max_actions] of var 0..num_flags: to_flag;
array[1..max_actions] of var bool: action_used;

% State after each action
array[0..max_actions, 0..num_flags-1] of var bool: flag_state;

% Constraints

% Initial state: flag[0] = true, all others false
constraint forall(i in 1..num_flags-1) (flag_state[0, i] = false);
constraint flag_state[0, 0] = true;

% Final state: flag[7] = true
constraint flag_state[max_actions, 7] = true;

% Action constraints
constraint forall(a in 1..max_actions) (
    if action_used[a] then
        % Action transfers flag from from_flag[a] to to_flag[a]
        % Precondition: flag[from_flag[a]] must be true before action
        flag_state[a-1, from_flag[a]] = true /\
        % Effect: flag[to_flag[a]] becomes true
        flag_state[a, to_flag[a]] = true /\
        % All other flags remain unchanged
        forall(f in 0..num_flags-1 where f != to_flag[a]) (
            flag_state[a, f] = flag_state[a-1, f]
        )
    else
        % Unused actions: state doesn't change
        forall(f in 0..num_flags-1) (
            flag_state[a, f] = flag_state[a-1, f]
        )
    endif
);

% Expected sequence: 0->1->2->3->4->5->6->7
% This means we need exactly 7 actions
constraint sum(a in 1..max_actions)(action_used[a]) = 7;

% First action: 0->1
constraint action_used[1] = true;
constraint from_flag[1] = 0;
constraint to_flag[1] = 1;

% Second action: 1->2
constraint action_used[2] = true;
constraint from_flag[2] = 1;
constraint to_flag[2] = 2;

% Third action: 2->3
constraint action_used[3] = true;
constraint from_flag[3] = 2;
constraint to_flag[3] = 3;

% Fourth action: 3->4
constraint action_used[4] = true;
constraint from_flag[4] = 3;
constraint to_flag[4] = 4;

% Fifth action: 4->5
constraint action_used[5] = true;
constraint from_flag[5] = 4;
constraint to_flag[5] = 5;

% Sixth action: 5->6
constraint action_used[6] = true;
constraint from_flag[6] = 5;
constraint to_flag[6] = 6;

% Seventh action: 6->7
constraint action_used[7] = true;
constraint from_flag[7] = 6;
constraint to_flag[7] = 7;

% Actions 8-10 are unused
constraint forall(a in 8..max_actions) (action_used[a] = false);

solve satisfy;

output [ "Action sequence:\n" ] ++
[
    "  " ++ show(a) ++ ": action_transfer_flag(" ++ show(from_flag[a]) ++ ", " ++ show(to_flag[a]) ++ ")\n"
    | a in 1..7
] ++ [ "\nFinal state:\n" ] ++
[
    "  flag[" ++ show(f) ++ "] = " ++ show(flag_state[max_actions, f]) ++ "\n"
    | f in 0..7
];
