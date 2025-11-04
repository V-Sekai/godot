#!/usr/bin/env python
"""
File Description: Sample test file. Tests the backtracking for a solution tree of depth 2.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import Methods, IPyHOP
from ipyhop_tests.test_action_models import actions_1 as actions
from ipyhop_tests.test_state_models import init_state_1 as init_state

methods = Methods()


def tm_1_1(state): return [('t_a', 0, 1), ('t_a', 1, 2), ('t_a', 3, 4)]
def tm_1_2(state): return [('t_a', 0, 1), ('t_a', 1, 2), ('t_a', 2, 3)]
def tm_1_3(state): return [('t_a', 0, 1), ('t_a', 1, 2), ('t_a', 2, 3), ('t_a', 3, 4)]


methods.declare_task_methods('tm_1', [tm_1_1, tm_1_2, tm_1_3])


def tm_2_1(state): return [('t_a', 3, 4), ('t_a', 4, 5), ('t_a', 6, 7)]
def tm_2_2(state): return [('t_a', 4, 5), ('t_a', 5, 6), ('t_a', 6, 7)]


methods.declare_task_methods('tm_2', [tm_2_1, tm_2_2])


# ******************************************        Main Program Start      ****************************************** #
def main():
    print(methods)
    print(actions)
    print(init_state)

    planner = IPyHOP(methods, actions)
    plan = planner.plan(init_state, [('tm_1',), ('tm_2',)], verbose=3)
    exp_0 = [('t_a', 0, 1), ('t_a', 1, 2), ('t_a', 2, 3), ('t_a', 3, 4), ('t_a', 4, 5), ('t_a', 5, 6), ('t_a', 6, 7)]
    assert plan == exp_0, "Result plan and expected plan are not same"
    print("First the planner backtracks to use the method \"tm_1_2\" for the task \"tm_1\". \n"
          "Then the planner backtracks agian to use the method \"tm_2_2\" for the task \"tm_2\". \n"
          "But the solution it finds so far doesn't satisfy the pre-conditions for the method \"tm_2_2\". \n"
          "So it backtracks again to use the method \"tm_1_3\" for the task \"tm_1\". \n"
          "And finally it backtracks again to use the method \"tm_2_2\" for the task \"tm_2\". \n")


# ******************************************        Main Program End        ****************************************** #
# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    try:
        main()
        print('\nFile executed successfully!\n')
    except KeyboardInterrupt:
        print('\nProcess interrupted by user. Bye!')

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""