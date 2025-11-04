#!/usr/bin/env python
"""
File Description: Sample test file. Tests the backtracking for an unsolvable solution.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import Methods, IPyHOP
from ipyhop_tests.test_action_models import actions_1 as actions
from ipyhop_tests.test_state_models import init_state_1 as init_state

methods = Methods()


def tm_1_1(state): return [('tm_2', ), ('t_a', 3, 4), ('t_a', 4, 5)]
def tm_1_2(state): return [('tm_2', ), ('t_a', 3, 4), ('t_a', 4, 5), ('t_a', 5, 6)]


methods.declare_task_methods('tm_1', [tm_1_1, tm_1_2, tm_1_2])


def tm_2_1(state): return [('t_a', 0, 1), ('t_a', 1, 2), ('t_a', 2, 3)]
def tm_2_2(state): return [('t_a', 0, 1), ('t_a', 1, 2), ('t_a', 2, 3), ('t_a', 3, 7)]


methods.declare_task_methods('tm_2', [tm_2_1, tm_2_2])


def tm_3_1(state): return [('t_a', 9, 10)]


methods.declare_task_methods('tm_3', [tm_3_1])


# ******************************************        Main Program Start      ****************************************** #
def main():
    print('\n\r', methods)
    print('\n\r', actions)
    print('\nInitial State: \n\r', init_state, '\n\r')

    planner = IPyHOP(methods, actions)
    plan = planner.plan(init_state, [('tm_1',), ('tm_3',)], verbose=3)
    exp_0 = []
    assert plan == exp_0, "Result plan and expected plan are not same"


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