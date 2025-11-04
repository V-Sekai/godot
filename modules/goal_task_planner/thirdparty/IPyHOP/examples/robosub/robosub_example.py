#!/usr/bin/env python
"""
File Description: Robosub example file. Run this file to solve the Robosub planning problem.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from __future__ import print_function
from examples.robosub.domain.robosub_mod_methods import methods
from examples.robosub.domain.robosub_mod_actions import actions
from examples.robosub.problem.robosub_mod_problem import init_state, task_list_1
from ipyhop import IPyHOP, planar_plot


# ******************************************        Main Program Start      ****************************************** #
def main():
    print(methods)
    print(actions)
    print(init_state)

    planner = IPyHOP(methods, actions)
    planner.blacklist_command(('a_touch_back_v', 'v1', 'l2'))
    planner.blacklist_command(('a_touch_front_v', 'v1', 'l2'))
    plan = planner.plan(init_state, task_list_1, verbose=0)
    graph = planner.sol_tree

    planar_plot(graph, root_node=0)

    print('Plan: ')
    for action in plan:
        print('\t', action)


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