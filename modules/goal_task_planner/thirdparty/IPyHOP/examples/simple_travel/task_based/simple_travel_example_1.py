#!/usr/bin/env python
"""
File Description: Simple Travel example file. Run this file to solve the Simple Travel planning problem.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from examples.simple_travel.task_based.simple_travel_domain import actions, methods
from examples.simple_travel.task_based.simple_travel_problem import init_state, task_list_1, task_list_2
from ipyhop import IPyHOP, planar_plot


# ******************************************        Main Program Start      ****************************************** #
def main():
    print(methods)
    print(actions)
    print(init_state)
    planner = IPyHOP(methods, actions)
    plan = planner.plan(init_state, task_list_1, verbose=1)
    graph = planner.sol_tree
    exp_1 = [('a_call_taxi', 'alice', 'home_a'), ('a_ride_taxi', 'alice', 'park'), ('a_pay_driver', 'alice', 'park')]
    assert plan == exp_1, "Result plan and expected plan are not same"
    planar_plot(graph, root_node=0)

    print('Plan: ')
    for action in plan:
        print('\t', action)

    # Lets try another, more elaborated task.
    plan = planner.plan(init_state, task_list_2, verbose=1)
    exp_2 = [('a_call_taxi', 'alice', 'home_a'), ('a_ride_taxi', 'alice', 'park'), ('a_pay_driver', 'alice', 'park'),
             ('a_walk', 'bob', 'home_b', 'park')]
    assert plan == exp_2, "Result plan and expected plan are not same"
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