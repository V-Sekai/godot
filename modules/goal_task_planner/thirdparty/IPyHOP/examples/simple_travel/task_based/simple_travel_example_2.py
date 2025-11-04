#!/usr/bin/env python
"""
File Description: Robosub example file. Run this file to solve the Robosub planning problem.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from examples.simple_travel.task_based.simple_travel_domain import actions, methods
from examples.simple_travel.task_based.simple_travel_problem import init_state, task_list_1
from ipyhop import IPyHOP


# ******************************************        Main Program Start      ****************************************** #
def main():
    print(methods)
    print(actions)
    print('\nInitial State: \n', init_state)

    planner = IPyHOP(methods, actions)
    print("\n- If verbose=0 (the default), Pyhop returns the solution but prints nothing.")
    plan = planner.plan(init_state, task_list_1, verbose=0)
    print('Plan: ', plan, '\n')

    print('\n- If verbose=1, Pyhop prints the problem and solution, and returns the solution:')
    plan = planner.plan(init_state, task_list_1, verbose=1)
    print('Plan: ', plan, '\n')

    print('\n- If verbose=2, Pyhop also prints a note at each iteration:')
    plan = planner.plan(init_state, task_list_1, verbose=2)
    print('Plan: ', plan, '\n')

    print('\n- If verbose=3, Pyhop also prints the intermediate states:')
    plan = planner.plan(init_state, task_list_1, verbose=3)
    print('Plan: ', plan, '\n')



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