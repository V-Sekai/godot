#!/usr/bin/env python
"""
File Description: Sample test file. Tests the replanning.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import Methods, IPyHOP
from ipyhop_tests.test_action_models import actions_1 as actions
from ipyhop_tests.test_state_models import init_state_1 as init_state

methods = Methods()


def tm_1_1(state): return [('tm_2', ), ('t_a', 3, 4), ('t_a', 4, 5)]
def tm_1_2(state): return [('tm_2', ), ('t_a', 3, 4), ('t_a', 4, 5), ('t_a', 5, 6)]


methods.declare_task_methods('tm_1', [tm_1_1, tm_1_2])


def tm_2_1(state): return [('t_a', 0, 1), ('t_a', 1, 2), ('t_a', 2, 3)]
def tm_2_2(state): return [('t_a', 0, 1), ('t_a', 1, 2), ('t_a', 2, 3), ('t_a', 3, 7)]
def tm_2_3(state): return [('t_a', 0, 1), ('t_a', 1, 3), ('t_a', 3, 7)]


methods.declare_task_methods('tm_2', [tm_2_1, tm_2_2, tm_2_3])


def tm_3_1(state): return [('t_a', 7, 8)]


methods.declare_task_methods('tm_3', [tm_3_1])


# ******************************************        Main Program Start      ****************************************** #
def main():
    print('\n\r', methods)
    print('\n\r', actions)
    print('\nInitial State: \n\r', init_state, '\n\r')

    planner = IPyHOP(methods, actions)
    plan = planner.plan(init_state, [('tm_1',), ('tm_3',)], verbose=3)
    exp_0 = [('t_a', 0, 1), ('t_a', 1, 2), ('t_a', 2, 3), ('t_a', 3, 7), ('t_a', 3, 4), ('t_a', 4, 5), ('t_a', 7, 8)]
    assert plan == exp_0, "Result plan and expected plan are not same."

    state_list = planner.simulate(init_state)
    fail_node = plan[2]
    after_fail_state = state_list[2]
    planner.blacklist_command(fail_node)

    # In practice the fail_node_id is directly tracked while executing the solution_tree.
    # This kind of search is just done for this test script.
    fail_node_id = 0
    for node in planner.sol_tree.nodes:
        if planner.sol_tree.nodes[node]['info'] == fail_node:
            fail_node_id = node
            break

    print("\nAssume that the action ", fail_node, "failed. \n"
          "We replan with the new state.\n")

    plan = planner.replan(after_fail_state, fail_node_id, verbose=3)
    exp_1 = [('t_a', 0, 1), ('t_a', 1, 3), ('t_a', 3, 7), ('t_a', 3, 4), ('t_a', 4, 5), ('t_a', 7, 8)]
    assert plan == exp_1, "Result plan and expected plan are not same."


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