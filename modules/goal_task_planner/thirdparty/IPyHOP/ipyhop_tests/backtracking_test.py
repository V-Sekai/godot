#!/usr/bin/env python
"""
File Description: Backtracking Test File. Serves as a basic test for correct backtracking.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import Methods, Actions, State, IPyHOP


def a_putv(state, flag_val):
    state.flag = flag_val
    return state


def a_getv(state, flag_val):
    if state.flag == flag_val:
        return state


actions = Actions()
actions.declare_actions([a_putv, a_getv])


def m_err(state):
    return [('a_putv', 0), ('a_getv', 1)]


def m0(state):
    return [('a_putv', 0), ('a_getv', 0)]


def m1(state):
    return [('a_putv', 1), ('a_getv', 1)]


methods = Methods()
methods.declare_task_methods('put_it', [m_err, m0, m1])


def m_need0(state):
    return [('a_getv', 0)]


def m_need1(state):
    return [('a_getv', 1)]


methods.declare_task_methods('need0', [m_need0])
methods.declare_task_methods('need1', [m_need1])
methods.declare_task_methods('need01', [m_need0, m_need1])
methods.declare_task_methods('need10', [m_need1, m_need0])


init_state = State("init_state")
init_state.flag = -1


# ******************************************        Main Program Start      ****************************************** #
def main():
    print(methods)
    print(actions)
    print(init_state)

    # two possible expected answers for check_result
    exp_0 = [('a_putv', 0), ('a_getv', 0), ('a_getv', 0)]
    exp_1 = [('a_putv', 1), ('a_getv', 1), ('a_getv', 1)]

    planner = IPyHOP(methods, actions)

    print("-- Four examples with verbose=3 to get a detailed account of the backtracking.")

    plan = planner.plan(init_state, [('put_it',), ('need0',)], verbose=3)
    assert plan == exp_0, "Result plan and expected plan are not same"
    print("Above, the planner backtracks once to use a different method for 'put_it'.\n")

    plan = planner.plan(init_state, [('put_it',), ('need01',)], verbose=3)
    assert plan == exp_0, "Result plan and expected plan are not same"
    print("The backtracking in the above example is the same as in the first one.\n")

    plan = planner.plan(init_state, [('put_it',), ('need10',)], verbose=3)
    assert plan == exp_0, "Result plan and expected plan are not same"
    print("Above, the planner backtracks to use a different method for 'put_it', \n"
          "and later it backtracks to use a different method for 'need10'.\n")

    plan = planner.plan(init_state, [('put_it',), ('need1',)], verbose=3)
    assert plan == exp_1, "Result plan and expected plan are not same"
    print("First, the planner backtracks to use a different method for 'put_it'. \n"
          "But the solution it finds for 'put_it' doesn't satisfy the preconditions of the \n"
          "method for 'need1', making it backtrack to use a third method for 'put_it'.\n")


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