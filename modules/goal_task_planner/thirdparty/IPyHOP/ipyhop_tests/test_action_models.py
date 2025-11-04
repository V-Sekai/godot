#!/usr/bin/env python
"""
File Description: Test action models. This dummy action definition is used for various tests of IPyHOP.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import Actions


def t_a(state, flag_key_1, flag_key_2):
    if state.flag[flag_key_1] is True:
        state.flag[flag_key_2] = True
        return state


actions_1 = Actions()
actions_1.declare_actions([t_a])


# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for test action model isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
