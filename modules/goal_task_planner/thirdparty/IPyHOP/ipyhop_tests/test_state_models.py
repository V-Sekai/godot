#!/usr/bin/env python
"""
File Description: Test state models. This dummy statesample_test_3.py definition is used for various tests of IPyHOP.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import State


init_state_1 = State("init_state_1")
init_state_1.flag = {0: True}
for i in range(1, 20):
    init_state_1.flag[i] = False


# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for test state model isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
