#!/usr/bin/env python
"""
File Description: File used for definition of State Class.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from copy import deepcopy


# ******************************************    Class Declaration Start     ****************************************** #
class State(object):
    """
    A state is just a collection of variable bindings.

    *   state = State('foo') tells IPyHOP to create an empty state object named 'foo'.
        To put variables and values into it, you should do assignments such as foo.var1 = val1
    """

    def __init__(self, name: str):
        self.__name__ = name

    # ******************************        Class Method Declaration        ****************************************** #
    def __str__(self):
        if self:
            var_str = "\r{state_name}.{var_name} = {var_value}\n"
            state_str = ""
            for name, val in self.__dict__.items():
                if name != "__name__":
                    _str = var_str.format(state_name=self.__name__, var_name=name, var_value=val)
                    _str = '\n\t\t'.join(_str[i:i+120] for i in range(0, len(_str), 120))
                    state_str += _str
            return state_str[:-1]
        else:
            return "False"

    # ******************************        Class Method Declaration        ****************************************** #
    def __repr__(self):
        return str(self.__class__) + ", " + self.__name__

    # ******************************        Class Method Declaration        ****************************************** #
    def update(self, state):
        self.__dict__.update(state.__dict__)
        return self

    # ******************************        Class Method Declaration        ****************************************** #
    def copy(self):
        return deepcopy(self)


# ******************************************    Class Declaration End       ****************************************** #
# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    print("Test instantiation of State class ...")
    test_state = State('test_state')
    test_state.test_var_1 = {'key1': 'val1'}
    test_state.test_var_2 = {'key1': 0}
    test_state.test_var_3 = {'key2': {'key3': 5}, 'key3': {'key2': 5}}
    print(test_state)

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
"""
