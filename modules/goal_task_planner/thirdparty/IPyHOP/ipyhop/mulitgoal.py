#!/usr/bin/env python
"""
File Description: File used for definition of Multigoal Class.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from copy import deepcopy
from typing import Union


# ******************************************    Class Declaration Start     ****************************************** #
class MultiGoal(object):
    """
    A MultiGoal is just a collection of variable bindings.

    *   mg = MultiGoal('foo') tells IPyHOP to create an empty MultiGoal object named 'foo'.
        To put variables and values into it, you should do assignments such as mg.var1 = val1
        It represents a conjunctive goal, i.e., the goal of achieving every state-variable binding in mg.

        example::
            mg = MultiGoal('multi_goal_1')
            mg.loc = {'a': 'r1', 'b': 'r2'}
            mg.loc['c'] = 'r3' ...
    """

    def __init__(self, name: str, goal_tag: Union[None, str] = None):
        self.__name__ = name
        self.goal_tag = goal_tag

    # ******************************        Class Method Declaration        ****************************************** #
    def __str__(self):
        if self:
            var_str = "\t{goal_name}.{var_name} = {var_value}\n"
            goal_str = ""
            for name, val in self.__dict__.items():
                if name != "__name__":
                    goal_str += var_str.format(goal_name=self.__name__, var_name=name, var_value=val)
            return goal_str[:-1]
        else:
            return "False"

    # ******************************        Class Method Declaration        ****************************************** #
    def __repr__(self):
        return str(self.__class__) + ", " + self.__name__

    # ******************************        Class Method Declaration        ****************************************** #
    def update(self, multigoal):
        self.__dict__.update(multigoal.__dict__)
        return self

    # ******************************        Class Method Declaration        ****************************************** #
    def copy(self):
        return deepcopy(self)


# ******************************************    Class Declaration End       ****************************************** #
# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    print("Test instantiation of MultiGoal class ...")
    test_goal = MultiGoal('test_goal')
    test_goal.test_var_1 = {'key1': 'val1'}
    test_goal.test_var_2 = {'key1': 0}
    test_goal.test_var_3 = {'key2': {'key3': 5}, 'key3': {'key2': 5}}
    print(test_goal)

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
"""
