#!/usr/bin/env python
"""
File Description: File used for definition of Actions Class.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from __future__ import print_function, division
from typing import List, Callable, Union, Any, Dict
from ipyhop.state import State


# ******************************************    Class Declaration Start     ****************************************** #
class Actions(object):
    """
    A class to store all the actions defined in a planning domain.

    *   actions = Actions() tells IPyHOP to create an empty actions container.
        To add actions into it, you should use actions.declare_actions(action_list).
        declare_actions([a1, a2, ..., ak]) tells IPyHOP that a1, a2, ..., ak are all of the planning actions.
        This supersedes any previous call to declare_actions([a1, a2, ..., ak]).

    All the actions are stored in a dictionary member variable named action_dict with the following structure::

        {op_name_1: [op_func_a, ...], op_name_2: [op_func_x, ...]...}

    Use the member function declare_actions to add actions to the action_dict.
    """

    def __init__(self):
        self.action_dict = dict()
        self.action_prob = dict()
        self.action_cost = dict()

    # ******************************        Class Method Declaration        ****************************************** #
    def __str__(self):
        a_str = '\n\rACTIONS: ' + ', '.join(self.action_dict)
        return a_str

    # ******************************        Class Method Declaration        ****************************************** #
    def __repr__(self):
        return self.__str__()

    _action_list_type = List[Callable[[Any], Union[State, bool]]]
    _act_prob_dict_type = Dict[str, List]
    _act_cost_dict_type = Dict[str, float]

    # ******************************        Class Method Declaration        ****************************************** #
    def declare_actions(self, action_list: _action_list_type):
        """
        declare_actions([a1, a2, ..., ak]) tells IPyHOP that [a1, a2, ..., ak] are all of the planning actions.
        This supersedes any previous call to declare_actions.

        :param action_list: List of actions in the
        """
        assert type(action_list) == list, "action_list must be a list."
        for action in action_list:
            assert callable(action), "action in action_list should be callable."
        self.action_dict.update({action.__name__: action for action in action_list})
        self.action_prob.update({action.__name__: [1, 0] for action in action_list})
        self.action_cost.update({action.__name__: 1.0 for action in action_list})

    # ******************************        Class Method Declaration        ****************************************** #
    def declare_action_models(self, act_prob_dict: _act_prob_dict_type, act_cost_dict: _act_cost_dict_type):
        self.action_prob.update({action: act_prob_dict[action] for action in act_prob_dict})
        self.action_cost.update({action: act_cost_dict[action] for action in act_cost_dict})

        assert(len(self.action_prob.keys()) == len(self.action_dict.keys()))
        assert (len(self.action_cost.keys()) == len(self.action_dict.keys()))


# ******************************************    Class Declaration End       ****************************************** #
# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    def test_action_1(): return False
    def test_action_2(): return False
    def test_action_3(): return False


    print("Test instantiation of Methods class ...")
    actions = Actions()
    actions.declare_actions([test_action_1, test_action_2, test_action_3])
    print(actions)

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
"""
