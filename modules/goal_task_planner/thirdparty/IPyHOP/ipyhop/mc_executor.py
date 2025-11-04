#!/usr/bin/env python
"""
File Description: File used for definition of monte-carlo plan executor
"""

# ******************************************    Libraries to be imported    ****************************************** #
from typing import List, Union, Tuple
from ipyhop.state import State
from ipyhop.actions import Actions
import numpy as np


# ******************************************    Class Declaration Start     ****************************************** #
class MonteCarloExecutor(object):
    def __init__(self, actions: Actions, seed: int = 10):
        self.actions = actions
        self.exec_list = None
        np.random.seed(seed)

    # ******************************        Class Method Declaration        ****************************************** #
    def execute(self, state: State, plan: List[str], actions: Union[Actions, None] = None) -> List[Tuple]:
        self.actions = actions if actions is not None else self.actions
        self.exec_list = [(None, state.copy())]
        state_copy = state.copy()
        for act_inst in plan:
            act_name = act_inst[0]
            act_params = act_inst[1:]
            act_func = self.actions.action_dict[act_name]
            act_prob = self.actions.action_prob[act_name]
            result = np.random.choice(len(act_prob), 1, p=act_prob)[0]
            result_state = None
            if result == 0:
                result_state = act_func(state_copy.copy(), *act_params)

            self.exec_list.append((act_inst, result_state))
            if result_state is None:
                return self.exec_list
            state_copy = result_state
        return self.exec_list


# ******************************************    Class Declaration End       ****************************************** #
# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for MonteCarloExecutor isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
"""
