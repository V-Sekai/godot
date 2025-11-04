#!/usr/bin/env python
"""
File Description: Blocks World problem file. Initial state and task list for the planning problem is defined here.
"""

from ipyhop import State, MultiGoal

# A state is a collection of all of the state variables and their values.
# Every state variable in the domain should have a value.
init_state_1 = State('init_state_1')
init_state_1.pos = {'a': 'b', 'b': 'table', 'c': 'table'}
init_state_1.clear = {'c': True, 'b': False, 'a': True}
init_state_1.holding = {'hand': False}

# A goal is a collection of some (but not necessarily all) of the state variables and their desired values.
# Below, both goal1a and goal1b specify c on b, and b on a.
# The difference is that goal1a also specifies that a is on table and the hand is empty.
goal1a = MultiGoal('goal1a')
goal1a.pos = {'c': 'b', 'b': 'a', 'a': 'table'}
goal1a.clear = {'c': True, 'b': False, 'a': False}
goal1a.holding = {'hand': False}

goal1b = MultiGoal('goal1b')
goal1b.pos = {'c': 'b', 'b': 'a'}

init_state_2 = State('init_state_2')
init_state_2.pos = {'a': 'c', 'b': 'd', 'c': 'table', 'd': 'table'}
init_state_2.clear = {'a': True, 'c': False, 'b': True, 'd': False}
init_state_2.holding = {'hand': False}

goal2a = MultiGoal('goal2a')
goal2a.pos = {'b': 'c', 'a': 'd', 'c': 'table', 'd': 'table'}
goal2a.clear = {'a': True, 'c': False, 'b': True, 'd': False}
goal2a.holding = {'hand': False}

goal2b = MultiGoal('goal2b')
goal2b.pos = {'b': 'c', 'a': 'd'}

init_state_3 = State('init_state_3')
init_state_3.pos = {1: 12, 12: 13, 13: 'table', 11: 10, 10: 5, 5: 4, 4: 14, 14: 15, 15: 'table', 9: 8, 8: 7, 7: 6,
                    6: 'table', 19: 18, 18: 17, 17: 16, 16: 3, 3: 2, 2: 'table'}
init_state_3.clear = {x: False for x in range(1, 20)}
init_state_3.clear.update({1: True, 11: True, 9: True, 19: True})
init_state_3.holding = {'hand': False}

goal3 = MultiGoal('goal3')
goal3.pos = {15: 13, 13: 8, 8: 9, 9: 4, 4: 'table', 12: 2, 2: 3, 3: 16, 16: 11, 11: 7, 7: 6, 6: 'table'}
goal3.clear = {17: True, 15: True, 12: True}

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
