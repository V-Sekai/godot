#!/usr/bin/env python
"""
File Description: Simple Travel problem file. Initial state and task list for the planning problem is defined here.
"""

from ipyhop import MultiGoal
from examples.simple_travel.task_based.simple_travel_problem import init_state

init_state = init_state

goal1 = [('loc', 'alice', 'park')]
goal2 = [('loc', 'alice', 'park'), ('loc', 'bob', 'park')]

multigoal1 = MultiGoal('multigoal1')
multigoal1.loc = {'alice': 'park'}

# bigger initial goal
multigoal2 = MultiGoal('multigoal2')
multigoal2.loc = {'alice': 'park', 'bob': 'park'}

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
