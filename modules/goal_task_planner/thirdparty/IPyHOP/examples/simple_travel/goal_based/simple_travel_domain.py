#!/usr/bin/env python
"""
File Description: Simple Travel domain file. All the actions and methods for Simple Travel planning domain are
    defined here.
Derived from: PyHOP example of Simple Travel domain by Dana Nau <nau@cs.umd.edu>, May 31, 2013
"""

from ipyhop import Methods, mgm_split_multigoal
from examples.simple_travel.task_based.simple_travel_domain import actions
from examples.simple_travel.task_based.simple_travel_domain import tm_do_nothing as gm_do_nothing
from examples.simple_travel.task_based.simple_travel_domain import tm_travel_by_foot as gm_travel_by_foot
from examples.simple_travel.task_based.simple_travel_domain import tm_travel_by_taxi as gm_travel_by_taxi

actions = actions
methods = Methods()
methods.declare_goal_methods('loc', [gm_do_nothing, gm_travel_by_foot, gm_travel_by_taxi])
methods.declare_multigoal_methods(None, [mgm_split_multigoal])

# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for Simple Travel Domain isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
