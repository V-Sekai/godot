#!/usr/bin/env python
"""
File Description: Blocks World methods file. All the methods for Blocks World planning domain are defined here.
Derived from: PyHOP example of Blocks World domain by Dana Nau <nau@cs.umd.edu>, November 15, 2012.

Each IPyHOP method is a Python function. The 1st argument is the current state (this is analogous to Python methods,
in which the first argument is the class instance). The rest of the arguments must match the arguments of the
task that the method is for. For example, the task ('get', b1) has a method "tm_get(state, b1)", as shown below.
"""

from ipyhop import Methods, mgm_split_multigoal


# ******************************************        Method Definitions        **************************************** #
# Create a IPyHOP Methods object. A Methods object stores all the methods defined for the planning domain.
methods = Methods()
methods.declare_multigoal_methods(None, [mgm_split_multigoal])


def gm_move1(state, b1, b2):
    """
    If goal is ('pos',b1,b2) and we're holding nothing,
    then assert goals to get b1 and put it on b2.
    """
    if b2 != 'hand' and not state.holding['hand']:
        if b2 == 'table':
            return [('clear', b1, True), ('pos', b1, 'hand'), ('pos', b1, b2)]
        else:
            return [('clear', b2, True), ('clear', b1, True), ('pos', b1, 'hand'), ('pos', b1, b2)]


def gm_make_clear(state, b2, truth):
    if truth == True:
        if b2 == 'table' or state.clear[b2]:
            return []
        else:
            above_b2 = [b1 for b1 in state.pos if state.pos[b1] == b2]
            b1 = above_b2[0]  # the block that's on b2
            return [('clear', b1, True), ('pos', b1, 'table')]


def gm_hold(state, b1, hand):
    """
    If goal is ('pos',b1,'hand') and b1 is clear and we're holding nothing
    Generate either a pickup or an unstack subtask for b1.
    """
    if hand == 'hand' and state.clear[b1] and state.holding[hand] == False:
        if state.pos[b1] == 'table':
            return [('a_pickup', b1)]
        else:
            return [('a_unstack', b1, state.pos[b1])]


def gm_put(state, b1, b2):
    """
    If goal is ('pos',b1,b2) and we're holding b1,
    Generate either a putdown or a stack subtask for b1.
    b2 is b1's destination: either the table or another block.
    """
    if b2 != 'hand' and state.pos[b1] == 'hand':
        if b2 == 'table':
            return [('a_putdown', b1)]
        elif state.clear[b2]:
            return [('a_stack', b1, b2)]


methods.declare_goal_methods('pos', [gm_move1, gm_hold, gm_put])
methods.declare_goal_methods('clear', [gm_make_clear])

# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for Blocks World Mehthods isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
