#!/usr/bin/env python
"""
File Description: Blocks World methods file. All the methods for Blocks World planning domain are defined here.
Derived from: PyHOP example of Blocks World domain by Dana Nau <nau@cs.umd.edu>, November 15, 2012.

Each IPyHOP method is a Python function. The 1st argument is the current state (this is analogous to Python methods,
in which the first argument is the class instance). The rest of the arguments must match the arguments of the
task that the method is for. For example, the task ('get', b1) has a method "tm_get(state, b1)", as shown below.
"""

from ipyhop import Methods
from examples.blocks_world.task_based.blocks_world_methods_1 import status


# ******************************************        Helper Functions        ****************************************** #
def all_clear_blocks(state):
    return [x for x in state.clear if state.clear[x] == True]


# ******************************************        Method Definitions        **************************************** #
def mgm_move_blocks(state, multigoal):
    """
    This method implements the following block-stacking algorithm:
    - If there's a clear block b that can be moved to a position where it will never need to be moved again,
    then do so, and invoke the multigoal again.
    - Otherwise, if there's a clear block b that needs to be moved out of the way in order to make another
    block accessible, then move b to the table, and invoke the multigoal again.
    - Otherwise, no blocks need to be moved.
    """
    for b1 in all_clear_blocks(state):
        s = status(b1, state, multigoal)
        if s == 'move-to-block':
            b2 = multigoal.pos[b1]
            return [('pos', b1, b2), multigoal]
        elif s == 'move-to-table':
            return [('pos', b1, 'table'), multigoal]
        else:
            continue
    #
    # if we get here, no blocks can be moved to their final locations
    for b1 in all_clear_blocks(state):
        if status(b1, state, multigoal) == 'waiting':
            return [('pos', b1, 'table'), multigoal]
    #
    # if we get here, there are no blocks that need moving
    return []


# Create a IPyHOP Methods object. A Methods object stores all the methods defined for the planning domain.
methods = Methods()

# declare_task_methods must be called once for each task name.
# Below, 'declare_task_methods('move_blocks', [tm_move_blocks])' tells IPyHOP that 'move_blocks' has one method,
# tm_move_blocks. Notice that 'move_blocks' is a quoted string, and tm_move_blocks is the actual function.
methods.declare_multigoal_methods(None, [mgm_move_blocks])


def gm_move1(state, b1, b2):
    """
    If goal is ('pos',b1,b2), b1 is clear, we're holding nothing, and b2 is
    either the table or clear, then assert goals to get b1 and put it on b2.
    """
    if b2 != 'hand' and state.clear[b1] and state.holding['hand'] == False:
        if b2 == 'table' or state.clear[b2]:
            return [('pos', b1, 'hand'), ('pos', b1, b2)]


def gm_get(state, b1, b2):
    """
    If goal is ('pos',b1,b2='hand') and b is clear and we're holding nothing
    Generate either a pickup or an unstack subtask for b1.
    """
    if b2 == 'hand' and state.clear[b1] and state.holding['hand'] == False:
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


methods.declare_goal_methods('pos', [gm_move1, gm_get, gm_put])

# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for Blocks World Mehthods isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
