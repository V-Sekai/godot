#!/usr/bin/env python
"""
File Description: Blocks World alternative methods file. Alternative methods for the planning domain are defined here.
Derived from: PyHOP example of Blocks World domain by Dana Nau <nau@cs.umd.edu>, November 15, 2012.

Each IPyHOP method is a Python function. The 1st argument is the current state (this is analogous to Python methods,
in which the first argument is the class instance). The rest of the arguments must match the arguments of the
task that the method is for. For example, the task ('get', b1) has a method "tm_get(state, b1)", as shown below.
"""

from ipyhop import Methods
from examples.blocks_world.task_based.blocks_world_methods_1 import tm_move_blocks, \
    tm_move1, tm_put

# Create a IPyHOP Methods object. A Methods object stores all the methods defined for the planning domain.
methods = Methods()

# declare_task_methods must be called once for each task name.
# Below, 'declare_task_methods('move_blocks', [tm_move_blocks])' tells
# IPyHOP that 'move_blocks' has one method, tm_move_blocks.
# Notice that 'move_blocks' is a quoted string, and tm_move_blocks is the actual function.
methods.declare_task_methods('move_blocks', [tm_move_blocks])

methods.declare_task_methods('move_one', [tm_move1])


def tm_get_by_pickup(state, b1):
    """
    Generate a pickup subtask.
    """
    if state.clear[b1]:
        return [('pickup_task', b1)]


def tm_get_by_unstack(state, b1):
    """
    Generate a pickup subtask.
    """
    if state.clear[b1]:
        return [('unstack_task', b1)]


methods.declare_task_methods('get', [tm_get_by_pickup, tm_get_by_unstack])


def tm_pickup(state, b1):
    """
    Generate a pickup subtask.
    """
    if state.clear[b1]:
        return [('a_pickup', b1)]


methods.declare_task_methods('pickup_task', [tm_pickup])


def tm_unstack(state, b1):
    """
    Generate a pickup subtask.
    """
    if state.clear[b1]:
        return [('a_unstack', b1, state.pos[b1])]


methods.declare_task_methods('unstack_task', [tm_unstack])

methods.declare_task_methods('put', [tm_put])

# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for Blocks World Mehthods isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
