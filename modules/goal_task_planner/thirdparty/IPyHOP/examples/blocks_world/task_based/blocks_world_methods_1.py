#!/usr/bin/env python
"""
File Description: Blocks World methods file. All the methods for Blocks World planning domain are defined here.
Derived from: PyHOP example of Blocks World domain by Dana Nau <nau@cs.umd.edu>, November 15, 2012.

Each IPyHOP method is a Python function. The 1st argument is the current state (this is analogous to Python methods,
in which the first argument is the class instance). The rest of the arguments must match the arguments of the
task that the method is for. For example, the task ('get', b1) has a method "tm_get(state, b1)", as shown below.
"""

from ipyhop import Methods


# ******************************************        Helper Functions        ****************************************** #
def is_done(b1, state, goal):
    if b1 == 'table': return True
    if b1 in goal.pos and goal.pos[b1] != state.pos[b1]:
        return False
    if state.pos[b1] == 'table': return True
    return is_done(state.pos[b1], state, goal)


def status(b1, state, goal):
    if is_done(b1, state, goal):
        return 'done'
    elif not state.clear[b1]:
        return 'inaccessible'
    elif not (b1 in goal.pos) or goal.pos[b1] == 'table':
        return 'move-to-table'
    elif is_done(goal.pos[b1], state, goal) and state.clear[goal.pos[b1]]:
        return 'move-to-block'
    else:
        return 'waiting'


def all_blocks(state):
    return state.clear.keys()


def find_if(cond, seq):
    """
    Return the first x in seq such that cond(x) holds, if there is one.
    Otherwise return None.
    """
    for x in seq:
        if cond(x):
            return x
    return None


# ******************************************        Method Definitions        **************************************** #
def tm_move_blocks(state, goal):
    """
    This method implements the following block-stacking algorithm:
    If there's a block that can be moved to its final position, then do so and call move_blocks recursively.
    Otherwise, if there's a block that needs to be moved and can be moved to the table, then do so and call move_blocks
    recursively. Otherwise, no blocks need to be moved.
    """
    for b1 in all_blocks(state):
        s = status(b1, state, goal)
        if s == 'move-to-table':
            return [('move_one', b1, 'table'), ('move_blocks', goal)]
        elif s == 'move-to-block':
            return [('move_one', b1, goal.pos[b1]), ('move_blocks', goal)]
        else:
            continue
    #
    # if we get here, no blocks can be moved to their final locations
    b1 = find_if(lambda x: status(x, state, goal) == 'waiting', all_blocks(state))
    if b1 is not None:
        return [('move_one', b1, 'table'), ('move_blocks', goal)]
    #
    # if we get here, there are no blocks that need moving
    return []


# Create a IPyHOP Methods object. A Methods object stores all the methods defined for the planning domain.
methods = Methods()

# declare_task_methods must be called once for each task name.
# Below, 'declare_task_methods('move_blocks', [tm_move_blocks])' tells IPyHOP that 'move_blocks' has one method,
# tm_move_blocks. Notice that 'move_blocks' is a quoted string, and tm_move_blocks is the actual function.
methods.declare_task_methods('move_blocks', [tm_move_blocks])


def tm_move1(state, b1, dest):
    """
    Generate subtasks to get b1 and put it at dest.
    """
    return [('get', b1), ('put', b1, dest)]


methods.declare_task_methods('move_one', [tm_move1])


def tm_get(state, b1):
    """
    Generate either a pickup or an unstack subtask for b1.
    """
    if state.clear[b1]:
        if state.pos[b1] == 'table':
            return [('a_pickup', b1)]
        else:
            return [('a_unstack', b1, state.pos[b1])]


methods.declare_task_methods('get', [tm_get])


def tm_put(state, b1, b2):
    """
    Generate either a putdown or a stack subtask for b1. b2 is b1's destination: either the table or another block.
    """
    if state.holding['hand'] == b1:
        if b2 == 'table':
            return [('a_putdown', b1)]
        else:
            return [('a_stack', b1, b2)]


methods.declare_task_methods('put', [tm_put])

# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for Blocks World Mehthods isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
