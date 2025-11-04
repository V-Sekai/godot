#!/usr/bin/env python
"""
File Description: Robosub mod problem file. Initial state and task list for the planning problem is defined here.
"""
# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import State


# ******************************************        Problem Definition      ****************************************** #
init_state = State('init_state')

init_state.loc = {'r': 'l0', 't1': 'r', 't2': 'r',
                  'g': 'l1', 'gm1': 'l1', 'cm1': 'l1', 'gp1': 'l1',
                  'v1': 'l2', 'v2': 'l2', 'gp2': 'l2',
                  'gm2': 'l3', 'c1': 'l3', 'ap1': 'l4', 'd1': 'l4',
                  'cm2': 'l5', 'ap2': 'l5', 's1': 'l5'}

init_state.found = {'g': False, 'gm1': False, 'cm1': False, 'gp1': False,
                    'v1': False, 'v2': False, 'gp2': False,
                    'gm2': False, 'c1': False, 'ap1': False, 'd1': False,
                    'cm2': False, 'ap2': False, 's1': False,
                    'l0': True, 'l1': False, 'l2': False, 'l3': False, 'l4': False, 'l5': False}

init_state.crossed_gate = {'g': False}
init_state.traversed_path = {'gp1': False, 'gp2': False}
init_state.vampire_touched = {'v1': False, 'v2': False}
init_state.coffin_filled = {'c1': []}
init_state.opened = {'c1': False}
init_state.staked_dracula = {'d1': []}
init_state.decapitated = {'d1': False}
init_state.surfaced = {'r': False}

rigid_relations = dict()
rigid_relations['adj'] = {'l0': ['l1'], 'l1': ['l0', 'l2'], 'l2': ['l1', 'l3'], 'l3': ['l2', 'l4'],
                          'l4': ['l3', 'l5'], 'l5': ['l4']}
rigid_relations['type'] = {'l0': 'l', 'l1': 'l', 'l2': 'l', 'l3': 'l', 'l4': 'l', 'l5': 'l', 'g': 'g', 'r': 'r',
                           'gm1': 'gm', 'gm2': 'gm', 'cm1': 'cm', 'cm2': 'cm', 'gp1': 'gp', 'gp2': 'gp', 'c1': 'c',
                           'v1': 'v', 'v2': 'v', 'd1': 'd', 'ap1': 'ap', 'ap2': 'ap', 's1': 's', 't1': 't', 't2': 't'}
init_state.rigid = rigid_relations

task_list_1 = [('pinger_task', ), ('main_task', ['l1', 'l2', 'l3', 'l4', 'l5'])]
task_list_2 = [('pinger_task', ), ('main_task', ['l1']), ('main_task', ['l2']), ('main_task', ['l3']),
               ('main_task', ['l4']), ('main_task', ['l5'])]


# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for Robosub Mod Problem isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""