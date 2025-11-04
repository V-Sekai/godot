#!/usr/bin/env python
"""
File Description: Robosub mod problem file. Initial state and task list for the planning problem is defined here.
"""
# ******************************************    Libraries to be imported    ****************************************** #
from random import sample, seed
from ipyhop import State


# ******************************************        Problem Definition      ****************************************** #
class StateSampler(object):
    def __init__(self, seed_val=None):
        self.options = dict()
        self.options['r'] = ['l0']
        self.options['t1'] = ['r']
        self.options['t2'] = ['r']
        self.options['g'] = ['l1']
        self.options['cm1'] = ['l1', 'l2', 'l3', 'l4', 'l5']
        self.options['cm2'] = ['l1', 'l2', 'l3', 'l4', 'l5']
        self.options['gp1'] = ['l2']
        self.options['gp2'] = ['l3']
        self.options['v1'] = ['l1', 'l2', 'l3', 'l4', 'l5']
        self.options['v2'] = ['l1', 'l2', 'l3', 'l4', 'l5']
        self.options['c1'] = ['l1', 'l2', 'l3', 'l4', 'l5']
        self.options['gm1'] = [None]    # Defined dynamically based on value of c1
        self.options['gm2'] = [None]    # Defined dynamically based on value of c1
        self.options['ap1'] = ['l4']
        self.options['ap2'] = ['l5']
        self.options['d1'] = ['l1', 'l2', 'l3', 'l4', 'l5']
        self.options['s1'] = ['l5']
        seed(seed_val)

    def sample(self, state_name="init_state"):
        state = State(state_name)
        rigid_relations = dict()
        rigid_relations['adj'] = \
            {'l0': ['l1'], 'l1': ['l0', 'l2'], 'l2': ['l1', 'l3'], 'l3': ['l2', 'l4'], 'l4': ['l3', 'l5'], 'l5': ['l4']}
        rigid_relations['type'] = \
            {'l0': 'l', 'l1': 'l', 'l2': 'l', 'l3': 'l', 'l4': 'l', 'l5': 'l', 'g': 'g', 'r': 'r', 'gm1': 'gm',
             'gm2': 'gm', 'cm1': 'cm', 'cm2': 'cm', 'gp1': 'gp', 'gp2': 'gp', 'c1': 'c', 'v1': 'v', 'v2': 'v',
             'd1': 'd', 'ap1': 'ap', 'ap2': 'ap', 's1': 's', 't1': 't', 't2': 't'}
        state.rigid = rigid_relations
        state.found = {'g': False, 'gm1': False, 'cm1': False, 'gp1': False,
                       'v1': False, 'v2': False, 'gp2': False,
                       'gm2': False, 'c1': False, 'ap1': False, 'd1': False,
                       'cm2': False, 'ap2': False, 's1': False,
                       'l0': True, 'l1': False, 'l2': False, 'l3': False, 'l4': False, 'l5': False}

        state.crossed_gate = {'g': False}
        state.traversed_path = {'gp1': False, 'gp2': False}
        state.vampire_touched = {'v1': False, 'v2': False}
        state.coffin_filled = {'c1': []}
        state.opened = {'c1': False}
        state.staked_dracula = {'d1': []}
        state.decapitated = {'d1': False}
        state.surfaced = {'r': False}

        state.loc = dict()
        for key in self.options:
            state.loc[key] = sample(self.options[key], 1)[0]
        valid_gm_locs = [x for x in ['l1', 'l2', 'l3', 'l4', 'l5'] if x <= state.loc['c1']]
        state.loc['gm1'] = sample(valid_gm_locs, 1)[0]
        state.loc['gm2'] = sample(valid_gm_locs, 1)[0]
        return state


# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    sampler = StateSampler()
    ss_1 = sampler.sample(state_name="ss_1")
    print(ss_1)
    ss_2 = sampler.sample(state_name="ss_2")
    print(ss_2)

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
