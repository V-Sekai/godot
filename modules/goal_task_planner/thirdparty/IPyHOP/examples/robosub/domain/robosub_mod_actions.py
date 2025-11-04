#!/usr/bin/env python
"""
File Description: Robosub actions file. All the actions for Robosub planning domain are defined here.
"""
# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import Actions


# ******************************************        Action Definitions      ****************************************** #
actions = Actions()


# search for a location in the field
def a_search_for(state, loc_):
    if loc_ in state.rigid['adj'][state.loc['r']]:
        state.found[loc_] = True
        return state


# search for an object at current location
def a_localize(state, obj_):
    if state.loc['r'] == state.loc[obj_]:
        state.found[obj_] = True
        return state


# acoustic pinger triangulation
def a_localize_ap(state, ap_):
    if state.rigid['type'][ap_] == 'ap':
        state.found[ap_] = True
        state.found[state.loc[ap_]] = True
        return state


# move to a recognized location
def a_move(state, loc_):
    if state.found[loc_] is True and state.rigid['type'][loc_] == 'l':
        state.loc['r'] = loc_
        return state


# cross the gate at current location from 40% side
def a_cross_gate_40(state, gate_):
    if state.loc['r'] == state.loc[gate_] and state.found[gate_] is True and state.rigid['type'][gate_] == 'g':
        state.crossed_gate[gate_] = 'T40'
        return state


# cross the gate at current location from 60% side
def a_cross_gate_60(state, gate_):
    if state.loc['r'] == state.loc[gate_] and state.found[gate_] is True and state.rigid['type'][gate_] == 'g':
        state.crossed_gate[gate_] = 'T60'
        return state


# pick an obj at current location (allowed objects: crucifix marker, garlic marker)
def a_pick(state, obj_):
    if state.loc['r'] == state.loc[obj_] and state.found[obj_] is True:
        if state.rigid['type'][obj_] == 'gm' or state.rigid['type'][obj_] == 'cm':
            state.loc[obj_] = 'r'
            return state


# trace a guide path at current location
def a_trace_guide_path(state, gp_):
    if state.loc['r'] == state.loc[gp_] and state.found[gp_] is True and state.rigid['type'][gp_] == 'gp':
        state.traversed_path[gp_] = True
        return state


# touch the back of a vampire at current location
def a_touch_back_v(state, v_):
    if state.loc['r'] == state.loc[v_] and state.found[v_] is True and state.rigid['type'][v_] == 'v':
        state.vampire_touched[v_] = 'Tb'
        return state


# touch the front of a vampire at current location.
def a_touch_front_v(state, v_):
    if state.loc['r'] == state.loc[v_] and state.found[v_] is True and state.rigid['type'][v_] == 'v':
        state.vampire_touched[v_] = 'Tf'
        return state


# open the coffin at current location
def a_open_c(state, c_):
    if state.loc['r'] == state.loc[c_] and state.found[c_] is True and state.rigid['type'][c_] == 'c':
        state.opened[c_] = True
        return state


# drop a garlic in an opened coffin at current location
def a_drop_garlic_open_coffin(state, gm_, c_):
    if state.loc[gm_] == 'r' and state.loc['r'] == state.loc[c_] and state.opened[c_] is True:
        if state.rigid['type'][c_] == 'c' and state.rigid['type'][gm_] == 'gm':
            state.loc[gm_] = c_
            state.coffin_filled[c_].append('1o')
            return state


# drop garlic on a closed coffin at current location
def a_drop_garlic_closed_coffin(state, gm_, c_):
    if state.loc[gm_] == 'r' and state.loc['r'] == state.loc[c_] and state.found[c_] is True:
        if state.rigid['type'][c_] == 'c' and state.rigid['type'][gm_] == 'gm':
            state.loc[gm_] = c_
            state.coffin_filled[c_].append('1c')
            return state


# decapitate a dracula at current location
def a_decap_d(state, d_):
    if state.loc['r'] == state.loc[d_] and state.found[d_] is True and state.rigid['type'][d_] == 'd':
        state.decapitated[d_] = True
        return state


# stake a decapitated dracula at current location
def a_stake_decap_d(state, t_, d_):
    if state.loc[t_] == 'r' and state.loc['r'] == state.loc[d_] and state.decapitated[d_] is True:
        if state.rigid['type'][t_] == 't' and state.rigid['type'][d_] == 'd':
            state.loc[t_] = d_
            state.staked_dracula[d_].append('1d')
            return state


# stake a normal dracula at current location
def a_stake_norm_d(state, t_, d_):
    if state.loc[t_] == 'r' and state.loc['r'] == state.loc[d_] and state.found[d_] is True:
        if state.rigid['type'][t_] == 't' and state.rigid['type'][d_] == 'd':
            state.loc[t_] = d_
            state.staked_dracula[d_].append('1n')
            return state


# surface in a surface zone at current location carrying a crucifix marker
def a_surface(state, cm, s):
    if state.loc['r'] == state.loc[s] and state.loc[cm] == 'r' and state.found[s] is True:
        if state.rigid['type'][cm] == 'cm' and state.rigid['type'][s] == 's':
            state.surfaced['r'] = True
            return state


actions.declare_actions([a_search_for, a_localize, a_localize_ap, a_move, a_cross_gate_40, a_cross_gate_60, a_pick,
                         a_touch_back_v, a_touch_front_v, a_trace_guide_path, a_open_c, a_drop_garlic_open_coffin,
                         a_drop_garlic_closed_coffin, a_decap_d, a_stake_decap_d, a_stake_norm_d, a_surface])

action_probability = {
    'a_cross_gate_40': [0.3, 0.7],
    'a_pick': [0.95, 0.05],
    'a_touch_back_v': [0.4, 0.6],
    'a_touch_front_v': [0.8, 0.2],
    'a_trace_guide_path': [0.85, 0.15],
    'a_open_c': [0.5, 0.5],
    'a_drop_garlic_open_coffin': [0.9, 0.1],
    'a_drop_garlic_closed_coffin': [0.9, 0.1],
    'a_decap_d': [0.4, 0.6],
    'a_stake_decap_d': [0.8, 0.2],
    'a_stake_norm_d': [0.8, 0.2],
}

action_cost = {
    'a_search_for': 2,
    'a_move': 5,
    'a_cross_gate_40': 10,
    'a_cross_gate_60': 8,
    'a_pick': 3,
    'a_touch_front_v': 3,
    'a_touch_back_v': 6,
    'a_trace_guide_path': 3,
    'a_open_c': 5,
    'a_drop_garlic_open_coffin': 2,
    'a_drop_garlic_closed_coffin': 2,
    'a_decap_d': 5,
    'a_stake_norm_d': 2,
    'a_stake_decap_d': 2,
    'a_surface': 3,
}

actions.declare_action_models(action_probability, action_cost)

# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for Robosub Mod Commands isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
