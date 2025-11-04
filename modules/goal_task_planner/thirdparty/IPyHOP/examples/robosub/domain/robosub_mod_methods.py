#!/usr/bin/env python
"""
File Description: Robosub methods file. All the methods for Robosub planning domain are defined here.
"""
# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import Methods

# ******************************************        Method Definitions      ****************************************** #
methods = Methods()


def tm1_move(state, loc_):
    if state.loc['r'] != loc_:
        if state.found[loc_] is True:
            return [('a_move', loc_)]
        else:
            if loc_ in state.rigid['adj'][state.loc['r']]:
                return [('a_search_for', loc_), ('a_move', loc_)]
            if loc_ > state.loc['r']:
                l_ = state.rigid['adj'][state.loc['r']][-1]
                return [('a_search_for', l_), ('a_move', l_), ('move_task', loc_)]
            if loc_ < state.loc['r']:
                l_ = state.rigid['adj'][state.loc['r']][0]
                return [('a_search_for', l_), ('a_move', l_), ('move_task', loc_)]
    return []


methods.declare_task_methods('move_task', [tm1_move])


def tm1_cross_gate(state, gate_):
    if state.crossed_gate[gate_] is False:
        if state.loc['r'] == state.loc[gate_]:
            if state.found[gate_] is True:
                return [('a_cross_gate_40', gate_)]
            return [('a_localize', gate_), ('a_cross_gate_40', gate_)]
        return [('move_task', state.loc[gate_]), ('cross_gate_task', gate_)]
    return []


def tm2_cross_gate(state, gate_):
    if state.crossed_gate[gate_] is False:
        if state.loc['r'] == state.loc[gate_]:
            if state.found[gate_] is True:
                return [('a_cross_gate_60', gate_)]
            return [('a_localize', gate_), ('a_cross_gate_60', gate_)]
        return [('move_task', state.loc[gate_]), ('cross_gate_task', gate_)]
    return []


methods.declare_task_methods('cross_gate_task', [tm1_cross_gate, tm2_cross_gate])


def tm1_pick(state, obj_):
    if state.loc[obj_] != 'r':
        if state.loc['r'] == state.loc[obj_]:
            if state.found[obj_]:
                return [('a_pick', obj_)]
            return [('a_localize', obj_), ('a_pick', obj_)]
        return [('move_task', state.loc[obj_]), ('pick_task', obj_)]
    return []


def tm2_pick(*_):
    return []   # skip the task


methods.declare_task_methods('pick_task', [tm1_pick, tm2_pick])


def tm1_trace_path(state, gp_):
    if state.traversed_path[gp_] is False:
        if state.loc['r'] == state.loc[gp_]:
            if state.found[gp_] is True:
                return [('a_trace_guide_path', gp_)]
            return [('a_localize', gp_), ('a_trace_guide_path', gp_)]
        return [('move_task', state.loc[gp_]), ('trace_path_task', gp_)]
    return []


def tm2_trace_path(*_):
    return []   # skip the task


methods.declare_task_methods('trace_path_task', [tm1_trace_path, tm2_trace_path])


def tm1_slay_vampire(state, v_):
    if state.vampire_touched[v_] is False:
        if state.loc['r'] == state.loc[v_]:
            if state.found[v_] is True:
                return [('a_touch_back_v', v_)]
            return [('a_localize', v_), ('a_touch_back_v', v_)]
        return [('move_task', state.loc[v_]), ('slay_vampire_task', v_)]
    return []


def tm2_slay_vampire(state, v_):
    if state.vampire_touched[v_] is False:
        if state.loc['r'] == state.loc[v_]:
            if state.found[v_] is True:
                return [('a_touch_front_v', v_)]
            return [('a_localize', v_), ('a_touch_front_v', v_)]
        return [('move_task', state.loc[v_]), ('slay_vampire_task', v_)]
    return []


def tm3_slay_vampire(*_):
    return []   # skip the task


methods.declare_task_methods('slay_vampire_task', [tm1_slay_vampire, tm2_slay_vampire, tm3_slay_vampire])


def tm1_drop_garlic(state, gm_, c_):
    if len(state.coffin_filled[c_]) < 2:
        if state.loc[gm_] == 'r':
            if state.loc['r'] == state.loc[c_]:
                if state.found[c_] is True:
                    if state.opened[c_] is True:
                        return [('a_drop_garlic_open_coffin', gm_, c_)]
                    return [('a_open_c', c_), ('a_drop_garlic_open_coffin', gm_, c_)]
                return [('a_localize', c_), ('a_open_c', c_), ('a_drop_garlic_open_coffin', gm_, c_)]
            return [('move_task', state.loc[c_]), ('drop_garlic_task', gm_, c_)]
        return
    return []


def tm2_drop_garlic(state, gm_, c_):
    if len(state.coffin_filled[c_]) < 2:
        if state.loc[gm_] == 'r':
            if state.loc['r'] == state.loc[c_]:
                if state.found[c_] is True:
                    return [('a_drop_garlic_closed_coffin', gm_, c_)]
                return [('a_localize', c_), ('a_drop_garlic_closed_coffin', gm_, c_)]
            return [('move_task', state.loc[c_]), ('drop_garlic_task', gm_, c_)]
        return
    return []


def tm3_drop_garlic(*_):
    return []   # skip the task


methods.declare_task_methods('drop_garlic_task', [tm1_drop_garlic, tm2_drop_garlic, tm3_drop_garlic])


def tm1_stake_heart(state, t, d):
    if len(state.staked_dracula[d]) < 2 and state.loc[t] == 'r':
        if state.loc['r'] == state.loc[d]:
            if state.found[d] is True:
                if state.decapitated[d] is True:
                    return [('a_stake_decap_d', t, d)]
                return [('a_decap_d', d), ('a_stake_decap_d', t, d)]
            return [('a_localize', d), ('a_decap_d', d), ('a_stake_decap_d', t, d)]
        return [('move_task', state.loc[d]), ('stake_heart_task', t, d)]
    return []


def tm2_stake_heart(state, t, d):
    if len(state.staked_dracula[d]) < 2 and state.loc[t] == 'r':
        if state.loc['r'] == state.loc[d]:
            if state.found[d] is True:
                return [('a_stake_norm_d', t, d)]
            return [('a_localize', d), ('a_stake_norm_d', t, d)]
        return [('move_task', state.loc[d]), ('stake_heart_task', t, d)]
    return []


def tm3_stake_heart(*_):
    return []   # skip the task


methods.declare_task_methods('stake_heart_task', [tm1_stake_heart, tm2_stake_heart, tm3_stake_heart])


def tm1_surface(state, s_):
    if state.surfaced['r'] is False:
        if state.loc['r'] == state.loc[s_]:
            if state.found[s_] is True:
                return [('a_surface', 'cm1', s_)]
            return [('a_localize', s_), ('a_surface', 'cm1', s_)]
        return [('move_task', state.loc[s_]), ('surface_task', s_)]
    return []


def tm2_surface(state, s_):
    if state.surfaced['r'] is False:
        if state.loc['r'] == state.loc[s_]:
            if state.found[s_] is True:
                return [('a_surface', 'cm2', s_)]
            return [('a_localize', s_), ('a_surface', 'cm2', s_)]
        return [('move_task', state.loc[s_]), ('surface_task', s_)]
    return []


def tm3_surface(*_):
    return []   # skip the task


methods.declare_task_methods('surface_task', [tm1_surface, tm2_surface, tm3_surface])


def tm1_pinger(state):
    tasks = []
    if state.found['ap1'] is False:
        tasks.append(('a_localize_ap', 'ap1'))
    if state.found['ap2'] is False:
        tasks.append(('a_localize_ap', 'ap2'))
    return tasks


methods.declare_task_methods('pinger_task', [tm1_pinger])


def _common_tasks(state, task_list, locs):
    if state.loc['g'] == locs and state.crossed_gate['g'] is False:
        task_list.append(('cross_gate_task', 'g'))

    if state.loc['gm1'] == locs:
        task_list.append(('pick_task', 'gm1'))
    if state.loc['gm2'] == locs:
        task_list.append(('pick_task', 'gm2'))
    if state.loc['c1'] == locs:
        if state.loc['gm1'] != 'c1':
            task_list.append(('drop_garlic_task', 'gm1', 'c1'))
        if state.loc['gm2'] != 'c1':
            task_list.append(('drop_garlic_task', 'gm2', 'c1'))

    if state.loc['cm1'] == locs:
        task_list.append(('pick_task', 'cm1'))
    if state.loc['cm2'] == locs:
        task_list.append(('pick_task', 'cm2'))

    if state.loc['v1'] == locs and state.vampire_touched['v1'] is False:
        task_list.append(('slay_vampire_task', 'v1'))
    if state.loc['v2'] == locs and state.vampire_touched['v2'] is False:
        task_list.append(('slay_vampire_task', 'v2'))

    if state.loc['d1'] == locs:
        if state.loc['t1'] == 'r':
            task_list.append(('stake_heart_task', 't1', 'd1'))
        if state.loc['t2'] == 'r':
            task_list.append(('stake_heart_task', 't2', 'd1'))

    if state.loc['gp1'] == locs and state.traversed_path['gp1'] is False:
        task_list.append(('trace_path_task', 'gp1'))
    if state.loc['gp2'] == locs and state.traversed_path['gp2'] is False:
        task_list.append(('trace_path_task', 'gp2'))

    if state.loc['s1'] == locs and state.surfaced['r'] is False:
        task_list.append(('surface_task', 's1'))


def tm1_main(state, loc_list):
    task_list = []
    for locs in loc_list:
        task_list.append(('move_task', locs))
        _common_tasks(state, task_list, locs)

    return task_list


methods.declare_task_methods('main_task', [tm1_main])


# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    raise NotImplementedError("Test run / Demo routine for Robosub Mod Methods isn't implemented.")

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
Organization: University of Maryland at College Park
"""
