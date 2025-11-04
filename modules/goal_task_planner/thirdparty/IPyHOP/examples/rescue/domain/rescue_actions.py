#!/usr/bin/env python
"""
File Description: Robosub actions file. All the actions for Robosub planning domain are defined here.
"""
# ******************************************    Libraries to be imported    ****************************************** #
from ipyhop import Actions


# ******************************************        Action Definitions      ****************************************** #
actions = Actions()


def a_free_robot(state, r):     # Added a command.
    state.status[r] = 'free'
    return state


def a_die_update(state, p):
    state.status[p] = 'dead'
    state.real_status[p] = 'dead'
    return None


def a_move_euclidean(state, r, l, l_, dist):        # TODO: Ask Sunandita why dist is used.
    (x1, y1) = l
    (x2, y2) = l_

    x_low = min(x1, x2)
    x_high = max(x1, x2)
    y_low = min(y1, y2)
    y_high = max(y1, y2)

    for obs in state.rigid['obstacles']:
        (ox, oy) = obs
        if x_low <= ox <= x_high and y_low <= oy <= y_high:
            if ox == x1 or x2 == x1:    # TODO: Ask Sunandita about this condition.
                return None
            elif abs((oy - y1)/(ox - x1) - (y2 - y1)/(x2 - x1)) <= 0.0001:  # Same slope
                return None

    if l == l_:
        return state

    elif state.loc[r] == l:
        state.loc[r] = l_
        return state

    return None


def a_move_manhattan(state, r, l, l_, dist):        # TODO: Ask Sunandita why dist is used.
    (x1, y1) = l
    (x2, y2) = l_

    x_low = min(x1, x2)
    x_high = max(x1, x2)
    y_low = min(y1, y2)
    y_high = max(y1, y2)

    for obs in state.rigid['obstacles']:
        (ox, oy) = obs
        # TODO: Why the inconsistent checking method for position difference.
        if abs(oy - y1) <= 0.0001 and x_low <= ox <= x_high:    # TODO: Ask Sunandita about this condition.
            return None
        elif abs(ox - x2) <= 0.0001 and y_low <= oy <= y_high:  # TODO: Ask Sunandita about this condition.
            return None

    if l == l_:
        return state

    elif state.loc[r] == l:
        state.loc[r] = l_
        return state

    return None


def a_move_curved(state, r, l, l_, dist):       # TODO: Ask Sunandita why dist is used.
    (x1, y1) = l
    (x2, y2) = l_
    centre_x = (x1 + x2) / 2
    centre_y = (y1 + y2) / 2

    for obs in state.rigid['obstacles']:
        (ox, oy) = obs
        r2 = (x2 - centre_x) ** 2 + (y2 - centre_y) ** 2
        ro = (ox - centre_x) ** 2 + (oy - centre_y) ** 2
        if abs(r2 - ro) <= 0.0001:      # TODO: Ask Sunandita about this condition.
            return None

    if l == l_:
        return state

    elif state.loc[r] == l:
        state.loc[r] = l_
        return state

    return None


def a_move_fly(state, r, l, l_):
    if l == l_:
        return state

    elif state.loc[r] == l:
        state.loc[r] = l_
        return state

    return None


def a_move_alt_fly(state, r, l, l_):
    if l == l_:
        return state

    elif state.loc[r] == l:
        state.loc[r] = l_
        return state

    return None


def a_inspect_person(state, r, p):      # TODO: Should check if robot is available to do that operation.
    state.status[p] = state.real_status[p]
    return state


def a_support_person(state, r, p):  # TODO: Should check if robot is available to do that operation.
    if state.status[p] != 'dead':
        state.status[p] = 'OK'
        state.real_status[p] = 'OK'
        return state
    return None


def a_support_person_2(state, r, p):  # TODO: Should check if robot is available to do that operation.
    if state.status[p] != 'dead':
        state.status[p] = 'OK'
        state.real_status[p] = 'OK'
        return state
    return None


def a_inspect_location(state, r, l):    # TODO: Should check if robot is available to do that operation.
    state.status[l] = state.real_status[l]
    return state


def a_clear_location(state, r, l):      # TODO: Should check if robot is available to do that operation.
    state.status[l] = 'clear'
    state.real_status[l] = 'clear'
    return state


def a_clear_location_2(state, r, l):      # TODO: Should check if robot is available to do that operation.
    state.status[l] = 'clear'
    state.real_status[l] = 'clear'
    return state


def a_replenish_supplies(state, r):
    if state.loc[r] == (1, 1):
        state.has_medicine[r] = 2       # TODO: Ask Sunandita the use of this state variable.
        return state
    return None


def a_transfer(state, r, r_):
    if state.loc[r] == state.loc[r_]:
        if state.has_medicine[r] > 0:
            state.has_medicine[r_] += 1
            state.has_medicine[r] -= 1
            return state
    return None


# NOTE: This is not a callable action.
def sense_image(state, r, camera, l):
    img = {'loc': None, 'person': None}
    visibility = False
    if state.weather[l] == 'rainy':
        if camera == 'bottom_camera' and state.altitude[r] == 'low':
            visibility = True
    elif state.weather[l] == 'foggy':
        if camera == 'front_camera' and state.altitude[r] == 'low':
            visibility = True
    elif state.weather[l] == 'dusts_storm':
        if state.altitude[r] == 'low':
            visibility = True
    elif state.weather[l] == 'clear':
        visibility = True

    if visibility:
        img['loc'] = l
        img['person'] = state.real_person[l]

    return img


def a_capture_image(state, r, camera, l):       # TODO: Should check if robot is available to do that operation.
    img = sense_image(state, r, camera, l)
    state.current_image[r] = img
    return state


def a_change_altitude(state, r, new_alt):
    if state.altitude[r] != new_alt:
        state.altitude[r] = new_alt
    return state


def a_check_real(state, l):
    p = state.real_person[l]

    if p is not None:
        if state.real_status[p] == 'injured' or state.real_status[p] == 'dead' or state.real_status[l] == 'has_debri':
            return a_die_update(state, p)
        else:
            return state
    else:
        return state


def a_engage_robot(state, r):
    state.status[r] = 'busy'
    state.new_robot[1] = r
    return state


def a_force_engage_robot(state):
    state.new_robot[1] = state.rigid['wheeled_robots'][0]
    state.status[state.rigid['wheeled_robots'][0]] = 'busy'
    return state


actions.declare_actions([a_free_robot, a_die_update, a_move_euclidean, a_move_manhattan, a_move_curved, a_move_fly,
                         a_move_alt_fly, a_inspect_person, a_support_person, a_support_person_2, a_inspect_location,
                         a_clear_location, a_clear_location_2, a_replenish_supplies, a_transfer, a_capture_image,
                         a_change_altitude, a_check_real, a_engage_robot, a_force_engage_robot])

# NOTE: Probabilistic actions shouldn't be called if there is no need of executing them. This could be done in methods.


action_probability = {
    'a_free_robot': [1, 0],
    'a_die_update': [1, 0],
    'a_move_euclidean': [0.8, 0.2],
    'a_move_manhattan': [0.9, 0.1],
    'a_move_curved': [0.9, 0.1],
    'a_move_fly': [0.9, 0.1],
    'a_move_alt_fly': [0.95, 0.05],
    'a_inspect_person': [1, 0],
    'a_support_person': [0.85, 0.15],
    'a_support_person_2': [0.95, 0.05],
    'a_inspect_location': [1, 0],
    'a_clear_location': [0.85, 0.15],
    'a_clear_location_2': [0.95, 0.05],
    'a_replenish_supplies': [0.95, 0.05],
    'a_transfer': [0.8, 0.2],
    'a_capture_image': [1, 0],
    'a_change_altitude': [1, 0],
    'a_check_real': [1, 0],
    'a_engage_robot': [0.9, 0.1],
    'a_force_engage_robot': [0.99, 0.01],
}

action_cost = {
    'a_free_robot': 1,
    'a_die_update': 1,
    'a_move_euclidean': 4,
    'a_move_manhattan': 6,
    'a_move_curved': 7,
    'a_move_fly': 10,
    'a_move_alt_fly': 15,
    'a_inspect_person': 1,
    'a_support_person': 5,
    'a_support_person_2': 7,
    'a_inspect_location': 1,
    'a_clear_location': 6,
    'a_clear_location_2': 8,
    'a_replenish_supplies': 5,
    'a_transfer': 3,
    'a_capture_image': 1,
    'a_change_altitude': 1,
    'a_check_real': 1,
    'a_engage_robot': 2,
    'a_force_engage_robot': 4,
}

actions.declare_action_models(action_probability, action_cost)




