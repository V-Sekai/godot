#!/usr/bin/env python
"""
File Description: Rescue problem file. Initial state and task list for the planning problem is defined here.
"""

from ipyhop import State

init_state = State('init_state')

rigid_relations = dict()
rigid_relations['obstacles'] = set()  # This should be a set of tuples
rigid_relations['wheeled_robots'] = ('r1', 'w1')
rigid_relations['drones'] = ('a1',)
rigid_relations['large_robots'] = ()

init_state.rigid = rigid_relations

init_state.loc = {'r1': (1, 1), 'w1': (5, 5), 'p1': (2, 2), 'a1': (2, 1)}
init_state.robot_type = {'r1': 'wheeled', 'w1': 'wheeled', 'a1': 'uav'}

init_state.has_medicine = {'a1': 0, 'w1': 0, 'r1': 0}
init_state.status = {'r1': 'free', 'w1': 'free', 'a1': 'unk', 'p1': 'unk', init_state.loc['p1']: 'unk'}
init_state.altitude = {'a1': 'high'}
init_state.current_image = {'a1': None}
init_state.real_status = {'r1': 'free', 'w1': 'free', 'a1': 'free', 'p1': 'injured', init_state.loc['p1']: 'clear'}
init_state.real_person = {init_state.loc['p1']: 'p1'}

init_state.new_robot = {1: None}
init_state.weather = {init_state.loc['p1']: 'clear'}

task_list_1 = [('move_task', 'r1', (5, 5)), ]
task_list_2 = [('survey_task', 'a1', (2, 2)), ]
