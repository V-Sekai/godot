"""
File Description: Rescue problem file. Initial state and task list for the planning problem is defined here.
"""
# ******************************************    Libraries to be imported    ****************************************** #
from random import sample, seed
from ipyhop import State
import itertools

# ******************************************        Problem Definition      ****************************************** #
class StateSampler(object):
    def __init__(self, seed_val=None):
        self.options = dict()
        self.options['locs'] = list(itertools.product([i for i in range(10)], repeat=2))
        self.options['locs'].remove((1, 1))
        seed(seed_val)

    def sample(self, state_name="init_state"):
        state = State(state_name)
        rigid_relations = dict()
        rigid_relations['wheeled_robots'] = ('r1', 'w1')
        rigid_relations['drones'] = ('a1',)
        rigid_relations['base_loc'] = (1, 1)
        rigid_relations['other_loc'] = sample(self.options['locs'], 5)
        rigid_relations['obstacles'] = []

        eu_obs_1_ind = sample([i for i in range(5)], 3)
        b = rigid_relations['base_loc']
        for ind in eu_obs_1_ind:
            a = rigid_relations['other_loc'][ind]
            rigid_relations['obstacles'].append(((a[0] + b[0])/2, (a[1] + b[1])/2))

        loc_pair_list = list(itertools.combinations(rigid_relations['other_loc'], 2))
        loc_pair_list = sample(loc_pair_list, 2)

        for lp in loc_pair_list:
            rigid_relations['obstacles'].append(((lp[0][0] + lp[1][0]) / 2, (lp[0][1] + lp[1][1]) / 2))

        state.rigid = rigid_relations

        state.loc = {'r1': rigid_relations['base_loc'], 'w1': rigid_relations['base_loc'],
                     'a1': rigid_relations['base_loc'], 'p1': rigid_relations['other_loc'][0],
                     'p2': rigid_relations['other_loc'][1], 'p3': rigid_relations['other_loc'][2]}

        state.robot_type = {'r1': 'wheeled', 'w1': 'wheeled', 'a1': 'uav'}

        state.has_medicine = {'a1': 0, 'w1': 2, 'r1': 2}

        state.status = {'r1': 'free', 'w1': 'free', 'a1': 'free', 'p1': 'unk', state.loc['p1']: 'unk',
                        'p2': 'unk', state.loc['p2']: 'unk', 'p3': 'unk', state.loc['p3']: 'unk',
                        rigid_relations['other_loc'][3]: 'unk', rigid_relations['other_loc'][4]: 'unk'}

        state.altitude = {'a1': 'high'}
        state.current_image = {'a1': None}

        state.real_status = {'r1': 'free', 'w1': 'free', 'a1': 'free', 'p1': 'injured', state.loc['p1']: 'clear',
                        'p2': 'injured', state.loc['p2']: 'clear', 'p3': 'injured', state.loc['p3']: 'clear',
                        rigid_relations['other_loc'][3]: 'debri', rigid_relations['other_loc'][4]: 'debri'}

        state.real_person = {rigid_relations['other_loc'][0]: 'p2', rigid_relations['other_loc'][1]: 'p2',
                             rigid_relations['other_loc'][2]: 'p3', rigid_relations['other_loc'][3]: None,
                             rigid_relations['other_loc'][4]: None}

        state.new_robot = {1: None}
        state.weather = {rigid_relations['other_loc'][0]: 'clear', rigid_relations['other_loc'][1]: 'clear',
                         rigid_relations['other_loc'][2]: 'clear', rigid_relations['other_loc'][3]: 'clear',
                         rigid_relations['other_loc'][4]: 'clear'}

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