#!/usr/bin/env python
"""
File Description: File used for definition of Methods Class.
"""

# ******************************************    Libraries to be imported    ****************************************** #
from __future__ import print_function, division
from typing import List, Callable, Union, Any


# ******************************************    Class Declaration Start     ****************************************** #
class Methods(object):
    """
    A class to store all the methods defined for all the tasks in a planning domain.

    *   methods = Methods() tells IPyHOP to create an empty methods container.
        To add tasks and associated task methods into it, you should use
        methods.declare_task_methods(task_name, method_list).
        To add tasks and associated goal methods into it, you should use
        methods.declare_goal_methods(goal_name, method_list).
        To add tasks and associated multigoal methods into it, you should use
        methods.declare_multigoal_methods(goal_tag, method_list).

    All the task methods are stored in a dictionary member variable named method_dict with the following structure:
        {task_name_1: [method_func_a, ...], task_name_2: [method_func_x, ...]...}
    All the goal methods are stored in a dictionary member variable named method_dict with the following structure:
        {gaol_name_1: [method_func_a, ...], goal_name_2: [method_func_x, ...]...}
    All the multigoal methods are stored in a dictionary member variable named method_dict with the following structure:
        {multigaol_tag_1: [method_func_a, ...], multigoal_tag_2: [method_func_x, ...]..., split: tm_split_multigaol}
    """

    def __init__(self):
        self.task_method_dict = dict()
        self.goal_method_dict = dict()
        self.multigoal_method_dict = {None: []}

    # ******************************        Class Method Declaration        ****************************************** #
    def __str__(self):
        m_str = '\n\r{:<30}{}'.format('TASK:', 'METHODS:')
        for task in self.task_method_dict:
            m_str += '\n{:<30}'.format(task) + ', '.join([f.__name__ for f in self.task_method_dict[task]])
        m_str += '\n\n\r{:<30}{}'.format('GOAL:', 'METHODS:')
        for goal in self.goal_method_dict:
            m_str += '\n{:<30}'.format(goal) + ', '.join([f.__name__ for f in self.goal_method_dict[goal]])
        m_str += '\n\n\r{:<30}{}'.format('MULTIGOAL:', 'METHODS:')
        for mgoal in self.multigoal_method_dict:
            m_str += '\n{:<30}'.format(str(mgoal)) + ', '.join([f.__name__ for f in self.multigoal_method_dict[mgoal]])
        return m_str

    # ******************************        Class Method Declaration        ****************************************** #
    def __repr__(self):
        return self.__str__()

    _method_list_type = List[Callable[[Any], Union[List[tuple], bool]]]

    # ******************************        Class Method Declaration        ****************************************** #
    def declare_task_methods(self, task_name: str, method_list: _method_list_type):
        """
        declare_task_methods('foo', [m1, m2, ..., mk]) tells IPyHOP that [m1, m2, ..., mk] is a complete list of
        all the methods for a tasks having task-name 'foo'. This supersedes any previous call to
        declare_task_methods('foo', [...]).

        :param task_name: Name of the task.
        :param method_list: List of method functions.
        """
        assert type(task_name) == str, "task_name must be a string."
        assert type(method_list) == list, "method_list must be a list."
        for method in method_list:
            assert callable(method), "method in method_list should be callable."
        self.task_method_dict.update({task_name: method_list})

    # ******************************        Class Method Declaration        ****************************************** #
    def declare_goal_methods(self, goal_name: str, method_list: _method_list_type):
        """
        declare_goal_methods('foo', [m1, m2, ..., mk]) tells IPyHOP that [m1, m2, ..., mk] is a complete list of
        all the methods for a gaol having gaol-name 'foo'. This supersedes any previous call to
        declare_goal_methods('foo', [...]).

        :param goal_name: Name of the gal.
        :param method_list: List of method functions.
        """
        assert type(goal_name) == str, "goal must be a string."
        assert type(method_list) == list, "method_list must be a list."
        for method in method_list:
            assert callable(method), "method in method_list should be callable."
        self.goal_method_dict.update({goal_name: method_list})

    # ******************************        Class Method Declaration        ****************************************** #
    def declare_multigoal_methods(self, multigoal_tag: Union[None, str], method_list: _method_list_type):
        """
        declare_multigoal_methods('foo', [m1, m2, ..., mk]) tells IPyHOP that [m1, m2, ..., mk] is a complete list of
        all the methods for a multigoal having multigoal-tag 'foo'. This supersedes any previous call to
        declare_multigoal_methods('foo', [...]).

        :param multigoal_tag: Optional tag for the multigoal.
        :param method_list: List of method functions.
        """
        assert type(multigoal_tag) == str or type(multigoal_tag) == type(None), "multigoal_tag must be a string or None"
        assert type(method_list) == list, "method_list must be a list."
        for method in method_list:
            assert callable(method), "method in method_list should be callable."
        self.multigoal_method_dict.update({multigoal_tag: method_list})


# ******************************************    Class Declaration End       ****************************************** #

# **************************************        Function Declaration        ****************************************** #
def _goals_not_achieved(state, multigoal):
    """
    _goals_not_achieved takes two arguments: a state s and a multigoal g.
    It returns a dictionary of the goals in g that aren't true in s.
    For example, suppose
        s.loc['c0'] = 'room0', g.loc['c0'] = 'room0',
        s.loc['c1'] = 'room1', g.loc['c1'] = 'room3',
        s.loc['c2'] = 'room2', g.loc['c2'] = 'room4'.
    Then _goals_not_achieved(s, g) will return
        {'loc': {'c1': 'room3', 'c2': 'room4'}}

    :param state:
    :param multigoal:
    :return:
    """
    unachieved = {}
    for name in vars(multigoal):
        if name == '__name__' or name == 'goal_tag':
            continue
        for arg in vars(multigoal).get(name):
            val = vars(multigoal).get(name).get(arg)
            if val != vars(state).get(name).get(arg):
                # want arg_value_pairs.name[arg] = val
                if not unachieved.get(name):
                    unachieved.update({name: {}})
                unachieved.get(name).update({arg: val})
    return unachieved


def mgm_split_multigoal(state, multigoal):
    """
    mgm_split_multigoal takes two arguments: the current state and a multigoal to achieve. mgm_split_multigoal
    separates the multigoal into a collection of individual goals. Then it repeatedly iterates through the list of
    individual goals, trying to achieve each goal that isn't already true. The purpose of the repetition is to overcome
    deleted-condition interactions (in which accomplishing a goal has a side-effect of falsifying another goal that was
    previously true).

    More specifically, if one or more of the individual goals is not true, then mgm_split_multigoal returns a goal list
    [g_1, ..., g_n, G], where g_1, ..., g_n are the goals that aren't true, and G is the multigoal. The list tells
    the planner to achieve g_1, ..., g_n sequentially, then invoke mgm_split_multigoal again to re-achieve any goals
    that have become false.

    The main problem with mgm_split_multigoal is that it isn't smart about choosing the order in which to achieve
    g1, ..., gn. Some orderings may work better than others. Thus it might be desirable to modify the method to use a
    heuristic function to choose a good order.

    :param state:
    :param multigoal:
    :return:
    """
    goal_dict = _goals_not_achieved(state, multigoal)
    goal_list = []
    for state_var_name in goal_dict:
        for arg in goal_dict[state_var_name]:
            val = goal_dict[state_var_name][arg]
            goal_list.append((state_var_name, arg, val))
    if goal_list:
        # achieve goals, then check whether they're all simultaneously true
        return goal_list + [multigoal]
    return goal_list


# ******************************************    Demo / Test Routine         ****************************************** #
if __name__ == '__main__':
    def test_method_1(): return False
    def test_method_2(): return False
    def test_method_3(): return False


    print("Test instantiation of Methods class ...")
    methods = Methods()
    methods.declare_task_methods('test_task_1', [test_method_1])
    methods.declare_task_methods('test_task_2', [test_method_2, test_method_3])

    methods.declare_goal_methods('goal_1', [test_method_1, test_method_2])

    methods.declare_multigoal_methods(None, [test_method_1, test_method_2, test_method_3])
    methods.declare_multigoal_methods('mg_1', [test_method_1, test_method_3])

    print(methods)

"""
Author(s): Yash Bansod
Repository: https://github.com/YashBansod/IPyHOP
"""
