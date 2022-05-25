from typing import Dict, List, Union

from numpy.lib import stride_tricks

import cloudpickle
import numpy as np
import torch


class CloudpickleWrapper(object):
    """A cloudpickle wrapper used in :class:`~tianshou.env.SubprocVectorEnv`"""

    def __init__(self, data):
        self.data = data

    def __getstate__(self):
        return cloudpickle.dumps(self.data)

    def __setstate__(self, data):
        self.data = cloudpickle.loads(data)


class BaseRewardLogger(object):
    """Class for keeping track of rewards accumulated in an environment.
    """

    def __init__(self) -> None:
        self.value = 0.0
        self.num = 0

    def add(self, x: Union[List, np.ndarray, torch.Tensor]) -> None:
        """Add a scalar into :class:`MovAvg`. You can add ``torch.Tensor`` with
        only one element, a python scalar, or a list of python scalar.
        """
        if isinstance(x, torch.Tensor):
            x = to_numpy(x)
        if isinstance(x, list) or isinstance(x, np.ndarray):
            self.value += np.mean(x)
            self.num += 1

    def log(self) -> Dict:
        """Get the average."""
        if self.num == 0:
            return {'rew': 0}
        return {'rew': self.value / self.num}


class SimpleTagRewardLogger(BaseRewardLogger):
    """Class for keeping track of rewards accumulated in an environment.
    """

    def __init__(self, num_chasers: int) -> None:
        self.num_chasers = num_chasers
        self.num = 0.0
        self.chaser_value = 0.0
        self.runner_value = 0.0

    def add(self, x: Union[List, np.ndarray, torch.Tensor]) -> None:
        """Add a scalar into :class:`MovAvg`. You can add ``torch.Tensor`` with
        only one element, a python scalar, or a list of python scalar.
        """
        if isinstance(x, torch.Tensor):
            assert self.ndim == x.size(0)
            x = to_numpy(x)
        if isinstance(x, list) or isinstance(x, np.ndarray):
            self.num += 1
            for i, v in enumerate(x):
                if i < self.num_chasers:
                    self.chaser_value += v
                else:
                    self.runner_value += v

    def log(self) -> Dict:
        """Get the average."""
        return {'rew/adv_rew': self.chaser_value / self.num,
                'rew/agt_rew': self.runner_value / self.num,
                'rew': (self.chaser_value + self.runner_value) / self.num}


# class SimpleTagBenchmarkLogger(object):
#     """Class for logging how many collissions happen.
#     """

#     def __init__(self, num_envs:int, num_chasers: int) -> None:
#         self.num_chasers = num_chasers
#         self.curr_collisions = np.zeros(num_envs)
#         self.total_collisions = 0
#         self.num_episodes = 0

#     def add(self, info: List) -> None:
#         for env_idx, elem in enumerate(info):
#             collisions = elem['n']
#             total_collisions = np.sum(collisions[:self.num_chasers])
#             self.curr_collisions[env_idx] += total_collisions

#     def log(self) -> None:
#         if self.num_episodes == 0:
#             return {'bench/collisions': 0}
#         return {'bench/collisions': self.total_collisions / self.num_episodes}

#     def episode_end(self, env_idx: int) -> None:
#         self.num_episodes += 1
#         self.total_collisions += self.curr_collisions[env_idx]
#         self.curr_collisions[env_idx] = 0.0

#     def reset(self):
#         self.curr_collisions[:] = 0

class SimpleTagBenchmarkLogger(object):
    """Class for logging how many collissions happen.
    """

    def __init__(self, num_envs:int, num_chasers: int, max_world_steps=25) -> None:
        self.num_chasers = num_chasers
        self.curr_collisions = np.zeros(num_envs)
        self.curr_cr_dists = np.zeros(num_envs)
        self.adv_ex_rews = np.zeros(num_envs)
        self.agt_ex_rews = np.zeros(num_envs)
        self.agt_pure_rews = np.zeros(num_envs)
        self.agt_bound_rews = np.zeros(num_envs)
        self.total_collisions = 0
        self.total_cr_dist = 0
        self.total_adv_ex_rews = 0
        self.total_agt_ex_rews = 0
        self.total_agt_pure_rews = 0
        self.total_agt_bound_rews = 0
        self.num_episodes = 0
        self.end_steps = max_world_steps

    def add(self, info: List) -> None:
        for env_idx, elem in enumerate(info):
            bench_data =  elem['n']
            collisions = [bench_tuple[0] for bench_tuple in bench_data]
            dists = [bench_tuple[1] for bench_tuple in bench_data]
            ex_rews = [bench_tuple[2] for bench_tuple in bench_data]
            pure_rews = [bench_tuple[3] for bench_tuple in bench_data]
            bound_rews = [bench_tuple[4] for bench_tuple in bench_data]
            total_collisions = np.sum(collisions[:self.num_chasers])
            total_dist = np.sum(dists[self.num_chasers:])
            adv_ex_rews = np.sum(ex_rews[:self.num_chasers])
            agt_ex_rews = np.sum(ex_rews[self.num_chasers:])
            agt_pure_rews = np.sum(pure_rews[self.num_chasers:])
            agt_bound_rews = np.sum(bound_rews[self.num_chasers:])
            self.curr_collisions[env_idx] += total_collisions
            self.curr_cr_dists[env_idx] += total_dist
            self.adv_ex_rews[env_idx] += adv_ex_rews
            self.agt_ex_rews[env_idx] += agt_ex_rews
            self.agt_bound_rews[env_idx] += agt_bound_rews
            self.agt_pure_rews[env_idx] += agt_pure_rews
    
    def log(self) -> None:
        if self.num_episodes == 0:
            return {'bench/collisions': 0, 'bench/cr_dist': 0, 'bench/adv_step_rew': 0,
            'bench/agt_step_rew': 0, 'bench/agt_pure_rew': 0, 'bench/agt_bound_rew': 0}
        return {'bench/collisions': self.total_collisions / self.num_episodes, \
            'bench/cr_dist': self.total_cr_dist / self.num_episodes,
            'bench/adv_step_rew': self.total_adv_ex_rews / self.num_episodes,
            'bench/agt_step_rew': self.total_agt_ex_rews / self.num_episodes,
            'bench/agt_pure_rew': self, 'bench/agt_bound_rew': 0}

    def episode_end(self, env_idx: int) -> None:
        self.num_episodes += 1
        self.total_collisions += self.curr_collisions[env_idx] / self.end_steps
        self.total_cr_dist += self.curr_cr_dists[env_idx] / self.end_steps
        self.total_adv_ex_rews += self.adv_ex_rews[env_idx] / self.end_steps
        self.total_agt_ex_rews += self.agt_ex_rews[env_idx] / self.end_steps
        self.total_agt_pure_rews += self.agt_pure_rews[env_idx] / self.end_steps
        self.total_agt_bound_rews += self.agt_bound_rews[env_idx] /self.end_steps
        self.curr_collisions[env_idx] = 0.0
        self.curr_cr_dists[env_idx] = 0.0
        self.adv_ex_rews[env_idx] = 0.0
        self.agt_ex_rews[env_idx] = 0.0
        self.agt_bound_rews[env_idx] = 0.0
        self.agt_pure_rews[env_idx] = 0.0

    def reset(self):
        self.curr_collisions[:] = 0
        self.curr_cr_dists[:] = 0
        self.adv_ex_rews[:] = 0
        self.agt_ex_rews[:] = 0
        self.agt_pure_rews[:] = 0
        self.agt_bound_rews[:] = 0

class SimpleAdversaryBenchmarkLogger(object):
    """Class for logging how many collissions happen.
    """

    def __init__(self, num_envs:int, num_adv: int, max_world_steps=25) -> None:
        self.num_adv = num_adv
        self.adv_data = np.zeros(num_envs)
        self.agt_data = np.zeros(num_envs)
        self.adv_occupied = np.zeros(num_envs)
        self.agt_occupied = np.zeros(num_envs)
        self.adv_ex_rews = np.zeros(num_envs)
        self.agt_ex_rews = np.zeros(num_envs)
        self.agt_pure_rews = np.zeros(num_envs)
        self.agt_bound_rews = np.zeros(num_envs)
        self.total_adv_data = 0
        self.total_agt_data = 0
        self.total_adv_occupied = 0
        self.total_agt_occupied = 0
        self.total_adv_ex_rews = 0
        self.total_agt_ex_rews = 0
        self.total_agt_pure_rews = 0
        self.total_agt_bound_rews = 0
        self.num_episodes = 0
        self.end_steps = max_world_steps

    def add(self, info: List) -> None:
        for env_idx, elem in enumerate(info):
            bench_data =  elem['n']
            ex_rews = [bench_tuple[0] for bench_tuple in bench_data]
            occupied = [bench_tuple[1] for bench_tuple in bench_data]
            dists = [bench_tuple[2] for bench_tuple in bench_data]
            pure_rews = [bench_tuple[3] for bench_tuple in bench_data]
            bound_rews = [bench_tuple[4] for bench_tuple in bench_data]
            sum_adv_data = np.sum(dists[:self.num_adv])
            sum_agt_data = np.sum(dists[self.num_adv:])
            sum_adv_occupied = np.sum(occupied[:self.num_adv])
            sum_agt_occupied = np.sum(occupied[self.num_adv:])
            adv_ex_rews = np.sum(ex_rews[:self.num_adv])
            agt_ex_rews = np.sum(ex_rews[self.num_adv:])
            agt_pure_rews = np.sum(pure_rews[self.num_adv:])
            agt_bound_rews = np.sum(bound_rews[self.num_adv:])
            self.adv_data[env_idx] += sum_adv_data
            self.agt_data[env_idx] += sum_agt_data
            self.adv_occupied[env_idx] += sum_adv_occupied
            self.agt_occupied[env_idx] += sum_agt_occupied
            self.adv_ex_rews[env_idx] += adv_ex_rews
            self.agt_ex_rews[env_idx] += agt_ex_rews
            self.agt_pure_rews[env_idx] += agt_pure_rews
            self.agt_bound_rews[env_idx] += agt_bound_rews

    def log(self) -> None:
        if self.num_episodes == 0:
            return {'bench/adv_occupied': 0, 'bench/agt_occupied': 0, 'bench/adv_dist': 0, 
            'bench/agt_dist': 0, 'bench/adv_step_rew': 0, 'bench/agt_step_rew': 0, 
            'bench/agt_pure_rew': 0, 'bench/agt_bound_rew': 0}
        return {'bench/adv_occupied': self.total_adv_occupied / self.num_episodes, 
            'bench/agt_occupied': self.total_agt_occupied / self.num_episodes,
            'bench/adv_dist': self.total_adv_data / self.num_episodes, 
            'bench/agt_dist': self.total_agt_data / self.num_episodes,
            'bench/adv_step_rew': self.total_adv_ex_rews / self.num_episodes,
            'bench/agt_step_rew': self.total_agt_ex_rews / self.num_episodes,
            'bench/agt_pure_rew': self.total_agt_pure_rews / self.num_episodes, 
            'bench/agt_bound_rew': self.total_agt_bound_rews / self.num_episodes}

    def episode_end(self, env_idx: int) -> None:
        self.num_episodes += 1
        self.total_adv_data += self.adv_data[env_idx] / self.end_steps
        self.total_agt_data += self.agt_data[env_idx] / self.end_steps
        self.total_adv_occupied += self.adv_occupied[env_idx] / self.end_steps
        self.total_agt_occupied += self.agt_occupied[env_idx] / self.end_steps
        self.total_adv_ex_rews += self.adv_ex_rews[env_idx] / self.end_steps
        self.total_agt_ex_rews += self.agt_ex_rews[env_idx] / self.end_steps
        self.total_agt_pure_rews += self.agt_pure_rews[env_idx] / self.end_steps
        self.total_agt_bound_rews += self.agt_bound_rews[env_idx] / self.end_steps
        self.adv_data[env_idx] = 0.0
        self.agt_data[env_idx] = 0.0
        self.adv_occupied[env_idx] = 0.0
        self.agt_occupied[env_idx] = 0.0
        self.adv_ex_rews[env_idx] = 0.0
        self.agt_ex_rews[env_idx] = 0.0
        self.agt_pure_rews[env_idx] = 0.0
        self.agt_bound_rews[env_idx] = 0.0

    def reset(self):
        self.adv_data[:] = 0
        self.agt_data[:] = 0
        self.adv_occupied[:] = 0
        self.agt_occupied[:] = 0
        self.adv_ex_rews[:] = 0
        self.agt_ex_rews[:] = 0
        self.agt_pure_rews[:] = 0
        self.agt_bound_rews[:] = 0

class SimpleSpreadBenchmarkLogger(object):
    """Class for logging benchmark data for simple_spread.
    """

    def __init__(self, num_envs:int, max_world_steps=25) -> None:
        self.curr_reward = np.zeros(num_envs)
        self.curr_min_dist = np.zeros(num_envs)
        self.curr_occupied = np.zeros(num_envs)
        # self.curr_collisions = np.zeros(num_envs)
        self.curr_pure_rew = np.zeros(num_envs)
        self.curr_bound_rew = np.zeros(num_envs)
        self.curr_end_steps = np.zeros(num_envs)
        self.reward = 0.0
        self.min_dist = 0.0
        self.occupied = 0.0
        # self.collisions = 0.0
        self.pure_rew = 0.0
        self.bound_rew = 0.0
        self.num_episodes = 0
        self.end_steps = 0
        self.max_steps = max_world_steps

    def add(self, info: List) -> None:
        for env_idx, elem in enumerate(info):
            bench_data = elem['n'][0]
            self.curr_reward[env_idx] += float(bench_data[0]) 
            # self.curr_collisions[env_idx] += float(bench_data[1]) 
            self.curr_min_dist[env_idx] += float(bench_data[1]) 
            self.curr_occupied[env_idx] += float(bench_data[2]) 
            self.curr_pure_rew[env_idx] += float(bench_data[4])
            self.curr_bound_rew[env_idx] += float(bench_data[5])
            end_step = float(bench_data[3])
            if end_step < self.curr_end_steps[env_idx]:
                self.curr_end_steps[env_idx] = end_step

    def log(self) -> None:
        if self.num_episodes == 0:
            return {
                    'bench/min_dist': 0.0,
                    'bench/step_reward': 0.0,
                    'bench/occupied': 0.0, 
                    'bench/end_steps': 0.0,
                    'bench/pure_rew': 0.0,
                    'bench/bound_rew': 0.0}
        return {
                'bench/min_dist': self.min_dist / self.num_episodes,
                'bench/step_reward': self.reward / self.num_episodes,
                'bench/occupied': self.occupied / self.num_episodes,
                'bench/end_steps': self.end_steps / self.num_episodes,
                'bench/pure_rew': self.pure_rew / self.num_episodes,
                'bench/bound_rew': self.bound_rew / self.num_episodes}

    def episode_end(self, env_idx: int) -> None:
        self.num_episodes += 1
        curr_end_steps = self.curr_end_steps[env_idx]
        self.end_steps += curr_end_steps
        self.reward += self.curr_reward[env_idx] / curr_end_steps
        self.min_dist += self.curr_min_dist[env_idx] / curr_end_steps
        self.occupied += self.curr_occupied[env_idx] / curr_end_steps
        self.pure_rew += self.curr_pure_rew[env_idx] / curr_end_steps
        self.bound_rew += self.curr_bound_rew[env_idx] / curr_end_steps
        # self.collisions += self.curr_collisions[env_idx] / curr_end_steps
        self.curr_reward[env_idx] = 0
        self.curr_min_dist[env_idx] = 0
        self.curr_occupied[env_idx] = 0
        self.curr_pure_rew[env_idx] = 0
        self.curr_bound_rew[env_idx] = 0
        # self.curr_collisions[env_idx] = 0
        self.curr_end_steps[env_idx] = self.max_steps

    def reset(self):
        self.curr_reward[:] = 0
        self.curr_min_dist[:] = 0
        self.curr_occupied[:] = 0
        self.curr_pure_rew[:] = 0
        self.curr_bound_rew[:] = 0
        # self.curr_collisions[:] = 0
        self.curr_end_steps[:] = self.max_steps

class SimpleCoopPushBenchmarkLogger(object):
    """Class for logging benchmark data for simple_spread.
    """

    def __init__(self, num_envs:int, max_world_steps=50) -> None:
        self.curr_reward = np.zeros(num_envs)
        self.curr_occupied = np.zeros(num_envs)
        self.curr_dist = np.zeros(num_envs)
        self.curr_collisions = np.zeros(num_envs)
        self.curr_end_steps = np.zeros(num_envs)
        self.reward = 0.0
        self.occupied = 0.0
        self.dist = 0.0
        self.collisions = 0.0
        self.num_episodes = 0
        self.end_steps = 0
        self.max_steps = max_world_steps

    def add(self, info: List) -> None:
        for env_idx, elem in enumerate(info):
            bench_data = elem['n'][0]
            self.curr_reward[env_idx] += float(bench_data[0]) 
            self.curr_collisions[env_idx] += float(bench_data[1]) 
            self.curr_dist[env_idx] += float(bench_data[2]) 
            self.curr_occupied[env_idx] += float(bench_data[3]) 
            end_step = float(bench_data[4])
            if end_step < self.curr_end_steps[env_idx]:
                self.curr_end_steps[env_idx] = end_step

    def log(self) -> None:
        if self.num_episodes == 0:
            return {'bench/ex_rew': 0.0,
                    'bench/collisions': 0.0,
                    'bench/avg_dist': 0.0,
                    'bench/occupied': 0.0,
                    'bench/end_steps': 0.0}
        return {'bench/ex_rew': self.reward / self.num_episodes,
                'bench/collisions': self.collisions / self.num_episodes,
                'bench/avg_dist': self.dist / self.num_episodes,
                'bench/occupied': self.occupied / self.num_episodes,
                'bench/end_steps': self.end_steps / self.num_episodes}

    def episode_end(self, env_idx: int) -> None:
        self.num_episodes += 1
        curr_end_steps = self.curr_end_steps[env_idx]
        self.end_steps += curr_end_steps
        self.reward += self.curr_reward[env_idx] / curr_end_steps
        self.occupied += self.curr_occupied[env_idx] 
        self.dist += self.curr_dist[env_idx] / curr_end_steps
        self.collisions += self.curr_collisions[env_idx] / curr_end_steps
        self.curr_reward[env_idx] = 0.
        self.curr_occupied[env_idx] = 0.
        self.curr_dist[env_idx] = 0.
        self.curr_collisions[env_idx] = 0.
        self.curr_end_steps[env_idx] = self.max_steps

    def reset(self):
        self.curr_reward[:] = 0.
        self.curr_occupied[:] = 0.
        self.curr_dist[:] = 0.
        self.curr_collisions[:] = 0.
        self.curr_end_steps[:] = self.max_steps
