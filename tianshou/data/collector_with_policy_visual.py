import gym
import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.utils import MovAvg, VecMovAvg, VecTotal
from tianshou.env import BaseVectorEnv
from tianshou.env import BaseRewardLogger
from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, ListReplayBuffer, to_numpy


class Collector(object):
    """The :class:`~tianshou.data.Collector` enables the policy to interact
    with different types of environments conveniently.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer`
        class, or a list of :class:`~tianshou.data.ReplayBuffer`. If set to
        ``None``, it will automatically assign a small-size
        :class:`~tianshou.data.ReplayBuffer`.
    :param function preprocess_fn: a function called before the data has been
        added to the buffer, see issue #42, defaults to ``None``.
    :param function initialize_state_fn: a function called before the data
        has been added to the buffer, see issue #42, defaults to ``None``.
    :param int stat_size: for the moving average of recording speed, defaults
        to 100.
    :param reward_aggregator: support either VecTotal or VecMovAvg.
    :param reward_logger: supports custom reward logging.

    The ``preprocess_fn`` is a function called before the data has been added
    to the buffer with batch format, which receives up to 7 keys as listed in
    :class:`~tianshou.data.Batch`. It will receive with only ``obs`` when the
    collector resets the environment. It returns either a dict or a
    :class:`~tianshou.data.Batch` with the modified keys and values. Examples
    are in "test/base/test_collector.py".

    Example:
    ::

        policy = PGPolicy(...)  # or other policies if you wish
        env = gym.make('CartPole-v0')
        replay_buffer = ReplayBuffer(size=10000)
        # here we set up a collector with a single environment
        collector = Collector(policy, env, buffer=replay_buffer)

        # the collector supports vectorized environments as well
        envs = VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(3)])
        buffers = [ReplayBuffer(size=5000) for _ in range(3)]
        # you can also pass a list of replay buffer to collector, for multi-env
        # collector = Collector(policy, envs, buffer=buffers)
        collector = Collector(policy, envs, buffer=replay_buffer)

        # collect at least 3 episodes
        collector.collect(n_episode=3)
        # collect 1 episode for the first env, 3 for the third env
        collector.collect(n_episode=[1, 0, 3])
        # collect at least 2 steps
        collector.collect(n_step=2)
        # collect episodes with visual rendering (the render argument is the
        #   sleep time between rendering consecutive frames)
        collector.collect(n_episode=1, render=0.03)

        # sample data with a given number of batch-size:
        batch_data = collector.sample(batch_size=64)
        # policy.learn(batch_data)  # btw, vanilla policy gradient only
        #   supports on-policy training, so here we pick all data in the buffer
        batch_data = collector.sample(batch_size=0)
        policy.learn(batch_data)
        # on-policy algorithms use the collected data only once, so here we
        #   clear the buffer
        collector.reset_buffer()

    For the scenario of collecting data from multiple environments to a single
    buffer, the cache buffers will turn on automatically. It may return the
    data more than the given limitation.

    .. note::

        Please make sure the given environment has a time limitation.
    """

    def __init__(self,
                 policy: BasePolicy,
                 env: Union[gym.Env, BaseVectorEnv],
                 buffer: Optional[Union[ReplayBuffer, List[ReplayBuffer]]]
                 = None,
                 preprocess_fn: Callable[[Any], Union[dict, Batch]] = None,
                 initialize_state_fn: Callable[[Any], Union[dict, Batch]]
                 = None,
                 stat_size: Optional[int] = 100,
                 num_agents: Optional[int] = 1,
                 reward_aggregator: object = VecTotal,
                 reward_logger: object = BaseRewardLogger,
                 benchmark_logger: object = None,
                 **kwargs) -> None:
        super().__init__()
        self.env = env
        self.env_num = 1
        self.num_agents = num_agents
        self.collect_time = 0
        self.collect_step = 0
        self.collect_episode = 0
        self.buffer = buffer
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self.initialize_state_fn = initialize_state_fn
        # if preprocess_fn is None:
        #     def _prep(**kwargs):
        #         return kwargs
        #     self.preprocess_fn = _prep
        self.process_fn = policy.process_fn
        self._multi_env = isinstance(env, BaseVectorEnv)
        self._multi_buf = False  # True if buf is a list
        # need multiple cache buffers only if storing in one buffer
        self._cached_buf = []
        if self._multi_env:
            self.env_num = len(env)
            if isinstance(self.buffer, list):
                assert len(self.buffer) == self.env_num, \
                    'The number of data buffer does not match the number of ' \
                    'input env.'
                self._multi_buf = True
            elif isinstance(self.buffer, ReplayBuffer) or self.buffer is None:
                self._cached_buf = [
                    ListReplayBuffer() for _ in range(self.env_num)]
            else:
                raise TypeError('The buffer in data collector is invalid!')
        self.stat_size = stat_size
        self.reward_aggregator = reward_aggregator
        self.reward_logger = reward_logger
        self.benchmark_logger = benchmark_logger
        self.reset()

    def reset(self) -> None:
        """Reset all related variables in the collector."""
        # state over batch is either a list, an np.ndarray, or a torch.Tensor
        self.reset_env()
        self.reset_buffer()
        self.step_speed = MovAvg(self.stat_size)
        self.episode_speed = MovAvg(self.stat_size)
        self.collect_step = 0
        self.collect_episode = 0
        self.collect_time = 0

    def reset_buffer(self) -> None:
        """Reset the main data buffer."""
        if self._multi_buf:
            for b in self.buffer:
                b.reset()
        else:
            if self.buffer is not None:
                self.buffer.reset()

    def get_env_num(self) -> int:
        """Return the number of environments the collector have."""
        return self.env_num

    def reset_env(self) -> None:
        """Reset all of the environment(s)' states and reset all of the cache
        buffers (if need).
        """
        self.state = None
        self._reset_state(list(range(self.get_env_num())))
        self._policy = None
        if self.state is not None:
            self._policy = Batch(_state=self.state)
        self._obs = self.env.reset()
        if not self._multi_env:
            self._obs = self._make_batch(self._obs)
        if self.preprocess_fn:
            result = self.preprocess_fn(Batch(
                obs=self._obs, policy=self._policy))
            self._obs = result.get('obs', self._obs)
            self._policy = result.get('policy', self._policy)
        self._act = self._rew = self._done = self._info = None
        if self._multi_env:
            self.reward = np.zeros((self.env_num, self.num_agents))
            self.length = np.zeros(self.env_num)
        else:
            self.reward = np.zeros((1, self.num_agents))
            self.length = 0
        self.reward_agg = self.reward_aggregator(
                ndim=self.env_num * self.num_agents)
        if self.benchmark_logger:
            self.benchmark_logger.reset()
        for b in self._cached_buf:
            b.reset()

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        """Reset all the seed(s) of the given environment(s)."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)

    def render(self, **kwargs) -> None:
        """Render all the environment(s)."""
        if hasattr(self.env, 'render'):
            return self.env.render(**kwargs)

    def close(self) -> None:
        """Close the environment(s)."""
        if hasattr(self.env, 'close'):
            self.env.close()

    def _make_batch(self, data: Any) -> Union[Any, np.ndarray]:
        """Return [data]."""
        if isinstance(data, np.ndarray):
            return data[None]
        else:
            return np.array([data])

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset self.state[id]."""
        if self.initialize_state_fn is not None:
            self.state = self.initialize_state_fn(self.state, id)
        elif self.state is None:
            return
        elif isinstance(self.state, list):
            self.state[id] = None
        elif isinstance(self.state, (dict, Batch)):
            for k in self.state.keys():
                if isinstance(self.state[k], list):
                    self.state[k][id] = None
                elif isinstance(self.state[k], (torch.Tensor, np.ndarray)):
                    self.state[k][id] = 0
        elif isinstance(self.state, (torch.Tensor, np.ndarray)):
            self.state[id] = 0

    def collect(self,
                n_step: int = 0,
                n_episode: Union[int, List[int]] = 0,
                random: bool = False,
                render: Optional[float] = None,
                render_mode: Optional[str] = None,
                log_fn: Optional[Callable[[dict], None]] = None
                ) -> Dict[str, float]:
        """Collect a specified number of step or episode.

        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect (in each
            environment).
        :param bool random: whether to use random policy for collecting data,
            defaults to ``False``.
        :type n_episode: int or list
        :param float render: the sleep time between rendering consecutive
            frames, defaults to ``None`` (no rendering).
        :param function log_fn: a function which receives env info, typically
            for tensorboard logging.

        .. note::

            One and only one collection number specification is permitted,
            either ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` the collected number of episodes.
            * ``n/st`` the collected number of steps.
            * ``v/st`` the speed of steps per second.
            * ``v/ep`` the speed of episode per second.
            * ``rew`` the mean reward over collected episodes.
            * ``len`` the mean length over collected episodes.
        """
        warning_count = 0
        if not self._multi_env:
            n_episode = np.sum(n_episode)
        start_time = time.time()
        assert sum([(n_step != 0), (n_episode != 0)]) == 1, \
            "One and only one collection number specification is permitted!"
        cur_step = 0
        cur_episode = np.zeros(self.env_num) if self._multi_env else 0
        reward_logger = self.reward_logger()
        length_sum = 0
        frames = []
        multiplier = 0
        init_act = np.ones((1,3))
        while True:
            if warning_count >= 100000:
                warnings.warn(
                    'There are already many steps in an episode. '
                    'You should add a time limitation to your environment!',
                    Warning)

            # Initialize batch.
            batch = Batch(
                obs=self._obs, act=self._act, rew=self._rew,
                done=self._done, obs_next=None, info=self._info,
                policy=self._policy)

            # Choose actions.
            if random:
                action_space = self.env.action_space
                if isinstance(action_space, list):
                    result = Batch(act=[a.sample() for a in action_space])
                else:
                    result = Batch(act=self._make_batch(action_space.sample()))
            else:
                with torch.no_grad():
                    result = self.policy(batch)

            # save hidden state to policy._state, in order to save into buffer
            self.state = result.get('state', None)
            if hasattr(result, 'policy'):
                self._policy = to_numpy(result.policy)
                if self.state is not None:
                    self._policy._state = self.state
            elif self.state is not None:
                self._policy = Batch(_state=self.state)
            else:
                self._policy = [{}] * self.env_num

            # Take environment step.
            self._act = to_numpy(result.act)
            
            # multiplier %= 5
            # self._act = init_act * multiplier
            # if cur_step % 10 == 0:
            #     multiplier += 1
            
            # elif cur_step % 5 == 1:
            #     self._act = np.ones((1,3)) 
            # elif cur_step % 5 == 2:
            #     self._act = np.ones((1,3)) * 2
            # elif cur_step % 5 == 3:
            #     self._act = np.ones((1,3)) * 3
            # else:
            #     self._act = np.ones((1,3)) * 4
            print("org act:", result.act)
            print("after act:", self._act.shape)
            print("self.multi_env:", self._multi_env)
            obs_next, self._rew, self._done, self._info = self.env.step(
                self._act if self._multi_env else self._act[0])
            obs_next_wm = []
            num_agents = 3
            # # obs_n = np.concatenate([self._obs[0][i,:] for i in range(num_agents)], axis=0)
            # # act_n = np.concatenate([self._act[:,i] for i in range(num_agents)], axis=0)
            # # print(obs_n.shape, np.expand_dims(act_n, axis=-1).shape)
            # Overwrite agents' positions
            # for i in range(num_agents):
            #     world_model = self.policy.world_models[i]
            #     obs_i = self._obs[0][i,:]
            #     act_i = self._act[:,i]
            #     # print(obs_i.shape, act_i.shape)
            #     inputs = np.expand_dims(np.concatenate((obs_i, act_i), axis=0), axis=0)
            #     pred_next_obs_n = world_model(inputs).squeeze(0).detach().numpy()
            #     # print("true next obs:", obs_next[i])
            #     # print("pred next obs:", pred_next_obs_n)
            #     # print("diff:", np.abs(obs_next[i] - pred_next_obs_n))
            #     obs_next_wm.append(pred_next_obs_n)
            #     # overwriting the agents' positions with the ones predicted by the world model
            #     self.env.world.entities[i].state.p_pos = pred_next_obs_n[:2]
            
            # Update batch elements.
            if not self._multi_env:
                obs_next = self._make_batch(obs_next)
                self._rew = self._make_batch(self._rew)
                self._done = self._make_batch(self._done)
                self._info = self._make_batch(self._info)

            # Logging and rendering.
            if log_fn:
                log_fn(self._info if self._multi_env else self._info[0])
            if self.benchmark_logger:
                self.benchmark_logger.add(self._info)
            if render:
                if render_mode:
                    frame = self.env.render(mode=render_mode)[0]
                    frames.append(frame)
                else:
                    self.env.render()
                if render > 0:
                    time.sleep(render)

            # Preprocess new batch of data.
            if self.preprocess_fn:
                result = self.preprocess_fn(Batch(
                    obs=self._obs, act=self._act, rew=self._rew,
                    done=self._done, obs_next=obs_next, info=self._info,
                    policy=self._policy))
                self._obs = result.get('obs', self._obs)
                self._act = result.get('act', self._act)
                self._rew = result.get('rew', self._rew)
                self._done = result.get('done', self._done)
                obs_next = result.get('obs_next', obs_next)
                self._info = result.get('info', self._info)
                self._policy = result.get('policy', self._policy)

            # Update cummulative rewards.
            self.length += 1
            if self._rew.ndim == 1:
                self.reward = self.reward_agg.add(self._rew).reshape(
                        self.env_num, 1)
            else:
                self.reward = self.reward_agg.add(self._rew.reshape(
                        self.env_num * self.num_agents)).reshape(
                                self.env_num, self.num_agents)

            if self._multi_env:
                for i in range(self.env_num):
                    data = {
                        'obs': self._obs[i], 'act': self._act[i],
                        'rew': self._rew[i], 'done': self._done[i],
                        'obs_next': obs_next[i], 'info': self._info[i],
                        'policy': self._policy[i]}
                    if self._cached_buf:
                        warning_count += 1
                        self._cached_buf[i].add(**data)
                    elif self._multi_buf:
                        warning_count += 1
                        self.buffer[i].add(**data)
                        cur_step += 1
                    else:
                        warning_count += 1
                        if self.buffer is not None:
                            self.buffer.add(**data)
                        cur_step += 1
                    if np.all(self._done[i]):
                        if n_step != 0 or np.isscalar(n_episode) or \
                                cur_episode[i] < n_episode[i]:
                            cur_episode[i] += 1
                            reward_logger.add(self.reward[i])
                            length_sum += self.length[i]
                            if self._cached_buf:
                                cur_step += len(self._cached_buf[i])
                                if self.buffer is not None:
                                    self.buffer.update(self._cached_buf[i])
                        self.reward = self.reward_agg.reset(
                                i*self.num_agents, (i+1)*self.num_agents)
                        if self.benchmark_logger:
                            self.benchmark_logger.episode_end(i)
                        self.length[i] = 0
                        if self._cached_buf:
                            self._cached_buf[i].reset()
                if np.any(self._done):
                    done_envs = np.where(self._done[:, 0])[0]
                    self._reset_state(done_envs)
                    if self.state is not None:
                        self._policy = Batch(_state=self.state)
                    obs_next = self.env.reset(done_envs)
                    if self.preprocess_fn:
                         resnext = self.preprocess_fn(
                            Batch(obs=obs_next, policy=self._policy),
                            id=done_envs)
                         obs_next = resnext.get('obs', obs_next)
                         self._policy = resnext.get('policy', self._policy)
                if n_episode != 0:
                    if isinstance(n_episode, list) and \
                            (cur_episode >= np.array(n_episode)).all() or \
                            np.isscalar(n_episode) and \
                            cur_episode.sum() >= n_episode:
                        break
            else:
                if self.buffer is not None:
                    self.buffer.add(
                        self._obs[0], self._act[0], self._rew[0],
                        self._done[0], obs_next[0], self._info[0],
                        self._policy[0])
                cur_step += 1
                if np.all(self._done):
                    cur_episode += 1
                    reward_logger.add(self.reward[0])
                    length_sum += self.length
                    self.reward = self.reward_agg.reset(0, self.num_agents)
                    if self.benchmark_logger:
                        self.benchmark_logger.episode_end(0)
                    self.length = 0
                    self._reset_state([0])
                    if self.state is not None:
                        self._policy = Batch(_state=self.state)
                    obs_next = self._make_batch(self.env.reset())
                    if self.preprocess_fn:
                        resnext = self.preprocess_fn(
                            Batch(obs=obs_next, policy=self._policy), id=0)
                        obs_next = resnext.get('obs', obs_next)
                        self._policy = resnext.get('policy', self._policy)
                if n_episode != 0 and cur_episode >= n_episode:
                    break
            if n_step != 0 and cur_step >= n_step:
                break
            self._obs = obs_next
        self._obs = obs_next

        # Collect values for next iterations
        if self._multi_env:
            cur_episode = sum(cur_episode)
        duration = max(time.time() - start_time, 1e-9)
        self.step_speed.add(cur_step / duration)
        self.episode_speed.add(cur_episode / duration)
        self.collect_step += cur_step
        self.collect_episode += cur_episode
        self.collect_time += duration
        n_episode = max(cur_episode, 1)
        output = {
            # 'n/ep': cur_episode,
            'n/st': cur_step,
            # 'v/st': self.step_speed.get(),
            # 'v/ep': self.episode_speed.get(),
            # 'len': length_sum / n_episode,
        }

        # Gather rewards to log.
        for k, v in reward_logger.log().items():
            output[k] = v

        # Gather benchmark data to log.
        if self.benchmark_logger:
            for k, v in self.benchmark_logger.log().items():
                output[k] = v

        # Store rendered frames.
        if render and render_mode == 'rgb_array':
            output['frames'] = frames
        return output

    def sample(self, batch_size: int, global_step: int = None) -> Batch:
        """Sample a data batch from the internal replay buffer. It will call
        :meth:`~tianshou.policy.BasePolicy.process_fn` before returning
        the final batch data.

        :param int batch_size: ``0`` means it will extract all the data from
            the buffer, otherwise it will extract the data with the given
            batch_size.
        :param global_step: what step of training are we on.
        """
        if self._multi_buf:
            if batch_size > 0:
                lens = [len(b) for b in self.buffer]
                total = sum(lens)
                batch_index = np.random.choice(
                    len(self.buffer), batch_size, p=np.array(lens) / total)
            else:
                batch_index = np.array([])
            batch_data = Batch()
            for i, b in enumerate(self.buffer):
                cur_batch = (batch_index == i).sum()
                if batch_size and cur_batch or batch_size <= 0:
                    batch, indice = b.sample(cur_batch)
                    batch = self.process_fn(batch, b, indice,
                                            global_step=global_step)
                    batch_data.append(batch)
        else:
            batch_data, indice = self.buffer.sample(batch_size)
            # batch_data = self.process_fn(batch_data, self.buffer, indice,
            #                              global_step=global_step)
            batch_data, intr_rews = self.process_fn(batch_data, self.buffer, indice)
        return batch_data, intr_rews
