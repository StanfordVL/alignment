# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple example of setting up a multi-agent version of GFootball with rllib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import argparse
from policy import AlignSACPolicy
import gfootball.env as football_env
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.agents.sac.sac_torch_policy import SACTorchPolicy
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.utils.deprecation import DEPRECATED_VALUE

from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()

parser.add_argument('--num-agents', type=int, default=3)
parser.add_argument('--num-policies', type=int, default=3)
parser.add_argument('--num-iters', type=int, default=10000)
parser.add_argument('--num-gpus', type=int, default=0)
parser.add_argument('--object-store-memory-GB', type=int, default=10)
parser.add_argument('--memory-GB',type=int, default=10)
parser.add_argument('--logdir', type=str, default='/vision2/u/zixianma/football_results')
parser.add_argument('--name', type=str, default='exp')
parser.add_argument('--simple', action='store_true')
parser.add_argument('--align-mode', type=str, default='elign_team')
parser.add_argument('--radius', type=float, default=float('inf'))
parser.add_argument('--seeds', type=int, nargs='+', help='a list of random seeds', required=True)

class RllibGFootball(MultiAgentEnv):
  """An example of a wrapper for GFootball to make it compatible with rllib."""

  def __init__(self, num_agents):
    self.env = football_env.create_environment(
        env_name='academy_3_vs_1_with_keeper', stacked=False,
        representation='simple115v2',
        rewards='scoring',
        env_radius=args.radius,
        logdir=os.path.join(tempfile.gettempdir(), 'rllib_test'),
        write_goal_dumps=False, write_full_episode_dumps=False, render=False,
        dump_frequency=0,
        number_of_left_players_agent_controls=num_agents,
        channel_dimensions=(42, 42))
    self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
    self.observation_space = gym.spaces.Box(
        low=self.env.observation_space.low[0],
        high=self.env.observation_space.high[0],
        dtype=self.env.observation_space.dtype)
    self.num_agents = num_agents
    self._agent_ids = [f'agent_{id}' for id in range(self.num_agents)]

  def reset(self):
    original_obs = self.env.reset()
    obs = {}
    for x in range(self.num_agents):
      if self.num_agents > 1:
        obs['agent_%d' % x] = original_obs[x]
      else:
        obs['agent_%d' % x] = original_obs
    return obs

  def step(self, action_dict):
    actions = []
    for key, value in sorted(action_dict.items()):
      actions.append(value)
    o, r, d, i = self.env.step(actions)
    rewards = {}
    obs = {}
    infos = {}
    for pos, key in enumerate(sorted(action_dict.keys())):
      infos[key] = i
      if self.num_agents > 1:
        rewards[key] = r[pos]
        obs[key] = o[pos]
      else:
        rewards[key] = r
        obs[key] = o
    dones = {'__all__': d}
    return obs, rewards, dones, infos


if __name__ == '__main__':
  args = parser.parse_args()
  ray.init(num_gpus=args.num_gpus, _memory=args.memory_GB * (1024 ** 3), object_store_memory=args.object_store_memory_GB * (1024 ** 3))

  # Simple environment with `num_agents` independent players
  register_env('gfootball', lambda _: RllibGFootball(args.num_agents))
  single_env = RllibGFootball(args.num_agents)
  obs_space = single_env.observation_space
  act_space = single_env.action_space

  def gen_policy(_):
    return (AlignSACPolicy if hasattr(args, 'align_mode') else SACTorchPolicy, obs_space, act_space, {'env_radius': args.radius, 'align_mode': args.align_mode})

  # Setup PPO with an ensemble of `num_policies` different policies
  policies = {
      'policy_{}'.format(i): gen_policy(i) for i in range(args.num_policies)
  }
  policy_ids = list(policies.keys())

  DefaultConfig = with_common_config({
    # === Model ===
    # Use two Q-networks (instead of one) for action-value estimation.
    # Note: Each Q-network will have its own target network.
    "twin_q": True,
    # Use a e.g. conv2D state preprocessing network before concatenating the
    # resulting (feature) vector with the action input for the input to
    # the Q-networks.
    "use_state_preprocessor": DEPRECATED_VALUE,
    # Model options for the Q network(s). These will override MODEL_DEFAULTS.
    # The `Q_model` dict is treated just as the top-level `model` dict in
    # setting up the Q-network(s) (2 if twin_q=True).
    # That means, you can do for different observation spaces:
    # obs=Box(1D) -> Tuple(Box(1D) + Action) -> concat -> post_fcnet
    # obs=Box(3D) -> Tuple(Box(3D) + Action) -> vision-net -> concat w/ action
    #   -> post_fcnet
    # obs=Tuple(Box(1D), Box(3D)) -> Tuple(Box(1D), Box(3D), Action)
    #   -> vision-net -> concat w/ Box(1D) and action -> post_fcnet
    # You can also have SAC use your custom_model as Q-model(s), by simply
    # specifying the `custom_model` sub-key in below dict (just like you would
    # do in the top-level `model` dict.
    "Q_model": {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": None,
        "custom_model": None,  # Use this to define custom Q-model(s).
        "custom_model_config": {},
    },
    # Model options for the policy function (see `Q_model` above for details).
    # The difference to `Q_model` above is that no action concat'ing is
    # performed before the post_fcnet stack.
    "policy_model": {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
        "post_fcnet_hiddens": [],
        "post_fcnet_activation": None,
        # "custom_model": AlignLossModel,  # Use this to define a custom policy model.
        # "custom_model_config": {
        #   "fcnet_hiddens": [128, 128],
        #   "fcnet_activation": "relu",
        #   "post_fcnet_hiddens": [],
        #   "post_fcnet_activation": None,
        # },
    },
    # Actions are already normalized, no need to clip them further.
    "clip_actions": False,

    # === Learning ===
    # Update the target by \tau * policy + (1-\tau) * target_policy.
    "tau": 5e-3,
    # Initial value to use for the entropy weight alpha.
    "initial_alpha": 1.0,
    # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
    # Discrete(2), -3.0 for Box(shape=(3,))).
    # This is the inverse of reward scale, and will be optimized automatically.
    "target_entropy": "auto",
    # N-step target updates. If >1, sars' tuples in trajectories will be
    # postprocessed to become sa[discounted sum of R][s t+n] tuples.
    "n_step": 1,
    # Number of env steps to optimize for before returning.
    "timesteps_per_iteration": 100,

    # === Replay buffer ===
    # Size of the replay buffer (in time steps).
    "buffer_size": DEPRECATED_VALUE,
    "replay_buffer_config": {
        "type": "MultiAgentReplayBuffer",
        "capacity": int(1e6),
    },
    # Set this to True, if you want the contents of your buffer(s) to be
    # stored in any saved checkpoints as well.
    # Warnings will be created if:
    # - This is True AND restoring from a checkpoint that contains no buffer
    #   data.
    # - This is False AND restoring from a checkpoint that does contain
    #   buffer data.
    "store_buffer_in_checkpoints": False,
    # If True prioritized replay buffer will be used.
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta": 0.4,
    "prioritized_replay_eps": 1e-6,
    "prioritized_replay_beta_annealing_timesteps": 20000,
    "final_prioritized_replay_beta": 0.4,
    # Whether to LZ4 compress observations
    "compress_observations": False,

    # The intensity with which to update the model (vs collecting samples from
    # the env). If None, uses the "natural" value of:
    # `train_batch_size` / (`rollout_fragment_length` x `num_workers` x
    # `num_envs_per_worker`).
    # If provided, will make sure that the ratio between ts inserted into and
    # sampled from the buffer matches the given value.
    # Example:
    #   training_intensity=1000.0
    #   train_batch_size=250 rollout_fragment_length=1
    #   num_workers=1 (or 0) num_envs_per_worker=1
    #   -> natural value = 250 / 1 = 250.0
    #   -> will make sure that replay+train op will be executed 4x as
    #      often as rollout+insert op (4 * 250 = 1000).
    # See: rllib/agents/dqn/dqn.py::calculate_rr_weights for further details.
    "training_intensity": None,

    # === Optimization ===
    "optimization": {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
    },
    # If not None, clip gradients during optimization at this value.
    "grad_clip": None,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1500,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 1,
    # Size of a batched sampled from replay buffer for training.
    "train_batch_size": 256,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 0,

    # === Parallelism ===
    # Whether to use a GPU for local optimization.
    "num_gpus": args.num_gpus,
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Whether to allocate GPUs for workers (if > 0).
    "num_gpus_per_worker": args.num_gpus,
    # Whether to allocate CPUs for workers (if > 0).
    "num_cpus_per_worker": 1,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent reporting frequency from going lower than this time span.
    "min_time_s_per_reporting": 1,

    # Whether the loss should be calculated deterministically (w/o the
    # stochastic action sampling step). True only useful for cont. actions and
    # for debugging!
    "_deterministic_loss": False,
    # Use a Beta-distribution instead of a SquashedGaussian for bounded,
    # continuous action spaces (not recommended, for debugging only).
    "_use_beta_distribution": False,
})
  extra_config = {
          'env': 'gfootball',
          'simple_optimizer': args.simple,
          'framework': 'torch',
          'log_level': 'ERROR', 
          'seed': tune.grid_search(args.seeds),
          'multiagent': {
              'policies': policies,
              'policy_mapping_fn': tune.function(
                  lambda agent_id: policy_ids[int(agent_id[6:])]),
    }
  }

  config = Trainer.merge_trainer_configs(
        DefaultConfig, extra_config, _allow_unknown_configs=True
    )

  tune.run(
      'SAC',
      stop={'training_iteration': args.num_iters},
      checkpoint_freq=500,
      config=config,
      local_dir=args.logdir,
      name=args.name,
      checkpoint_at_end=True,
  )
