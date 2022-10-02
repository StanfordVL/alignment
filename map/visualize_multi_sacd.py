import json
import os
import torch
import argparse
import numpy as np

from models.world_model import WorldModel
from models.sac_nets import Critic, Actor, CentralizedCritic
from tianshou.data import Dict2Obj
from tianshou.env import make_multiagent_env
from tianshou.env import BaseRewardLogger, SimpleTagRewardLogger, SimpleTagBenchmarkLogger, SimpleSpreadBenchmarkLogger, SimpleAdversaryBenchmarkLogger
from tianshou.env import create_video
from tianshou.env.multiagent.multi_discrete import MultiDiscrete
from tianshou.policy import SACDMultiPolicy, SACDMultiCCPolicy, SACDMultiWMPolicy, SACDMultiCCWMPolicy
from tianshou.data import Collector


def get_args():
    parser = argparse.ArgumentParser()

    # State arguments.
    parser.add_argument('--save-video', action='store_true', default=False)
    parser.add_argument('--video-file', type=str, default='videos/simple.mp4')
    parser.add_argument('--save-img', action='store_true', default=False)
    parser.add_argument('--img-dir', type=str, default='imgs/')
    parser.add_argument('--auto-dir', action='store_true', default=False)
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--amb-init', type=int, default=0)
    parser.add_argument('--final-model', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument('--num-render', type=int, default=2)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_known_args()[0]
    return args


def visualize_multi_sacd(args=get_args()):
    torch.set_num_threads(1)  # for poor CPU
    params = Dict2Obj(json.load(
        open(os.path.join(args.logdir, "args.json"), "r")))
    params.device = args.device
    task_params = {'num_good_agents': params.num_good_agents,
                   'num_adversaries': params.num_adversaries,
                   'obs_radius': params.obs_radius,
                   # use args. because params are inherited from the training setting
                   # but we might test in ambiguous settings even not trained in these
                   'amb_init': args.amb_init,
                   'rew_shape': False
                   }
    env = make_multiagent_env(
        params.task, benchmark=True, optional=task_params)
    num_agents = len(env.world.agents)
    # to account for setups where each agent might have a different action space
    action_space_n = []
    act_dims = []
    for act_space in env.action_space:
        if isinstance(act_space, MultiDiscrete):
            total_dim = np.sum(act_space.high - act_space.low + 1)
            act_dims.append(act_space.high[0] - act_space.low[0] + 1)
        else:
            total_dim = act_space.n
            act_dims.append(act_space.n)
        action_space = max(total_dim, max(action_space_n)
                           if len(action_space_n) > 0 else total_dim)
        action_space_n.append(action_space)
    # seed
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    if 'spread' in params.task:
        reward_logger = BaseRewardLogger
        log_keys = ['rew']
    else:
        def reward_logger(): return SimpleTagRewardLogger(len(
            [a for a in env.world.agents if a.adversary]))
        log_keys = ['rew', 'rew/adv_rew', 'rew/agt_rew']

    # model
    actors = [Actor(params.layer_num, params.state_shape, action_space_n[i],
                    softmax=True, device=params.device).to(params.device)
              for i in range(num_agents)]
    critic1s = [CentralizedCritic(params.layer_num, params.state_shape, action_space_n[i], num_agents,
                                  device=params.device).to(params.device) if params.centralized else Critic(params.layer_num, params.state_shape,
                                                                                                            action_space_n[i],
                                                                                                            device=params.device).to(params.device)
                for i in range(num_agents)]
    critic2s = [CentralizedCritic(params.layer_num, params.state_shape, action_space_n[i], num_agents,
                                  device=params.device).to(params.device) if params.centralized else
                Critic(params.layer_num, params.state_shape,
                       action_space_n[i],
                       device=params.device).to(params.device)
                for i in range(num_agents)]

    # World Model
    world_models = [WorldModel(num_agents, params.layer_num, params.state_shape, params.action_shape,  device=params.device).to(params.device)
                    for _ in range(num_agents)]

    # Setup the benchmark loggers.
    benchmark_logger = None
    if 'simple_tag' in params.task:
        benchmark_logger = SimpleTagBenchmarkLogger(
            params.test_num,
            len([a for a in env.world.agents if a.adversary]))
        log_keys.extend(['bench/collisions', 'bench/cr_dist',
                        'bench/adv_step_rew', 'bench/agt_step_rew'])
    elif 'spread' in params.task:
        benchmark_logger = SimpleSpreadBenchmarkLogger(params.test_num)
        log_keys.extend(['bench/min_dist',
                         'bench/step_reward', 'bench/occupied'])
    else:
        benchmark_logger = SimpleAdversaryBenchmarkLogger(
            params.test_num, len([a for a in env.world.agents if a.adversary]))
        log_keys.extend(['bench/adv_occupied', 'bench/agt_occupied', 'bench/adv_dist',
                        'bench/agt_dist', 'bench/adv_step_rew', 'bench/agt_step_rew'])

    # Policy
    dist = torch.distributions.Categorical
    comm_agts = [i for i in range(len(env.world.agents)) if (
        not env.world.agents[i].silent) and env.world.agents[i].leader]
    if params.centralized:
        if params.intr_rew == 'no':
            policy = SACDMultiCCPolicy(
                actors, None,
                critic1s, None,
                critic2s, None,
                dist, params.tau, params.gamma, params.alpha,
                reward_normalization=params.rew_norm,
                ignore_done=params.ignore_done,
                estimation_step=params.n_step,
                act_dims=act_dims)
        else:
            policy = SACDMultiCCWMPolicy(
                actors, None,
                critic1s, None,
                critic2s, None,
                world_models, None,
                dist_fn=dist,
                tau=params.tau,
                gamma=params.gamma,
                alpha=params.alpha,
                reward_normalization=params.rew_norm,
                ignore_done=params.ignore_done,
                estimation_step=params.n_step,
                num_adv=params.num_adversaries,
                intr_rew_options=params.intr_rew,
                num_landmark=len(env.world.landmarks),
                obs_radii=[env.world.obs_radius] * num_agents)
    else:
        if params.intr_rew == 'no':
            policy = SACDMultiPolicy(
                actors, None,
                critic1s, None,
                critic2s, None,
                dist_fn=dist,
                tau=params.tau,
                gamma=params.gamma,
                alpha=params.alpha,
                reward_normalization=params.rew_norm,
                ignore_done=params.ignore_done,
                estimation_step=params.n_step)
        else:
            policy = SACDMultiWMPolicy(
                actors, None,
                critic1s, None,
                critic2s, None,
                world_models, None,
                dist_fn=dist,
                tau=params.tau,
                gamma=params.gamma,
                alpha=params.alpha,
                reward_normalization=params.rew_norm,
                ignore_done=params.ignore_done,
                estimation_step=params.n_step,
                num_adv=params.num_adversaries,
                intr_rew_options=params.intr_rew,
                num_landmark=len(env.world.landmarks),
                obs_radii=[env.world.obs_radius] * num_agents)

    # Load model parameters.
    if args.final_model:
        policy.load(args.logdir, type='final')
    else:
        policy.load(args.logdir)
    policy.eval()

    # Change max steps for a longer visualization
    env.world.max_steps = 250
    env.world.random_init = True
    collector = Collector(policy, env, num_agents=num_agents,
                          reward_logger=reward_logger,
                          benchmark_logger=benchmark_logger)
    if args.amb_init:
        video_file = args.logdir + '_amb_init'
    else:
        video_file = args.logdir
    if args.final_model:
        video_file += '_final'
    video_file = video_file.replace(
        'log', 'videos') + '.mp4' if args.auto_dir else args.video_file
    img_dir = args.logdir.replace(
        'log', 'imgs') if args.auto_dir else args.img_dir

    for i in range(args.num_render):
        collector.reset()
        result = collector.collect(
            n_episode=1, render=args.render, render_mode='rgb_array')
        for k in log_keys:
            print(f'{k}: {result[k]}')
        if args.save_video:
            if args.save_img:
                create_video(result['frames'], video_file.replace(
                    '.mp4', f'_{i}.mp4'), img_dir + f'_{i}')
            else:
                create_video(result['frames'], video_file.replace(
                    '.mp4', f'_{i}.mp4'))
    collector.close()


if __name__ == '__main__':
    visualize_multi_sacd()
