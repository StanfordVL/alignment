import json
import os
import torch
import wandb
import argparse
import numpy as np
import csv
from models.world_model import WorldModel
from models.sac_nets import Critic, Actor, CentralizedCritic
from tianshou.data import Dict2Obj
from tianshou.env import make_multiagent_env

from tianshou.env import VectorEnv
from tianshou.env import BaseRewardLogger
from tianshou.env import SimpleTagRewardLogger
from tianshou.env import SimpleTagBenchmarkLogger
from tianshou.env import SimpleSpreadBenchmarkLogger
from tianshou.env import SimpleAdversaryBenchmarkLogger
from tianshou.env.multiagent.multi_discrete import MultiDiscrete
from tianshou.policy import (
    SACDMultiPolicy, 
    SACDMultiCCPolicy, 
    SACDMultiCCWMPolicy,
    SACDMultiCCWMOthersPolicy, 
    SACDMultiWMPolicy, 
    SACDMultiCommPolicy, 
    SACDMultiCommWMPolicy
)
from tianshou.data import Collector


def get_args():
    parser = argparse.ArgumentParser()

    # State arguments.
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--savedir', type=str, default='./result')
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--eval-num', type=int, default=100)
    parser.add_argument('--amb-init', type=int, default=0)
    parser.add_argument('--test-final', action='store_true', default=False)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_known_args()[0]
    return args


def evaluate_multi_sacd(args=get_args()):
    torch.set_num_threads(1)  # for poor CPU
    # wandb_dir = '/scr/zixianma/multiagent/' if torch.cuda.is_available() else 'log/'
    # wandb.init(dir=wandb_dir, project='multiagent-eval')
    params = Dict2Obj(json.load(
            open(os.path.join(args.logdir, "args.json"), "r")))
    params.device = args.device
    # wandb.config.update(params)
    # wandb.config.update(args, allow_val_change=True)
    task_params = {'num_good_agents': params.num_good_agents,
                'num_adversaries': params.num_adversaries,
                'obs_radius': params.obs_radius,
                'with_comm': params.with_comm,
                # use args. because params are inherited from the training setting
                # but we might test in ambiguous settings even not trained in these
                'amb_init': args.amb_init,
                'rew_shape': False}
    env = make_multiagent_env(params.task, benchmark=True, optional=task_params)
    num_agents = len(env.world.agents)
    test_envs = VectorEnv(
        [lambda: make_multiagent_env(params.task, benchmark=True, optional=task_params)
            for _ in range(args.eval_num)])
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
        action_space = max(total_dim, max(action_space_n) if len(action_space_n) > 0 else total_dim)
        action_space_n.append(action_space)
    # seed
    np.random.seed(105)
    torch.manual_seed(105)

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
    # if 'wm_noise_level' in params:
    #     world_models = [WorldModel(num_agents, params.layer_num, params.state_shape, action_space_n[i],  device=params.device, wm_noise_level=params.wm_noise_level).to(params.device) for i in range(num_agents)]
    # else:
    world_models = [WorldModel(num_agents, params.layer_num, params.state_shape, action_space_n[i],  device=params.device).to(params.device) for i in range(num_agents)]
    

    # Setup the reward loggers and the corresponding keys
    if 'spread' in params.task:
        reward_logger = BaseRewardLogger
        log_keys = ['rew'] 
    else:
        reward_logger = lambda: SimpleTagRewardLogger(len(
                [a for a in env.world.agents if a.adversary]))
        log_keys = ['rew', 'rew/adv_rew', 'rew/agt_rew']

    # Setup the benchmark loggers.
    benchmark_logger = None
    if 'simple_tag' in params.task:
        benchmark_logger = SimpleTagBenchmarkLogger(
            args.eval_num,
            len([a for a in env.world.agents if a.adversary]))
        log_keys.extend(['bench/collisions', 'bench/cr_dist', 'bench/adv_step_rew', 'bench/agt_step_rew', 'bench/agt_pure_rew', 'bench/agt_bound_rew'])
    elif 'spread' in params.task:
        benchmark_logger = SimpleSpreadBenchmarkLogger(args.eval_num)
        log_keys.extend(['bench/min_dist',
                         'bench/step_reward', 'bench/occupied', 'bench/pure_rew', 'bench/bound_rew'])
    else:
        benchmark_logger = SimpleAdversaryBenchmarkLogger(args.eval_num, len([a for a in env.world.agents if a.adversary]))
        log_keys.extend(['bench/adv_occupied', 'bench/agt_occupied','bench/adv_dist', 'bench/agt_dist', 'bench/adv_step_rew', 'bench/agt_step_rew', 'bench/agt_pure_rew', 'bench/agt_bound_rew'])

    # Policy
    dist = torch.distributions.Categorical
    
    # in simple world comm, we make all the agents communicative, but only the leader communciates something meaningful
    comm_agts = [i for i in range(len(env.world.agents)) if (not env.world.agents[i].silent) and env.world.agents[i].leader]
    if params.centralized:
        if params.others_model:
            policy = SACDMultiCCWMOthersPolicy(
                            actors, None,
                            critic1s, None,
                            critic2s, None,
                            world_models, None,
                            dist, params.tau, params.gamma, params.alpha,
                            reward_normalization=params.rew_norm,
                            ignore_done=params.ignore_done,
                            estimation_step=params.n_step,
                            act_dims=act_dims,
                            num_adv=params.num_adversaries,
                            intr_rew_options=params.intr_rew,
                            num_landmark=len(env.world.landmarks),
                            obs_radii=[env.world.obs_radius] * num_agents)
        else:
            if params.intr_rew == '000':
                policy = SACDMultiCCPolicy(
                            actors, None,
                            critic1s, None,
                            critic2s, None,
                            dist, params.tau, params.gamma, params.alpha,
                            reward_normalization=params.rew_norm,
                            ignore_done=params.ignore_done,
                            estimation_step=params.n_step,
                            act_dims=act_dims,
                            intr_rew_options=params.intr_rew)
            else:
                policy = SACDMultiCCWMPolicy(
                            actors, None,
                            critic1s, None,
                            critic2s, None,
                            world_models, None,
                            dist, params.tau, params.gamma, params.alpha,
                            reward_normalization=params.rew_norm,
                            ignore_done=params.ignore_done,
                            estimation_step=params.n_step,
                            act_dims=act_dims,
                            num_adv=params.num_adversaries,
                            intr_rew_options=params.intr_rew,
                            num_landmark=len(env.world.landmarks),
                            obs_radii=[env.world.obs_radius] * num_agents)
    else:
        if params.intr_rew == 0 or params.intr_rew == '000':
            if len(comm_agts) > 0:
                policy = SACDMultiCommPolicy(actors, None,
                    critic1s, None,
                    critic2s, None,
                    dist, params.tau, params.gamma, params.alpha,
                    reward_normalization=params.rew_norm,
                    ignore_done=params.ignore_done,
                    estimation_step=params.n_step,
                    comm_agts=comm_agts,
                    act_dims=act_dims)
            else:
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
            if len(comm_agts) > 0:
                policy = SACDMultiCommWMPolicy(actors, None,
                    critic1s, None,
                    critic2s, None,
                    world_models, None,
                    dist, params.tau, params.gamma, params.alpha,
                    reward_normalization=params.rew_norm,
                    ignore_done=params.ignore_done,
                    estimation_step=params.n_step,
                    num_adv=params.num_adversaries,
                    intr_rew_options=params.intr_rew,
                    comm_agts=comm_agts,
                    act_dims=act_dims)
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

    run_name = args.logdir[args.logdir.rfind('/') + 1:]
    # wandb.run.name = run_name + '_amb_init' if args.amb_init else run_name
    # Load model parameters.
    policy.load(args.logdir)
    policy.eval()

    # Change max steps for a longer visualization
    collector = Collector(policy, test_envs, num_agents=num_agents,
                          reward_logger=reward_logger,
                          benchmark_logger=benchmark_logger)
    result = collector.collect(n_episode=args.num_episodes)
    collector.close()
    # wandb.log(result)
    log_path = args.logdir 
    # start = log_path.find('log') + 4
    # end = log_path.find('epochs') + 6 if log_path.find('epochs') != -1 else max(log_path.find('rep'), 0) - 1
    start = log_path.rfind('/') + 1
    end = log_path.find('intr') - 1 if log_path.find('intr') != -1 else max(log_path.find('rep'), 0) - 1
    dir = os.path.join(args.savedir, params.task[:-3])
    if not os.path.exists(dir):
        os.makedirs(dir)
    if args.amb_init:
        filename = os.path.join(dir, log_path[start:end]) + '_amb_init.csv'
    else:
        filename = os.path.join(dir, log_path[start:end]) + '.csv'
    row = [log_path[start:]]
    for k in log_keys:
        print(f'{k}: {result[k]}')
        row.append(result[k])
    assert len(row) == len(log_keys) + 1

    fields = []
    get_name = lambda x: x[x.find('/') + 1:]
    if not os.path.exists(filename):
        fields = ['exp_name'] + [get_name(metric) for metric in log_keys]
    
    with open(filename, 'a') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
            
        # writing the fields  
        if len(fields) > 0:
            csvwriter.writerow(fields)  
        # writing the data rows  
        csvwriter.writerow(row)

    if args.test_final:
        # Load model parameters.
        policy.load(args.logdir, type='final')
        policy.eval()

        # Change max steps for a longer visualization
        collector = Collector(policy, test_envs, num_agents=num_agents,
                            reward_logger=reward_logger,
                            benchmark_logger=benchmark_logger)
        result = collector.collect(n_episode=args.num_episodes)
        collector.close()
        
        log_path = args.logdir 
        # start = log_path.find('log') + 4
        # end = log_path.find('epochs') + 6 if log_path.find('epochs') != -1 else max(log_path.find('rep'), 0) - 1
        start = log_path.rfind('/') + 1
        end = log_path.find('intr') - 1 if log_path.find('intr') != -1 else max(log_path.find('rep'), 0) - 1
        dir = os.path.join(args.savedir, params.task[:-3])
        if not os.path.exists(dir):
            os.makedirs(dir)
        if args.amb_init:
            filename = os.path.join(dir, log_path[start:end]) + '_amb_init_final.csv'
        else:
            filename = os.path.join(dir, log_path[start:end]) + '_final.csv'
        row = [log_path[start:]]
        for k in log_keys:
            print(f'{k}: {result[k]}')
            row.append(result[k])
        assert len(row) == len(log_keys) + 1

        fields = []
        get_name = lambda x: x[x.find('/') + 1:]
        if not os.path.exists(filename):
            fields = ['exp_name'] + [get_name(metric) for metric in log_keys]
        
        with open(filename, 'a') as csvfile:  
            # creating a csv writer object  
            csvwriter = csv.writer(csvfile)  
                
            # writing the fields  
            if len(fields) > 0:
                csvwriter.writerow(fields)  
            # writing the data rows  
            csvwriter.writerow(row)
    

if __name__ == '__main__':
    evaluate_multi_sacd()
